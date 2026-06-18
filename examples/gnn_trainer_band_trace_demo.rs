// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Runs a small `ModuleTrainer::train_epochs_loader()` pass and exports
//! band-labelled GNN replay telemetry produced during `backward_bands()`.

#[path = "_shared/backend.rs"]
mod backend;

use serde_json::{json, Value};
use st_core::plugin::{
    global_registry, PluginEvent, PluginEventJsonlWriter, PluginEventJsonlWriterConfig,
};
use st_core::telemetry::xai::{GraphFlowTracer, GraphLayerReport, GraphRoundtableTrace};
use st_nn::{
    Dataset, EpochStats, GraphActivation, GraphBatchReadoutErrorTrace, GraphBatchReadoutTrace,
    GraphContext, GraphLayerSpec, GraphReadout, MeanSquaredError, ModuleTrainer,
    NeighborhoodAggregation, RoundtableConfig, Tensor, TensorError, TrainingRunConfig,
    ZSpaceGraphBatchRegressor, ZSpaceGraphNetworkBuilder,
};
use std::collections::BTreeMap;
use std::env;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_NODES: usize = 4;
const DEFAULT_FEATURES: usize = 2;

#[derive(Debug, Clone)]
struct Args {
    run_dir: PathBuf,
    trace_json: Option<PathBuf>,
    events: Option<PathBuf>,
    backend: String,
    epochs: usize,
    patience: usize,
    train_graphs: usize,
    validation_graphs: usize,
    batch_size: usize,
    nodes: usize,
    features: usize,
    seed: u64,
    learning_rate: f32,
    curvature: f32,
    top_k: u32,
    mid_k: u32,
    bottom_k: u32,
    here_tolerance: f32,
}

impl Args {
    fn parse() -> st_nn::PureResult<Self> {
        let mut argv = env::args().skip(1).peekable();
        let mut args = Self {
            run_dir: default_run_dir(),
            trace_json: None,
            events: None,
            backend: "auto".to_string(),
            epochs: 4,
            patience: 2,
            train_graphs: 12,
            validation_graphs: 4,
            batch_size: 2,
            nodes: DEFAULT_NODES,
            features: DEFAULT_FEATURES,
            seed: 2025,
            learning_rate: 0.05,
            curvature: -1.0,
            top_k: 1,
            mid_k: 1,
            bottom_k: 1,
            here_tolerance: 1e-5,
        };

        while let Some(flag) = argv.next() {
            match flag.as_str() {
                "--run-dir" => args.run_dir = PathBuf::from(take_arg(&mut argv, "--run-dir")?),
                "--trace-json" => {
                    args.trace_json = Some(PathBuf::from(take_arg(&mut argv, "--trace-json")?))
                }
                "--events" => args.events = Some(PathBuf::from(take_arg(&mut argv, "--events")?)),
                "--backend" => args.backend = take_arg(&mut argv, "--backend")?,
                "--epochs" => args.epochs = take_parse(&mut argv, "--epochs")?,
                "--patience" => args.patience = take_parse(&mut argv, "--patience")?,
                "--train-graphs" => args.train_graphs = take_parse(&mut argv, "--train-graphs")?,
                "--validation-graphs" => {
                    args.validation_graphs = take_parse(&mut argv, "--validation-graphs")?
                }
                "--batch" => args.batch_size = take_parse(&mut argv, "--batch")?,
                "--nodes" => args.nodes = take_parse(&mut argv, "--nodes")?,
                "--features" => args.features = take_parse(&mut argv, "--features")?,
                "--seed" => args.seed = take_parse(&mut argv, "--seed")?,
                "--lr" => args.learning_rate = take_parse(&mut argv, "--lr")?,
                "--curvature" => args.curvature = take_parse(&mut argv, "--curvature")?,
                "--top-k" => args.top_k = take_parse(&mut argv, "--top-k")?,
                "--mid-k" => args.mid_k = take_parse(&mut argv, "--mid-k")?,
                "--bottom-k" => args.bottom_k = take_parse(&mut argv, "--bottom-k")?,
                "--here-tolerance" => {
                    args.here_tolerance = take_parse(&mut argv, "--here-tolerance")?
                }
                "--help" | "-h" => {
                    return Err(TensorError::Generic(usage().to_string()));
                }
                other => {
                    return Err(TensorError::Generic(format!(
                        "unknown flag: {other}. Try --help"
                    )));
                }
            }
        }

        if args.epochs == 0
            || args.train_graphs == 0
            || args.validation_graphs == 0
            || args.batch_size == 0
            || args.nodes == 0
            || args.features == 0
        {
            return Err(TensorError::InvalidValue {
                label: "gnn_trace_invalid_dims",
            });
        }
        if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "gnn_trace_learning_rate",
            });
        }
        if !args.curvature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "gnn_trace_curvature",
                value: args.curvature,
            });
        }
        if args.top_k == 0 || args.mid_k == 0 || args.bottom_k == 0 {
            return Err(TensorError::InvalidValue {
                label: "gnn_trace_roundtable_k",
            });
        }
        if !args.here_tolerance.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "gnn_trace_here_tolerance",
                value: args.here_tolerance,
            });
        }
        if args.here_tolerance < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "gnn_trace_here_tolerance",
            });
        }
        Ok(args)
    }
}

fn usage() -> &'static str {
    "usage: cargo run -p st-nn --example gnn_trainer_band_trace_demo -- \
     [--run-dir PATH] [--trace-json PATH] [--events PATH] \
     [--backend auto|wgpu|cuda|hip|cpu] [--epochs N] [--patience N] \
     [--train-graphs N] [--validation-graphs N] [--batch N] \
     [--nodes N] [--features N] [--seed N] [--lr F] [--curvature F] \
     [--top-k N] [--mid-k N] [--bottom-k N] [--here-tolerance F]"
}

fn take_arg<I>(argv: &mut std::iter::Peekable<I>, flag: &str) -> st_nn::PureResult<String>
where
    I: Iterator<Item = String>,
{
    argv.next()
        .ok_or_else(|| TensorError::Generic(format!("{flag} requires a value")))
}

fn take_parse<I, T>(argv: &mut std::iter::Peekable<I>, flag: &str) -> st_nn::PureResult<T>
where
    I: Iterator<Item = String>,
    T: std::str::FromStr,
{
    let raw = take_arg(argv, flag)?;
    raw.parse::<T>()
        .map_err(|_| TensorError::Generic(format!("{flag} could not parse value '{raw}'")))
}

fn default_run_dir() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    PathBuf::from(format!("models/runs/gnn_band_trace_probe/{stamp}"))
}

fn nonzero(value: usize, label: &'static str) -> st_nn::PureResult<NonZeroUsize> {
    NonZeroUsize::new(value)
        .ok_or(TensorError::InvalidDimensions {
            rows: 1,
            cols: value,
        })
        .map_err(|_| TensorError::InvalidValue { label })
}

fn build_context(nodes: usize) -> st_nn::PureResult<GraphContext> {
    let mut adjacency = vec![0.0f32; nodes * nodes];
    for idx in 0..nodes {
        if idx + 1 < nodes {
            adjacency[idx * nodes + (idx + 1)] = 1.0;
            adjacency[(idx + 1) * nodes + idx] = 1.0;
        }
    }
    GraphContext::from_adjacency(Tensor::from_vec(nodes, nodes, adjacency)?)
}

fn build_model(
    context: GraphContext,
    tracer: Arc<Mutex<GraphFlowTracer>>,
    nodes: usize,
    features: usize,
    curvature: f32,
    learning_rate: f32,
) -> st_nn::PureResult<ZSpaceGraphBatchRegressor> {
    let mut builder = ZSpaceGraphNetworkBuilder::new(
        context,
        nonzero(features, "gnn_trace_features")?,
        curvature,
        learning_rate,
    )
    .with_tracer(tracer);
    builder.push_layer(
        GraphLayerSpec::new(nonzero(features * 2, "gnn_trace_hidden")?).with_aggregation(
            NeighborhoodAggregation::multi_hop_sum(nonzero(2, "gnn_trace_hops")?)
                .with_include_self(true)
                .with_attenuation(0.6),
        ),
    );
    builder.push_layer(
        GraphLayerSpec::new(nonzero(features, "gnn_trace_output")?)
            .with_activation(GraphActivation::Relu),
    );
    let network = builder.build("trainer_band_trace")?;
    Ok(ZSpaceGraphBatchRegressor::new(
        network,
        GraphReadout::Mean,
        nonzero(nodes, "gnn_trace_nodes")?,
    ))
}

fn graph_sample(
    context: &GraphContext,
    nodes: usize,
    features: usize,
    seed: u64,
) -> st_nn::PureResult<(Tensor, Tensor)> {
    let input = Tensor::random_uniform(nodes, features, -1.0, 1.0, Some(seed))?;
    let propagated = context.propagate(&input)?;
    let node_target = input.scale(0.25)?.add(&propagated.scale(0.75)?)?;
    let graph_target = GraphReadout::Mean.forward(&node_target)?;
    Ok((input, graph_target))
}

fn graph_samples(
    context: &GraphContext,
    nodes: usize,
    features: usize,
    seed: u64,
    count: usize,
) -> st_nn::PureResult<Vec<(Tensor, Tensor)>> {
    (0..count)
        .map(|idx| graph_sample(context, nodes, features, seed + idx as u64))
        .collect()
}

fn roundtable_trace_json(trace: &GraphRoundtableTrace) -> Value {
    json!({
        "signal": {
            "above": trace.signal.above,
            "here": trace.signal.here,
            "beneath": trace.signal.beneath,
            "drift": trace.signal.drift,
            "band_sizes": [
                trace.signal.band_sizes.0,
                trace.signal.band_sizes.1,
                trace.signal.band_sizes.2,
            ],
            "sheet_index": trace.signal.sheet_index,
            "sheet_confidence": trace.signal.sheet_confidence,
            "curvature": trace.signal.curvature,
            "spin": trace.signal.spin,
            "energy": trace.signal.energy,
        },
        "influence": {
            "above_multiplier": trace.influence.above_multiplier,
            "here_multiplier": trace.influence.here_multiplier,
            "beneath_multiplier": trace.influence.beneath_multiplier,
            "drift_bias": trace.influence.drift_bias,
        },
        "aggregation": {
            "base_coefficients": trace.aggregation.base_coefficients,
            "step_scales": trace.aggregation.step_scales,
            "band_pass_scales": trace.aggregation.band_pass_scales,
            "effective_coefficients": trace.aggregation.effective_coefficients,
        },
        "band_pass": trace.band_pass.as_ref().map(|pass| json!({
            "band": pass.band.as_str(),
            "gradient_l1": pass.gradient_l1,
            "gradient_l2": pass.gradient_l2,
            "gradient_rms": pass.gradient_rms,
        })),
    })
}

fn report_json(report: &GraphLayerReport) -> Value {
    json!({
        "layer": report.layer,
        "curvature": report.curvature,
        "total_flow_energy": report.total_flow_energy(),
        "weight_update_magnitude": report.weight_update_magnitude,
        "bias_update_magnitude": report.bias_update_magnitude,
        "node_flows": report.node_flows.iter().map(|flow| {
            json!({
                "node_index": flow.node_index,
                "incoming_weight": flow.incoming_weight,
                "aggregated_norm": flow.aggregated_norm,
                "energy": flow.energy(),
            })
        }).collect::<Vec<_>>(),
        "roundtable": report.roundtable.as_ref().map(roundtable_trace_json),
    })
}

fn signal_json(signal: &st_nn::RoundtableBandSignal) -> Value {
    json!({
        "above": signal.energy().above,
        "here": signal.energy().here,
        "beneath": signal.energy().beneath,
        "drift": signal.energy().drift,
        "band_sizes": [signal.band_sizes().0, signal.band_sizes().1, signal.band_sizes().2],
        "spectral": {
            "sheet_index": signal.energy().spectral.sheet_index,
            "sheet_confidence": signal.energy().spectral.sheet_confidence,
            "curvature": signal.energy().spectral.curvature,
            "spin": signal.energy().spectral.spin,
            "energy": signal.energy().spectral.energy,
        }
    })
}

fn epoch_stats_json(stats: &EpochStats) -> Value {
    json!({
        "batches": stats.batches,
        "total_loss": stats.total_loss,
        "average_loss": stats.average_loss,
        "tensor_backend": stats.tensor_backend,
    })
}

fn readout_trace_json(trace: &GraphBatchReadoutTrace) -> Value {
    json!({
        "readout": trace.readout.as_str(),
        "graph_count": trace.graph_count,
        "total_rows": trace.total_rows,
        "output_shape": [trace.output_shape.0, trace.output_shape.1],
        "entries": trace.entries.iter().map(|entry| json!({
            "graph_index": entry.graph_index,
            "row_start": entry.row_start,
            "row_end": entry.row_end,
            "node_l2": entry.node_l2,
            "prediction_l2": entry.prediction_l2,
        })).collect::<Vec<_>>(),
    })
}

fn readout_error_json(trace: &GraphBatchReadoutErrorTrace) -> Value {
    json!({
        "graph_count": trace.graph_count,
        "output_shape": [trace.output_shape.0, trace.output_shape.1],
        "mean_squared_error": trace.mean_squared_error,
        "target_mean_square": trace.target_mean_square,
        "normalized_mean_squared_error": trace.normalized_mean_squared_error,
        "entries": trace.entries.iter().map(|entry| json!({
            "graph_index": entry.graph_index,
            "prediction_l2": entry.prediction_l2,
            "target_l2": entry.target_l2,
            "target_mean_square": entry.target_mean_square,
            "residual_l2": entry.residual_l2,
            "mean_squared_error": entry.mean_squared_error,
            "normalized_mean_squared_error": entry.normalized_mean_squared_error,
        })).collect::<Vec<_>>(),
    })
}

fn normalized_mean_squared_error(mse: f32, target_mean_square: f32) -> Option<f32> {
    if !mse.is_finite() || !target_mean_square.is_finite() {
        return None;
    }
    if target_mean_square > f32::EPSILON {
        return Some(mse / target_mean_square);
    }
    if mse.abs() <= f32::EPSILON {
        return Some(0.0);
    }
    None
}

fn validation_readout_json(
    model: &ZSpaceGraphBatchRegressor,
    validation_samples: &[(Tensor, Tensor)],
    batch_size: usize,
) -> st_nn::PureResult<Value> {
    let mut batches = Vec::new();
    let mut graph_count = 0usize;
    let mut total_rows = 0usize;
    let mut weighted_mse = 0.0f32;
    let mut weighted_target_mean_square = 0.0f32;

    for (batch_index, chunk) in validation_samples.chunks(batch_size).enumerate() {
        let input = Tensor::cat_rows(
            &chunk
                .iter()
                .map(|(input, _)| input.clone())
                .collect::<Vec<_>>(),
        )?;
        let target = Tensor::cat_rows(
            &chunk
                .iter()
                .map(|(_, target)| target.clone())
                .collect::<Vec<_>>(),
        )?;
        let (prediction, readout_trace) = model.forward_with_trace(&input)?;
        let readout_error = readout_trace.compare_predictions(&prediction, &target)?;
        graph_count += readout_error.graph_count;
        total_rows += readout_trace.total_rows;
        weighted_mse += readout_error.mean_squared_error * readout_error.graph_count as f32;
        weighted_target_mean_square +=
            readout_error.target_mean_square * readout_error.graph_count as f32;
        batches.push(json!({
            "batch_index": batch_index,
            "input_shape": [input.shape().0, input.shape().1],
            "target_shape": [target.shape().0, target.shape().1],
            "prediction_shape": [prediction.shape().0, prediction.shape().1],
            "prediction_l2": prediction.squared_l2_norm().sqrt(),
            "trace": readout_trace_json(&readout_trace),
            "error": readout_error_json(&readout_error),
        }));
    }

    let mean_squared_error = if graph_count == 0 {
        0.0
    } else {
        weighted_mse / graph_count as f32
    };
    let target_mean_square = if graph_count == 0 {
        0.0
    } else {
        weighted_target_mean_square / graph_count as f32
    };
    Ok(json!({
        "batch_count": batches.len(),
        "graph_count": graph_count,
        "total_rows": total_rows,
        "mean_squared_error": mean_squared_error,
        "target_mean_square": target_mean_square,
        "normalized_mean_squared_error": normalized_mean_squared_error(
            mean_squared_error,
            target_mean_square,
        ),
        "batches": batches,
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    std::fs::create_dir_all(&args.run_dir)?;
    std::fs::write(
        args.run_dir.join("command.txt"),
        env::args().collect::<Vec<_>>().join(" "),
    )?;

    let trace_path = args
        .trace_json
        .clone()
        .unwrap_or_else(|| args.run_dir.join("gnn_band_trace.json"));
    if let Some(parent) = trace_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let events_path = args
        .events
        .clone()
        .unwrap_or_else(|| args.run_dir.join("trainer_trace.jsonl"));
    if let Some(parent) = events_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let backend_sel = backend::parse_backend(Some(args.backend.as_str()))?;
    let backend_runtime = backend::prepare_backend_runtime(&backend_sel)?;
    let _events_writer = PluginEventJsonlWriter::subscribe(
        global_registry().event_bus().clone(),
        &events_path,
        PluginEventJsonlWriterConfig::default(),
    )?;
    global_registry()
        .event_bus()
        .publish(&PluginEvent::BackendChanged {
            backend: backend_sel.label.clone(),
        });

    let context = build_context(args.nodes)?;
    let train_samples = graph_samples(
        &context,
        args.nodes,
        args.features,
        args.seed,
        args.train_graphs,
    )?;
    let validation_samples = graph_samples(
        &context,
        args.nodes,
        args.features,
        args.seed + 10_000,
        args.validation_graphs,
    )?;
    let train_loader = Dataset::from_vec(train_samples)
        .loader()
        .shuffle(args.seed)
        .batched(args.batch_size)
        .prefetch(2);
    let validation_loader = Dataset::from_vec(validation_samples.clone())
        .loader()
        .batched(args.batch_size);

    let tracer = Arc::new(Mutex::new(GraphFlowTracer::new()));
    let mut model = build_model(
        context.clone(),
        tracer.clone(),
        args.nodes,
        args.features,
        args.curvature,
        args.learning_rate,
    )?;
    let mut trainer = ModuleTrainer::new(
        backend_sel.caps,
        args.curvature,
        args.learning_rate,
        args.learning_rate * 0.2,
    );
    trainer.prepare(&mut model)?;
    let schedule = trainer.roundtable(
        args.batch_size as u32,
        args.features as u32,
        RoundtableConfig::default()
            .with_top_k(args.top_k)
            .with_mid_k(args.mid_k)
            .with_bottom_k(args.bottom_k)
            .with_here_tolerance(args.here_tolerance),
    );
    let tensor_policy = backend::tensor_backend_policy_meta(backend_sel.caps);
    let roundtable_backend_audit = backend::roundtable_backend_audit(backend_sel.caps, &schedule);

    let mut loss = MeanSquaredError::new();
    let training = trainer.train_epochs_loader(
        &mut model,
        &mut loss,
        &train_loader,
        Some(&validation_loader),
        &schedule,
        TrainingRunConfig::new(args.epochs)
            .with_validation_patience(Some(args.patience))
            .with_min_delta(1e-5)
            .with_epoch_shuffle_seed(Some(args.seed))
            .with_restore_best(true),
    )?;

    let signal = trainer
        .gnn_roundtable_signal()
        .expect("trainer should emit a roundtable signal");
    let reports = tracer
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .drain();

    let probe_graphs = validation_samples.len().min(args.batch_size);
    let probe_input = Tensor::cat_rows(
        &validation_samples
            .iter()
            .take(probe_graphs)
            .map(|(input, _)| input.clone())
            .collect::<Vec<_>>(),
    )?;
    let probe_target = Tensor::cat_rows(
        &validation_samples
            .iter()
            .take(probe_graphs)
            .map(|(_, target)| target.clone())
            .collect::<Vec<_>>(),
    )?;
    let (graph_prediction, readout_trace) = model.forward_with_trace(&probe_input)?;
    let readout_error = readout_trace.compare_predictions(&graph_prediction, &probe_target)?;
    let validation_readout = validation_readout_json(&model, &validation_samples, args.batch_size)?;

    let mut by_band = BTreeMap::<String, Vec<Value>>::new();
    for report in &reports {
        let Some(trace) = report.roundtable.as_ref() else {
            continue;
        };
        let Some(pass) = trace.band_pass.as_ref() else {
            continue;
        };
        by_band
            .entry(pass.band.as_str().to_string())
            .or_default()
            .push(json!({
                "layer": report.layer,
                "gradient_l1": pass.gradient_l1,
                "gradient_l2": pass.gradient_l2,
                "gradient_rms": pass.gradient_rms,
                "base_coefficients": trace.aggregation.base_coefficients,
                "effective_coefficients": trace.aggregation.effective_coefficients,
                "step_scales": trace.aggregation.step_scales,
                "band_pass_scales": trace.aggregation.band_pass_scales,
                "total_flow_energy": report.total_flow_energy(),
            }));
    }

    let payload = json!({
        "schema": "st.gnn.band_trace.v2",
        "run": {
            "run_dir": args.run_dir,
            "trace_json": trace_path,
            "events_path": events_path,
            "backend": backend_sel.label.clone(),
            "device_caps": backend::DeviceCapsMeta::from(backend_sel.caps),
            "backend_runtime": backend_runtime,
            "tensor_policy": tensor_policy,
            "roundtable_backend_audit": roundtable_backend_audit,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "train_graphs": args.train_graphs,
            "validation_graphs": args.validation_graphs,
            "batch_size": args.batch_size,
            "nodes": args.nodes,
            "features": args.features,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "curvature": args.curvature,
            "roundtable": {
                "top_k": args.top_k,
                "mid_k": args.mid_k,
                "bottom_k": args.bottom_k,
                "here_tolerance": args.here_tolerance,
            },
        },
        "trainer": {
            "epochs_run": training.epochs_run(),
            "best_epoch": training.best_epoch().map(|epoch| epoch.epoch),
            "best_score": training.best_epoch().map(|epoch| epoch.score),
            "last_epoch": training.last_epoch().map(|epoch| epoch.epoch),
            "stopped_early": training.stopped_early,
            "restored_best": training.restored_best,
            "history": training.epochs.iter().map(|epoch| json!({
                "epoch": epoch.epoch,
                "train": epoch_stats_json(&epoch.train),
                "validation": epoch.validation.as_ref().map(epoch_stats_json),
                "score": epoch.score,
                "improved": epoch.improved,
            })).collect::<Vec<_>>(),
        },
        "signal": signal_json(&signal),
        "readout": {
            "prediction_shape": [graph_prediction.shape().0, graph_prediction.shape().1],
            "prediction_l2": graph_prediction.squared_l2_norm().sqrt(),
            "trace": readout_trace_json(&readout_trace),
            "error": readout_error_json(&readout_error),
        },
        "validation_readout": validation_readout,
        "reports": reports.iter().map(report_json).collect::<Vec<_>>(),
        "band_replays": by_band,
    });

    std::fs::write(&trace_path, serde_json::to_string_pretty(&payload)?)?;
    std::fs::write(
        args.run_dir.join("run.json"),
        serde_json::to_string_pretty(&payload["run"])?,
    )?;

    println!("trace_json={}", trace_path.display());
    println!(
        "events_jsonl={}",
        payload["run"]["events_path"].as_str().unwrap_or("")
    );
    println!(
        "trainer: epochs_run={} best_epoch={:?} best_score={:?} restored_best={}",
        training.epochs_run(),
        training.best_epoch().map(|epoch| epoch.epoch),
        training.best_epoch().map(|epoch| epoch.score),
        training.restored_best
    );
    println!(
        "readout: graph_count={} mean_mse={:.6} prediction_shape={:?}",
        readout_trace.graph_count,
        readout_error.mean_squared_error,
        graph_prediction.shape()
    );
    println!(
        "validation_readout: graph_count={} mean_mse={:.6} nmse={:.6}",
        payload["validation_readout"]["graph_count"]
            .as_u64()
            .unwrap_or(0),
        payload["validation_readout"]["mean_squared_error"]
            .as_f64()
            .unwrap_or(0.0),
        payload["validation_readout"]["normalized_mean_squared_error"]
            .as_f64()
            .unwrap_or(0.0)
    );
    println!(
        "signal: above={:.4} here={:.4} beneath={:.4} drift={:.4}",
        signal.energy().above,
        signal.energy().here,
        signal.energy().beneath,
        signal.energy().drift
    );
    for (band, entries) in &by_band {
        let shown = entries.len().min(4);
        println!("band={} entries={} showing={}", band, entries.len(), shown);
        for entry in entries.iter().take(shown) {
            println!(
                "  layer={} grad_l2={:.6} effective={}",
                entry["layer"].as_str().unwrap_or("layer"),
                entry["gradient_l2"].as_f64().unwrap_or(0.0),
                entry["effective_coefficients"],
            );
        }
    }

    Ok(())
}
