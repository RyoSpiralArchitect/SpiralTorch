// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Sequence model-zoo: legacy LSTM backend debt probe.

#[path = "_shared/backend.rs"]
mod backend;

use serde_json::{json, Value};
use st_core::backend::device_caps::DeviceCaps;
use st_core::plugin::{
    global_registry, PluginEvent, PluginEventJsonlWriter, PluginEventJsonlWriterConfig,
};
use st_nn::{
    load_json, push_backend_policy, save_json, BackendPolicy, EpochStats, Linear, Loss, Lstm,
    MeanSquaredError, Module, ModuleTrainer, Relu, RoundtableConfig, Sequential, Tensor,
    TensorError,
};
use std::env;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct Args {
    run_dir: PathBuf,
    events: Option<PathBuf>,
    backend: String,
    epochs: usize,
    batches_per_epoch: usize,
    steps: usize,
    hidden: usize,
    seed: u64,
    learning_rate: f32,
    curvature: f32,
}

impl Args {
    fn parse() -> st_nn::PureResult<Self> {
        let mut argv = env::args().skip(1).peekable();
        let mut args = Self {
            run_dir: default_run_dir(),
            events: None,
            backend: "auto".to_string(),
            epochs: 3,
            batches_per_epoch: 3,
            steps: 10,
            hidden: 6,
            seed: 404,
            learning_rate: 1.5e-2,
            curvature: -1.0,
        };

        while let Some(flag) = argv.next() {
            match flag.as_str() {
                "--run-dir" => args.run_dir = PathBuf::from(take_arg(&mut argv, "--run-dir")?),
                "--events" => args.events = Some(PathBuf::from(take_arg(&mut argv, "--events")?)),
                "--backend" => args.backend = take_arg(&mut argv, "--backend")?,
                "--epochs" => args.epochs = take_parse(&mut argv, "--epochs")?,
                "--batches" => args.batches_per_epoch = take_parse(&mut argv, "--batches")?,
                "--steps" => args.steps = take_parse(&mut argv, "--steps")?,
                "--hidden" => args.hidden = take_parse(&mut argv, "--hidden")?,
                "--seed" => args.seed = take_parse(&mut argv, "--seed")?,
                "--lr" => args.learning_rate = take_parse(&mut argv, "--lr")?,
                "--curvature" => args.curvature = take_parse(&mut argv, "--curvature")?,
                "--help" | "-h" => return Err(TensorError::Generic(usage().to_string())),
                other => {
                    return Err(TensorError::Generic(format!(
                        "unknown flag: {other}. Try --help"
                    )));
                }
            }
        }

        if args.epochs == 0 || args.batches_per_epoch == 0 || args.steps == 0 || args.hidden == 0 {
            return Err(TensorError::InvalidValue {
                label: "lstm_probe_invalid_dims",
            });
        }
        if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "lstm_probe_learning_rate",
            });
        }
        if !args.curvature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lstm_probe_curvature",
                value: args.curvature,
            });
        }
        Ok(args)
    }
}

fn usage() -> &'static str {
    "usage: cargo run -p st-nn --example modelzoo_lstm_sequence_probe -- \
     [--run-dir PATH] [--events PATH] [--backend auto|wgpu|cuda|hip|cpu] \
     [--epochs N] [--batches N] [--steps N] [--hidden N] [--seed N] \
     [--lr F] [--curvature F]"
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
    PathBuf::from(format!("models/runs/lstm_sequence_probe/{stamp}"))
}

fn build_model(hidden: usize) -> st_nn::PureResult<Sequential> {
    let mut model = Sequential::new();
    model.push(Lstm::new("lstm_probe", 1, hidden)?);
    model.push(Relu::new());
    model.push(Linear::new("head", hidden, 1)?);
    Ok(model)
}

fn signal_value(seed: u64, batch_idx: usize, step: usize) -> f32 {
    let phase = seed as f32 * 0.013 + batch_idx as f32 * 0.37 + step as f32 * 0.29;
    let trend = ((batch_idx + step) % 7) as f32 * 0.015;
    0.6 * phase.sin() + 0.3 * (phase * 0.43).cos() + trend
}

fn build_batch(steps: usize, seed: u64, batch_idx: usize) -> st_nn::PureResult<(Tensor, Tensor)> {
    let input = Tensor::from_fn(steps, 1, |step, _| signal_value(seed, batch_idx, step))?;
    let target = Tensor::from_fn(steps, 1, |step, _| {
        let here = signal_value(seed, batch_idx, step);
        let next = signal_value(seed, batch_idx, step + 1);
        0.7 * next + 0.2 * here + 0.1 * (next - here)
    })?;
    Ok((input, target))
}

fn build_epoch(
    batches_per_epoch: usize,
    steps: usize,
    seed: u64,
) -> st_nn::PureResult<Vec<(Tensor, Tensor)>> {
    (0..batches_per_epoch)
        .map(|idx| build_batch(steps, seed.wrapping_add(idx as u64 * 101), idx))
        .collect()
}

fn evaluate_epoch_loss_with_policies(
    model: &dyn Module,
    batches: &[(Tensor, Tensor)],
    forward_policy: BackendPolicy,
    loss_policy: BackendPolicy,
) -> st_nn::PureResult<Option<f32>> {
    if batches.is_empty() {
        return Ok(None);
    }
    let mut loss = MeanSquaredError::new();
    let mut total = 0.0f32;
    for (input, target) in batches {
        let prediction = {
            let _policy_guard = push_backend_policy(forward_policy);
            model.forward(input)?
        };
        let loss_value = {
            let _policy_guard = push_backend_policy(loss_policy);
            loss.forward(&prediction, target)?
        };
        total += loss_value.data().iter().copied().sum::<f32>();
    }
    Ok(Some(total / batches.len() as f32))
}

fn epoch_stats_json(stats: &EpochStats) -> Value {
    json!({
        "batches": stats.batches,
        "total_loss": stats.total_loss,
        "average_loss": stats.average_loss,
        "tensor_backend": stats.tensor_backend,
    })
}

fn run_json_path(run_dir: &PathBuf, name: &str) -> String {
    run_dir.join(name).display().to_string()
}

fn run() -> st_nn::PureResult<()> {
    let args = Args::parse()?;
    std::fs::create_dir_all(&args.run_dir).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    std::fs::write(
        args.run_dir.join("command.txt"),
        env::args().collect::<Vec<_>>().join(" "),
    )
    .map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;

    let backend_sel = backend::parse_backend(Some(args.backend.as_str()))?;
    let backend_runtime = backend::prepare_backend_runtime(&backend_sel)?;
    let events_path = args
        .events
        .clone()
        .unwrap_or_else(|| args.run_dir.join("trainer_trace.jsonl"));
    if let Some(parent) = events_path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;
    }
    let mut trainer = ModuleTrainer::new(
        backend_sel.caps,
        args.curvature,
        args.learning_rate,
        args.learning_rate,
    );
    let schedule = trainer.roundtable(
        args.steps as u32,
        1,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );
    let tensor_policy = backend::tensor_backend_policy_meta(backend_sel.caps);
    let roundtable_backend_audit = backend::roundtable_backend_audit(backend_sel.caps, &schedule);

    let mut model = build_model(args.hidden)?;
    model.attach_hypergrad(args.curvature, args.learning_rate)?;

    let epochs = (0..args.epochs)
        .map(|idx| {
            build_epoch(
                args.batches_per_epoch,
                args.steps,
                args.seed.wrapping_add(idx as u64 * 1_000),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
    let selected_policy = BackendPolicy::from_device_caps(backend_sel.caps);
    let pretrain_cpu_reference_loss =
        evaluate_epoch_loss_with_policies(&model, &epochs[0], cpu_policy, cpu_policy)?;
    let pretrain_selected_forward_cpu_loss =
        evaluate_epoch_loss_with_policies(&model, &epochs[0], selected_policy, cpu_policy)?;
    let pretrain_cpu_forward_selected_loss =
        evaluate_epoch_loss_with_policies(&model, &epochs[0], cpu_policy, selected_policy)?;
    let pretrain_loss =
        evaluate_epoch_loss_with_policies(&model, &epochs[0], selected_policy, selected_policy)?;
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

    let mut loss = MeanSquaredError::new();
    let mut first_loss = None;
    let mut last_loss = None;
    let mut epochs_json = Vec::with_capacity(args.epochs);
    for (epoch_idx, epoch) in epochs.into_iter().enumerate() {
        let stats = trainer.train_epoch(&mut model, &mut loss, epoch, &schedule)?;
        if first_loss.is_none() {
            first_loss = Some(stats.average_loss);
        }
        last_loss = Some(stats.average_loss);
        println!(
            "epoch={} batches={} avg_loss={:.6} tensor_wgpu={} tensor_cpu={} tensor_naive={}",
            epoch_idx + 1,
            stats.batches,
            stats.average_loss,
            stats.tensor_backend.backend_wgpu,
            stats.tensor_backend.backend_cpu,
            stats.tensor_backend.backend_naive,
        );
        epochs_json.push(json!({
            "epoch": epoch_idx + 1,
            "stats": epoch_stats_json(&stats),
        }));
    }

    let weights_path = args.run_dir.join("weights.json");
    save_json(&model, &weights_path)?;
    let mut reloaded = build_model(args.hidden)?;
    load_json(&mut reloaded, &weights_path)?;
    let (sanity_x, _) = build_batch(args.steps, args.seed.wrapping_add(999_999), 0)?;
    let sanity_output = {
        let _policy_guard = push_backend_policy(selected_policy);
        reloaded.forward(&sanity_x)?
    };

    let payload = json!({
        "schema": "st.sequence.lstm_probe.v1",
        "run": {
            "run_dir": args.run_dir.display().to_string(),
            "events_path": events_path.display().to_string(),
            "weights_path": weights_path.display().to_string(),
            "backend": backend_sel.label.clone(),
            "device_caps": backend::DeviceCapsMeta::from(backend_sel.caps),
            "backend_runtime": backend_runtime,
            "tensor_policy": tensor_policy,
            "roundtable_backend_audit": roundtable_backend_audit,
            "epochs": args.epochs,
            "batches_per_epoch": args.batches_per_epoch,
            "steps": args.steps,
            "hidden": args.hidden,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "curvature": args.curvature,
        },
        "summary": {
            "pretrain_loss": pretrain_loss,
            "pretrain_cpu_reference_loss": pretrain_cpu_reference_loss,
            "pretrain_selected_forward_cpu_loss": pretrain_selected_forward_cpu_loss,
            "pretrain_cpu_forward_selected_loss": pretrain_cpu_forward_selected_loss,
            "pretrain_backend_gap": match (pretrain_loss, pretrain_cpu_reference_loss) {
                (Some(selected), Some(cpu_reference)) => Some(selected - cpu_reference),
                _ => None,
            },
            "pretrain_forward_gap": match (pretrain_selected_forward_cpu_loss, pretrain_cpu_reference_loss) {
                (Some(selected_forward), Some(cpu_reference)) => Some(selected_forward - cpu_reference),
                _ => None,
            },
            "pretrain_loss_gap": match (pretrain_cpu_forward_selected_loss, pretrain_cpu_reference_loss) {
                (Some(selected_loss), Some(cpu_reference)) => Some(selected_loss - cpu_reference),
                _ => None,
            },
            "first_loss": first_loss,
            "last_loss": last_loss,
            "loss_delta": match (first_loss, last_loss) {
                (Some(first), Some(last)) => Some(last - first),
                _ => None,
            },
            "sanity_output_shape": [sanity_output.shape().0, sanity_output.shape().1],
            "sanity_output_l2": sanity_output.squared_l2_norm().sqrt(),
        },
        "epochs": epochs_json,
    });
    let trace_path = args.run_dir.join("lstm_trace.json");
    let trace_json =
        serde_json::to_string_pretty(&payload).map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;
    std::fs::write(&trace_path, trace_json).map_err(|err| TensorError::IoError {
        message: err.to_string(),
    })?;
    let run_json =
        serde_json::to_string_pretty(&payload["run"]).map_err(|err| TensorError::IoError {
            message: err.to_string(),
        })?;
    std::fs::write(args.run_dir.join("run.json"), run_json).map_err(|err| {
        TensorError::IoError {
            message: err.to_string(),
        }
    })?;

    println!(
        "trace_json={}",
        run_json_path(&args.run_dir, "lstm_trace.json")
    );
    println!("events_jsonl={}", events_path.display());
    println!(
        "summary: first_loss={} last_loss={} delta={}",
        first_loss
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "-".to_string()),
        last_loss
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "-".to_string()),
        match (first_loss, last_loss) {
            (Some(first), Some(last)) => format!("{:.6}", last - first),
            _ => "-".to_string(),
        }
    );

    Ok(())
}

fn main() -> st_nn::PureResult<()> {
    run()
}
