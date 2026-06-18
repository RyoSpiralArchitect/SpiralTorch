// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Training model-zoo: SpiralLightning + self-supervised InfoNCE (minimal).

#[cfg(feature = "selfsup")]
#[path = "_shared/backend.rs"]
mod backend;

#[cfg(not(feature = "selfsup"))]
fn main() {
    eprintln!("This example requires building `st-nn` with the `selfsup` feature.");
}

#[cfg(feature = "selfsup")]
mod demo {
    use serde_json::{json, Value};
    use st_core::backend::device_caps::DeviceCaps;
    use st_core::distributed::{AccumulatorSyncError, AccumulatorSynchronizer};
    use st_core::plugin::{
        global_registry, PluginEvent, PluginEventJsonlWriter, PluginEventJsonlWriterConfig,
    };
    use st_nn::trainer::selfsup::InfoNCELoss;
    use st_nn::{
        load_json, push_backend_policy, save_json, BackendPolicy, EpochStats, InfoNCEConfig,
        LightningConfig, Linear, Loss, Module, Relu, RoundtableConfig, SelfSupBatch, SelfSupEpoch,
        SelfSupEpochTelemetry, SelfSupObjective, SelfSupStage, Sequential, SpiralLightning,
        SpiralSession, Tensor, TensorError,
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
        pair_batch: usize,
        input_dim: usize,
        embed_dim: usize,
        seed: u64,
        learning_rate: f32,
        curvature: f32,
        temperature: f32,
        normalize: bool,
        accumulator_sync: String,
    }

    impl Args {
        fn parse() -> st_nn::PureResult<Self> {
            let mut argv = env::args().skip(1).peekable();
            let mut args = Self {
                run_dir: default_run_dir(),
                events: None,
                backend: "auto".to_string(),
                epochs: 6,
                batches_per_epoch: 4,
                pair_batch: 4,
                input_dim: 8,
                embed_dim: 6,
                seed: 7_000,
                learning_rate: 2e-2,
                curvature: -1.0,
                temperature: 0.2,
                normalize: true,
                accumulator_sync: "none".to_string(),
            };

            while let Some(flag) = argv.next() {
                match flag.as_str() {
                    "--run-dir" => args.run_dir = PathBuf::from(take_arg(&mut argv, "--run-dir")?),
                    "--events" => {
                        args.events = Some(PathBuf::from(take_arg(&mut argv, "--events")?))
                    }
                    "--backend" => args.backend = take_arg(&mut argv, "--backend")?,
                    "--epochs" => args.epochs = take_parse(&mut argv, "--epochs")?,
                    "--batches" => args.batches_per_epoch = take_parse(&mut argv, "--batches")?,
                    "--pair-batch" => args.pair_batch = take_parse(&mut argv, "--pair-batch")?,
                    "--input-dim" => args.input_dim = take_parse(&mut argv, "--input-dim")?,
                    "--embed-dim" => args.embed_dim = take_parse(&mut argv, "--embed-dim")?,
                    "--seed" => args.seed = take_parse(&mut argv, "--seed")?,
                    "--lr" => args.learning_rate = take_parse(&mut argv, "--lr")?,
                    "--curvature" => args.curvature = take_parse(&mut argv, "--curvature")?,
                    "--temperature" => args.temperature = take_parse(&mut argv, "--temperature")?,
                    "--normalize" => args.normalize = take_bool(&mut argv, "--normalize")?,
                    "--accumulator-sync" => {
                        args.accumulator_sync = take_arg(&mut argv, "--accumulator-sync")?
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
                || args.batches_per_epoch == 0
                || args.pair_batch == 0
                || args.input_dim == 0
                || args.embed_dim == 0
            {
                return Err(TensorError::InvalidValue {
                    label: "selfsup_invalid_dims",
                });
            }
            if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "selfsup_learning_rate",
                });
            }
            if !args.curvature.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "selfsup_curvature",
                    value: args.curvature,
                });
            }
            if !args.temperature.is_finite() || args.temperature <= 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "selfsup_temperature",
                });
            }
            match args.accumulator_sync.as_str() {
                "none" | "local" => {}
                _ => {
                    return Err(TensorError::InvalidValue {
                        label: "selfsup_accumulator_sync",
                    });
                }
            }
            Ok(args)
        }
    }

    fn usage() -> &'static str {
        "usage: cargo run -p st-nn --example modelzoo_lightning_selfsup_minimal -- \
         [--run-dir PATH] [--events PATH] [--backend auto|wgpu|cuda|hip|cpu] \
         [--epochs N] [--batches N] [--pair-batch N] [--input-dim N] \
         [--embed-dim N] [--seed N] [--lr F] [--curvature F] \
         [--temperature F] [--normalize true|false] [--accumulator-sync none|local]"
    }

    #[derive(Debug, Clone, Copy)]
    struct LocalAccumulatorSynchronizer {
        rank: usize,
        world_size: usize,
    }

    impl LocalAccumulatorSynchronizer {
        fn new() -> Self {
            Self {
                rank: 0,
                world_size: 1,
            }
        }
    }

    impl AccumulatorSynchronizer for LocalAccumulatorSynchronizer {
        fn rank(&self) -> usize {
            self.rank
        }

        fn world_size(&self) -> usize {
            self.world_size
        }

        fn synchronize_accumulators(
            &self,
            _gradients: &mut [f32],
        ) -> Result<(), AccumulatorSyncError> {
            Ok(())
        }
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

    fn take_bool<I>(argv: &mut std::iter::Peekable<I>, flag: &str) -> st_nn::PureResult<bool>
    where
        I: Iterator<Item = String>,
    {
        let raw = take_arg(argv, flag)?;
        match raw.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(TensorError::Generic(format!(
                "{flag} expected true|false, got '{raw}'"
            ))),
        }
    }

    fn default_run_dir() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        PathBuf::from(format!("models/runs/lightning_selfsup_minimal/{stamp}"))
    }

    fn build_model(input_dim: usize, embed_dim: usize) -> st_nn::PureResult<Sequential> {
        let mut model = Sequential::new();
        model.push(Linear::new("enc1", input_dim, embed_dim)?);
        model.push(Relu::new());
        model.push(Linear::new("enc2", embed_dim, embed_dim)?);
        Ok(model)
    }

    fn build_epoch(
        pairs_per_epoch: usize,
        pair_batch: usize,
        input_dim: usize,
        seed: u64,
    ) -> st_nn::PureResult<SelfSupEpoch> {
        let mut batches = Vec::with_capacity(pairs_per_epoch);
        for idx in 0..pairs_per_epoch {
            let anchors = Tensor::random_uniform(
                pair_batch,
                input_dim,
                -1.0,
                1.0,
                Some(seed.wrapping_add(idx as u64)),
            )?;
            let jitter = Tensor::random_uniform(
                pair_batch,
                input_dim,
                -0.05,
                0.05,
                Some(seed.wrapping_add(10_000).wrapping_add(idx as u64)),
            )?;
            let positives = anchors.add(&jitter)?;
            batches.push(SelfSupBatch::from_pairs(anchors, positives)?);
        }
        Ok(SelfSupEpoch::new(batches))
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

    fn evaluate_info_nce_epoch_with_policies(
        model: &dyn Module,
        epoch: &SelfSupEpoch,
        forward_policy: BackendPolicy,
        loss_policy: BackendPolicy,
        temperature: f32,
        normalize: bool,
    ) -> st_nn::PureResult<Option<f32>> {
        let mut loss = InfoNCELoss::new(temperature, normalize);
        let mut total = 0.0f32;
        let mut batches = 0usize;
        for batch in epoch.batches() {
            let prediction = {
                let _policy_guard = push_backend_policy(forward_policy);
                model.forward(batch.combined())?
            };
            let target = Tensor::zeros(1, 1)?;
            let value = {
                let _policy_guard = push_backend_policy(loss_policy);
                loss.forward(&prediction, &target)?
            };
            total += value.data()[0];
            batches += 1;
        }
        Ok(if batches == 0 {
            None
        } else {
            Some(total / batches as f32)
        })
    }

    pub fn run() -> st_nn::PureResult<()> {
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

        let backend_sel = crate::backend::parse_backend(Some(args.backend.as_str()))?;
        let backend_runtime = crate::backend::prepare_backend_runtime(&backend_sel)?;
        let tensor_policy = crate::backend::tensor_backend_policy_meta(backend_sel.caps);
        let events_path = args
            .events
            .clone()
            .unwrap_or_else(|| args.run_dir.join("trainer_trace.jsonl"));
        if let Some(parent) = events_path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| TensorError::IoError {
                message: err.to_string(),
            })?;
        }
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

        let combined_rows = args.pair_batch * 2;
        let session = SpiralSession::builder(backend_sel.caps)
            .with_curvature(args.curvature)
            .with_hyper_learning_rate(args.learning_rate)
            .with_fallback_learning_rate(args.learning_rate)
            .build()?;

        let roundtable = RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5);

        let cfg = LightningConfig::new(combined_rows as u32, args.embed_dim as u32)
            .with_roundtable(roundtable);
        let mut lightning = SpiralLightning::with_config(session, cfg.clone());
        if args.accumulator_sync == "local" {
            lightning
                .trainer_mut()
                .set_training_device(LocalAccumulatorSynchronizer::new());
        }
        let mut model = build_model(args.input_dim, args.embed_dim)?;

        let objective =
            SelfSupObjective::InfoNCE(InfoNCEConfig::new(args.temperature, args.normalize));
        let epochs = (0..args.epochs)
            .map(|idx| {
                build_epoch(
                    args.batches_per_epoch,
                    args.pair_batch,
                    args.input_dim,
                    args.seed + idx as u64 * 100,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let selected_policy = BackendPolicy::from_device_caps(backend_sel.caps);
        let pretrain_cpu_reference_info_nce = evaluate_info_nce_epoch_with_policies(
            &model,
            &epochs[0],
            cpu_policy,
            cpu_policy,
            args.temperature,
            args.normalize,
        )?;
        let pretrain_selected_forward_cpu_loss_info_nce = evaluate_info_nce_epoch_with_policies(
            &model,
            &epochs[0],
            selected_policy,
            cpu_policy,
            args.temperature,
            args.normalize,
        )?;
        let pretrain_cpu_forward_selected_loss_info_nce = evaluate_info_nce_epoch_with_policies(
            &model,
            &epochs[0],
            cpu_policy,
            selected_policy,
            args.temperature,
            args.normalize,
        )?;
        let pretrain_info_nce = evaluate_info_nce_epoch_with_policies(
            &model,
            &epochs[0],
            selected_policy,
            selected_policy,
            args.temperature,
            args.normalize,
        )?;
        let pretrain_backend_gap = match (pretrain_info_nce, pretrain_cpu_reference_info_nce) {
            (Some(selected), Some(cpu_reference)) => Some(selected - cpu_reference),
            _ => None,
        };
        let pretrain_forward_gap = match (
            pretrain_selected_forward_cpu_loss_info_nce,
            pretrain_cpu_reference_info_nce,
        ) {
            (Some(selected_forward), Some(cpu_reference)) => Some(selected_forward - cpu_reference),
            _ => None,
        };
        let pretrain_loss_gap = match (
            pretrain_cpu_forward_selected_loss_info_nce,
            pretrain_cpu_reference_info_nce,
        ) {
            (Some(selected_loss), Some(cpu_reference)) => Some(selected_loss - cpu_reference),
            _ => None,
        };
        let stage =
            SelfSupStage::with_epochs(cfg.clone(), objective, epochs).with_label("selfsup.minimal");
        let report = lightning.fit_selfsup_plan(&mut model, [stage])?;

        let weights_path = args.run_dir.join("weights.json");
        save_json(&model, &weights_path)?;
        let mut reloaded = build_model(args.input_dim, args.embed_dim)?;
        load_json(&mut reloaded, &weights_path)?;
        let sanity = Tensor::random_uniform(
            combined_rows,
            args.input_dim,
            -1.0,
            1.0,
            Some(args.seed.wrapping_add(999)),
        )?;
        let sanity_output = reloaded.forward(&sanity)?;

        let mut first_loss = None;
        let mut last_loss = None;
        let stages_json = report
            .stages()
            .iter()
            .enumerate()
            .map(|(stage_idx, stage)| {
                let label = stage.label().unwrap_or("selfsup.stage");
                println!("stage[{stage_idx}] label={label}");
                let epochs = stage
                    .epochs()
                    .iter()
                    .enumerate()
                    .map(|(epoch_idx, epoch)| {
                        let stats = epoch.stats();
                        let info_nce = epoch.telemetry().map(|telemetry| match telemetry {
                            SelfSupEpochTelemetry::InfoNCE(metrics) => {
                                if first_loss.is_none() {
                                    first_loss = Some(metrics.mean_loss);
                                }
                                last_loss = Some(metrics.mean_loss);
                                json!({
                                    "mean_loss": metrics.mean_loss,
                                    "batches": metrics.batches,
                                })
                            }
                        });
                        let info_loss = info_nce
                            .as_ref()
                            .and_then(|value| value.get("mean_loss"))
                            .and_then(|value| value.as_f64());
                        println!(
                            "  epoch[{epoch_idx}] batches={} avg_loss={:.6} info_nce={}",
                            stats.batches,
                            stats.average_loss,
                            info_loss
                                .map(|value| format!("{value:.6}"))
                                .unwrap_or_else(|| "-".to_string())
                        );
                        json!({
                            "epoch": epoch_idx + 1,
                            "stats": epoch_stats_json(stats),
                            "info_nce": info_nce,
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "stage_index": stage_idx,
                    "label": label,
                    "config": {
                        "rows": stage.config().rows(),
                        "cols": stage.config().cols(),
                    },
                    "epochs": epochs,
                })
            })
            .collect::<Vec<_>>();

        let loss_delta = match (first_loss, last_loss) {
            (Some(first), Some(last)) => Some(last - first),
            _ => None,
        };
        let payload = json!({
            "schema": "st.selfsup.lightning_trace.v1",
            "run": {
                "run_dir": args.run_dir.display().to_string(),
                "events_path": events_path.display().to_string(),
                "weights_path": weights_path.display().to_string(),
                "backend": backend_sel.label.clone(),
                "device_caps": crate::backend::DeviceCapsMeta::from(backend_sel.caps),
                "backend_runtime": backend_runtime,
                "tensor_policy": tensor_policy,
                "epochs": args.epochs,
                "batches_per_epoch": args.batches_per_epoch,
                "pair_batch": args.pair_batch,
                "input_dim": args.input_dim,
                "embed_dim": args.embed_dim,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "curvature": args.curvature,
                "temperature": args.temperature,
                "normalize": args.normalize,
                "accumulator_sync": args.accumulator_sync,
            },
            "summary": {
                "pretrain_info_nce": pretrain_info_nce,
                "pretrain_cpu_reference_info_nce": pretrain_cpu_reference_info_nce,
                "pretrain_selected_forward_cpu_loss_info_nce": pretrain_selected_forward_cpu_loss_info_nce,
                "pretrain_cpu_forward_selected_loss_info_nce": pretrain_cpu_forward_selected_loss_info_nce,
                "pretrain_backend_gap": pretrain_backend_gap,
                "pretrain_forward_gap": pretrain_forward_gap,
                "pretrain_loss_gap": pretrain_loss_gap,
                "first_info_nce": first_loss,
                "last_info_nce": last_loss,
                "info_nce_delta": loss_delta,
                "sanity_output_shape": [sanity_output.shape().0, sanity_output.shape().1],
                "sanity_output_l2": sanity_output.squared_l2_norm().sqrt(),
            },
            "stages": stages_json,
        });
        let trace_path = args.run_dir.join("selfsup_trace.json");
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
            run_json_path(&args.run_dir, "selfsup_trace.json")
        );
        println!("events_jsonl={}", events_path.display());
        println!(
            "summary: first_info_nce={} last_info_nce={} delta={}",
            first_loss
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string()),
            last_loss
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string()),
            loss_delta
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string())
        );

        Ok(())
    }
}

#[cfg(feature = "selfsup")]
fn main() -> st_nn::PureResult<()> {
    demo::run()
}
