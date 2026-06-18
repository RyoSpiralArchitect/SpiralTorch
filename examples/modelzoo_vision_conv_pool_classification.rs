// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Vision model-zoo: Conv2d + MaxPool2d binary classification probe.

#[path = "_shared/backend.rs"]
mod backend;

use serde_json::{json, Value};
use st_core::backend::device_caps::DeviceCaps;
use st_core::plugin::{
    global_registry, PluginEvent, PluginEventJsonlWriter, PluginEventJsonlWriterConfig,
};
use st_nn::{
    load_json, push_backend_policy, save_json, BackendPolicy, Conv2d, EpochStats,
    HyperbolicCrossEntropy, Linear, Loss, MaxPool2d, Module, ModuleTrainer, Relu, RoundtableConfig,
    Sequential, Tensor, TensorError,
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
    batch: usize,
    seed: u64,
    learning_rate: f32,
    curvature: f32,
    height: usize,
    width: usize,
    out_channels: usize,
}

impl Args {
    fn parse() -> st_nn::PureResult<Self> {
        let mut argv = env::args().skip(1).peekable();
        let mut args = Self {
            run_dir: default_run_dir(),
            events: None,
            backend: "auto".to_string(),
            epochs: 4,
            batches_per_epoch: 3,
            batch: 8,
            seed: 777,
            learning_rate: 2e-2,
            curvature: -1.0,
            height: 8,
            width: 8,
            out_channels: 4,
        };

        while let Some(flag) = argv.next() {
            match flag.as_str() {
                "--run-dir" => args.run_dir = PathBuf::from(take_arg(&mut argv, "--run-dir")?),
                "--events" => args.events = Some(PathBuf::from(take_arg(&mut argv, "--events")?)),
                "--backend" => args.backend = take_arg(&mut argv, "--backend")?,
                "--epochs" => args.epochs = take_parse(&mut argv, "--epochs")?,
                "--batches" => args.batches_per_epoch = take_parse(&mut argv, "--batches")?,
                "--batch" => args.batch = take_parse(&mut argv, "--batch")?,
                "--seed" => args.seed = take_parse(&mut argv, "--seed")?,
                "--lr" => args.learning_rate = take_parse(&mut argv, "--lr")?,
                "--curvature" => args.curvature = take_parse(&mut argv, "--curvature")?,
                "--height" => args.height = take_parse(&mut argv, "--height")?,
                "--width" => args.width = take_parse(&mut argv, "--width")?,
                "--out-channels" => args.out_channels = take_parse(&mut argv, "--out-channels")?,
                "--help" | "-h" => return Err(TensorError::Generic(usage().to_string())),
                other => {
                    return Err(TensorError::Generic(format!(
                        "unknown flag: {other}. Try --help"
                    )));
                }
            }
        }

        if args.epochs == 0
            || args.batches_per_epoch == 0
            || args.batch == 0
            || args.height == 0
            || args.width == 0
            || args.out_channels == 0
        {
            return Err(TensorError::InvalidValue {
                label: "vision_conv_probe_invalid_dims",
            });
        }
        if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "vision_conv_probe_learning_rate",
            });
        }
        if !args.curvature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "vision_conv_probe_curvature",
                value: args.curvature,
            });
        }
        Ok(args)
    }
}

fn usage() -> &'static str {
    "usage: cargo run -p st-nn --example modelzoo_vision_conv_pool_classification -- \
     [--run-dir PATH] [--events PATH] [--backend auto|wgpu|cuda|hip|cpu] \
     [--epochs N] [--batches N] [--batch N] [--seed N] [--lr F] \
     [--curvature F] [--height N] [--width N] [--out-channels N]"
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
    PathBuf::from(format!("models/runs/vision_conv_pool_probe/{stamp}"))
}

fn conv_output_hw(
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> (usize, usize) {
    let (h, w) = input_hw;
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;
    (oh, ow)
}

fn pool_output_hw(
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> (usize, usize) {
    let (h, w) = input_hw;
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;
    (oh, ow)
}

fn build_batch(
    batch: usize,
    hw: (usize, usize),
    base_seed: u64,
) -> st_nn::PureResult<(Tensor, Tensor)> {
    let pixels = hw.0 * hw.1;
    let mut data = Vec::with_capacity(batch * pixels);
    let mut targets = Vec::with_capacity(batch);

    for idx in 0..batch {
        let seed = base_seed.wrapping_add(idx as u64);
        let mut sample = Tensor::random_uniform(1, pixels, 0.0, 0.20, Some(seed))?
            .data()
            .to_vec();
        let class_one = idx % 2 == 1;
        if class_one {
            for r in hw.0 / 2..hw.0 {
                for c in hw.1 / 2..hw.1 {
                    sample[r * hw.1 + c] += 0.9;
                }
            }
            targets.push(1.0);
        } else {
            for r in 0..hw.0 / 2 {
                for c in 0..hw.1 / 2 {
                    sample[r * hw.1 + c] += 0.9;
                }
            }
            targets.push(0.0);
        }
        data.extend_from_slice(&sample);
    }

    let x = Tensor::from_vec(batch, pixels, data)?;
    let y = Tensor::from_vec(batch, 1, targets)?;
    Ok((x, y))
}

fn build_epoch(
    batches_per_epoch: usize,
    batch: usize,
    hw: (usize, usize),
    seed: u64,
) -> st_nn::PureResult<Vec<(Tensor, Tensor)>> {
    (0..batches_per_epoch)
        .map(|idx| build_batch(batch, hw, seed.wrapping_add(idx as u64 * 100)))
        .collect()
}

fn build_model(
    input_hw: (usize, usize),
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    pool_kernel: (usize, usize),
    pool_stride: (usize, usize),
) -> st_nn::PureResult<Sequential> {
    let conv_out_hw = conv_output_hw(input_hw, kernel, stride, padding);
    let pool_out_hw = pool_output_hw(conv_out_hw, pool_kernel, pool_stride, (0, 0));
    let pooled_features = out_channels * pool_out_hw.0 * pool_out_hw.1;

    let mut model = Sequential::new();
    model.push(Conv2d::new(
        "conv1",
        1,
        out_channels,
        kernel,
        stride,
        padding,
        (1, 1),
        input_hw,
    )?);
    model.push(Relu::new());
    model.push(MaxPool2d::new(
        out_channels,
        pool_kernel,
        pool_stride,
        (0, 0),
        conv_out_hw,
    )?);
    model.push(Relu::new());
    model.push(Linear::new("head", pooled_features, 1)?);
    Ok(model)
}

fn evaluate_epoch_loss_with_policies(
    model: &dyn Module,
    batches: &[(Tensor, Tensor)],
    forward_policy: BackendPolicy,
    loss_policy: BackendPolicy,
    curvature: f32,
) -> st_nn::PureResult<Option<f32>> {
    if batches.is_empty() {
        return Ok(None);
    }
    let mut loss = HyperbolicCrossEntropy::new(curvature)?;
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

    let input_hw = (args.height, args.width);
    let kernel = (3, 3);
    let stride = (1, 1);
    let padding = (1, 1);
    let pool_kernel = (2, 2);
    let pool_stride = (2, 2);

    let mut model = build_model(
        input_hw,
        args.out_channels,
        kernel,
        stride,
        padding,
        pool_kernel,
        pool_stride,
    )?;
    model.attach_hypergrad(args.curvature, args.learning_rate)?;

    let mut trainer = ModuleTrainer::new(
        backend_sel.caps,
        args.curvature,
        args.learning_rate,
        args.learning_rate,
    );
    let schedule = trainer.roundtable(
        args.batch as u32,
        1,
        RoundtableConfig::default()
            .with_top_k(1)
            .with_mid_k(1)
            .with_bottom_k(1)
            .with_here_tolerance(1e-5),
    );
    let tensor_policy = backend::tensor_backend_policy_meta(backend_sel.caps);
    let roundtable_backend_audit = backend::roundtable_backend_audit(backend_sel.caps, &schedule);

    let epochs = (0..args.epochs)
        .map(|idx| {
            build_epoch(
                args.batches_per_epoch,
                args.batch,
                input_hw,
                args.seed.wrapping_add(idx as u64 * 1_000),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
    let selected_policy = BackendPolicy::from_device_caps(backend_sel.caps);
    let pretrain_cpu_reference_loss = evaluate_epoch_loss_with_policies(
        &model,
        &epochs[0],
        cpu_policy,
        cpu_policy,
        args.curvature,
    )?;
    let pretrain_selected_forward_cpu_loss = evaluate_epoch_loss_with_policies(
        &model,
        &epochs[0],
        selected_policy,
        cpu_policy,
        args.curvature,
    )?;
    let pretrain_cpu_forward_selected_loss = evaluate_epoch_loss_with_policies(
        &model,
        &epochs[0],
        cpu_policy,
        selected_policy,
        args.curvature,
    )?;
    let pretrain_loss = evaluate_epoch_loss_with_policies(
        &model,
        &epochs[0],
        selected_policy,
        selected_policy,
        args.curvature,
    )?;

    let mut loss = HyperbolicCrossEntropy::new(args.curvature)?;
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
    let mut reloaded = build_model(
        input_hw,
        args.out_channels,
        kernel,
        stride,
        padding,
        pool_kernel,
        pool_stride,
    )?;
    load_json(&mut reloaded, &weights_path)?;
    let (sanity_x, _) = build_batch(args.batch, input_hw, args.seed.wrapping_add(999_999))?;
    let sanity_output = {
        let _policy_guard = push_backend_policy(selected_policy);
        reloaded.forward(&sanity_x)?
    };

    let payload = json!({
        "schema": "st.vision.conv_pool_trace.v1",
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
            "batch": args.batch,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "curvature": args.curvature,
            "height": args.height,
            "width": args.width,
            "out_channels": args.out_channels,
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
    let trace_path = args.run_dir.join("vision_trace.json");
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
        run_json_path(&args.run_dir, "vision_trace.json")
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
