# Migrating from PyTorch to SpiralTorch

SpiralTorch is designed to feel familiar to PyTorch users while exposing Z-space native training
capabilities and curvature-aware numerics. This guide mirrors common PyTorch snippets with their
SpiralTorch counterparts and highlights the deeper Z-space features that differentiate the
framework.

## Core layer mapping

| PyTorch | SpiralTorch |
| --- | --- |
| `nn.Linear(in, out)` | `st_nn::layers::Linear::new("fc", in, out)` |
| `nn.BatchNorm1d(feats)` | `st_nn::layers::BatchNorm1d::new("bn", feats, 0.1, 1e-5)` |
| `nn.BatchNorm1d(feats)` + Z-space | `st_nn::layers::ZSpaceBatchNorm1d::new("bnz", feats, -1.0, 0.1, 1e-5)?` |
| `nn.LayerNorm(feats)` | `st_nn::layers::LayerNorm::new("ln", feats, -1.0, 1e-5)` |
| `nn.LayerNorm(feats)` + Z-space | `st_nn::layers::ZSpaceLayerNorm::new("lnz", feats, -1.0, 1e-5)?` |
| `nn.LSTM(in, hidden)` | `st_nn::layers::Lstm::new("lstm", in, hidden)` |
| `nn.MaxPool2d` / `nn.AvgPool2d` | `st_nn::layers::conv::MaxPool2d`, `st_nn::layers::conv::AvgPool2d` |

`BatchNorm1d` and `Lstm` cache gradients and running statistics so you can flip between training
and evaluation using `set_training(true/false)` exactly like PyTorch. The LSTM forward expects a
time-major 2-D tensor and exposes `set_state` / `reset_state` helpers to manage the hidden and cell
states explicitly.

The Z-space batch and layer norm variants extend their Euclidean counterparts with a curvature-aware
projector. You can blend Euclidean and hyperbolic activations via `with_projector_gain` or adjust the
gain during training using `adapt_projector_gain` to keep the projected radius within a desired range.

## Optimisers and schedulers

SpiralTorch ships a high-level `ZSpaceOptimizer` paired with the `WarmupCosineScheduler`, covering
the usual `torch.optim` experience while wiring in curvature-specific behaviour.

```rust
use st_nn::{optim::{OptimizerMode, WarmupCosineScheduler, ZSpaceOptimizer}, layers::Linear, Tensor};

let mut layer = Linear::new("fc", 4, 2)?;
let mut opt = ZSpaceOptimizer::new(1e-2)?;
opt.set_mode(OptimizerMode::hypergrad(-1.0, 5e-3)?);
opt.prepare_module(&mut layer)?; // attach the hyper-gradient tape

let mut scheduler = WarmupCosineScheduler::new(1e-2, 1e-4, 100, 1000)?;
let lr = scheduler.step_optimizer(&mut opt, &mut layer)?; // cosine update with warmup
```

`OptimizerMode` switches between Euclidean, Realgrad, and Hypergrad updates, automatically
injecting the curvature-aware step size logic required by Z-space training. The warmup cosine
scheduler mirrors `torch.optim.lr_scheduler.CosineAnnealingLR` with an integrated burn-in phase.

## Mixed precision training

SpiralTorch offers an AMP-style API through `mixed_precision::GradScaler` and the `AutocastGuard`
utility, closely matching `torch.cuda.amp` semantics.

```rust
use st_nn::{mixed_precision::GradScaler, optim::{OptimizerMode, ZSpaceOptimizer}, Tensor};

let mut scaler = GradScaler::new(2.0, 2.0, 0.5, 200)?.with_limits(1.0, 1024.0);
let mut opt = ZSpaceOptimizer::new(1e-3)?;
opt.set_mode(OptimizerMode::realgrad(1e-3)?);
opt.prepare_module(&mut model)?;

let scaled_loss = scaler.scale_loss(loss_value);
let stepped = scaler.step(&mut opt, &mut model)?; // automatically skips on overflow
```

Guard scopes give you precise control over automatic casting when mixing FP16/FP32 operations.

## Z-space telemetry and interpretability

`ZSpaceBatchNorm1d` and `ZSpaceLayerNorm` record detailed telemetry every forward pass. Batch norm
captures per-feature radius while layer norm captures per-sample curvature, in addition to the
projection Jacobian, whitened activations, and blended outputs. Call `telemetry()` to retrieve a
`ZSpaceBatchNormTelemetry` or `ZSpaceLayerNormTelemetry` snapshot, or `last_ball_radius()` for a
quick summary. These diagnostics unlock curvature-aware regularisation heuristics:

```rust
let telemetry = layer.telemetry().expect("forward pass executed");
let avg_radius: f32 = telemetry.radius().iter().copied().sum::<f32>()
    / telemetry.radius().len() as f32;
if avg_radius > 0.8 {
    layer.set_projector_gain(0.5)?; // tighten the projection if the batch drifts
}
```

When you prefer an automated adjustment, `adapt_projector_gain` nudges the projector gain towards a
target radius with exponential smoothing, enabling self-regulating Z-space pipelines across both
batch and layer normalisation workflows.

For vision models, `st_vision::xai::GradCam` interoperates seamlessly with Z-space activations, so
you can visualise curvature-aware representations without additional glue code.

## Summary

* Native Rust implementations of `BatchNorm1d`, `ZSpaceBatchNorm1d`, `ZSpaceLayerNorm`, `LayerNorm`, and `Lstm` mirror
  PyTorch ergonomics while exposing curvature-aware controls.
* `ZSpaceOptimizer` and `WarmupCosineScheduler` cover the common optimisation rituals inside
  SpiralTorch.
* `mixed_precision::GradScaler` plus `AutocastGuard` bring AMP-style workflows to Rust.
* Z-space telemetry APIs simplify monitoring, debugging, and adapting hyperbolic training loops when
  migrating from PyTorch.
