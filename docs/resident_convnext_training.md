# Resident ConvNeXt Training

With `st-vision/wgpu`, `ConvNeXtBackbone::compile_resident_training(device, batch)`
creates a GPU-owned, fixed-batch training model from the backbone's current host
weights. Its stem, depthwise blocks, affine normalization, MLPs, residuals,
downsampling and final normalization participate in forward, VJP and plain SGD.
The next forward uses the updated GPU weights, not host inference caches.

This is the existing SpiralTorch ConvNeXt-style architecture, not a claim of
torchvision checkpoint/architecture compatibility or pretrained weights.

## Ownership And Execution

- Compilation snapshots the source model. Later host mutations cannot overwrite
  GPU updates, and device learning does not mutate the original host model.
- Host parameters with pending gradients or attached hypergrad/realgrad tapes
  are rejected before transfer. Even a zeroed-but-attached gradient must be
  resolved explicitly; optimizer state is never silently discarded or migrated.
- `forward` accepts a same-device `[batch, C, H, W]` resident tensor, including
  offset/strided views. Its owning prediction is `[batch, output_features]`.
- `backward` accepts an arbitrary same-shape resident cotangent, returns the
  input derivative and all parameter derivatives in source `Module` order, and
  binds them to that model's weight version. It applies no loss scaling or extra
  batch average. The loss is responsible for its own reduction.
- Only the latest forward can use the reusable intermediate tape. A new forward
  invalidates previous tape tokens, while predictions and derivatives own their
  values. Foreign model and stale weight-version tokens are rejected.
- `sgd` submits one `ResidentParameters` all-parameter decision. Non-finite
  arithmetic or inherited failure rejects the entire update, preserving every
  original weight bit. A zero rate still validates derivatives. Attempted
  updates, including rejection/zero-rate, invalidate old gradient tokens.
- Shape/device/rate errors detected before dispatch preserve the relevant
  state. Numerical acceptance is deferred until a receipt is explicitly read;
  it is not implied by successful submission.

```rust,ignore
let mut model = backbone.compile_resident_training(device.clone(), batch)?;
let forward = model.forward(&resident_images)?;
let loss = forward.prediction().mean_squared_error(&resident_targets)?;
let derivatives = model.backward(&forward, loss.prediction_gradient())?;
let update = model.sgd(&derivatives, learning_rate)?;
let next = model.forward(&resident_images)?;
// No host synchronization above. Observe acceptance explicitly when needed:
let attempted_revision = update.snapshot()?.read()?; // native
// Browser: update.snapshot()?.read_async().await?
```

An update receipt reports an attempted revision, not an accepted-step count.
`parameter_snapshot()` owns the current GPU tensors; `parameter_names()` and
their canonical shapes retain the source Module order. Neither is a serialized
training checkpoint. Retained snapshots, gradients and receipts survive later
updates and destruction of the compiled model.

## Shared Backend, Not A Second Optimizer

Convolution topology is captured by `Conv2d::resident_spec()` and
`DepthwiseConv2d::resident_spec()`; current values are supplied separately.
Linear/LayerNorm/GELU use the existing `InferencePlan` lowering and
`ResidentGraphAutograd`. Its `set_parameter_tensors` packs strided values and
copies them GPU-to-GPU into the compiled bindings, preserving validity guards
and invalidating the old tape. The guard-capture pipeline is cached across
rebindings. Update arithmetic and atomic acceptance come from the shared
`ResidentParameters` SGD contract.

There are still per-step tensor/candidate allocations and GPU-to-GPU parameter
copies. This implementation makes no throughput or allocation-efficiency claim.
It is not momentum, clipping, accumulation, tied-weight support, ModuleTrainer
policy migration, checkpoint/resume, or a Python/JavaScript model binding.

## Validation And Next Gates

The shared native/browser fixture is
`crates/st-vision/examples/support/convnext_learning_checks.rs`; its bounded
results and replay commands are recorded in
[the learning result](../benchmarks/results/2026-09-30-convnext-resident-learning.md).
The next gates are explicit host/checkpoint transfer, a resident classification
head and real-image data path, thin Python/WASM model bindings, and matched
accuracy/throughput comparisons. A loss decrease on this tiny synthetic fixture
does not prove real-image generalization or long-run optimization stability.
