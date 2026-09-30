# Resident Parameter Ownership

`st_backend_wgpu::resident_training::parameters::ResidentParameters` owns a
nonempty set of N-D GPU tensors independently of a particular model topology.
It is the plain-SGD update boundary for models assembled from resident tensor
operations, including convolution. It does not export or mutate a host `Module`.

## Update Contract

1. Construct an owner from existing tensors on one device/queue. No upload or
   readback is implicit. Each entry is independent, not a tied-weight alias.
2. Take an immutable `snapshot()` and use its `values()` for forward and VJP.
3. Call that snapshot's `bind_gradients(...)` with derivatives in the same order
   and shapes. This validates shape/device compatibility and binds an opaque
   owner/version identity. The caller remains responsible for computing the
   derivatives from that version; binding is not proof of differentiation.
4. Call `owner.sgd(&gradients, rate)`. Packing, candidate calculation, the shared
   acceptance decision, and all output selection are one queue submission.
5. Use the new snapshot directly in the next forward. Read an update receipt
   explicitly when numerical acceptance must be observed.

The same `st-kernel-contracts::sgd` CPU/WGSL rule used by graph training checks
the gradient, multiplication, and subtraction separately. Inherited tensor
failure guards also reject the entire update. Every parameter retains its old
bits when any candidate fails. A zero rate still validates every derivative and
preserves signed zero. Rejected gradient guards are recorded in the receipt,
not propagated into otherwise valid retained parameters.

An attempted device update advances the revision even if rejected or zero-rate,
so its old gradient tokens cannot be submitted again. Host validation errors do
not advance it. This follows the graph learner's identity semantics, using the
same internal owner/version type. Revision is not an accepted-step counter.

Snapshots, gradient tensors, and update receipts survive subsequent updates and
owner destruction. This immutable implementation allocates candidate and output
storage for each step; it does not yet provide optimizer-state or buffer pools.

## Convolution Example

The native/browser fixture in
`crates/st-backend-wgpu/examples/support/resident_parameters_checks.rs` executes:

```rust,ignore
let parameters = owner.snapshot();
let weights = &parameters.values()[0];
let bias = &parameters.values()[1];
let prediction = input.conv2d(weights, bias, (1, 1), (0, 0), (1, 1))?;
let loss = prediction.mean_squared_error(&target)?;
let vjp = input.conv2d_vjp(
    weights, loss.prediction_gradient(), (1, 1), (0, 0), (1, 1),
)?;
let derivatives = parameters.bind_gradients(vjp[1..].to_vec())?;
let update = owner.sgd(&derivatives, 0.03)?;
// The next forward consumes owner.snapshot().values(), not the old snapshot.
```

On native platforms, `update.snapshot()?.read()?` confirms the attempted
revision or returns `TrainingError::Rejected`. In a browser, use
`update.snapshot()?.read_async().await?`. Either operation is an explicit
synchronization boundary; neither is required to submit the next step.

## Boundaries And Next Connection

- This is exact plain SGD. Reduction, clipping, momentum, decay, and geometric
  optimizer policy are not inferred or silently applied.
- A parameter snapshot is not a serialized training checkpoint. Host weight
  handoff and optimizer-state resume still require explicit ownership contracts.
- The separate [compiled ConvNeXt training model](resident_convnext_training.md)
  uses this owner for its forward, VJP and next-step parameters. Its source host
  model and inference caches remain independent. This is not implicit host
  synchronization or migration of attached optimizer state.
- Python/WASM model bindings and matched real-image accuracy/throughput remain
  open. The browser example executes the Rust API, not a new JavaScript model API.
