# Query-Coordinate Pullback Control

The Rust-owned resident byte decoder supports an explicit training ablation:
`ByteDecoderQueryPullback::{Full, StopCoordinates}`. It applies to contextual
geometry mixtures, including the four-way
[projective family](complex_projective_geometry.md). It does not add a new
metric or change attention's forward scores.

```rust
use st_nn::resident::ByteDecoderQueryPullback;

let forward = model.forward(&batch)?;
let loss = forward.next_byte_loss(loss_spec)?;
let full = model.backward(&forward, loss.prediction_gradient())?;
let stopped = model.backward_with_query_pullback(
    &forward,
    loss.prediction_gradient(),
    ByteDecoderQueryPullback::StopCoordinates,
)?;
// Choose one ablation arm for this update, not both.
let update = model.sgd(&stopped, learning_rate)?;
```

For shared causal coordinates `z`, routing is `softmax(query(z) + logits)`.
`StopCoordinates` is equivalent to using `query(stop_gradient(z))` while
keeping `metric(z_i, z_j)` differentiable. Query-projection parameters still
receive their gradients. Metric endpoints, gains, shared mixture logits and
ordinary QKV/head paths remain differentiable. The wave/encoder and embeddings
lose only the query-input contribution, not the metric contribution.

This is **not** freezing the router, removing geometry, detaching the entire
score bias, or computing an exact gradient of the unchanged full objective.
It is an intentional selective stop-gradient learning rule. Ordinary
`backward` always selects `Full`. Models without contextual routers reject
`StopCoordinates`, and the returned VJP records the selected policy.

The same Rust implementation is compiled for native WGPU and browser WASM.
The new browser export is a synthetic validation harness, not a general JS
training-session API; no new Python facade is claimed here.

## Update And Resume Contract

Owner/revision checks, invalid-gradient propagation and the atomic parameter
update are unchanged. An update invalidates previously captured forwards and
VJPs as before. A retained forward can be used to compare Full, StopCoordinates
and Full again before any update.

The policy is per backward call, not persistent model state. Checkpoint schemas
and contents are unchanged. A caller doing stopped training **must record and
reapply the policy after restore**, along with data order, learning rate and
runtime/kernel options. The checkpoint alone is not a complete training recipe.
Native tests cover stopped CE updates and exact continued/restored checkpoints.
Browser evidence covers paired VJPs and the original Full training/resume path,
not stopped-mode browser training/resume.

## Why Pair The Control

To isolate the router contribution, compare `(GPU Full - GPU Stop)` against
`(CPU Full - CPU Stop)` at the same weights, forward and cotangent. Subtracting
CPU Stop from GPU Full instead mixes total runtime rounding differences into
the much smaller router contribution. The independent CPU f64 diagnostic also
fails that old mixed-runtime gate on these cases; it is not an exact-arithmetic
oracle because it retains the existing wave constants.

The [paired study](../benchmarks/results/2026-10-13-query-pullback-paired-v1/README.md)
retains the original `.002` relative-L2 threshold and rejects zero/wrong-sign
controls. It also checks every stopped parameter VJP against independent Torch
autograd. Original projective-v1 qualification stays **4/6**, including the
phase-gradient trajectory failure and old router gate. Neither the new paired
pass nor same-state diagnostics rewrite those failures or establish corpus
quality, useful FT, speed, or a general advantage over ordinary attention.
