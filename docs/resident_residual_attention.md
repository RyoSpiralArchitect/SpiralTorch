# Resident residual attention training

Source addition after the published 0.4.29. The Rust `st_nn::resident` API
connects existing graph plans and attention without another Python/WASM math
implementation:

```text
u = pre(x)
y = x + attention(u, z_bias, pair_bias)
output = y + feed_forward(y)
```

A conventional pre-norm block supplies LayerNorm for `pre`, and
LayerNorm -> Linear -> GELU -> Linear for `feed_forward`. Explicit Scaler,
ReLU and ToposResonator operations are also preserved by the existing graph
contract. This is not a complete language model: embeddings, position encoding,
the LM head, token loss, KV-cache, dropout and optimizer history are not added.

## Composition and ownership

`ResidualAttentionPlan::from_plans(&pre, &attention, &feed_forward)` clones
immutable snapshots. It checks exact logical layouts on every edge and both
residuals, not just flattened sizes. The input/output layout is `[B,T,C]`.
Attention's internal projected width may differ from `C`.

```rust
let plan = ResidualAttentionPlan::from_plans(&pre, &attention, &feed_forward)?;
let mut block = plan.compile_training_wgpu(runtime)?;
let forward = block.forward(&input, z_bias.as_ref(), pair_bias.as_ref())?;
let loss = forward.prediction().mean_squared_error(&target)?;
let gradients = block.backward(&forward, loss.prediction_gradient())?;
let update = block.sgd(&gradients, 0.02)?;
// Native: explicit acceptance readback. In WASM use snapshot().read_async().await.
let accepted_revision = update.snapshot()?.read()?;
```

The block has **one** `ResidentParameters` owner, not separately updated
attention and MLP optimizers. Parameter order is the pre-graph's parameter order,
fused QKV weight/bias, attention output weight/bias, then the feed-forward graph's
parameter order. A conventional block has 12 tensors. The tested Topos recipe
adds a learned gate between GELU and the second MLP Linear, for 13 tensors.
Parameter-free pre/feed-forward graphs are supported too.

Backward is an exact arbitrary-cotangent VJP through both skip paths, with no
implicit row average. Optional Z/pair biases retain the attention contract;
their logical gradients are returned but the biases remain caller-owned.

Every update uses the shared all-or-none SGD transaction. A numerical rejection
preserves all weights and advances the attempted revision, invalidating old
gradient tokens. A receipt read proves acceptance; submission alone does not.
Foreign/stale forward tokens are rejected. Only the latest forward tape is
reusable, while returned outputs, snapshots and gradients remain owned values.
Repeated cotangents do not overwrite earlier gradients. Graph parameter
rebinding and activation transfers between block parts stay on GPU.
The final prediction guard participates in backward, and every returned
derivative shares the whole-VJP guard, including overflow in the final input
gradient addition. `TensorDevice::guard_together` joins only flag words and
returns zero-copy value aliases; it does not change existing handles or values.

This composition reuses the same private attention autograd execution as
`ResidentAttentionTraining`; there is no second attention derivative. It does
not mutate the original Modules/plans, migrate optimizer history, synchronize
host parameters or introduce a checkpoint format.

The [Python/WASM clients](resident_residual_attention_clients.md) expose this
same owner and VJP through thin handles, including browser LayerNorm/Topos
module construction. They are also unreleased source additions after 0.4.29.

## Verification

`tools/generate_residual_attention_torch_fixture.py` is an independent CPU-f32
PyTorch reference. Its frozen fixture covers 60 forward/VJP conditions:
three shapes, two masks, five bias modes and Topos on/off. Both plain and Topos
recipes also have 32 plain-SGD updates. The fixed tolerance is
`3e-6 + 5e-5 * abs(reference)` for every compared value.
The [recorded native/browser results](../benchmarks/results/2026-10-09-resident-residual-attention/README.md)
retain the individual conditions, guard outcomes and provenance. Plain and Topos
trajectories are separate correctness recipes, not a matched quality ablation.

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-nn \
  --no-default-features --features wgpu --test resident_residual_attention \
  -- --test-threads=1 --nocapture
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-nn \
  --no-default-features --features wgpu --lib resident:: -- --test-threads=1
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --target wasm32-unknown-unknown --example resident_residual_attention_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_residual_attention_browser.wasm \
  --target web --out-dir target/residual-attention-web
python3 -I -S -m http.server 8771 --bind 127.0.0.1
```

Use the wasm-bindgen CLI matching Cargo.lock (0.2.129). Open
`http://127.0.0.1:8771/crates/st-nn/tests/residual_attention_browser.html`.
The browser runs the same Rust probe, including strided inputs/cotangents,
foreign/stale token rejection, retained values, nonfinite rejection and recovery.
Native fault injection rejects each of the 13 parameter candidates atomically.
Both terminal-overflow probes also verify the optional Z/pair gradient guards.
Backend regressions alternate good and failed aliases through cached graph
bindings, including views, GPU packing, recovery and retained old outputs.
The 32-update loops have no intermediate readback, including acceptance flags.
CI runs native checks and compiles the browser probe; browser execution remains
a separate live validation. These are synthetic block correctness checks, not
throughput results, a complete LM training run or evidence of language quality.
