# Resident attention projection training

`st_nn::resident::ResidentAttentionTraining` connects fused QKV projection,
multi-head attention, output projection, their VJPs, and one all-or-none plain-SGD
update. The same Rust implementation runs on native WGPU and browser WebGPU. It
extends the [attention VJP primitive](resident_attention_learning.md), not the
host `Module::backward` path. This is a source addition after 0.4.29; no Python or
JavaScript training wrapper, complete decoder, or language-quality claim is added.

## Use and ownership

Start with an `AttentionInferencePlan` from the
[frozen projection API](resident_zspace_attention.md). Compile a separate trainable
owner, upload inputs/targets once, and retain outputs only when needed:

```rust,ignore
let mut model = plan.compile_training_wgpu(runtime.clone())?;
for _ in 0..16 {
    let forward = model.forward(&input, z_bias.as_ref(), pair_bias.as_ref())?;
    let loss = forward.prediction().mean_squared_error(&target)?;
    let gradients = model.backward(&forward, loss.prediction_gradient())?;
    let update = model.sgd(&gradients, 0.03)?;
    receipts.push(update);
}
// Only now inspect losses, parameters and acceptance receipts, if wanted.
let parameters = model.parameter_snapshot();
```

`compile_training_wgpu_with_options` selects the existing matmul tile, kernel and
accumulation. The default is the scalar kernel; this change selects no new fast
path. A resident forward has an owning prediction and a token for the most recent
forward at the same parameter owner/revision. Starting another forward invalidates
the old tape, but not its prediction. Repeated cotangents on the current tape
produce independently retained gradients. Foreign tokens, old tapes and stale
gradient revisions are rejected.

All four optimized tensors belong to **one** `ResidentParameters` owner:

| Index | Parameter | Shape |
| --- | --- | --- |
| 0 | Fused QKV weight | `[input_width, 3 * heads * head_dim]` |
| 1 | Fused QKV bias | `[3 * heads * head_dim]` |
| 2 | Output weight | `[heads * head_dim, output_width]` |
| 3 | Output bias | `[output_width]` |

Q, K and V occupy consecutive column groups. The backward uses shared
`ResidentTensor::concatenate` to assemble their logical cotangents on GPU,
including strided head views. Concatenation validates shapes/context, allocates a
fresh contiguous output, and preserves failures even from empty operands. There
is one submission for the join and no host mapping or implicit broadcasting.

`sgd` submits one accept/reject decision across both projections. A bad gradient
or any overflowing candidate preserves **all** parameter bits. A submitted update
advances the attempted revision even when the GPU rejects it, making old gradients
stale; a host-side validation error does not submit an update. The returned receipt
must be read to establish acceptance. A zero rate still checks validity. The next
forward rebinds projection buffers from the new owner snapshot through GPU copies.
The original plan, source Modules, and retained snapshots never change.

Z-bias and pair-bias remain caller-owned inputs, not hidden optimizer parameters.
Their optional logical gradients are returned by `z_bias_gradient()` and
`pair_bias_gradient()`. The input gradient is also logical; reduce any broadcast
dimensions when returning it to a shared source. No implicit geometry update,
source Module synchronization, optimizer history, checkpoint format, dropout,
KV-cache, or decoder loss is introduced. These explicit routes reject any active
tensor execution-plan binding, as the attention inference route does.

## Verification

The independent CPU-f32 PyTorch 2.12.1 oracle uses separate ordinary Q/K/V/output
projections and autograd. It covers three shapes, both masks and five bias modes:
30 matched forward/input/parameter/geometry-gradient conditions. The geometry
variants within a shape/mask share input, weights and upstream. A separate
16-step MSE/SGD trajectory updates projection parameters with fixed geometry;
native and browser probes inspect losses/receipts only after the loop. The fixed
comparison gate is `abs(error) <= 3e-6 + 5e-5 * abs(reference)`.

Additional native tests cover finite-gradient overflow isolated to either
projection, held-gradient reuse, foreign/stale owners, zero-rate validation,
source/snapshot immutability, and empty/strided/multi-workgroup concatenation.
See the [dated correctness evidence](../benchmarks/results/2026-10-09-resident-attention-projection-training/README.md).
This small synthetic test is not a throughput comparison or an FT-quality result.

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
  cargo test --locked --release -p st-nn --no-default-features --features wgpu \
  --lib resident::attention::training -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
  cargo test --locked --release -p st-nn --no-default-features --features wgpu \
  --test resident_attention_training -- --nocapture --test-threads=1
```

Native GPU tests return early without the opt-in, so a default green run is not
GPU execution evidence. The standalone Rust browser probe builds as follows:

```bash
cargo build --locked -p st-nn --no-default-features --features wgpu \
  --example resident_attention_training_browser --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/debug/examples/resident_attention_training_browser.wasm \
  --target web --out-dir target/attention-projection-training-web
python3 -I -S -m http.server 8769 --bind 127.0.0.1
```

Use wasm-bindgen CLI 0.2.129, matching Cargo.lock. Open
`http://127.0.0.1:8769/crates/st-nn/tests/attention_training_browser.html` and require
`passed: true`, 30 checks, 16 accepted updates and both rejection checks. The page
downloads the exact JSON result. CI runs the native regressions and checks the
WASM example; browser execution is recorded separately, not inferred from build
success. The generator `tools/generate_resident_attention_training_torch_fixture.py`
refuses to overwrite an existing fixture. Regenerate to a fresh path with the
pinned PyTorch and optional global patches disabled before comparing bytes.
