# Portable resident ConvNeXt checkpoint/restart

## Scope And Contract

This extends the [model-owned learning path](2026-09-30-convnext-resident-learning.md)
with an explicit capture/mapping boundary, portable JSON, host handoff and
resident restart. It is a **plain-SGD model checkpoint**, not a ModuleTrainer or
DataLoader checkpoint. Loss, learning-rate schedule, data cursor, augmentation
RNGs and trainer policy remain caller-owned. See
[the API contract](../../docs/resident_convnext_training.md#portable-model-checkpoint).

The checkpoint owns architecture, fixed batch size, canonical parameter names,
shapes and f32 values, optimizer kind and attempted-update revision. Restoring
always creates a fresh owner identity, so old gradients and forward tokens are
invalid even at the same numeric revision. Host synchronization is explicit and
preflights the entire parameter set before committing any value.

## Observed Checks

The same Rust fixture (`examples/support/convnext_checkpoint_checks.rs` under
`st-vision`) runs natively and in Chrome WebGPU. Two synthetic `[2, 8, 8]` images,
two stages of widths `[3, 4]` and depths `[1, 1]`, a `[2, 2]` stem, curvature -1
and epsilon 0.001 exercise all 22 backbone parameters. Inputs and targets vary
with an explicit caller-owned cursor; the rate is `0.0001 / (1 + cursor * 0.1)`.

The fixture submits eight forward/MSE/VJP/SGD steps, captures a checkpoint after
four, continues learning and drops the original model **before mapping that
capture**. It restarts at cursor four and compares every resumed prediction and
all updated weights with uninterrupted learning. No mapping occurs inside these
learning segments; checkpoint transfer and validation do explicitly read back.

| Execution | Same-runtime resumed predictions/weights | Aggregate maximum scaled error |
| --- | --- | --- |
| Native GPU | Bitwise identical at all four resumed steps | 4.761782e-7 |
| Chrome WebGPU | Bitwise identical at all four resumed steps | 4.151297e-7 |
| Chrome-produced checkpoint imported into native GPU, four resumed steps | Bounded f32 parity with native uninterrupted run | 4.761782e-7 |

The aggregate error also includes the host-forward comparison; the cross-runtime
row is **not** a cross-runtime bitwise claim. The fixed tolerance is
`abs(actual - expected) / (1 + abs(expected)) <= 2e-4`. Payload transfer itself
preserves all stored f32 bits. Snapshot/JSON/host transfer and same-runtime
restart separately require exact bits.

Fifteen checks cover metadata, capture ownership after later updates and model
drop, all parameter bits, source-host isolation, explicit host handoff, CPU
forward parity, fresh owner identities, old-forward rejection, resumed
predictions/weights, acceptance receipts, and rejection/retry. An inherited
non-finite gradient is rejected even at zero rate: all weights stay unchanged,
revision advances from 8 to 9, that state is saved/restored, and a valid retry
commits revision 10 identically to an uninterrupted retry.

The [browser report](2026-09-30-convnext-resident-checkpoint-browser.json) records
Chrome 154.0.8037.58, the actual Rust BrowserWebGpu runtime, a separate non-fallback
Apple/Metal adapter probe, and hashes of the fixture, generated JS/WASM and
checkpoint. The separate probe does not attest the Rust runtime's physical
device. There is still no public Python/JavaScript ConvNeXt model binding.

The exact browser-produced checkpoint imported by the native test is 9,948 bytes,
SHA-256 `64d782162b2e5cac48289270b7aed4b6c0f4203f7cd857697245e26a88f22c8a`.
Its raw JSON, generated module and complete logs are retained locally; only the
result, conditions, hashes and replay instructions are published. These are tiny
synthetic correctness checks, not real-image accuracy, PyTorch speed comparisons,
or long-running optimization/resumption evidence.

## Regression And Replay

- GPU-enabled libraries: backend 243 and vision 78 passed; one pre-existing
  vision test remains ignored.
- CPU-only vision: 65 passed. Host backbone integration: 14 passed.
- Four CPU checkpoint tests cover topology budgets/overflow, signed zero,
  subnormal/MAX f32 bit preservation, malformed schemas/metadata/weights, and
  all-parameter preflight preserving the target on a late validation failure.
- Strict backend Clippy passes natively and for wasm32. Vision all-target Clippy
  passes non-strictly with existing NN warnings and two existing needless-borrow
  warnings in a ConvNeXt VJP test; no new checkpoint diagnostics.
- Scoped nightly rustfmt, whitespace and harness JavaScript syntax checks pass.
  CI includes the native GPU restart fixture and wasm32 example compilation.

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-backend-wgpu -p st-vision --features st-vision/wgpu --lib -- --test-threads=1
cargo +1.98.0 test --locked -p st-vision --no-default-features --features nn --lib
cargo +1.98.0 test --locked -p st-vision --test backbones
cargo +1.98.0 build --locked --release -p st-vision --features wgpu --target wasm32-unknown-unknown --example convnext_resident_checkpoint_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/convnext_resident_checkpoint_browser.wasm --target web --out-dir /tmp/convnext-checkpoint-web --out-name spiraltorch_wasm
node tools/test_resident_browser.cjs /tmp/convnext-checkpoint-web "$CHROME_EXECUTABLE" /tmp/convnext-checkpoint-new-report.json '' '' '' '' convnext-checkpoint-resident
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_CONVNEXT_CHECKPOINT_IMPORT=/tmp/convnext-checkpoint-new-report.json.checkpoint.json cargo +1.98.0 test --locked -p st-vision --features wgpu --lib resident_convnext_checkpoint_resumes_with_fresh_identity -- --nocapture
```

Use wasm-bindgen 0.2.104 and Playwright. Report and sibling `.checkpoint.json`
paths must be new; the harness refuses to overwrite prior evidence. These replay
commands recreate the same deterministic conditions, not necessarily identical
floating-point results or artifact hashes on another compiler/device.
