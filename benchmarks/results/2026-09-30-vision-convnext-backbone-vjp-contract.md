# Resident ConvNeXt backbone VJP contract

The Rust `ResidentTensor::conv2d_vjp` supplies deterministic dense NCHW
input, weight, and bias gradients with one shared validity guard. It supports
stride, padding, dilation, and non-contiguous sources without floating-point
atomics or intermediate host readback. `Conv2d::vjp_resident` reuses the
module's parameter cache and refreshes it after host edits.

`ConvNeXtBackbone::vjp_resident` composes this primitive with the resident
depthwise and block VJPs and graph-autograd final LayerNorm. A two-stage
backbone with a stem and downsampling returns one input gradient and 22
parameter gradients in `Module::visit_parameters` order and host parameter
shapes. Forward activations are recomputed and retained on the GPU until the
caller explicitly snapshots a result. No host gradient accumulation, loss
reduction, or optimizer update is performed.

## Checks

- Native Apple Metal: `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-backend-wgpu -p st-nn -p st-vision --features st-nn/wgpu,st-vision/wgpu --lib` passed 1,096 tests (236 backend, 788 NN, 72 vision; one existing ignored vision test). The new tests compare the dense primitive, model-owned Conv2d, and all 23 backbone gradient tensors with CPU references. They also cover parameter edits, non-contiguous views, empty or mismatched shapes, invalid-source propagation, and recovery.
- CPU-only: `cargo +1.98.0 test --locked -p st-vision --no-default-features --features nn --lib` passed 61 tests.
- Browser: build the `st-vision` `convnext_backbone_vjp_browser` example for `wasm32-unknown-unknown --release`, generate web bindings with `wasm-bindgen 0.2.104`, then run `tools/test_resident_browser.cjs` with fixture `convnext-backbone-vjp-resident`. The [Chrome WebGPU report](2026-09-30-vision-convnext-backbone-vjp-browser.json) records CPU parity for all 23 gradient tensors, exact asset hashes, a separate non-fallback Apple Metal adapter probe, and no page or console errors. The probe does not attest the Rust runtime device.

The browser report records the maximum absolute error and the maximum
`abs(gpu - cpu) / (1 + abs(cpu))` across every gradient element. Both references
and results must be finite, and the scaled error must not exceed `0.01`.
Chrome 154.0.8037.58 observed maximum absolute error `0.00031661987` and
maximum scaled error `0.00004087983` on this two-image synthetic backbone.
The readback count describes the implemented execution path, not a GPU profiler
measurement: host mapping occurs only when the completed gradients are checked.

With `wasm-bindgen 0.2.104` and Playwright installed, replay from the repository
root (set `CHROME_EXECUTABLE` to the installed browser executable):

```sh
cargo +1.98.0 build --locked -p st-vision --features wgpu --example convnext_backbone_vjp_browser --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/examples/convnext_backbone_vjp_browser.wasm --target web --out-dir /tmp/spiraltorch-backbone-vjp-web --out-name spiraltorch_wasm
node tools/test_resident_browser.cjs /tmp/spiraltorch-backbone-vjp-web "$CHROME_EXECUTABLE" /tmp/spiraltorch-backbone-vjp-report.json '' '' '' '' convnext-backbone-vjp-resident
```

This is whole-backbone VJP correctness on a small synthetic shape, not a
GPU-owned optimizer step, real-dataset training comparison, or speedup claim.
The next gate is to keep parameter and optimizer state resident across steps,
then compare accuracy and throughput at matched training conditions.
