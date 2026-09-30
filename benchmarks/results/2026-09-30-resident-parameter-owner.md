# Model-independent resident parameter ownership

## Scope

`ResidentParameters` owns immutable GPU weight versions and executes all-or-none
plain SGD with the shared Rust/WGSL candidate rule. Packing, candidate generation,
one global decision, and output selection use one queue submission. It rejects
stale/foreign owner tokens before dispatch and preserves every parameter when a
device-side check fails. No parameter or gradient readback is implicit.

This is a building block for model-owned ConvNeXt training, not that completed
model integration. The current host ConvNeXt caches are not yet bound to this
owner. No Python/JavaScript model wrapper, optimizer-state checkpoint, or full-model
performance claim is introduced. See [the API contract](../../docs/resident_parameters.md).

## Evidence

- Native GPU: all 240 `st-backend-wgpu` library tests passed with
  `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1`. This includes the existing graph learner
  after sharing its owner/version identity type with the new parameter owner.
- Mandatory `st-core` library tests: 1,001 passed.
- GPU-enabled `st-nn` and `st-vision` library tests: 788 and 72 passed;
  one existing vision test remains ignored.
- Native and wasm32 strict Clippy passed for all backend targets.
- The [real-browser report](2026-09-30-resident-parameter-owner-browser.json)
  passed in Chrome 154.0.8037.58, with no page or console errors.
- Native and browser execute the same Rust fixture. Thirteen named checks cover
  offset/strided values, strided gradients, signed zero, stale tokens, distinct
  owners at the same revision, shape/rate validation, whole-update rejection,
  invalid zero-rate gradients, recovery, retained snapshots/receipts, and multiple
  workgroups. Additional native checks reject foreign devices and preserve
  invalid initial-parameter guards.

The fixture also runs 16 consecutive synthetic Conv2d/MSE/VJP/SGD steps with
**zero readbacks inside the training loop**, then validates the retained update
receipts, final weights, and next forward against a Rust scalar reference.
The four input values are `[1, 2, 3, 4]`, targets are half the input, and the
learned model is one 1x1 kernel plus bias. At learning rate 0.03, MSE changes
from **1.875** to **0.0029072959441691637**. The final kernel and bias match the
reference at `0.5448817014694214` and `-0.1319359689950943`.

Numerical comparisons use `abs(actual - expected) <= 2e-5 * (1 + abs(expected))`;
retention checks compare f32 bits exactly. This is a bounded synthetic learning
contract, not real-image generalization or a speed comparison. The Rust browser
runtime reports `BrowserWebGpu`/`Other` with an empty device name. A separate
browser probe confirms a non-fallback Apple/Metal adapter but does not attest
the physical device used by that Rust runtime.

## Replay

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-backend-wgpu --lib resident_training::parameters
cargo +1.98.0 build --locked -p st-backend-wgpu --example resident_parameters_browser --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_parameters_browser.wasm --target web --out-dir /tmp/spiraltorch-parameters-web --out-name spiraltorch_wasm
node tools/test_resident_browser.cjs /tmp/spiraltorch-parameters-web "$CHROME_EXECUTABLE" /tmp/spiraltorch-parameters-new-report.json '' '' '' '' parameters-resident
```

Use `wasm-bindgen 0.2.104`, Playwright, and a new output path on each run. The
harness refuses to overwrite evidence and records the hashes of all served
module assets and the fixture page. Generated binaries remain outside the repo.
