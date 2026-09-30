# Resident Vision Input: Bounded Contract Result

This slice connects Rust-owned Normalize and DataLoader batches to the existing
resident ConvNeXt classifier. Python and JavaScript expose the same transform
implementation. These are synthetic correctness checks, not real-image quality
or matched PyTorch performance evidence.

## What Ran

The shared native/browser classifier fixture adds four steps of
DataLoader -> Normalize -> Resize -> Flip -> Crop -> ConvNeXtClassifier -> CE -> SGD.
Two generated 1x12x14 images become 1x8x8 model inputs. The same small two-stage
classifier starts from the same weights on CPU and GPU. Every normalized input,
logit, loss, and all 24 parameter gradients/updated tensors are compared. All
observations occur after the four submitted steps, with no image or training-loop
readbacks between updates. Invalid normalization separately rejects the entire
update and preserves all classifier weights.

| Check | Native WGPU | Chrome WebGPU |
| --- | --- | --- |
| Normalized-input classifier steps | 4 passed | 4 passed |
| Maximum scaled error across normalized-input checks | 1.004829e-7 | 1.586298e-7 |
| First / fourth observed CE | 0.701423 / 0.693488 | 0.701423 / 0.693488 |
| Invalid normalization preserves all weights | Passed | Passed |
| Existing classifier learning/resume checks | All 15 passed | All 15 passed |

Scaled error is abs(actual - reference) / (1 + abs(reference)); the shared
classifier tolerance is 2e-4 and every compared value must be finite. The
existing, separate eight-step classifier case has maximum scaled error
3.118535e-5 native / 1.458478e-4 browser. Its within-runtime restart comparison
remains bitwise. This does not claim bitwise equality across runtimes or a
checkpoint of the DataLoader cursor/RNG.

The public browser transform fixture compares four normalized batches against
Rust CPU (maximum absolute error 1.788140e-7), plus resident continuation,
invalid statistics, inherited overflow guards after cropping, retry RNG, and
subnormal statistics. It also retains the 12 existing seeded geometry cases.
Chrome 154.0.8037.58 reports the actual Rust backend as BrowserWebGpu; the
separate Apple metal-3 adapter probe is not device-identity attestation.

Full conditions, losses, browser errors and generated-asset hashes are in
[the public-client report](browser-clients.json) and
[the shared classifier report](browser-classifier.json).

## Numerical Failure Found And Fixed

The first extreme-value test rejected the finite quotient 1e-40 / 1e-40 as
NonFinite. Checked WGSL division now uses integer significands and
round-to-nearest-even for extreme exponents, preserving subnormal operands and
results without shader-f64 or CPU fallback. Ordinary exponents retain hardware
division. Subtraction reuses the integer-defined rounded-add helper.

The native regression checks more than 1,000 finite signed exponent/fraction
combinations against CPU quotient bits. Additional checks cover zero
denominators, invalid intermediates, fused/batched/sequential execution, broadcast
VJPs, and zero cotangents that must not hide invalid forward values. The browser
public API independently checks normalization with subnormal statistics. This is
not an exhaustive f32 proof or a claim about untested GPU implementations.

## Other Validation

- Shared kernel contracts: 37 tests passed.
- Native runtime-enabled WGPU backend: 245 passed; vision: 89 passed, one existing ignored.
- CPU-only vision: 72 passed; portable graph tests: 7 passed, including v4 round-trip and downgrade rejection.
- Fresh default-feature wheel: 18 Python tests passed without skips, covering classifier, transforms, pointwise, pointwise cotangents and stream-frame APIs.
- Python binding with no default features and python-default enabled: cargo check passed.
- Strict kernel-contract/backend Clippy passed natively; strict backend Clippy passed for wasm32.
- Workspace rustfmt, generated/shipped resident TypeScript checks, and git diff --check passed.

The default native dispatcher embeds transform shaders, so installed wheels no
longer require the original checkout's shader files. Existing explicit custom
shader-directory construction remains available.

## Boundaries

- Rust/Python expose the resident DataLoader; browser callers supply batches.
  There is no JavaScript DataLoader or Python/JS resident ConvNeXt training
  binding in this slice. The browser learning fixture calls the real Rust model.
- Batches are homogeneous NCHW. Resident ColorJitter and mixed image sizes fail
  explicitly. Geometry transforms images, not associated boxes/masks.
- CPU or mapped-async failures preserve image/cursor/RNG as documented.
  Successful resident submission advances cursor/RNG before deferred GPU validity
  is observed; a returned tensor handle is not an accepted training update.
- Model checkpoints do not contain data cursor, augmentation RNG or trainer policy.
- The public-client report includes existing geometry-only timing cases and
  every condition in the six-case batch sweep. Those are not Normalize timings
  or PyTorch comparisons. On its 3x128x128 geometry-only case, CPU median was
  0.100 ms and WebGPU including readback was 0.800 ms; no general speedup is claimed.

See [the input API guide](../../../docs/resident_vision_input.md).

## Evidence Hashes

Logs, wheels, and original generated modules are retained locally. Published
reports are byte-for-byte copies, not rewritten summaries of raw observations.

| Evidence | SHA-256 |
| --- | --- |
| Original failed extreme-division test | 2ec678ad293fa24dbf4b173b3abefe568a36435da0ece996e229c6f06dbcca33 |
| Fixed extreme-division sweep | d83009febf65c614f08e91e527dabbd5a4b67a678463af36a6d39b874a088e5d |
| Kernel-contract tests | 34d45bfb5befd4919b27482f886f74f61275982b9e31de28be5533811beb42e9 |
| Native backend/vision tests | ca820bed033af10ba4aaf7e92307e06fb195cc9d049bf7bcd15ca4b14feed023 |
| Native classifier fixture | e59db67de9d0f294894feff546d34e466a8489ff92c94578cee7a864ff40e75e |
| CPU-only vision tests | fefbadf5b8f837016c5f9611df02afcf085451c7a7bd1076c1d6b2cbefd28305 |
| Portable graph tests | 54f43d9d57ca2af8dfad8da20e98f3409a9272d562a878ad74f692b62399f248 |
| Fresh-wheel 18-test regression | 1a04388c6ed7046c55949bb9cdb930395e5e844285939f091a3e29797b8a4829 |
| No-WGPU Python binding check | 21f3d1021d32de3704cc27b5ab5f2095d50a3a10acbc50feb8f326364073bd6a |
| Native strict Clippy | f1441d17e3f8de9392df782994ebb68758929cf51c3b4d5424df26afcc58797c |
| WASM strict Clippy | a27098c91f43b2752684343b30bd9c8743fad57ed6a2a3044640447b44f7c116 |
| Public-client browser report | e4a649d001d7c20002be4c44d0f17b024dfa5babfc3f71b811b1d53c891f8f2f |
| Classifier browser report | baadb54245f5a88b78ee22b9037d8b9562fdb7c64c8c0df9ed76e57b5d2ab959 |
| Tested macOS arm64 abi3 wheel (0.4.27) | 2f70562e27468f4a037e9ad8c1aa244ae744dd0399626897b65b1fe225ecbe32 |

## Replay

Run from the repository root with Rust 1.98.0, wasm-bindgen CLI 0.2.104, a real
Chrome executable in CHROME, and Playwright on NODE_PATH. The harness creates
an isolated browser profile and refuses to overwrite an existing report.

```sh
cargo +1.98.0 test --locked -p st-kernel-contracts --lib
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-backend-wgpu -p st-vision --features st-vision/wgpu --lib
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-vision --features wgpu --lib resident_classifier_learning_checkpoint_and_common_entry -- --nocapture
cargo +1.98.0 test --locked -p st-vision --lib
cargo +1.98.0 test --locked -p st-nn --features wgpu --lib resident::portable
cargo +1.98.0 check --locked -p spiraltorch-py --no-default-features --features python-default

CLIENT_DIR="$(mktemp -d)"
cargo +1.98.0 build --locked --release --target wasm32-unknown-unknown -p spiraltorch-wasm --features webgpu
wasm-bindgen --target web --out-dir "$CLIENT_DIR" --out-name spiraltorch_wasm target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm
node tools/test_resident_browser.cjs "$CLIENT_DIR" "$CHROME" "$CLIENT_DIR/report.json" '' '' '' '' vision-transforms

CLASSIFIER_DIR="$(mktemp -d)"
cargo +1.98.0 build --locked --release --target wasm32-unknown-unknown -p st-vision --features wgpu --example convnext_classifier_browser
wasm-bindgen --target web --out-dir "$CLASSIFIER_DIR" --out-name spiraltorch_wasm target/wasm32-unknown-unknown/release/examples/convnext_classifier_browser.wasm
node tools/test_resident_browser.cjs "$CLASSIFIER_DIR" "$CHROME" "$CLASSIFIER_DIR/report.json" '' '' '' '' convnext-classifier-resident

maturin build --locked --release --manifest-path bindings/st-py/Cargo.toml --out /tmp/spiraltorch-input-wheels
python -m pip install --force-reinstall /tmp/spiraltorch-input-wheels/spiraltorch-*.whl
python -m pip install pytest numpy
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -m pytest --import-mode=importlib -q bindings/st-py/tests/test_vision_classifier_native.py bindings/st-py/tests/test_vision_wgpu_pipeline.py bindings/st-py/tests/test_wgpu_pointwise.py bindings/st-py/tests/test_nn_pointwise_cotangent.py bindings/st-py/tests/test_vision_stream_frame_native.py
```

Use an isolated Python environment. This validates a locally built wheel, not a
new PyPI release. Next gates are thin resident classifier clients and matched
real-image training/throughput/restart evidence.
