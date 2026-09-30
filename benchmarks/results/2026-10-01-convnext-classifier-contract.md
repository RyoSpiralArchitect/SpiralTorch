# ConvNeXt Classifier: One Model From Learning To Inference

## Scope

Correctness checks for the actual SpiralTorch ConvNeXt-style backbone, spatial
global-average pool, and Linear head. This is not a throughput benchmark,
torchvision architecture/checkpoint equivalence, pretrained weights, or a
real-image generalization result.

The shared native/WASM fixture uses two synthetic 1x8x8 stripe images, two
stages of widths 2/4 and depths 1/1, a 2x2 stem, epsilon 1e-3, seed 7, two
classes, mean cross entropy, and eight plain-SGD steps at rate 0.01. All 24
parameter **tensors** (not 24 scalar values) share one resident owner and
acceptance/revision clock. There are no readbacks inside those eight update
submissions; explicit observations follow the loop. CPU reference arithmetic
is interleaved for validation, so this fixture must not be timed as throughput.

## Results

| Check | Native GPU | Real Chrome WebGPU |
| --- | --- | --- |
| Initial cross entropy | 0.6629087925 | 0.6629087925 |
| CE after eight updates | 0.4883580506 | 0.4883580506 |
| Maximum scaled CPU error | 0.00003118535 | 0.00014584778 |
| Fixed error bound | 0.0002 | 0.0002 |
| Updated parameter groups | 6/6 | 6/6 |
| Checkpoint at step 4, resume to 8 | Every prediction and weight bitwise | Every prediction and weight bitwise |
| Invalid label, zero-rate update | Rejects, every weight unchanged | Rejects, every weight unchanged |
| Valid update after rejection | Accepted; all weights match CPU within bound | Accepted; all weights match CPU within bound |

Scaled error is `abs(actual - reference) / (1 + abs(reference))`; every value
must be finite. Bitwise restart is **within each runtime**, not cross-runtime
bitwise equivalence. The six groups are stem, first block, downsample, second
block, final normalization, and head. The final checkpoint is restored to a
host classifier and moved into the ordinary `VisionModel` inference interface;
its logits are compared with the trained resident model.

The final browser fixture has 15 checks. Two independent final browser launches
passed and produced identical report bytes. Chrome 154.0.8037.58 used the Rust
`BrowserWebGpu` backend. A separate adapter probe reported Apple `metal-3`,
non-fallback; that separate probe is not attestation of the Rust device identity.
The native test rejects CPU/software adapters. See the
[browser report](2026-10-01-convnext-classifier-browser.json) for asset hashes,
all losses, actual runtime adapter, launch flags, and empty error/console arrays.

## Browser Failure Found And Fixed

The original browser artifact failed twice in WGPU's `BufferMappedRange::drop`
with a detached/out-of-bounds `ArrayBuffer` during `ResidentParameters::sgd`
uniform upload. The local vendored WGPU implementation created an unsafe
`Uint8Array::view`, returned it to Rust, then passed it back to JavaScript for
`set`. Externref allocation can grow WASM memory in that interval and detach
the view. It now calls `Uint8Array::copy_from` with the Rust slice directly,
creating the view within the JavaScript call instead. No CPU fallback,
readback insertion, memory preallocation workaround, or reduced fixture was used.

The fixed 14-check artifact passed, then the extended final 15-check fixture
(including valid retry after rejection) passed twice. Original failure records
remain local, along with the original/fixed WASM and JS artifacts. Their hashes
are retained below rather than overwriting the failed observation.

## Other Validation

- Native GPU backend: 243 tests passed; vision: 83 passed, one existing ignored.
- CPU-only vision: 69 tests passed; backbone integration: 14 passed.
- Pooling CPU/GPU tests: 2 passed, including non-contiguous input, layout parity,
  inherited invalid guards, and large finite VJPs without unused filter overflow.
- CPU classifier example: two-image CE 0.627618 -> 0.356605 over 20 steps.
- Fresh default-feature wheel: 8 tests passed (classifier factory and existing
  WGPU transform/handoff tests), with no skips. The three-class ConvNeXt factory
  owns 27,888,003 scalar parameters and reports no pretrained weights. This
  verifies factory construction and preprocessing, not a full Python training API.
- The native shared 15-check fixture passed separately after adding retry.
- Strict st-nn rustdoc and no-NN vision compilation passed. The latter does not
  silently substitute SimpleCnn for an unavailable ConvNeXt model.

Source: `crates/st-vision/examples/support/convnext_classifier_checks.rs`.
See [the API/ownership guide](../../docs/resident_convnext_training.md).

## Evidence Hashes

Raw logs and original build artifacts are retained locally; only results,
validation records, hashes and replay instructions are published.

| Local evidence | SHA-256 |
| --- | --- |
| Original browser failure | `02837e715190aec217828a887f5d03ac4fdf4c34b561d6afce354a5c688b9b30` |
| Original artifact failure replay | `c27d726407c668d4b35975c47d3136394641e634085edf0be6fcc5243f9c5964` |
| Fixed 14-check browser report | `327e75906f326930e02a4c94b88b7f1deed49b84025dab96394226947fb50477` |
| Final browser report (both launches) | `93bea5a7ef5b66dbe895fc60d5beb7f0182c408b2a57c2a34334309c7d26a034` |
| Native 15-check log | `bd3d6abb509daddd6541e705bb55d417546fddc2dc6a2263d7b1d2157b063796` |
| Native backend/vision regression log | `85470ccb92ccc50efc97f5552546de7de35e7592543f498414173fa57803fef1` |
| CPU vision/backbone log | `ed591903076532c60fe3def57ee3498dc1f13660f39409852472771fa3c8cd01` |
| Fresh-wheel 8-test log | `ffcbb10fd620312ebbe610b07ec434d165efbc0422917a94152843a371f3bc56` |

## Replay

Run from the repository root. Rust 1.98.0 and wasm-bindgen CLI 0.2.104 were
used. `CHROME` is a real Chrome executable; `NODE_PATH` must include Playwright.
Use a new report path for each launch: the harness refuses to overwrite it.

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-vision --features wgpu --lib resident_classifier_learning_checkpoint_and_common_entry -- --nocapture
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-nn --features wgpu --lib global_pool -- --nocapture
cargo +1.98.0 test --locked -p st-vision --no-default-features --features nn --lib --test backbones
cargo +1.98.0 run --locked --release -p st-vision --example convnext_image_classification

MODULE_DIR="$(mktemp -d)"
cargo +1.98.0 build --locked --release --target wasm32-unknown-unknown -p st-vision --features wgpu --example convnext_classifier_browser
wasm-bindgen --target web --out-dir "$MODULE_DIR" --out-name spiraltorch_wasm target/wasm32-unknown-unknown/release/examples/convnext_classifier_browser.wasm
node tools/test_resident_browser.cjs "$MODULE_DIR" "$CHROME" "$MODULE_DIR/report.json" '' '' '' '' convnext-classifier-resident
```

Fresh-wheel checks, using an isolated Python environment and the current wheel:

```sh
maturin build --locked --release --manifest-path bindings/st-py/Cargo.toml --out /tmp/spiraltorch-classifier-wheels
python -m pip install --force-reinstall /tmp/spiraltorch-classifier-wheels/spiraltorch-*.whl
python -m pip install pytest numpy
python -I -m pytest --import-mode=importlib -q bindings/st-py/tests/test_vision_classifier_native.py bindings/st-py/tests/test_vision_wgpu_pipeline.py
```

Remaining gates: resident Normalize/DataLoader integration, Python/JavaScript
resident classifier clients, caller-owned cursor/RNG/rate restart, real-image
held-out accuracy, matched PyTorch numerical/throughput comparisons, and
longer/larger training. No result above establishes those gates.
