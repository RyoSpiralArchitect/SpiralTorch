# Vision NCHW resident batch: bounded browser result

This slice starts from `origin/main` at
`513494b535e046733791e5edaecab8f91e96bb68` (PR #2133). It adds a
homogeneous NCHW geometry path through `st-backend-wgpu` and `st-vision` to a
guarded `ResidentTensor`, with Rust, Python, and WASM entries. Image values
are uploaded once for the whole batch; small shader-parameter and per-image
flip-mask buffers are separate uploads. Adjacent geometry stages share one
transform submission, followed by the tensor finite-value guard. There is no
geometry-to-NN host readback. Seeded flip choices follow per-image CPU order,
and a rejected batch does not advance the pipeline RNG.

## Matched end-to-end browser comparison

The actual Chrome 153.0.8010.54 fixture used a non-fallback Apple WebGPU
adapter probe. The separate Rust runtime reported `BrowserWebGpu` and device
type `Other`; the probe is **not** an attestation of identical adapter identity.
Each case used identical packed float32 inputs, seed 811, resize to
`H/2+8 x W/2+8`, random horizontal flip at probability 0.5, center crop to
`H/2 x W/2`, then `Sequential(Scaler x1.1)`. The baseline called
`applyResident` and the same NN once per image; the candidate called
`applyResidentBatch` and the NN once for the batch. The two arms used
separate NN instances with identical Scaler weights, avoiding shape-cache
competition. Both stayed on WebGPU until their NN outputs were read. The
baseline therefore has N terminal
readbacks, while the candidate has one. Wall time includes JavaScript, upload,
GPU dispatch, and **all** NN output readbacks; it is not kernel-only timing.
Each run had 5 warmups and 20 paired samples per condition, alternating arm
order. Three independent browser processes were measured. Values below are
per-run medians in milliseconds (`per-image / batch`):

| Input NCHW | Run 1 | Run 2 | Run 3 | Max output error |
| --- | ---: | ---: | ---: | ---: |
| 1 x 1 x 64 x 64 | 0.5 / 0.5 | 0.6 / 0.5 | 0.5 / 0.6 | 0 |
| 4 x 1 x 64 x 64 | 2.0 / 0.6 | 2.0 / 0.6 | 2.1 / 0.5 | 0 |
| 16 x 1 x 64 x 64 | 8.1 / 0.7 | 7.9 / 0.6 | 8.2 / 0.7 | 0 |
| 1 x 3 x 128 x 128 | 0.8 / 0.6 | 0.8 / 0.6 | 0.8 / 0.6 | 0 |
| 4 x 3 x 128 x 128 | 3.1 / 0.9 | 3.2 / 0.9 | 3.1 / 0.9 | 0 |
| 16 x 3 x 128 x 128 | 13.4 / 2.3 | 12.6 / 2.2 | 13.3 / 2.2 | 0 |

The browser also checked a 5 x 2 x 9 x 11 batch against sequential CPU
transforms, then fed its 5 x 2 x 6 x 6 resident result into an NN `Scaler`.
The maximum NN output error was `1.1920928955078125e-7`. A separate
`GraphLearner(Scaler)` accepted the same resident image batch without an
intermediate readback. One supervised update at rate 0.01 reduced its MSE
from `0.06856946087143044` to `0.06637772586507311` in all three runs.
This is a connectivity and one-step update check, **not** backpropagation
through image transforms or evidence of full image-model training quality.
Malformed length, non-finite values, invalid crop, and seeded retry were
rejected/checked by the browser fixture; native Rust and the fresh Python
wheel additionally checked ragged image batches.

## Evidence and replay

Complete raw reports, including all 20 paired timing values per condition,
page errors, runtime metadata, asset hashes, and the earlier single-image
controls, are retained locally under
`~/Library/Logs/SpiralTorch/vision-resident-batch-v1/`. The three
`browser-fair-run-*` reports are the claim basis; earlier exploratory reports
remain local but were not used for the table. Verification confirmed all
three have six complete 20-pair conditions, zero page errors, zero sweep
output mismatch, and a decreasing one-step MSE. Their SHA-256 values:

| Local artifact | SHA-256 |
| --- | --- |
| `browser-fair-run-1.json` | `2217092dd2b3a5cfbc5ae852a8c50b945263e22d9642835fc32678872b6db4c6` |
| `browser-fair-run-2.json` | `b7500973ac9b42a12d1ef31b5b0aa376eca9cea969834b38ab359d0a6ad4359a` |
| `browser-fair-run-3.json` | `0d3470d0fd0bac15b0e2b3f38098d80f2d1e44c0fd10b4f86e34c75f4646b674` |
| `module-fair-final/spiraltorch_wasm_bg.wasm` | `7e9cf2e9ee96e48825cf39698358c3346eb1606ce75e91fb4a886191b9b64333` |
| fresh Python 0.4.27 wheel (`wheels-fair-final/`) | `8bd98f50f37587ed75387ec5ac2addaf9fa4aa9a32d9425568f171e53242983c` |
| `bindings/st-wasm/tests/vision_transforms.html` | `a1b5f1a2fbd932f66f3bf6fcbdb6651735cf8a2144d5543fdfeab6df39555f5d` |

From the repository root, rebuild and run the same browser fixture, choosing
fresh output paths because the harness refuses to overwrite a report:

```sh
wasm-pack build bindings/st-wasm --target web --out-dir /tmp/vision-batch-module --release -- --features webgpu
NODE_PATH=/path/to/playwright/node_modules node tools/test_resident_browser.cjs \
  /tmp/vision-batch-module /path/to/Chrome /tmp/vision-batch-report.json \
  '' '' '' '' vision-transforms
```

Verification from this source used `cargo test -p st-vision --features wgpu
--lib` (67 tests), CPU-only `st-vision` tests (45), `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test -p
st-backend-wgpu transform::tests::` (6 tests), a WASM release build and three
real-browser passes, and four `test_vision_wgpu_pipeline.py` tests from a
newly built wheel in an isolated venv outside the source tree. The wheel and
generated WASM package remain local, not in the Git repository.
The checked-in TypeScript declaration passed `tsc --noEmit --strict`; the
generated declaration passed with the `esnext,dom` library set because
wasm-bindgen emits `Symbol.dispose`.

Limits: homogeneous batches and resize/crop/flip only; no Normalize,
ColorJitter, variable-size padding, differentiable geometry, full-model
training, PyTorch comparison, cross-device generalization, or kernel-only
speedup claim. The N=1 control did not consistently improve.
