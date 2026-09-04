# Resident Matmul Pilot

Furnace: one RTX 5090, NVIDIA driver 595.84, WGPU Vulkan and PyTorch
2.13.0+cu132. The Vulkan ICD was pinned to NVIDIA. The existing CPU llama-server
was left running; no competing GPU process was present before this work.

## Native Resident Comparison

`resident-vs-torch.json` passed all 12 numerical gates: four shapes, three
seeds, float32 inputs, CPU float64 references. Each timing sample is 16
Python-loop dispatch calls followed by that backend's synchronization; five
warmups and 20 randomized paired samples are retained per case.

Median of the three seed medians, milliseconds **per 16-call sample**:

| M / K / N | SpiralTorch WGPU | PyTorch CUDA |
| --- | ---: | ---: |
| 64 / 64 / 64 | 0.386 | 0.127 |
| 256 / 256 / 256 | 0.572 | 0.158 |
| 512 / 512 / 512 | 1.943 | 0.238 |
| 1024 / 64 / 32 | 0.383 | 0.118 |

Both implementations retain device inputs/output. Allocation, uploads, and
full readbacks are outside timing; API, command submission, and host waiting
remain inside. The current ST tile is a portable 8x8x16, not an optimized
device-specific choice. PyTorch remains faster, especially at 512x512.
Do not turn this into a kernel-only, training-quality, confidence-interval,
or host-to-host speedup claim. This first report predates the browser-only
dependency backport and the stricter single-CUDA-device harness gate; its
original module and script hashes are retained, not relabeled as a later build.

`resident-vs-torch-final.json` repeats all 12 gates with a fresh private wheel
including the dependency backport and the final harness. It reports one visible
CUDA device and matching NVIDIA adapter names. The seed-median results are
0.385 / 0.127, 0.573 / 0.158, 1.917 / 0.238, and 0.383 / 0.118 ms
(SpiralTorch / PyTorch, in the same shape order). The performance conclusion
is unchanged. Its script/helper hashes match the committed harness sources.

## Browser Execution

`browser-resident.json`: Chrome 152.0.7977.77, 15/15 numerical cases and
26 input/ownership assertions passed. This includes concurrent workspace
creation, cross-realm Float32Array upload, invalid-update atomicity, overlapping
readbacks, and GPU chaining with readback after workspace free. Maximum absolute
error across cases was 1.057e-5, within the elementwise mixed tolerance.

Median of three seed medians, milliseconds **per 16-call sample**:

| M / K / N | Browser WebGPU |
| --- | ---: |
| 7 / 63 / 65 | 1.00 |
| 64 / 64 / 64 | 0.60 |
| 256 / 256 / 256 | 5.10 |
| 512 / 512 / 512 | 36.70 |
| 1024 / 64 / 32 | 0.95 |

The boundary includes a four-byte map-completion fence and browser/API costs.
`performance.now()` is visibly quantized; these are not GPU timestamps.
Do not compare these Mac browser numbers directly with Furnace CUDA numbers.
The separate browser adapter probe reported `apple / metal-3`, non-fallback.
The actual Rust runtime reported `BrowserWebGpu` but no adapter identity,
reflecting wgpu 0.20's metadata limitation. These are distinct observations.

`browser-resident-typed.json` reruns all 15 cases and 26 assertions after the
TypeScript transport fix. Generated declarations now expose number dimensions,
Float32Array operands, optional dispatch repetitions, and typed readback/fence
promises; `resident_types.cjs` verifies that generated API. The final page hash
matches the committed browser fixture. The earlier browser result is retained.

## Provenance And Validation

The native module path/hash, adapter identity, input hashes, script/helper
hashes, and raw samples are in the report. This is a private 0.4.27 development
wheel in a new isolated venv, not a replacement of public PyPI artifacts.
The local/system SpiralTorch installation was not changed.

Furnace source is an isolated detached worktree based on
`9beeca30acb2575a43d640ce9d6e5588266e7646`, plus the transferred resident
implementation and benchmark files. The complete implementation patch and
build/test logs live in
`~/.local/state/spiraltorch/benchmarks/resident-matmul-v1/`.
Git HEAD alone is not the identity of these uncommitted development builds.

Verified on Furnace:

- Resident backend: 3 tests including real-device edge precision, invalid-input
  atomicity, stale-output rejection, overlapping snapshots, and two-layer GPU
  chaining after workspace drops.
- Python: 2 tests with real WGPU enabled, public import identity and GPU chaining.
- Existing Tensor: 44 matmul-filtered tests with strict real WGPU enabled after
  moving its shader to the shared backend.
- Backend Clippy: native and wasm32, all targets, warnings denied.
- Browser-feature WASM release build and real browser execution succeeded.
- All seven existing Node WASM test scripts passed with a WebGPU-enabled artifact
  (numeric ingress, FFT, autograd SGD/nonlinear, LayerNorm, indexing, classification).
- The 3 resident Rust tests passed again after the dependency backport.
- Workspace inventory and pinned-nightly workspace formatting passed.
- Final private Python wheel: 2/2 real-WGPU tests and 12/12 comparison cases.
- Dependency-free benchmark identity preflight: 3/3 tests.

A broader Clippy run over all st-wasm bindings found 19 existing warnings in
unchanged files (cobol/cosmology/fractal_field/scale_stack/fft/mellin bridges).
The existing CI checks are preserved; the added binding feature gate uses
`cargo check` and a release build rather than claiming those warnings passed.

## Preserved Failures

The first two browser attempts timed out because the test server omitted
wasm-bindgen's generated JS snippets. Those runner failures are retained locally;
the server now serves an explicit asset allowlist and fails promptly on missing
module assets. They are not evidence of a GPU timeout.

`browser-device-request-negative.json` is the subsequent real browser failure:
the unpatched wgpu 0.20 sent the removed `maxInterStageShaderComponents` limit
and Chrome rejected device creation. The [minimal backport](../../../vendor/wgpu-0.20.1/SPIRALTORCH_PATCH.md)
changes only the upstream WebGPU limit mapping. All other vendored Rust sources
and both licenses match the crates.io archive; its SHA-256 was verified.

Source inspection separately exposed wgpu 0.20's unimplemented browser
`on_submitted_work_done`. The resident browser path uses a four-byte
map-completion fence instead. That transfer is included in browser timings;
it is not a CPU computation fallback. The negative device-request fixture
failed before it could exercise queue completion.
