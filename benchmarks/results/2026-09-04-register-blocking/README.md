# Resident Matmul: Explicit Register Blocking

Rust now supports `MatmulKernel::Register2x2`: four scalar accumulators per
invocation reuse two LHS and two RHS values per K step. Cooperative tile loads
use invocation coordinates, not output coordinates; edge outputs are guarded
individually. Each output keeps ascending-K float32 accumulation. Python and
WASM expose this same implementation and its actual invocation geometry.

**Scalar remains the default on every target.** Register blocking is explicitly
selected, requires even output tile M/N, and does not relax the existing
256-output-cell tile limit. Host Tensor still selects scalar kernels. Its
candidate validation now checks scratch storage and overflow as well as
invocations; autotune revision 5 avoids reusing older shader timings.

## Same-Binary Paired Measurements

Each final v3 report compares both kernels across six M/N/K tiles: 8x8x16,
8x16x16, 16x8x16, 16x16x16, 16x16x32 and 16x16x64. Seeds are 17, 29, 43;
five warmups and 20 samples per variant, with randomized variant order each
iteration. The two repeats use identical binaries and harness hashes. All
variants use identical input bytes and a float64 correctness reference.

Times below are **milliseconds per 16 separate API calls plus synchronization**,
excluding allocation, upload and full readback. Each endpoint is the median
of three seed medians in one run; ranges span two runs, not confidence intervals.
These are not GPU timestamps, training steps, or model-quality measurements.

### Browser / WASM

Isolated headless Chrome 152.0.7977.77 on Mac. The separate browser probe
reported Apple / metal-3, non-fallback; Rust wgpu 0.20 reports BrowserWebGpu
without exact adapter identity. The probe is not an identity attestation for
the Rust device. `--enable-unsafe-webgpu` and the four-byte map completion
fence are recorded and included in this boundary.

512x512x512, same tile and binary on both sides:

| Tile M/N/K | Scalar | Register 2x2 |
| --- | ---: | ---: |
| 8x8x16 | 36.60 | 34.00 |
| 8x16x16 | 34.80-34.85 | 15.80-15.90 |
| 16x8x16 | 34.80 | 16.10-16.20 |
| 16x16x16 | 32.80 | 14.30 |
| 16x16x32 | 33.60-33.65 | 13.90 |
| 16x16x64 | 32.90 | 14.70-14.80 |

The 16x16x16 blocked variant is about 2.29x as fast in this measured browser
boundary. At 256 cubed, the same tile goes from 4.60 to 2.30 ms. Small-case
timings vary materially across repeats (for example, 7x63x65 scalar is
0.85 then 0.60 ms); clock quantization and API/wait costs limit interpretation.
This does not justify making blocking the default on all browser GPUs.

Both final browser reports pass 180 shape/seed/tile/kernel cases and 403
input/ownership/geometry assertions. No browser PyTorch control is present;
do not divide these timings by Furnace's CUDA measurements.

### Furnace: WGPU Vulkan Versus PyTorch CUDA

RTX 5090, driver 595.84, PyTorch 2.13.0+cu132. Both paths use float32 and
TF32 is disabled. Exactly one visible CUDA device and matching adapter names
are required, not a portable cross-API UUID attestation. GPU was idle before
the experiment; the existing CPU llama-server was left untouched.

Same explicit 16x16x16 tile, milliseconds per 16-call sample:

| Shape M/K/N | WGPU Scalar | WGPU Register 2x2 | PyTorch CUDA |
| --- | ---: | ---: | ---: |
| 64x64x64 | 0.2622-0.2630 | 0.2956-0.2959 | 0.1288-0.1292 |
| 256x256x256 | 0.4783-0.4791 | 0.6385-0.6619 | 0.1600-0.1610 |
| 512x512x512 | 1.2934-1.2940 | 1.1155-1.1179 | 0.2409-0.2429 |
| 1024x64x32 | 0.2623-0.2627 | 0.2960-0.2968 | 0.1202-0.1205 |

At 512 cubed this cuts sample latency by about 14%, but PyTorch remains
roughly 4.6x faster. At 256 cubed blocking is **33-38% slower**; 8x8x16
blocking is also substantially worse on NVIDIA. All six tiles, including
losing choices, remain in the reports. No default policy or universal winner
is inferred from backend names.

Each native report passes 12 shape/seed cases, each with 12 SpiralTorch
variants and a PyTorch control. The harness keyword `auto` means the API
default, not autotuning; these measurements request both kernels explicitly.

## Long-K Precision Follow-Up: Not Yet Qualified

An additional Transformer-shaped sweep uses 1x768x768, 32x768x768,
128x768x3072, and 128x3072x768. `native-transformer-shapes.json.gz` retains
all 12 cases: **10 pass, two fail** the unchanged `rtol=1e-4, atol=1e-5`
float64-reference gate. The failing FFN-contraction cases use K=3072 with
seeds 17 and 43. No timings are admitted for those failed cases.

`diagnose_long_k.py` bypasses the benchmark's early correctness stop to inspect
both kernels at 8x8x16 and 16x16x16, plus a CPU sequential-f32 accumulator
and PyTorch CUDA. It records errors, not performance claims. The exact same
script and inputs also run against the pre-change PR #2061 wheel; its native
hash matches `../2026-09-04-resident-tiles/native-six-tiles-row-major.json`.
The baseline's `None` kernel label means its constructor has no kernel option.

| Seed | Baseline / current WGPU failed cells | WGPU max absolute error | PyTorch CUDA failed cells |
| --- | ---: | ---: | ---: |
| 17 | 1 / 1 | 4.49e-5 | 0 |
| 29 | 0 / 0 | 4.18e-5 | 0 |
| 43 | 2 / 2 | 3.96e-5 | 0 |

Both current kernels and the older scalar wheel reproduce the same error
metrics and worst failing-cell values. Sequential CPU f32 also fails this
gate (it is not bitwise identical to the GPU). This is evidence of an existing
long-reduction accuracy limitation, not a new register-blocking regression.
Do not use this comparison to relax the tolerance or claim all FFN shapes
are qualified. A more stable Rust/WASM accumulation path, benchmarked against
the unchanged fast path, remains follow-up work.

## Negative Prototype And Provenance

The first prototype automatically enabled blocking for even tiles with at
least 256 output cells. Its native regressions motivated the explicit API.
`prototype-source.patch.gz` applies to
`4837ddea8cb7143b8c3d80a41ae485620c12f46c` and preserves that rejected policy.
Its v2 native and two browser reports are retained separately as
`prototype-*.json.gz`. They are not final API/binary measurements and do not
belong in the same-binary v3 comparison above.

Final v3 reports record native/WASM, input or page, and harness hashes, actual
kernel/tile/geometry, adapters, and every timing sample. `source-files.sha256`
identifies the final implementation and harness files; `files.sha256` covers
the retained experiment artifacts. Private builds are 0.4.27 development
wheels, not replacements for public PyPI wheels or system installations.
Build/test logs and private wheels are under
`~/.local/state/spiraltorch/benchmarks/register-blocking-v1/` on Mac/Furnace.

## Validation

- Rust resident backend: six tests pass with real NVIDIA/Vulkan opted in,
  including both kernels, odd tile rejection, edge outputs, shared-memory
  geometry, chained workspaces and independent snapshot lifetimes.
- All 991 `st-core` library tests pass.
- Host Tensor: 46 matmul-filtered library tests plus the f32 precision
  integration test pass. Eight tile geometries execute real scalar f32/int8
  fixtures. Blocked int8 is shader-validated only, not runtime-qualified or
  exposed by the f32-only resident API.
- Python: two real-WGPU tests pass. Benchmark preflight: five tests pass.
- WASM release build and generated TypeScript contract pass, as do all seven
  existing Node numeric/autograd/classification regression scripts and the
  two real-browser runs above.
- CPU-only Python binding check and strict backend/Tensor all-targets Clippy
  pass on native and wasm32. Pinned-nightly formatting and workspace inventory
  checks pass. Existing vendored wgpu dependency warnings are not suppressed.

See [the API and reproduction guide](../../../docs/resident_webgpu_matmul.md).
