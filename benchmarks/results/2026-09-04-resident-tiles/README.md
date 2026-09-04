# Resident Matmul: Tiles And Shared-Memory Layout

This experiment uses the same Rust shader from native WGPU, browser WASM, and
host Tensor matmul. The optimization changes only the RHS workgroup-memory
address from `column * TILE_K + k` to `k * TILE_N + column`, for both stores and
loads. Global RHS packing, f32 accumulation and fusions are unchanged.

Rust now owns an explicit checked `MatmulTile`; Python exposes `tile_mnk` and
WASM exposes `createWithTile` / `tileMNK`. The default stays **8x8x16**.
Device limits are checked before allocation. No Python-side tuning heuristic,
implicit CPU fallback, TF32, or approximate math was introduced. Host Tensor
autotune revision 4 prevents reuse of timings from the old layout.

## Paired Measurements

Six tiles were tested in each run: 8x8x16, 8x16x16, 16x8x16, 16x16x16,
16x16x32, and 16x16x64 (M/N/K). Each shape uses seeds 17, 29 and 43;
five warmups and 20 samples per variant; variant order is shuffled each
iteration. Inputs and outputs stay resident. Each sample contains **16 API
dispatch calls followed by a completion wait**, not one GPU-timestamp interval.

Both layouts were measured twice, in baseline / row-major / baseline-repeat /
row-major-repeat order. Before/after runs are separate processes, not an
interleaved comparison of binaries. Native script/helper/input hashes match
across all four runs; browser page hashes match across all four browser runs.
Repeated runs retain the identical native/WASM binary hashes for each layout.

Each native report passes 12 shape/seed cases with six SpiralTorch variants
and a PyTorch control; each browser report passes 90 shape/seed/tile cases and
123 input/ownership assertions. All eight reports are retained, including
unfavorable tile choices. These repetitions are not confidence intervals.

### Same-GPU Native Comparison

Furnace: RTX 5090, driver 595.84, WGPU **Vulkan** versus PyTorch 2.13.0+cu132
**CUDA**, float32 throughout and TF32 disabled. One visible CUDA device and
matching adapter names were required; this is not a cross-API UUID attestation.
No competing GPU process was observed before the experiment. The existing
CPU llama-server was left untouched.

512x512x512 results, ms per 16-call sample. Each endpoint is the median of
three seed medians from one run; ranges span the two runs of each layout:

| Tile M/N/K | Before | Row-Major RHS |
| --- | ---: | ---: |
| 8x8x16 (default) | 1.781-1.806 | 1.703-1.706 |
| 8x16x16 | 2.007-2.133 | 1.482-1.484 |
| 16x8x16 | 1.703-1.705 | 1.482-1.484 |
| 16x16x16 | 2.001-2.009 | 1.287-1.341 |
| 16x16x32 | 3.583-3.793 | 1.281-1.284 |
| 16x16x64 | 3.431-3.514 | 1.280-1.283 |

The same-run PyTorch control after the change is 0.240-0.241 ms. PyTorch
remains substantially faster. The 16x16x16 layout change cuts measured sample
latency by roughly 33-36%; it is not a claim that the unchanged default tile
gets that speedup. Increasing K alone was particularly bad before the layout
change, so "bigger tiles are faster" was not a safe optimization rule.

Other shapes (64 cubed, 256 cubed, and 1024x64x32) remain in the raw reports.
Small cases remain dominated by API/queue/wait costs. Even the default tile's
256-cubed timings vary enough between runs that a broad default-tile speedup
claim would be misleading.

### Browser / WASM

Isolated headless Chrome 152.0.7977.77 on Mac; the separate browser probe
reported Apple / metal-3, non-fallback. Rust wgpu 0.20 reports BrowserWebGpu but
no exact adapter identity. The browser probe is not the Rust device's identity.
The recorded `--enable-unsafe-webgpu` flag and four-byte map-completion fence
are part of this test boundary, not hidden GPU-kernel timing.

512x512x512, the same seed-median/range convention and ms per 16-call sample:

| Tile M/N/K | Before | Row-Major RHS |
| --- | ---: | ---: |
| 8x8x16 (default) | 36.60-37.70 | 36.60-36.70 |
| 16x16x16 | 32.70 | 32.80 |
| 16x16x32 | 34.40 | 33.60 |
| 16x16x64 | 36.90 | 32.90 |

The layout helps the wider-K variants, but does not improve 16x16x16 on this
browser. Choosing 16x16x16 explicitly still improves on the default tile for
this shape; the layout itself must not be credited for that tile-choice gain.
At 64 cubed the first changed run rose from about 0.6 to 0.7-0.8 ms, but the
repeat returned to about 0.6 ms. Browser clocks are visibly quantized; small
differences are inconclusive. Do not divide these timings by Furnace's CUDA
numbers or infer an LLM training-quality improvement.

## Validation And Provenance

- Rust resident backend: 5 tests pass on NVIDIA/Vulkan, including odd tile
  3x5x7, edge dimensions, ownership, chained workspaces and snapshot lifetimes.
- Host Tensor: 45 matmul-filtered tests pass with strict WGPU enabled. A new
  test explicitly dispatches both f32 and int8 weights over four tile shapes;
  dyadic, exactly quantizable fixtures isolate addressing from quantization
  error. Existing scaled, transposed and fused matmul regressions also pass.
- Python: 2 real-WGPU tests pass; CPU-only binding `cargo check` passes.
- WASM release build, generated TypeScript contract and four real-browser
  passes succeed; all seven existing Node WASM regression scripts also pass.
  Benchmark preflight has 4 passing dependency-free tests.
- Backend Clippy passes with warnings denied on native and wasm32 targets;
  the `st-tensor` CPU/WGPU all-targets strict Clippy gate also passes.
  Vendored upstream wgpu dependency warnings are not suppressed or repaired by
  this change. Pinned-nightly workspace formatting and inventory checks pass.

Baseline source is parent `95e8e8d01ca3d75ba2c6dff23dc528c8825546b3` plus
`tile-api-before-layout.patch.gz` (decompress before applying in an isolated
checkout). `baseline-source.sha256` records the pre-layout kernel, backend,
and harness files. The after-layout builds add the two RHS address changes
and Tensor autotune revision 4. Later test-only additions and Rust formatting
do not change benchmark semantics; the binary hashes in each report remain
the authoritative identities of the measured artifacts.

Native reports record module/input/script/helper hashes and actual tile and
adapter data. Browser reports record WASM/module/page hashes and all raw
samples. `files.sha256` records these retained artifacts. Build/test logs and
private candidate wheels are under `~/.local/state/spiraltorch/benchmarks/resident-tiles-v1/`
on the Mac/Furnace respectively. These are private 0.4.27 development wheels,
not replacements of public PyPI wheels or the user's system installation.

Reproduction commands are in [the API guide](../../../docs/resident_webgpu_matmul.md).
