# MidK Rank-Bound Pruning

Baseline executable source: `1682ee04951f664677206e176343c4ad9a643d6e`.
Candidate executable source: `09c64a7fe2624828b26f8e60b5cd478111f229b8`.
Later evidence/documentation commits do not change the measured implementation.

## Shared Kernel Change

Parallel MidK computes each candidate's global rank by adding lower-bound
predecessor counts from sorted tiles. Once the partial rank reaches the retained
band's upper bound, remaining nonnegative counts cannot bring it back into the
band. The shader stops that candidate's searches, but still computes the exact
rank for every retained candidate. Signed zeros, source-index tie ordering,
non-finite exclusion and missing-output padding are unchanged.

This is one loop condition in the Rust-owned shader shared by native, WASM and
Python. No planner threshold, tile geometry, buffer, dispatch, client flag,
reward, or default-runtime change is introduced. The <=32-tile parallel route
is unchanged; fragmented MidK, TopK and BottomK keep their existing algorithms.

## Browser Comparison

Three sequential browser runs use A/B, A/A, A/B source order, 54 cases per run,
three deterministic seeds (one quantized), and separate devices in the same
browser. Arms alternate within each case. Four paired warmups precede 16 retained
pairs of 64 resident operations plus completion. Upload, allocation and exact
value/source-index readback validation are outside timing. The A/A run serves
the same frozen baseline module to both arms. Assets and fixture are hashed.
The 64-row fixture uses cols=8191, tile=256 and k=7/65; full per-cell geometries
and seeds are retained in the summary rather than generalized to other shapes.

| Group | Cells/Run | First A/B | Repeated A/B | A/A |
| --- | ---: | ---: | ---: | ---: |
| Parallel MidK, 2 rows | 30 | 0.968 | 0.968 | 1.000 |
| Parallel MidK, 64 rows | 6 | 0.911 | 0.907 | 1.010 |
| Unchanged routes | 12 | 1.002 | 1.001 | 1.000 |
| Single-tile controls | 6 | 1.001 | 1.002 | 1.001 |

Entries are medians over cells of candidate/baseline **mean-latency ratios**.
The 64-row cases consistently improve by about 9%; this is not an all-shape
claim. The 2-row group ranges from 0.871 to 1.021 in the first run and 0.873 to
1.021 in the repeat. The small cols=1023, tile=32, k=7 cases are about 1-2% slower
in the repeat, and remain included. The 64-row A/A range is 0.997-1.013.
Chrome reports an anonymous BrowserWebGpu adapter. These are host API/fence
timings on an active desktop, not pure GPU events, independent new seeds for
every batch, or a physical GPU identity claim.

## Native And PyTorch CUDA

Furnace runs WGPU/Vulkan and PyTorch **2.13.0+cu132 CUDA** on the same named
RTX 5090. Four source-frozen process blocks run A-B-B-A; each whole native/CUDA
run ends before the next begins. This separate matrix crosses rows 2/4 at
cols 8193/16385/32769, rows 64 at cols 8193, and rows 512 at cols 1025. It uses
three seeds, k=65, four fixed tile controls, UCB, and 16 real policy observations
per case. The complete study passes **96 cases and 1,536 Rust observations**.
Fixed controls are not fed to the policy; source-index-correct CUDA uses stable
sort. No source or executable changed during measurement.

| Fixed-Control Group | Cells/Pair | First Pair | Reverse Pair |
| --- | ---: | ---: | ---: |
| Parallel MidK, 2 rows | 3 | 0.987 | 0.990 |
| Parallel MidK, 4 rows | 3 | 1.001 | 1.002 |
| Parallel MidK, 64 rows | 3 | 0.979 | 0.977 |
| Parallel MidK, 512 rows | 9 | 0.994 | 0.995 |
| Unchanged routes | 78 | 1.001 | 0.999 |

Entries again summarize fixed-control mean-latency ratios, not online policy
wins. Native effects are smaller than the browser effects and mixed across
cells. Unchanged-route ranges are 0.964-1.038 / 0.951-1.028. Even the
hindsight-best fixed candidate remains **1.567-3.841x CUDA** across these cells.
That is not a PyTorch win, a model-quality claim, or a SpiralTorch CUDA backend
implementation. The native and browser matrices differ; do not combine their
effect sizes or extrapolate between devices.

GPU availability was inspected before each launch after the prior external
training ended. The runner also checks contention at the start/end of each
combined run, not continuously. The existing CPU llama-server and system
configuration were not changed. Owned GPU jobs and same-host build/timing work
did not overlap.

## Validation And Replay

- Both macOS and Furnace pass 93 live-enabled backend tests plus WGSL parsing.
  The new regression covers eight geometries, three k values and six input
  patterns, including ascending/reversed values, tied signed zeros, extremes,
  non-finite values, all-NaN input and the 32/33-tile route boundary.
- The newly built and byte-checked Python wheel passes 103 tests. Browser
  standard validation passes 71 cases; profiling passes 12 cases, the rank-only
  1024-repetition boundary and 145 validated timestamp reports, including real
  invalid-query and synthetic internal/scope-rejection failure cases.
- Native/WASM strict Clippy, fresh WASM/wheel builds, and generated/shipped
  TypeScript contracts pass. No new Python or WASM API is required.

[summary.json](summary.json) retains every measured cell and raw-file hashes.
[raw-logs.tar.xz](raw-logs.tar.xz) contains reports, source bundle, build/test
logs, product identities, GPU-only availability observations and replay scripts.
Large binaries and unrelated remote session listings are excluded; full status
logs remain local, with their hashes retained in the public GPU observations.
Native build/source and executable identities are checked independently.
Python/WASM attribution is build logs and product hashes, not embedded Git
attestation. After extraction, no GPU, browser or Torch installation is needed:

```sh
python -I analyze_pruning.py --repo /path/to/SpiralTorch --output recomputed.json
```

Extraction and replay reproduced the summary byte-for-byte.

Archive SHA-256: `4a58b12d07fe23b15d3a36d2a6809dcfd19cdb887120ad53937c229e16c982ad`.
Summary SHA-256: `7b5f1ba8ee412a278ece203cc416190d311e83c2e1e1e2c1920d574206f3e97b`.
