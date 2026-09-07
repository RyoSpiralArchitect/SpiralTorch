# Finite Tile Bounds

Native baseline: `6da35010c48f3461024680862735edc9c4530c61`.
Bracketed implementation: `8a3624f89b60b1586167e5c2978b2d19848dadeb`.
Later commits add evidence, not compiled library changes. Browser baseline is
the frozen `322f831a` product from the parallel-prefix study. Its backend/WASM
Rust sources equal `6da35010`; it is not relabeled as a fresh build. Raw compiler
WASM and served products have separate hashes, without a reproducible-build claim.

## Change And Invariants

After exact tile sorting, lane zero used to scan every finite index to count
the run. Finite indices already precede every INVALID_INDEX, independently of
TopK/MidK/BottomK value direction. The replacement checks empty/full endpoints,
brackets a sparse prefix by capped exponential steps, then finds the first
invalid index by binary search. Capping each step at the known invalid upper
bound also prevents unsigned overflow. The bound is the actual tile length,
not its padded storage stride, so full partial tiles take the endpoint path.

Workgroup-memory and storage-memory helpers use the same lower-bound invariant
with their respective address spaces. Only lane zero publishes the count.
No comparator, float arithmetic, sorting-network stage, barrier, dispatch grid,
buffer size or binding layout changes. This is the shared Rust backend shader
used by resident Rust, Python and WASM clients, not a Python reconstruction.
SpiralK plan meanings, Black Cat rewards and Golden's ordered-f64 mean remain
unchanged. No new public API, dependency, CI job or package release is introduced.

Initial binary-only `b4bf9399` passed all 117 browser cases but showed a sparse
regression signal: one finite value per tile had median latency ratio 1.031.
Full-width binary search did unnecessary probes for such short runs. The
bracketed candidate removes that search-cost problem without a fixed density
threshold. The initial report is retained as `browser-ab-first.json`; it is
not mislabeled as the final candidate. All runs use the same fixture hash.

## Fixed-Control Results

Entries are medians across nine cells (three kinds, three seeds) of
candidate/baseline **sample-mean latency**. Smaller is faster. They are not
confidence intervals, policy speedups, or universal workload guarantees.
All cells, including slower cases, remain in [summary.json](summary.json).

Browser mixed fixtures, k=65:

| Rows | Cols | Tile | First | Repeat |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1023 | 32 | 0.970 | 0.975 |
| 2 | 4095 | 128 | 0.897 | 0.898 |
| 2 | 8191 | 256 | 0.837 | 0.837 |
| 2 | 8193 | 512 | 0.664 | 0.664 |
| 2 | 8193 | 1024 | 0.448 | 0.451 |
| 2 | 8193 | 2048 | 0.390 | 0.393 |
| 64 | 8191 | 256 | 0.880 | 0.876 |
| 64 | 8193 | 512 | 0.832 | 0.835 |

Controlled finite counts, rows=2, cols=1025, tile=512, k=65:

| Finite Per Tile | First | Repeat |
| --- | ---: | ---: |
| 0 | 0.996 | 1.000 |
| 1 | 0.992 | 1.008 |
| 7 | 1.016 | 1.000 |
| 511 | 0.434 | 0.435 |
| 512 | 0.429 | 0.427 |

The final partial tile caps the finite count at its length. Each browser run
has 117 cases: 72 mixed-input shapes and 45 density controls. Seeds include
quantized ties and signed zeros; mixed fixtures contain non-finite inputs,
which ranking excludes from the finite candidate set.
Four warmup pairs precede 16 retained pairs of 64 resident operations plus
completion, alternating arms within each case. Input creation and exact
value/source-index readbacks are outside timing. Browser order is initial A/B,
A/A, bracketed A/B, bracketed A/B repeat. A/A median is 1.000, range 0.954-1.032.
Tiny density cases are noisy: a bracketed empty case reaches 1.073 and a
one-finite case 0.914. These are retained, not turned into tiny-case speed claims.
Chrome reports anonymous BrowserWebGpu identities. Timings are host API/fence
measurements on an active desktop, not GPU events.

Furnace finite fixtures, cols=8193, k=65:

| Rows | Tile | First | Reverse |
| --- | ---: | ---: | ---: |
| 2 | 32 | 0.993 | 0.995 |
| 2 | 128 | 0.992 | 0.987 |
| 2 | 256 | 0.950 | 0.940 |
| 2 | 512 | 0.754 | 0.802 |
| 2 | 1024 | 0.615 | 0.616 |
| 2 | 2048 | 0.653 | 0.662 |
| 64 | 32 | 0.913 | 0.931 |
| 64 | 128 | 0.962 | 0.950 |
| 64 | 256 | 0.882 | 0.893 |
| 64 | 512 | 0.729 | 0.733 |
| 64 | 1024 | 0.625 | 0.623 |
| 64 | 2048 | 0.682 | 0.677 |

Native WGPU/Vulkan and PyTorch **2.13.0+cu132 CUDA** use the same named RTX 5090.
A-B-B-A source order covers 72 requests, 432 fixed-control cells and 1,152 real
Rust UCB observations. Each request has six rotated SpiralK controls and 16
correctness-gated observations; fixed controls never seed Black Cat. Immutable
images embed clean source identity, retain before/after hashes and pass exact
outputs/source-index and feedback-receipt validation. No fallback is permitted.

The hindsight-best fixed WGPU candidate ranges **0.734-1.766x** the selected
CUDA reference across both candidate runs. This needs an important qualification:
the existing reference chooses stable_sort for MidK or any duplicate anywhere
within any input row, even outside the retained top/bottom set. It uses topk
only for tie-free TopK/BottomK. All eight below-one cases per candidate run are
against stable_sort. Against topk, ratios remain **1.551-1.766x**. This is not a
win over the fastest possible PyTorch implementation, an online-policy win,
or a new SpiralTorch CUDA kernel. Strengthening the retained-set CUDA reference
is a separate next step. Native/CUDA timings are separate process blocks, not
interleaved or GPU-event timing; non-finite density controls are browser-only.

## Validation And Replay

macOS and Furnace each pass 96 live backend tests plus WGSL syntax, with runtime
and timestamp tests enabled. New full-rank and k=7 coverage includes finite
counts 0/1/2/3/7/8/9/half/almost-full/full, 13 tile lengths across power-of-two
boundaries, the storage-sort boundary, partial tails, signed zeros, extreme
finite values and mixed/all-NaN/mixed reupload. No stale count survives reupload.
Native/WASM strict Clippy, canonical shader sync, generated TypeScript and
formatting pass. The source-built, wheel-byte-matched Python extension passes
103 tests; standard browser and timestamp/failure suites pass 71/12 cases.
Six receipt-validator unittests pass.

Owned GPU jobs never overlap, including across hosts. No same-host build overlaps
benchmark timing. Furnace availability and runner gates are point checks, not
continuous monitoring. Existing CPU workloads and system configuration remain
untouched. Private session listings are excluded; GPU-only observations retain
the original private-status hashes.

[raw-logs.tar.xz](raw-logs.tar.xz) retains generators, request hashes, full reports,
source bundles, product identities, initial negative signals and build/test logs.
Large executables, wheels and generated modules are excluded. Extract and replay
using the unchanged repository validators, without Torch, a browser or GPU:

```sh
python -I analyze_counts.py --repo /path/to/SpiralTorch --output recomputed.json
```

The summary records hashes for all 59 raw files. A separate extraction and
offline replay reproduced the published summary byte-for-byte.

SHA-256:

```text
854288752148b60c75c9ef7a57d5dae3380c9ebafef33e7db5f2fb29e0dfb1a0  raw-logs.tar.xz
a6514eb5288221b9d544596ffcb5556172f8b7bcaf0719e51c20b2243c790fcd  summary.json
```
