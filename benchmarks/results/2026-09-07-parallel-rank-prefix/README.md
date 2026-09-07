# Parallel Rank Prefix

Native baseline: `593c4be79104e74b7638a3602551980b90ab5d32`.
Guarded implementation: `322f831a0c16688862e76b87893af8360456b661`.
Later commits add evidence and a runner usage string, not compiled library
changes. Browser baseline is the frozen `920193b9` build from the pair-lanes
study: its backend/WASM Rust sources equal `593c4be7`. It is not relabeled as
a fresh build. Raw compiler WASM and served products have separate hashes;
there is no byte-reproducible-build claim.

## Change And Boundary

The existing parallel MidK lower-bound merge also serves short TopK/BottomK
prefixes. Each candidate's global rank is its own tile offset plus predecessor
counts in other sorted tiles. Only the first k candidates per tile can occur
in the global first k; their total (value, source-index) ranks give unique
destinations. Missing-value padding is written only into the disjoint tail.

Rust owns admission, the matching dispatch grid and the uniform mode tag.
WGSL follows that tag rather than independently reconstructing the choice.
The private uniform remains 32 bytes; its last word now carries the mode.
There is no extra dispatch, public API, planner tile rewrite or policy reward
change. Python and WASM execute the same backend implementation. Golden's
separate ordered-f64 arithmetic mean is untouched, not replaced by GPU/f32
arithmetic. This does not change Golden's plan-execution boundary.

Initial `5138ec3d` admitted k=2..256. All 99 browser cases were exact, but k=7
regressed: median ratios 1.200 (2 rows) and 1.127 (64 rows), tile=512. This
negative run and its distinct fixture are retained. The final route admits
only **2..32 tiles and k=64..256**, leaving single tiles, smaller/larger k and
more fragmented rows on the original path. The existing MidK modes are retained.
The expanded fixture measures k=63/64 and 256/257 explicitly.

## Matched Results

Ratios below are medians across six cells (TopK/BottomK, three seeds) of
candidate/baseline **sample-mean latency**, not confidence intervals. Smaller
is faster. All cells, including slower controls, are in [summary.json](summary.json).

Browser, cols=8193, tile=512:

| Rows | k | First | Repeat |
| --- | ---: | ---: | ---: |
| 2 | 7 (unchanged) | 1.008 | 1.003 |
| 2 | 63 (unchanged) | 1.000 | 1.002 |
| 2 | 64 | 0.523 | 0.524 |
| 2 | 65 | 0.520 | 0.520 |
| 2 | 256 | 0.181 | 0.181 |
| 2 | 257 (unchanged) | 0.999 | 0.999 |
| 64 | 64 | 0.994 | 0.995 |
| 64 | 65 | 0.993 | 0.994 |
| 64 | 256 | 0.713 | 0.709 |

Each guarded browser run has 117 cases, including 33-tile, single-tile and
MidK controls. Three deterministic seeds include quantized ties, non-finite
exclusion and signed zeros. Four warmup pairs precede 16 retained pairs of
64 resident operations plus completion; arms alternate inside each case.
Exact value/source-index readbacks and input creation are outside timing.
A/A between the guarded runs has median 1.000 and range 0.989-1.012.
The strongest targeted group is about 82% shorter, not every workload.
64-row k=64/65 is essentially unchanged. Small-k/unchanged cells still vary;
the repeat MidK control is 1.009, and some cells exceed the A/A range slightly.
These are host API/fence timings on an active desktop, not GPU-event timing.
Chrome exposes anonymous BrowserWebGpu adapter identities.

Furnace, cols=8193, k=65, TopK/BottomK combined:

| Rows | Tile | First | Reverse |
| --- | ---: | ---: | ---: |
| 2 | 512 | 0.468 | 0.462 |
| 2 | 1024 | 0.533 | 0.531 |
| 2 | 2048 | 0.788 | 0.819 |
| 64 | 512 | 0.559 | 0.563 |
| 64 | 1024 | 0.561 | 0.562 |
| 64 | 2048 | 0.814 | 0.827 |

Native WGPU/Vulkan and PyTorch **2.13.0+cu132 CUDA** use the same named RTX 5090.
A-B-B-A source order covers 72 requests and **1,152 real Rust UCB observations**.
Each request has six rotated SpiralK fixed controls, which never seed Black Cat,
and 16 correctness-gated policy observations. Source/build/image identities,
exact outputs, source indices and feedback receipts all pass. Controls include
unchanged tiles 32/128/256 and all MidK modes; their modest noise is retained.
The storage sort at tile=2048 is unchanged, but its TopK/BottomK merge changes.

CUDA uses stable sort for MidK/ties and topk for tie-free TopK/BottomK, with
exact source-index validation. Native/CUDA timings are separate process blocks,
not interleaved or GPU events. Even the hindsight-best fixed WGPU candidate is
still **1.079-2.370x CUDA** across the candidate runs (baseline A1: 1.350-5.158x).
This narrows the gap, not a PyTorch win, an online-policy win, a model-quality
result or a new SpiralTorch CUDA kernel. Performance admission is bounded by
these fixtures, not a universal speed guarantee for arbitrary finite sparsity.

## Validation And Replay

macOS and Furnace each pass 95 live backend tests plus WGSL parsing, with GPU
and timestamp tests enabled. New coverage checks Rust mode/grid boundaries,
non-power-of-two/partial tiles, k=63/64 and 256/257, signed zeros, extreme finite
values, ties, all-NaN/sparse rows, missing padding and mixed/all-NaN/mixed reupload.
Both complete WGSL modules share the validated canonical prelude. Strict native
and WASM Clippy, generated TypeScript checks and formatting pass. The fresh,
wheel-byte-matched Python extension passes 103 tests; browser standard/profile
suites pass 71/12 cases, including timestamp failures. Six receipt-validator
unittests pass. An extra pytest invocation collected an unrelated ancestor
package and failed before those tests; its log and the direct-unittest recovery
are retained. No unrelated package, dependency or CI job is changed.

Owned GPU jobs never overlap, including across hosts. No same-host build overlaps
benchmark timing. Each Furnace launch has an availability check; these and the
runner gates are point checks, not continuous monitoring. Existing CPU workloads
remain untouched. Private session listings are excluded; GPU-only observations
retain their original hashes.

[raw-logs.tar.xz](raw-logs.tar.xz) contains fixtures, request hashes, reports,
source bundles, product identities, negative runs, build/test logs and replay
scripts. Large executables, wheels and generated modules are not embedded.
Extract and replay against the unchanged repository validators, without Torch,
a browser or GPU:

```sh
python -I analyze_prefix.py --repo /path/to/SpiralTorch --output recomputed.json
```

The summary records 59 raw-file SHA-256 hashes. Separate extraction and replay
reproduce the committed summary byte-for-byte.

Archive SHA-256: `35f0d331b4fc9c4e31f4b7ad8e267ea62412319a87cf2313b97243e532f92619`.
Summary SHA-256: `8e831ea5f185aedaac307d156e24c5f9d39e4663a970ebf807bd4a3d30645b55`.
