# Rank Tile Pair Lanes

Baseline native source: `2e254bb25e47da8340f73655b37e1f97f0215520`.
Guarded implementation: `920193b92c9c7bf6cfb481170edab5b2cd102835`.
Later commits add documentation/evidence and a runner usage string, not compiled
library changes. Baseline WASM is the frozen `2928e357` build from the preceding
tensor-mean study; its backend/WASM Rust sources equal `2e254bb2`. It is not
relabeled as a newly built baseline. Raw compiler WASM and served assets are
frozen and hashed separately; no byte-reproducible-build claim is made.

## Change And Negative Candidate

The shared Rust-owned WGSL enumerates each bitonic compare/exchange pair once
instead of visiting both endpoints and leaving one inactive. Inserting a zero
bit at the current network distance gives disjoint pairs. The comparator,
network stage order, barriers, signed-zero/source-index ordering, non-finite
exclusion and missing-value padding are unchanged.

This only helps the loop structure above the 256-lane width. Initial candidate
`47d88571` applied it to every local tile: browser median latency ratios for
32/128 tiles regressed to 1.028/1.022. That run is retained, not promoted as a
win. The final candidate uses pair lanes only for padded strides 512/1024 and
keeps the original <=256 path. Larger storage-memory tiles are untouched.
No planner tile rewrite, public API, policy reward or additional dispatch is
introduced. Golden's separate ordered-f64 arithmetic mean remains unchanged:
this is not a lower-precision substitute or a Golden GPU-reduction claim.

## Fixed-Control Results

Entries are medians across cells of candidate/baseline **sample-mean latency
ratios**, not confidence intervals or online-policy speedups. Every cell,
including slower controls, remains in [summary.json](summary.json).

| Padded Tile | Browser First | Browser Repeat | Native First | Native Reverse |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1.001 | 0.999 | 1.001 | 1.001 |
| 128 | 1.001 | 0.999 | 1.000 | 1.000 |
| 256 | 1.000 | 0.998 | 0.999 | 1.013 |
| 512 | 0.958 | 0.958 | 0.874 | 0.874 |
| 1024 | 0.967 | 0.965 | 0.870 | 0.865 |
| 2048 (unchanged) | 0.994 | 0.992 | 0.989 | 0.992 |

Browser: 72 cases/run, TopK/MidK/BottomK, three seeds (including quantized ties),
rows 2/64, cols 1023/4095/8191/8193, six tile widths, k=65. The full matrix is
in the replay script. Four warmup pairs precede 16 retained pairs of 64 resident
operations plus completion; arms alternate inside each case. Inputs and exact
value/index readback validation are outside timing. An A/A run precedes the two
guarded A/B runs: median ratio 0.998, range 0.988-1.004. The unchanged 2048 path
also varies slightly. These are host API/fence timings on an active desktop,
not GPU events; Chrome reports an anonymous BrowserWebGpu adapter. The strongest
512-tile cells improve by about 11-12%, not every shape by that amount.

Furnace: native WGPU/Vulkan and PyTorch **2.13.0+cu132 CUDA** run on the same
named RTX 5090, in separate process blocks with A-B-B-A source order. Each run
has 18 requests: three seeds, three kinds, rows 2/64, cols=8193, k=65, six rotated
fixed tile controls and 16 real UCB observations. Across four runs, **72 requests
and 1,152 Rust policy observations** pass. Controls never seed the policy. All
source/build/image identities, exact outputs, source indices and feedback
receipts are checked. A source-index-correct CUDA operation is selected and
recorded: stable sort for MidK/ties, topk for tie-free TopK/BottomK.

Native 512 ratios range 0.853-1.003 / 0.842-1.006, so a few cells do not improve.
Native 1024 ranges are 0.818-0.945 / 0.823-0.912. Unchanged-route noise includes
a 128-tile cell at 1.070 in the reverse pair; it remains included. Even the
hindsight-best fixed WGPU candidate is **1.358-5.109x CUDA** across the candidate
runs. This does not establish a PyTorch win, general framework superiority,
model quality, or a new SpiralTorch CUDA kernel.

## Validation And Replay

macOS and Furnace each pass 94 backend tests with live GPU/timestamp tests
enabled, plus WGSL parsing. New tests exhaust pair-address coverage and expand
non-power-of-two/partial-tile boundaries around 256/512/1024. Full-width ranks,
signed zeros, ties, all-NaN rows and the storage-sort boundary are checked.
The fresh, wheel-byte-matched Python extension passes 103 tests. Browser standard
validation passes 71 cases; the GPU timestamp/failure-injection suite passes
12 cases. Native/WASM strict Clippy, generated TypeScript checks and formatting
pass. No new CI job or dependency is added.

Owned GPU jobs never overlap. No same-host build overlaps measured intervals.
Furnace availability is inspected before each GPU launch; these and the runner
contention gates are point checks, not continuous monitoring. Unrelated CPU
workloads and system configuration remain unchanged. Private session listings
are excluded from the archive; GPU-only observations retain original hashes.

[raw-logs.tar.xz](raw-logs.tar.xz) retains fixture generators, request hashes, reports,
negative candidates, source bundles, product identities, build/test logs and
replay scripts. Large binaries/wheels/generated modules are excluded. Extract
and replay with the unchanged repository validators, without Torch or a GPU:

```sh
python -I analyze_pairs.py --repo /path/to/SpiralTorch --output recomputed.json
```

The summary records raw-file SHA-256 hashes. Extraction/replay reproduced the
summary byte-for-byte. Compiled sources remain unchanged throughout measurement.

Archive SHA-256: `91fa05e149c3b2aed22d645c6002aaf7819fc940b1094701a042989f324cf53c`.
Summary SHA-256: `bb2bf572d327a67b748d229bd255b8c3228d0f259ed0c494f303daae7c80f975`.
