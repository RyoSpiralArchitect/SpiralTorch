# Resident Rank Adaptation: Rust, Python, and Browser WebGPU

SpiralK candidates now allocate the exact resident rank workspace selected by
Rust's Black Cat session. This is an execution connection, not a second
Python/JavaScript planner. TopK's new `rank_tile` is independent of FFT's
`tile_cols`; MidK/BottomK retain `ctile`. Existing defaults are unchanged.

The follow-up [cached MidK tournament study](../2026-09-07-midk-tournament/README.md)
uses these controls to optimize the shared Rust/WASM kernel, including
many-row regression evidence and matched browser comparisons.

## Measurements

Two complete repetitions on Furnace's RTX 5090 passed: 72 cases and 4,608
correctness-gated adaptive observations. Each run covers seeds 17/29/43,
TopK/MidK/BottomK, two shapes, and UCB/Thompson sampling. Seed 43 includes ties.
Every result is checked against canonical stable values AND source indices.

Candidates request tiles 32/128/256/512. Rust clamps tiles to the column count.
Each case has rotating equal-count controls (12 measured batches per candidate)
and a separate, initially uncredited 64-observation policy loop.
Each batch dispatches 16 resident rank pairs and waits for completion.

The table reports adaptive **mean** per-operation time divided by the fixed-256
control mean, including all 64 exploration choices. Entries are medians and
ranges over three seeds x two repetitions, NOT six independent seeds.
Rows are always 2; shapes are columns/k. Below 1 is faster than fixed 256.

| Kind | Columns/k | Policy | Mean Ratio | Range |
| --- | --- | --- | ---: | --- |
| TopK | 257/7 | UCB | 0.919 | 0.909-0.926 |
| TopK | 257/7 | Thompson | 0.944 | 0.939-0.968 |
| MidK | 257/7 | UCB | 0.906 | 0.900-0.920 |
| MidK | 257/7 | Thompson | 0.932 | 0.912-0.989 |
| BottomK | 257/7 | UCB | 0.917 | 0.903-0.919 |
| BottomK | 257/7 | Thompson | 0.974 | 0.914-0.978 |
| TopK | 8193/65 | UCB | 1.079 | 1.071-1.108 |
| TopK | 8193/65 | Thompson | 1.098 | 1.074-1.123 |
| MidK | 8193/65 | UCB | 0.657 | 0.651-0.706 |
| MidK | 8193/65 | Thompson | 0.818 | 0.803-0.837 |
| BottomK | 8193/65 | UCB | 1.082 | 1.056-1.100 |
| BottomK | 8193/65 | Thompson | 1.080 | 1.057-1.118 |

Large MidK benefits from tile 512, and UCB selects it more frequently.
Large TopK/BottomK often already favor 256, so exploration adds cost.
Do not replace these means with the more favorable median-step statistic:
for example, large MidK/UCB's median-step ratio is about 0.456, but its mean
ratio including exploration is 0.657.

Even the hindsight-best fixed WGPU controls remain **1.68-5.89x slower** than
their PyTorch 2.13.0+cu132 CUDA references by median latency. This is not a
PyTorch win. CUDA uses stable sort for MidK or tied rows, and topk otherwise.
Native WGPU/Vulkan and PyTorch CUDA execute in separate process blocks, never
overlapping; this is not cross-framework interleaving or GPU-event timing.

Allocation, upload, policy work, validation readbacks, and the control sweep
are outside adaptive timing. These are dispatch diagnostics, not total
cold-start tuning cost, convergence proof, or training-quality improvements.
Readbacks occur between batches and before each adaptive observation is credited.

## Portable Validation

- Browser WASM/WebGPU: 18 cases, 432 adaptive observations, both policies, exact
  values/indices, ties and nonfinite filtering. The browser adapter is anonymous;
  no physical-GPU identity or browser/native performance equivalence is claimed.
- Packaged Python on Apple M4/Metal: 24 tests passed, one optional local PyTorch
  interoperability test skipped. Real rank allocation/dispatch/readback, hint
  synthesis, and resident matmul regressions are included. No PyPI publication.
- Rust on macOS and Furnace/Linux: st-core 1,025 and st-kdsl 78 tests passed
  on each host. WASM host-side tests: 169 passed.
- Native example strict Clippy, formatting, benchmark admission tests (34),
  generated tuner table equality, and Python feature-isolation checks passed.
- GPU-free CI subset: 8 passed, 6 real-GPU cases intentionally skipped.
  The new six-test benchmark audit also runs in isolated Python in CI.

Full WASM-binding strict Clippy still has 19 pre-existing diagnostics.
A fresh baseline target reproduced exactly the same messages and locations;
the candidate adds zero. The first shared-target baseline attempt reused stale
SpiralK metadata and is retained as an invalid build attempt, not a baseline.

CI additionally caught a missing optional field in a WASM-only test initializer.
That test fixture is fixed; both default and WebGPU `wasm32 --all-targets` checks
now pass locally. [ci-validation.tar.gz](ci-validation.tar.gz) preserves the
initial CI failure and both successful checks, plus strict SpiralK Clippy.
No measured runtime code changed in this follow-up.

## Review Fix and Separate Replay

Review found that the TopK-only `rank_tile` hint also replaced the generic tile
used by BottomK refinement. A regression reproduced `ctile` changing from 512
to 128 from that hint alone. Commit
`5caf31829ecc668493e2a08d832b7e611745fd10` gates the hint on TopK in both heuristic
conversion and explicit RankPlan rewrites. MidK/BottomK keep their existing
tile sources; no default tile or GPU fallback policy was changed.

Fresh artifacts from that clean source passed a separate replay:

- Furnace: 36 cases and 2,304 correctness-gated adaptive observations.
- Browser: 18 cases and 432 observations, with exact values/stable indices.
- Packaged Python, Apple M4/Metal: 24 passed, one optional PyTorch test skipped.
- macOS Rust: st-core 1,027 and st-kdsl 78 passed. Linux with `wgpu-rt,kdsl`:
  st-core 1,053 passed, one live-adapter test ignored, and st-kdsl 78 passed.
- Native example and SpiralK strict Clippy, WASM/WebGPU all-targets check,
  formatting, and 21 rank benchmark admission tests passed.

Large MidK/UCB's mean ratio is 0.656 (range 0.646-0.713 across three seeds).
Large TopK/BottomK still pay exploration cost: 1.057-1.125 across both policies.
The hindsight-best fixed WGPU control remains 1.69-5.37x slower than CUDA.
This is one post-fix repetition, not additional independent seeds or a pooled
extension of the earlier table. The same timing exclusions still apply.

[review-replay-summary.json](review-replay-summary.json) records every case,
native source/build identity, Python extension identity, browser asset hashes,
and log hashes. [review-replay.tar.gz](review-replay.tar.gz) retains raw reports,
the failing regressions, successful checks, build logs, and the recomputation
script. The first regression attempt failed to compile because the test tried
to compare a type without PartialEq; the second reproduced the actual bugs.
After extraction, run
`python -I analyze-review-fix.py --repo /path/to/SpiralTorch`.
Python/WASM source labels rely on retained build logs and artifact hashes;
unlike the native executable, they do not claim embedded Git attestation.

Replay archive SHA-256:
`480a03fd9fb72e837b36dfa96f904564154e992713d70881d27075ea6f7ca1b0`.
Replay summary SHA-256:
`fac378126b809d25919c004c7495c5f700b3ecbdad4272ef4ebd197c412f9074`.

## Provenance

The original table's native executable and browser WASM were built from clean
`510a3b0487ea4959e34071d3e555d1e1191e12b7`.
Both native reports validate the embedded Rust build manifest against the exact
clean checkout before and after execution, and hash the copied execution image.
The private final Python wheel uses `365cd1fdebd3da7a450b5ccf1ae586f0da2cccb7`,
which adds the Python hint-field admission; native measured code is unchanged.
The browser fixture's typed-array comparison was corrected at
`25cd923c3ea0fc346c0da4b8c6ec3903756fcb90`. Later tests/CI/docs changes precede
the runtime review fix and its separate replay described above.
Raw failed fixture attempts remain preserved and are not counted as passed runs.

[summary.json](summary.json) contains all per-case means, medians, choices,
ratios, and raw-report SHA-256s. [raw-logs.tar.gz](raw-logs.tar.gz) includes the
unmodified reports, successful and failed test/build logs, and `analyze.py`.
After extracting the archive, recompute with
`python -I analyze.py --repo /path/to/SpiralTorch`.

Archive SHA-256:
`d7bd3434d88f001e19ece1c397a7d6c6afcd2253216119884a53886933b42ca4`.
Summary SHA-256:
`6078f2ed7153e52d15e1837bf8ac5c14e89ad65bf11cb305d3fc35b3798af273`.
