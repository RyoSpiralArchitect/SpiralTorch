# Window-Aware CUDA Admission

Corrected compiled/native/benchmark source:
`2effcf948692e921a358b4f6d9e60d32d4709537`. Later commits only add evidence/docs.
The [initial dc840c97 study](../2026-09-07-canonical-cuda-rank/README.md) remains
unchanged, including its raw reports and summary. Its replay now explicitly
uses a pinned measured checkout rather than the corrected source.

## Review Correction

[Review comment](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2088#discussion_r3946424315)
identified over-rejection of stable sort when mixed signed zeros cannot affect
the returned window. For example, TopK k=1 over `[3, -0, +0]` has the same exact
value and index under stable numeric sort and total-f32 order. Already-canonical
zero ordering can also make the full selected window safe.

Admission now compares the numeric stable sort's returned source-index window
with the canonical total-f32 window. Index repair uses the same window check
and still requires an untied numeric cutoff; it cannot repair an omitted
canonical source index at a tied cutoff. Incorrect zero order inside the
returned window remains rejected. The additional sort is needed only for rows
containing both zero signs, outside timing for the immutable admitted fixture.

No timed `RankControl.run`/`measure` body, Rust/WGPU/WASM implementation,
dependency, public API, or Black Cat feedback rule changes in this correction.
See the [reference contract](../../../docs/development/cuda_rank_reference.md).

## Repeated Measurements

All **98 cases** were rerun with new source-bound mode-555 native images on the
RTX 5090: two 18-case adaptation probes, the standard 54 cases and two targeted
four-case tail-tie probes. Both generators are byte-identical to the initial
study. All 98 request hashes/admission lists remain identical; the mixed-zero
correction is exercised by the live tests rather than reclassified timings.

There are 318 WGPU fixed-control cells, 256 CUDA controls and 704 real Rust UCB
observations. All controls and slower cases remain in [summary.json](summary.json).
WGPU completes before CUDA in each run; CUDA controls rotate within each case.
Each retained sample includes sixteen complete operations plus completion.
Key encoding/repair/gather and internal Torch workspace costs are inside timing;
validation readbacks are outside. Best fixed is hindsight, not online policy,
a confidence interval, GPU-event timing or the fastest possible PyTorch claim.

| Run | CUDA Winners: Topk / Repair / Stable / Packed | WGPU / CUDA Range |
| --- | --- | --- |
| First 18 | 4 / 4 / 10 / 0 | 0.730-1.722 |
| Repeat 18 | 4 / 3 / 11 / 0 | 0.739-1.740 |
| Standard 54 | 36 / 0 / 18 / 0 | 0.950-3.917 |

The first two rows compare hindsight-best fixed WGPU and CUDA controls. The
standard suite has one fixed WGPU tile per request and must not be pooled with
them. Repair/stable-sort ratios are 0.982-0.989 initially and 0.991-1.001 on
repeat; the small difference is not a robust repair-speedup claim. Packed keys
remain correct but slower than the selected controls in all timed fixtures.

The targeted tail-tie probes again select plain topk in all eight cases. Its
latency is **0.488-0.591x** the old stable-sort reference. Apparent WGPU/CUDA
ratios of **0.773-0.825** against the old rule become **1.317-1.674** against
the stronger reference. These are synthetic admission diagnostics, not broad
workload wins, a backend regression or revisions of old published results.

## Validation And Replay

Live Torch CPU **2.12.1** and CUDA **2.13.0+cu132** each pass 22 tests. Added
coverage includes zeros outside the window, already-canonical zero ordering,
unsafe MidK windows, and exact live stable-sort/index-repair results. One
intermediate positive test accidentally intersected reversed zeros; its failed
log is retained, and that input remains covered as a negative test. Final runs
pass. Six adaptation and six matmul-rank validator tests also pass.

Rust 1.98.1 native images embed clean source identity and retain unchanged
before/after hashes. Owned GPU jobs never overlap; no same-host build overlaps
timing. Availability checks are point observations, not exclusive reservations.
Existing CPU workloads and system configuration remain untouched.

[raw-logs.tar.xz](raw-logs.tar.xz) retains reports, generators, tests/build logs,
the review finding, identities and a compact source delta. The delta requires
`e952513661c1956582ea719e73fb4cbdabe2527e`; the original full handoff is retained
separately rather than nesting prior evidence into another archive. Large
executables and private session listings are not published.

Extract and replay against unchanged measured benchmark files, without Torch,
a browser or GPU:

```sh
python -I analyze_references.py --repo /path/to/SpiralTorch --output recomputed.json
```

All 37 raw files are hashed in the summary. Publication verifies separate archive
extraction and byte-identical replay.

SHA-256:

- Archive: `ab42ab4b58290995b9aba6293a2beede1c67204e9096c52b6eaccef959129952`
- Summary: `03a8ea5936c2725f48efb49b78bd6734002ffab5bde0e4ed948c54f56b312805`
