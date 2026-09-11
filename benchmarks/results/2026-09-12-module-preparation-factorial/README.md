# Isolating Resident Module Preparation

The previous combined preparation patch reduced allocations but regressed some
burst timings. This record retains a complete 2x2 isolation, not just the best
candidate. **Checked assembly is selected for main-worktree confirmation;
bulk parameter comparison is removed from the current implementation.**
This is a regression-avoidance decision, not an established universal speedup.

The subsequent [main-worktree confirmation](../2026-09-12-module-assembly-confirmation/README.md)
passes its complete pipeline and four fresh paired matrices against the original
baseline. The selected assembly-only source remains approximately baseline in
group-level burst timing; individual regressions are retained there as well.

See [API](../../../docs/module_resident_forward.md), [summary](summary.json),
[manifest](manifest.json), and [previous negative record](../2026-09-12-module-descriptor-assembly/README.md).

## Four Frozen Variants

| Variant | Source Commit | Descriptor Assembly | Parameter Comparison |
| --- | --- | --- | --- |
| Baseline | `542c744593dedf86e949aa82772e435acf35db04` | Per-leaf vectors | Per-element bits |
| Assembly | `dedd955ed37013cb57ba0357c72970d6d624fc44` | Checked shared vector | Per-element bits |
| Comparison | `a50239bcda35ae529f57d5b5b2818026a5abd912` | Per-leaf vectors | Exact byte slices |
| Both | `d65b87f622c38a7dc3034fce90efda124d3ee18c` | Checked shared vector | Exact byte slices |

Git tree/blob identities verify both factors independently: assembly code
matches its selected parent, comparison code and its Cargo dependency match
the other selected parent. GPU backend, Tensor and binding directory trees are
identical across all four variants. Tests and experimental documentation can
differ. This does not assert identical machine code or eliminate build-layout
effects. The selected code requires a fresh main-worktree build and confirmation.

The two isolation worktrees were committed cleanly before builds. The baseline
and combined products are the previously verified frozen products, not rebuilt
or substituted during measurement. All **48 products** were checked before and
after the experiment. No source changed during a run.

## Checks And Protocol

- All **60 serial steps** pass: 20 build/correctness steps, followed by 40 timing
  and validation steps. No retries, omitted trials, concurrent owned GPU job,
  hidden CPU fallback, or automatic tuning.
- Each isolation passes 752 Rust NN tests, 34 existing Python tests, 7 original
  Module Python tests, 111 browser assertions, and TypeScript checks. NaN,
  foreign DLPack writes, signed zero, old output ownership and GPU error guards
  remain covered. These configuration counts are not unique-test totals.
- Assembly-only passes four allocation/compatibility tests in each of CPU-only
  and WGPU-enabled builds. The former assembly algorithm with the same checked
  leaf descriptors allocates 69 times; the new 64-descriptor assembly allocates
  once. Nested caller-capacity assembly allocates zero times in each of 20 calls.
  This is test-only allocator instrumentation, not a GPU timing measurement or
  an allocation count from the old production binary.
- Twelve paired browser matrices contain 108 shape/seed cases and 1,114,992
  output comparisons. Maximum absolute error is `2.384185791015625e-7` against
  the admitted CPU reference. Every original-Module cache reports one compile,
  108 hits and 109 submitted forwards for each timed case.

Fixture: seeds 17/29/43; shapes [2,3,7], [2,8,64], [4,8,128]; 2/8/16
Scaler -> Linear -> GELU -> ReLU blocks. Three warmups and nine retained rotated
blocks per route. Each burst is eight independent fixed-resident-input forwards
plus one completed terminal host read. Compilation is separate; parameter
checks, assembly and output handling are timed. This is not recurrent decoding.

Native version orders are B/A/D/C, A/C/B/D, C/D/A/B, D/B/C/A, where B is baseline,
A is assembly, C is comparison and D is both. Each version occupies every order
position once; each directed adjacent pair occurs once. Each native worker is
a separate process with eager Torch controls. The baseline worker is shared
across the three candidate comparisons within a trial, so they are correlated.

Browser runs are same-page rotated baseline/candidate pairs, one pair for each
candidate in each trial. They are not a simultaneous four-variant measurement.
Exact physical browser GPU and host exclusivity remain UNKNOWN; timing is
visibly quantized. No control ratio normalizes away drift.

## Paired Burst Ratios

Each cell is the median candidate/baseline ratio across 12 seed-run pairs.
Lower is faster. The group medians are not ratios of pooled absolute times.

| Variant | Python Small | Python Middle | Python Large | Browser Small | Browser Middle | Browser Large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Assembly | 0.992 | 0.992 | 0.998 | 1.000 | 1.000 | 1.000 |
| Comparison | 0.993 | 0.968 | 1.016 | 1.000 | 1.000 | 1.000 |
| Both | 0.997 | 0.989 | 1.014 | 1.000 | 1.083 | 1.000 |

Assembly-only native ratio ranges are 0.723-1.291, 0.978-1.010 and
0.953-1.085; browser ranges are 0.833-1.200, 0.923-1.083 and 0.971-1.029.
It is approximately baseline at the group-median level, not regression-free:
11/36 native and 13/36 browser burst ratios are strictly greater than one.
All raw values, including sub-resolution differences, remain in the record.

Comparison-only regresses in all 12 large native burst pairs: median 1.016,
range 1.002-1.109. Both regresses in 22/36 native and 17/36 browser burst pairs;
its middle browser median remains 1.083, reproducing the earlier adverse result.
The comparison-only browser median is approximately one, so these measurements
do not prove that every bulk comparison alone causes a browser slowdown.

Bulk comparison has a real tradeoff in these measurements: large native
single-call d2h ratio medians are 0.944 for comparison-only and 0.948 for both,
while assembly-only is 0.999. The single-call benefit does not erase the burst
regressions. It is not an isolated CPU-phase or scheduling explanation.

Small native controls drift substantially. For assembly-only, explicit-graph
control ratios span 0.665-1.533 and Torch ratios span 0.661-1.334. Middle/large
explicit-graph medians are 0.994/1.003 and Torch medians 0.999/1.004. Browser
control medians are approximately one. `summary.json` retains all controls and
native conditional factor ratios without claiming statistical significance.

## Decision And Archive

Keep the checked single-vector assembly and its custom-module compatibility;
restore the original per-element comparison and remove the extra direct
dependency. This preserves the measured allocation reduction without retaining
the combined candidate's adverse group-level burst behavior. Verify it again
in the main worktree before treating it as the final runtime result. Numerical
checks stay enabled, and no layer silently falls back to a host implementation.

GPU arithmetic, guard ownership, output reuse and supported model set do not
change. Generic autograd/ModuleTrainer is not automatically migrated. No
CUDA/Furnace run, push, merge, release or fastest-PyTorch claim in this isolation.
The two local isolation worktrees and all negative results are retained.

The archive contains **124 records**, 111,153,788 raw bytes compressed to
2,641,504 bytes. Raw/compressed hashes and round-trip checks bind every member.
Commit metadata, source patches, the common fixture, drivers and all four trials
are included. Production binaries are identified by hashes, not committed.
