# Topos forward slice traversal: not adopted

The candidate replaces repeated drive/state indexing with zipped slice traversal
inside the same iteration-major finite Picard loop. Arithmetic, finite guards,
first-error order, capture sensitivity and audit accumulation remain unchanged.
There is no unsafe access, new API, or client-side mathematical implementation.

Baseline: `71b7a05dff1d18db3d1b20dbf88f8c7389ab5703`.
Candidate: `dcad72c1949e3d82d898b987f597a48c2ca3c517`.
Restoration: `35feb769b0af7951eb0999a395fd6e0fd37d4a55`.
The restoration commit's entire Git tree equals the baseline tree, before adding
these result files. **The candidate is not a production optimization.** Its patch
is retained in the bundle and Git history; no prior experiment was rewritten.

## Frozen decision and result

Before any timing, the plan required at least 5% shorter browser learning-step
medians in two of the three 256x768 conditions, with no greater than 5% regression
in the third, and exact saved-state parity. Passing that screen would still
require a separate full native NN/Torch comparison and fresh client validation.
This is a local screening rule, not a statistical significance test.

The screen **failed**: the three large browser conditions changed by -1.6%,
-1.2%, and +0.6%. The code was restored instead of changing the criterion after
seeing results. Native gains remain a separate follow-up hypothesis, not grounds
for retroactively passing the WASM screen or claiming complete NN speedup.

Apple M4, macOS 26.4.1, Rust 1.97.0 release, wasm-bindgen 0.2.104 and
Chrome 154.0.8037.98. Both packages and both native executables were built from
their pinned source before measurement. This is scalar WASM, **not WebGPU**.

Four fresh browser processes ran ABBA in forward/reverse/forward/reverse case
order. The unchanged harness alternates prepared-snapshot reads and actual
learning blocks, with two warmups and 20 measured blocks per route. There are
32 snapshot repetitions below volume 8192, otherwise eight, and four updates
per learning block. Aggregates are the median of the two process medians per
arm. All nine conditions and all samples remain in the bundle.

| Rows x features | K | Browser step before, ms | After, ms | Change |
| --- | ---: | ---: | ---: | ---: |
| 8 x 3 | 1 | unresolved | unresolved | n/a |
| 8 x 3 | 5 | unresolved | unresolved | n/a |
| 8 x 3 | 16 | unresolved | unresolved | n/a |
| 64 x 128 | 1 | 0.131250 | 0.125000 | -4.8% |
| 64 x 128 | 5 | 0.300000 | 0.293750 | -2.1% |
| 64 x 128 | 16 | 0.725000 | 0.731250 | +0.9% |
| 256 x 768 | 1 | 3.131250 | 3.081250 | -1.6% |
| 256 x 768 | 5 | 7.031250 | 6.950000 | -1.2% |
| 256 x 768 | 16 | 17.300000 | 17.406250 | +0.6% |

The complete browser record contains 1,440 timings, including 560 zero clock
samples. Ratios are null if either arm median is zero. The unchanged snapshot
control also varies: large-case ratios are approximately 0.91, 0.83 and 1.09.
Do not attribute those changes to the rewritten recurrence or call zero a free
operation. These observations do not establish a general or significant gain.

Every one of the 3,168 saved update losses agrees exactly across arms, as do
initial input/target/snapshot hashes, final gate/output hashes and block-end
output/input-gradient/gate-gradient/gate hashes. The latter are recorded every
four updates, not at every intermediate VJP. Every condition reduces target
MSE. This is synthetic execution parity, not pretrained-model quality evidence.

## Native diagnostics and limits

The unchanged native four-scope probe ran all nine conditions in the same ABBA
and case order: 36 reports, 3,456 timings, two warmups and 24 samples per scope.
It checks forward/gradient bits and both audits against its prepared NN result
after calls. It is not an independent mathematical reference. Scopes overlap;
they must not be added or subtracted as a phase-cost decomposition.

| Rows x features | K | Core capture before, ms | After, ms | NN forward before, ms | After, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 x 3 | 1 | 0.000396 | 0.000167 | 0.000771 | 0.000334 |
| 8 x 3 | 5 | 0.000583 | 0.000312 | 0.000948 | 0.000500 |
| 8 x 3 | 16 | 0.001042 | 0.000563 | 0.001386 | 0.000740 |
| 64 x 128 | 1 | 0.056875 | 0.041354 | 0.058573 | 0.042688 |
| 64 x 128 | 5 | 0.104823 | 0.083323 | 0.106365 | 0.084271 |
| 64 x 128 | 16 | 0.225073 | 0.194750 | 0.224875 | 0.194750 |
| 256 x 768 | 1 | 0.985229 | 0.961281 | 1.000302 | 0.977938 |
| 256 x 768 | 5 | 1.985115 | 1.890063 | 2.009657 | 1.906584 |
| 256 x 768 | 16 | 4.716906 | 4.406073 | 4.729500 | 4.455542 |

Validation and captured-VJP scopes, process medians and raw timings are also
retained, including tiny/noisy observations. No complete native NN/Torch timing,
fresh Python extension, GPU throughput or pretrained-model experiment was run
for this rejected candidate. Native forward-only results do not establish a
complete learning-step gain and must not be relabeled as a PyTorch comparison.

The candidate passed 24 core tests, including the frozen legacy recurrence's
bitwise/error-order controls and shared-gate tests, and the exact CI nightly
formatter. Both Node packages passed 27 transport cases / 469 checks / 240
updates and six ownership cases / 121 checks / 12 memory growths. Independent
read-only source review found no actionable P1/P2; it did not execute tests.

## Records and reproduction

`measurements.json.gz` contains the unmodified plan, build records, all browser
and native reports, derived summaries, harness sources, candidate patch and Node
contracts. `verification.json` inventories 139 retained originals by byte length
and SHA-256. Full packages, binaries and logs remain local. Initial package-name
setup and inline-script parse failures are retained separately; neither ran a
measurement or caused a timing phase to be restarted.

Run the standalone saved-record verifier, including its negative controls:

```sh
python -I -S -B benchmarks/results/2026-10-08-topos-core-forward-traversal/verify.py
```

This verifies retained evidence, not new runtime execution, and does not add a
new CI job or alter production tests. For a fresh comparison, build each pinned
revision with the same release flags, preserve separate output directories and
use the recorded harness bytes and complete ABBA plan. Native executable:
`cargo build --locked --offline --release -p st-nn --example topos_shared_phase_probe`.
WASM: `cargo build --locked --offline --release -p spiraltorch-wasm --target wasm32-unknown-unknown --features webgpu`.
Generate matching JS/WASM packages with wasm-bindgen 0.2.104 and explicit
`--out-name spiraltorch_wasm`; use the `topos-snapshot-bench` browser fixture.
Always use fresh result names. Never overwrite the preserved records.
