# Paired Fractional Forward: Shared Rust, Native Python And WASM

This changes the implementation of the **same finite causal GL operator**, not
its mathematics, adapter capacity or training policy. The two-pass baseline is
`98027f0177a2ae60f90d52c2a4622d7ddb229601`. Both versions retain the selective
alpha VJP introduced in that parent. Native identities and the frozen benchmark
identity are recorded in `timing-plan.json` and every primary result.

## Implementation And Correctness

`st-frac::learning::FractionalGlKernel` now computes output and cached order
differential together. It borrows the checked C-order input, shares each sample
read, skips out-of-range zero padding and tiles contiguous trailing features in
groups of 64. Two fixed-size f64 scratch arrays replace strided lane traversal;
their size does not grow with feature width. Each scalar accumulator keeps its
increasing-lag order and checked f32 output conversion. Full/history semantics,
shape/product limits, empty-history behavior and public VJP/JVP APIs remain.
No unsafe code, new dependencies, threads or client-side production formula are
introduced. General ND operators remain unchanged as regression references.
Failure domains remain guarded, but ordering/indexing of simultaneous invalid
input/coefficient diagnostics is not promised to be identical.

- Rust: 123 tests pass in both dev and release. The new tests compare both
  outputs, full VJP and JVP bit-for-bit against the old two-pass recipe across
  axes, ranks through 16, feature tile boundaries, K=1/2/5/32, integer and
  noninteger orders, non-unit spacing, signed zero and subnormal values.
- Native Python: 354 regression tests pass, including actual autograd/adapter
  learning, selective pullbacks, optional-dependency and prior geometry tests.
- WASM: 128 real wasm32 tiled/scalar-composition cases and 54 selective-VJP
  cases pass; 100-update full and 500-update history synthetic fits still pass.
  The tiled fixture is wired into CI. This is not a browser/GPU speed result.
- Pretrained regression: two separate native binaries each load copies of six
  completed GPT-2 adapters and their Adam state. For each adapter, the same
  final scheduled training batch is replayed twice. All losses, every gradient,
  resulting parameters and Adam tensors agree bit-for-bit across binaries.
  These 24 auxiliary updates do not change the frozen base or original studies,
  score held-out data, add independent seeds or establish new model quality.

`pretrained-parity.json` binds both native binaries, checkpoints, frozen probe
and comparator sources. Private tensor payloads are hashed but not published.
The supplied probe scripts are the frozen verification sources, not CI jobs;
repeating them requires the original local completed studies/model/data. The
comparator expects its two probe payloads beside the script in a private output
directory, and refuses to overwrite an existing receipt. Do not run it directly
inside this published results directory.

## Matched Python-Inclusive Timing

Median of three process medians, milliseconds, lower is better:

| Map / BxTxF | Baseline selective | Paired selective | Torch reference in baseline processes | Torch reference in paired processes |
| --- | ---: | ---: | ---: | ---: |
| Full / 2x128x768 | 22.167 | 16.620 | 3.624 | 3.631 |
| History / 2x128x768 | 21.849 | 16.509 | 3.477 | 3.719 |
| Full / 1x256x768 | 21.224 | 16.073 | 3.007 | 3.170 |
| History / 1x256x768 | 21.450 | 16.124 | 3.166 | 3.172 |

The paired path takes about 24-25% less time in this local comparison, with
baseline/paired ratios 1.32-1.33. It is still **4.44-5.08 times slower** than the
ordinary Torch convolution reference in its matched process groups. Do not
turn this into a general-library, model-throughput or GPU claim. Joint-backward
timings and all individual intervals are preserved too, not selectively omitted.

All 24 planned processes finished successfully: two shapes, two maps, two
native binaries and three repetitions. Each process uses two warmups per route
and 12 rounds containing all six route permutations twice. Binary order is
baseline/paired on odd repetitions, reversed on even repetitions. Output and
alpha-gradient SHA256 values agree across native binaries in every condition.
Torch computes the same polynomial but uses different accumulation; the
predeclared rtol=atol=3e-5 comparison passes outside each timed interval.

This is a **complete fresh rerun**, not a splice of the interrupted matrix.
The earlier 12-result partial attempt is retained locally and excluded; its
filenames/hashes and the reason are in `excluded-diagnostics.json`. Initial
single-pass Rust diagnostics and exploratory staged timings are also excluded
from the complete matrix, regardless of their measured values.

Conditions: Apple M4, 10 cores, 24 GiB; Rust 1.98.0 release, Python 3.12,
Torch 2.12.1; CPU float32 interfaces, K=32, alpha=0.9, h=1, seed=239. Torch uses
two intra-op threads; Rust's loops are serial. No own builds/tests ran alongside
timing. Other OS/application load remained: pre-process samples across the
primary and Rust-only runs ranged from 33.33% to 70.62% idle. No isolated host,
thermal control, confidence interval or portable performance guarantee is
claimed. Only numeric CPU samples/log hashes are published, not process names.

## Rust And Transport Diagnostics

Six separate Rust-only processes, three per binary, measure snapshot forward
plus the requested alpha VJP. Output inspection, bindings, model execution and
snapshot drop are outside these intervals. Cases have fixed order within each
process; binary order alternates by repetition. Median of process medians:

| Shape / axis | Map | Baseline ms | Paired ms |
| --- | --- | ---: | ---: |
| 2x128x768 / 1 | Full | 6.839 | 1.490 |
| 2x128x768 / 1 | History | 6.813 | 1.415 |
| 2x768x128 / 2 | Full | 6.314 | 3.294 |
| 2x768x128 / 2 | History | 6.078 | 3.304 |
| 196608 / 0 | Full | 5.446 | 3.671 |
| 196608 / 0 | History | 5.503 | 3.680 |

These are supplemental operator diagnostics, not the Python-inclusive numbers.
Output FNV checksums and alpha-gradient bits match across all six process cases;
the stronger bitwise reference tests above establish the numerical regression
coverage. The baseline profiler binary was built before formatting and the
untimed even-median expression changed to `is_multiple_of(2)`; its timed loop is
unchanged. `rust-plan.json` binds both binaries and the final profiler source,
not an assertion that their complete source files were byte-identical.

The separately instrumented paired Python stages retain explicit production
dtype/device arguments: output-list to Tensor alone has a median near 5.1 ms,
while native forward including input-list extraction is near 2.4 ms. These
staged measurements include their own boundaries and are **not an additive
decomposition** of public autograd latency. An earlier version omitted explicit
dtype/device and inflated conversion cost; it is excluded with its hash, not
used as production evidence. List-based transport is the next optimization
target, with snapshot ownership/lifetimes and true derivatives kept in Rust.

## Reproduce And Verify

Build each native revision in release with the same features/toolchain, keep
their packages separate, and use the same version of
`tools/benchmark_fractional_learning.py` for both. Pin imports to the intended
package, verify its hash, and run the plan's full/history shapes and repetitions
in fresh processes without overwriting earlier outputs:

```sh
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PACKAGE_ROOT" \
  python -P -B tools/benchmark_fractional_learning.py \
  --shape 2 128 768 --kernel-len 32 --alpha 0.9 --step 1 --seed 239 \
  --threads 2 --warmup 2 --rounds 12 --map history --native-profile release \
  --output "$NEW_TIMING_JSON"
```

Use the same public `learning_profile.rs` example with each Rust core revision
and run `cargo run --release -p st-frac --example learning_profile -- 12` for a
new standalone comparison. Do not time concurrent builds. The frozen staged
profiler is included for investigating transport, not a performance gate.

`SHA256SUMS` covers the published artifacts. The benchmark test file rebuilds
all process/group medians, matches the plan and output identities, and binds
probe receipts without executing hardware speed tests. It does **not** rerun
the private pretrained comparison. Weights, corpus text, native packages and
raw host/build logs remain local; `verification.json` records their relevant
hashes, commands and validation scope.
