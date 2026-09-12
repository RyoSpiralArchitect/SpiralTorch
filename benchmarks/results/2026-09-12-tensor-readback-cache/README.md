# Tensor Snapshot Staging: Two Rejected Cache Strategies

**Decision: keep fresh Tensor staging.** Both bounded cache prototypes passed
their correctness checks, but small/middle sustained browser reads consistently
became slower. Neither cache is present in the selected production implementation.
The stronger native, Python and browser snapshot tests remain.

| Version | Source commit | Disposition |
| --- | --- | --- |
| Original fresh staging | `6703df35e8389c3ebb9ddf5eb099f440995a930d` | Runtime baseline |
| One idle slot | `aaf4d4a91805878bacccb2750e818671366c1adf` | Rejected |
| Two alternating slots | `67b6cdad40cea9c348b2b434a77d80d61b47c71a` | Rejected |
| Restored source and retained tests | `350493470acac03d23ad357472a02017d602494f` | Selected |

The selected production library source and manifests equal the original
baseline. In particular, `runtime.rs` and `resident_tensor.rs` are byte-identical
to it, not merely configured to disable a remaining cache. Differences since
that baseline are tests, example fixtures, measurement tools and documentation.
The existing graph snapshot pool and high-level resident Module route are
unchanged. There is no new selected-runtime performance claim here.

## Hypotheses And Protocol

The [previous interval record](../2026-09-12-module-completed-intervals/README.md)
identified fresh staging allocation in `ResidentTensor::snapshot` as a candidate,
not a demonstrated bottleneck. The first prototype retained one exact-sized
idle staging buffer per TensorDevice and its clones, capped at 32 MiB. Busy or
oversized captures allocated independently. Existing exclusive leases, weak
ownership and map cancellation rules were retained.

After its browser result, a second prototype alternated between two exact-sized
slots, each capped at 16 MiB. This tested whether avoiding immediate reuse of the
same buffer would help. It did not establish that hypothesis. These are separate
experiments against the same original runtime, not a simultaneous three-way
comparison. Neither outcome identifies the browser's internal allocation policy
or proves allocation irrelevant for other workloads.

Both used the unchanged, fixed completed-read browser protocol: four matrices,
nine shape/seed cases per matrix (seeds 17, 29, 43), two warmup and nine retained
intervals per route. Each interval contains **256 independent forwards, each with
its own completed host read and release**. One outer clock covers the interval;
all returned typed arrays are retained until its end and then numerically checked.
Setup, upload, cold compilation and comparison are excluded. This measures the
completed host observation workload, not GPU kernel time or recurrent decoding.

Original/candidate Module routes and their explicit graph controls alternate
order. Context setup and case order also alternate. No retry, adaptive sample
count, trimming or discarded run was used. The control is not used to normalize
away drift. The one-slot prototype additionally received four paired short native
Python/PyTorch/browser matrices, preserved as separate endpoints.

## Sustained Browser Results

Primary endpoint: median of 12 case-median candidate/original ratios per shape.
Lower is faster. Ranges and regressions count the same 12 pairs.

| Shape | One slot | Range | Slower pairs | Two slots | Range | Slower pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `[2,3,7]` | 1.0140 | 0.9967-1.0216 | 11/12 | 1.0125 | 1.0062-1.0204 | 12/12 |
| `[2,8,64]` | 1.0194 | 1.0087-1.0273 | 12/12 | 1.0170 | 1.0039-1.0215 | 12/12 |
| `[4,8,128]` | 1.0040 | 0.9960-1.0391 | 10/12 | 0.9992 | 0.9597-1.0071 | 5/12 |

Pooled elapsed ratios and explicit graph controls are separate statistics:

| Shape | One-slot pooled Module | One-slot control median | Two-slot pooled Module | Two-slot control median |
| --- | ---: | ---: | ---: | ---: |
| `[2,3,7]` | 1.0116 | 1.0024 | 1.0076 | 1.0032 |
| `[2,8,64]` | 1.0229 | 1.0029 | 1.0215 | 1.0019 |
| `[4,8,128]` | 1.0088 | 1.0054 | 1.0000 | 0.9997 |

Every interval and slow tail is retained in the archive. The shortest intervals
were 59.0 ms and 59.3 ms respectively, versus an observed smallest positive clock
delta of about 0.1 ms. That is a granularity diagnostic, not an uncertainty bound.
Chrome was `152.0.7977.84`, `crossOriginIsolated=false`; both reported
`BrowserWebGpu` / `Other` with no physical adapter name. Browser physical GPU
identity and host exclusivity remain **UNKNOWN**.

Each experiment checked 405,504 completed reads including warmup and 697,737,216
returned values. Together: **811,008 reads and 1,395,474,432 values**, with maximum
absolute reference error `2.384185791015625e-7`. This correctness evidence does
not turn either timing result into an optimization win.

## Separate Native And Short-Call Evidence

The one-slot candidate's four short native matrices gave completed-read medians
of 0.9925 / 0.9890 / 0.9870 and eight-forward, final-read-only burst medians of
1.0191 / 0.9957 / 0.9968 across the three shapes. Small native completed-read
ratios ranged from 0.6325 to 1.5522, and small burst ratios from 0.6550 to 1.4514.
These noisy results do not establish a universal native improvement. The short
browser Module medians were approximately 1.0 with a coarse clock.

All paired ratios, ranges, regression counts and untargeted scalar/PyTorch MPS
controls are in [summary.json](summary.json). Raw reports retain the per-sample
elapsed times, outputs, device checks and source hashes. Native runs used Apple
M4/Metal and eager PyTorch 2.12.1 with `PYTORCH_ENABLE_MPS_FALLBACK=0`; no
fastest-PyTorch claim is made. The two-slot candidate did not receive a native
Python/PyTorch performance qualification after failing its WASM-first screen.

## Verification And Retained Tests

- One slot: all 55 full verification stages passed (557.2 s), followed by all
  14 replication/interval stages (240.1 s). Backend: 133 tests; Tensor: 449 CPU
  and 497 WGPU tests with one unchanged ignored test; NN: 707 CPU and 759 WGPU
  tests including the parameter-storage regression; Python GPU suites: 60 tests.
- One-slot native/browser graph training and VJP fixtures passed 120 eager
  PyTorch CPU/MPS cases, 17,136 comparisons, maximum absolute error
  `5.7220458984375e-6`. CPU-only surfaces, browser/Python parameter handoff,
  learner/autograd/pointwise/fusion clients and type declarations also passed.
- Two slots: all ten backend/WASM screen and interval stages passed (283.1 s).
  Backend: 133 tests; browser Module: 210 assertions. This is deliberately not
  labeled a full native/Python qualification.
- Selected source: all 13 stages passed (93.7 s). A fresh backend build passed
  132 tests. Newly built native and browser N-D fixtures passed. Source-identical
  original frozen Python/WASM libraries passed the strengthened client checks:
  Python 35 + 9 tests, browser Module 210 assertions, and eight CPU-only surface
  tests. This reused verified libraries, not newly built selected client packages
  or a new selected performance measurement.

The shared N-D fixture retains 24 mixed-shape captures, performs 48 intervening
discard/read cycles, checks two retained invalid-value guards (including an empty
view), exact values and negative-zero bits, and reads a snapshot after its tensor
and TensorDevice are dropped. Browser Rust additionally enters and cancels four
pending map futures, then verifies subsequent reads. Native blocking reads do
not claim those cancellation cases. Browser Module tests also start reads before
freeing their JS snapshot wrappers. Existing stage-error ordering and ownership
guards remain checked. Cache-specific identity-reuse tests are retained with the
rejected source snapshots, not imposed on the selected fresh-allocation route.

## Archive And Scope

[manifest.json](manifest.json) binds 218 compressed records: 145,655,708 raw bytes
and 2,538,300 compressed bytes, round-trip checked. It includes every run's logs,
receipts, fixtures and reports, both complete streamed interval sets, measurement
drivers, validators, and exact source blobs from each relevant Git commit. The
rejected source is retrieved from its own commit, never from the restored files.

All 54 frozen product records across original/one-slot/two-slot/selected versions
were rehashed during archival. Binaries remain in local verification directories;
the selected native N-D executable was copied out of the shared build directory
and verified against the unchanged original receipt. Served JavaScript, fixtures
and product hashes are archived. Both interval reports were independently
reaggregated, with streamed rows checked against the final reports; the original
and one-slot native fixtures have equal non-timing case contents.

This record does not change shader math, dispatch scheduling, generic autograd,
ModuleTrainer, fallback policy or the existing resident forward API. No CUDA or
Furnace run, release, push or merge is included. A possible next experiment is an
explicit terminal-forward API that combines computation and snapshot copy in
one queue submission. That is **not implemented or measured here**, and the cache
results alone do not establish submission overhead as the bottleneck.
