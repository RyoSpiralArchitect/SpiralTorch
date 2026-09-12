# Public Pointwise Plans and Learner Seed Fusion

## Connection and Correctness

Adopted source: `f583e5dc60879d1c45d88f7de5cd67345f4002d9`, with implementation
in `123e0337` and the public-constructor type-test correction in `f583e5dc`.
Baseline products are the preceding packed-capture verification at
`fd4147fb0a6de3687a251b71728558ff7d3dbf85`. These are separately frozen,
source-bound products, not an installed wheel or two labels for one binary.

`WgpuPointwiseInputs` and `WgpuPointwisePlan` expose the existing Rust pointwise
implementation to Python and WASM. Input handles can be replaced without
uploading/copying GPU values, submitting work or reading them back. Exact
layouts and device/queue identity are enforced in Rust; failed replacement is
atomic. The same operation vocabulary selects sequential, batched or fused
execution. Plans retain layouts and the device, not original input values.

The new public-client tests connect a fused cubic loss cotangent to
`ResidentGraphLearner::backward` and SGD: a gain of 2 on inputs `[1,2]` gives
gradient 68 and an accepted update to approximately 1.32 at rate 0.01.
Both Python and actual browser WebGPU execute this update. Tests also cover
rank-three strided/offset inputs, broadcasting, ownership after parent release,
scalar/empty layouts, malformed recipes, invalid intermediate values masked by
ReLU/empty views, and recovery. Rust additionally tests foreign device contexts.
See [the public API examples](../../../docs/resident_pointwise_clients.md).

The adopted `verified-c` run passes all 31 checks and binds 19 generated products.
Rust: contracts 23, NN 734, backend 117, integration 6, **880 total**.
Python: **43 tests without skips**, including five new pointwise tests, loading
the frozen extension rather than an installed package. WASM pointwise clients
pass 38 assertions; existing forward/fusion/autograd/learner clients, generated
and shipped types, CPU-only build checks, formatting and admission tests pass.

The wider native/browser fixture retains twelve 64-update learner cases per
runtime, with bit-identical resume and both gain-gradient/fusion policies.
Independent Torch 2.12.1 CPU/MPS replay checks 120 cases and 17,136 comparisons,
maximum absolute error `5.7220458984375e-6`, within its declared tolerances.
The weighted-learner subset has 5,904 comparisons and maximum absolute error
`5.960464477539063e-8`. These fixtures are not a model-quality experiment.

The initial `verified` run failed a type-test assumption that the deliberately
constructible input collection was opaque. The corrected `verified-b` run was
interrupted during Python compilation; its process handle was missing and a
live process check found no remaining driver/build/test process. Neither run is
adopted. Their logs are retained alongside the complete `verified-c` rerun.

## End-to-End Timing

Two treatment rounds cross seeds 17/29/43 with `[2,16,32]`/depth 2,
`[2,129,32]`/depth 4, and `[4,32,64]`/depth 8. The mixed model contains
Linear/GELU/gain/ReLU stages. Weights, inputs, targets, objective, ordered
gradient weights 0.75/0.25, exact VJPs, SGD rate 0.01 and matmul choices match.
Only the candidate compiles `e * e * e * norm` into one prepared dispatch,
instead of three elementwise tensor submissions. Model-graph fusion stays off.

Each interval resets the model and times eight nonzero updates, including two
GPU-derived cotangents, two VJPs, weighted SGD and every update acceptance
receipt. Immediate mode reads each receipt immediately; deferred mode reads
all receipts before the timer ends. Compilation, reset, zero-rate warmup and
initial/final full-state/objective probes are excluded. Python call overhead
is not timed. Each cadence has two warmup and eight retained blocks, rotating
native A/B/Torch or browser A/B order. These are repeated eight-step trajectories,
not one long training run.

Ratios below are geometric means of baseline median / candidate median across
all nine cases. Above 1 favors the candidate. The control uses the same custom
learner without seed fusion in either lane; it is not ordinary MSE or A/A.

| Round | Native Immediate | Native Deferred | Browser Immediate | Browser Deferred |
| --- | ---: | ---: | ---: | ---: |
| Fusion 1 | 0.915 | 1.062 | 1.078 | 1.187 |
| Fusion 2 | 0.971 | 1.219 | 1.032 | 1.080 |
| Unfused Control | 1.060 | 0.971 | 0.983 | 1.032 |

**This does not establish a universal speedup.** Native immediate-cadence
aggregates regress in both treatment rounds. The `[2,16,32]`, seed-17 case has
ratios 0.573 / 0.694 in those rounds. Native deferred minima are 0.830 / 0.616.
Browser aggregates improve, but immediate minima are 0.831 / 0.863 and deferred
minima 0.808 / 0.737. Favorable aggregates do not erase these delays.

The unfused control also varies: native immediate ratios span 0.942..1.304,
deferred 0.723..2.015; browser immediate 0.869..1.101, deferred 0.886..1.186.
Because the control uses different source-bound products, it is not an
identical-product variability measurement or a scalar correction to treatment
ratios. No isolated-kernel causal attribution or latency-regression-free claim
is made. Keep this fusion an explicit caller choice, not an automatic route.

Eager Torch has lower aggregate time in every native round/cadence. In the two
treatment rounds, Torch/candidate ratios span 0.581..0.706, roughly 1.42..1.72x
faster for Torch in those aggregates. Torch synchronizes completion without
equivalent Rust finite-check, rollback or acceptance-receipt guarantees. This
is neither fastest-Torch/torch.compile nor browser-vs-Torch timing.

All nine final-state fingerprints match baseline/candidate on both runtimes
in all three rounds. Predictions, both input/parameter VJPs and updated weights
match independent Torch replay within maximum absolute error
`1.7695128917694092e-8`. Each interval is checked live for reset consistency and
accepted updates; full final states are retained per lane/cadence, not as a
separate parameter dump for every update.

## Evidence and Limits

[summary.json](summary.json) separates treatment and control. Together they
retain 2,700 raw intervals, 2,160 post-warmup intervals and 21,600 nonzero timed
updates across all lanes, including warmup intervals. Rust updates have checked
acceptance receipts; Torch completion is synchronized, not guard-certified.
[manifest.json](manifest.json) binds 147 compressed artifacts, raw/compressed
hashes and sizes. Every stream was decompressed and verified. Generated products
remain at their receipt paths; the external Python loader is archived separately
at collection time. No credentials were needed for these local runs.

Native hardware is Apple M4 Metal/MPS, with Torch fallback disabled. Browser
reports WebGPU, not an attested physical adapter. Owned GPU jobs ran serially;
host contention remains UNKNOWN. Prior ordinary-MSE native timing regressions
and the earlier parallel GPU VJP one-off remain unresolved, not cleared by this
unfused-learner control. See the [preceding evidence](../2026-09-12-graph-packed-capture/README.md).

This exposes forward pointwise execution; the pointwise VJP API itself is not
newly bound, and callers still define the correct loss cotangent. There is no
automatic `ModuleTrainer`, generic autograd, GNN or `pure::Tensor` migration,
no Adam/momentum/mixed precision, CUDA/Furnace run, wheel release, push, merge,
or whole-repository CI claim.
