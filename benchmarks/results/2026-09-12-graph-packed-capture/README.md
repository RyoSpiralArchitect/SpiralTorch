# Prepared Graph VJP Capture

## Implementation and Correctness

Baseline: `f1215d1e228517bf4a63778a64796f1f9daf7bae`, using the immutable
products from the preceding composition verification. Adopted candidate:
`fd4147fb0a6de3687a251b71728558ff7d3dbf85`. The core optimization is
`5309d53f`; the adopted revision also updates browser/Torch fixture admission.

`ResidentGraphAutograd::backward` now snapshots its input and parameter gradients
through a prepared contiguous-copy pipeline, shared by `ResidentGraphLearner`.
Source bindings and shape uniforms persist. All copy dispatches use one compute
pass, independent output value buffers, and a fresh shared finite guard per VJP.
No per-output shape uploads or separate guard buffers are needed. Every value
and upstream guard is still checked. Late failures invalidate every sibling,
including detached tensor handles/views. A later backward never reuses the flag.

**Value allocations and copies remain.** This is not zero-copy autograd, a buffer
pool, a fused optimizer, or a change to tape identity/update policy. Cold pipeline
preparation is not the measured hot loop. General tensor operations retain their
shader math; internal flag ownership changes, and independent general-tensor
throughput has not been benchmarked here. Python/WASM use the Rust implementation
without a new public API. See [the contract](../../../docs/resident_graph_autograd.md).

The adopted `verified-b` run passes all 29 checks, with 19 hashed generated
products. Native Metal and browser WebGPU retain six arbitrary-VJP cases, twelve
autograd guard checks, and twelve 64-update learner cases with both gain policies
and fusion choices. Resume remains bit-identical. A new rank-three, 257-column
fixture queues four VJPs, detaches the tensor handles, overflows a late bias,
recovers, replaces the input, and reads only after workspace destruction.
Direct native tests additionally cover scalar negative zero, empty outputs,
unflagged nonfinite source data, the final upstream flag, invalid layouts and
independent value/guard ownership across repeated captures.

Rust: contracts 22, NN 734, backend 116, integration 6 (878 total). Python:
38 resident tests, no skips. Public WASM forward/fusion/autograd/learner clients,
generated/shipped types, formatting, CPU-only builds and benchmark admission pass.
Independent Torch 2.12.1 CPU/MPS replay checks 120 cases and 17,136 comparisons;
maximum absolute error is `5.7220458984375e-6` within `atol=2e-5, rtol=2e-4`.
The weighted learner subset has 5,904 comparisons, maximum `5.960464477539063e-8`.

The initial `verified` run failed browser admission because the page still
required eleven autograd guards. Its receipt, error and asset hashes are retained.
Admission now requires twelve guards and the explicit detached-capture case;
`verified-b`, not the failed attempt, is the adopted evidence.

## End-to-End Learner Timing

Two rounds cross seeds 17/29/43 with `[2,16,32]`/depth 2,
`[2,129,32]`/depth 4, and `[4,32,64]`/depth 8. Each timed interval performs
eight nonzero SGD updates, each with GPU-derived quadratic/quartic seeds, two
exact VJPs and ordered weights 0.75/0.25. Pointwise fusion is disabled in both
lanes. Immediate mode reads every update acceptance receipt immediately;
deferred mode reads every receipt before the timer ends. Reset, compilation,
zero-rate warmup and initial/final full-state/objective probes are outside timing.

Each cadence records two warmup and eight retained blocks with rotating
native A/B/Torch or browser A/B order. Ratios are geometric means of baseline
median / candidate median across all nine cases; above 1 favors the candidate.

| Runtime | Immediate, Rounds 1 / 2 | Deferred, Rounds 1 / 2 |
| --- | --- | --- |
| Native Metal | 1.323 / 1.266 | 1.317 / 1.252 |
| Chrome WebGPU | 1.593 / 1.589 | 1.576 / 1.582 |

Every learner case/cadence has a ratio above 1 in both rounds. The minimum
observed learner ratio is 1.117 native and 1.327 browser. Terminal predictions,
both input/parameter VJPs and updated weights match the independent Torch replay
within a maximum absolute error of `1.7695128917694092e-8`. Per-interval state
fingerprints also match across resets; full states are retained per lane/cadence,
not as separate parameter dumps for every update.

**Eager Torch MPS remains about 1.75..1.87 times faster** in these aggregates.
Torch/candidate ratios are 0.535..0.572. Torch synchronizes completion but does
not implement equivalent finite guards, rollback or acceptance receipts. This is
not fastest-Torch/torch.compile, browser-vs-Torch timing, or a model-quality result.

## Controls and Limits

Two ordinary MSE/SGD control rounds use the same frozen products but their
original workload, including per-step loss reads. Native geomeans are
0.913 / 0.978 immediate and 0.996 / 0.976 deferred. Minima are 0.659 / 0.732
and 0.858 / 0.735 respectively. Browser geomeans are 0.990 / 1.005 and
1.004 / 0.998. All terminal states pass, maximum error `1.4901161193847656e-8`.
**The individual native delays remain unresolved.** The ordinary MSE preparation,
step encoding, shaders, readback code and benchmark loop have no source diff.

An additional same-product A/A ordinary control uses the identical candidate
binary/WASM hashes in both lanes. Native immediate ratios still range
0.746..1.423 (geomean 1.034), deferred 0.989..1.236 (1.059). Browser immediate
ranges 0.980..1.033 (1.002), deferred 0.985..1.841 (1.079). This demonstrates
source-independent measurement variability; it does **not** prove the A/B delays
are harmless noise, quantify uncertainty for the learner workload, or establish
regression freedom. The A/A round is a control, not an optimization gain.

Native hardware is Apple M4 Metal/MPS with Torch fallback disabled. Browser
reports WebGPU, not an attested physical adapter. Owned GPU jobs are serial;
host contention and the earlier parallel GPU VJP one-off remain UNKNOWN.
There is no automatic `ModuleTrainer`, generic autograd, GNN or `pure::Tensor`
migration, CUDA/Furnace run, wheel release, push, merge or whole-repository CI claim.

[summary.json](summary.json) separates the two learner rounds, two ordinary
A/B controls and one A/A control. Together they preserve 4,500 raw intervals,
3,600 retained intervals and 36,000 nonzero timed updates, including warmup
intervals. [manifest.json](manifest.json) binds raw/compressed hashes and sizes;
every compressed stream is decompressed and verified. Failed admission and raw
negative controls are retained. Generated products remain at their receipt paths.
