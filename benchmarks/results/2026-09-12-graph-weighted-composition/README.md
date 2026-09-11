# Prepared Resident Gradient Composition

## Change and Boundary

Benchmark baseline: `d0b3923f4964d774e35687ca290d3fdaea35527d`.
Optimized implementation: `f1215d1e228517bf4a63778a64796f1f9daf7bae`.
The baseline adds the custom learner benchmark without changing the prior learner.
Its native/WASM products were frozen before the implementation change.

`ResidentGraphLearner::sgd_weighted` now composes up to four source gradients per
parameter dispatch, directly into the graph's prepared gradient buffers. Weight,
source-flag and shape buffers persist across updates. There are no explicit
per-update tensor-sized composition allocations or intermediate scaled tensors.
For one unit-weight contribution, the direct-copy path remains. Bind groups,
small host coefficient vectors and immutable VJP snapshots still have costs.

Every product and ordered intermediate sum uses the shared checked-elementwise
shader semantics. The source guard includes zero-weight inputs and parameterless
graphs. All candidates still pass the existing global decision before commit;
explicit gain policy, tape invalidation and owning receipts are unchanged.
Python/WASM automatically use this same Rust implementation; their APIs do not
reimplement the arithmetic. See [the learner API](../../../docs/resident_graph_learner.md).

This does not migrate `ModuleTrainer`, generic autograd or host `pure::Tensor`.
There is no default routing change, new optimizer policy, cross-forward
accumulation, CUDA/Furnace run, wheel release, push or merge. Ordinary MSE/SGD
shader math, encoding and preparation are not changed by this optimization.

## End-to-End Learner Measurements

Two complete standard-matrix rounds use nine shape/depth/seed combinations each:
`[2,16,32]`/2 layers, `[2,129,32]`/4, `[4,32,64]`/8, crossed with seeds 17/29/43.
Both lanes use the same unfused pointwise plan and eight nonzero-rate updates per
interval. Each step constructs quadratic/quartic cotangents on the GPU, performs
two exact VJPs, combines them with weights 0.75/0.25 and applies SGD at 0.01.
Thus this measures the whole resident learning step, not an isolated kernel.

Rust captures and reads every update acceptance receipt. Immediate mode reads
after each step; deferred mode enqueues the steps first and reads all receipts
before timing ends. Torch synchronizes completion per step or at the end, without
equivalent finite guards/rollback or Rust acceptance receipts. No per-step loss
is calculated/read inside these timers. Initial/final objective and full-state
probes, reset, compilation and a zero-rate warmup are outside timing.

All two warmups and eight retained blocks per cadence are recorded, rotating
native A/B/Torch order and browser A/B order. Ratios below are geometric means
of baseline median / candidate median across all nine cases; above 1 favors the
candidate. Both rounds are retained, not a best-run selection.

| Runtime | Immediate Update Reads | Deferred Update Reads |
| --- | --- | --- |
| Native Metal | 1.247 / 1.233 | 1.241 / 1.242 |
| Chrome WebGPU | 1.375 / 1.368 | 1.584 / 1.580 |

Every case/cadence improved in both rounds. Minimum native immediate ratios are
1.164 / 1.086 and deferred 1.207 / 1.206; browser immediate minima are 1.149 /
1.269, deferred 1.477 / 1.473. This supports a bounded hot-loop improvement,
not a universal speedup, cold-start improvement or a model-quality claim.

**Eager Torch MPS is still faster.** Torch/candidate geomeans are 0.373..0.405,
about 2.47..2.68 times faster than the candidate across these rounds/cadences.
This is not fastest-PyTorch or torch.compile, and guard/ownership overhead is not
matched. Reducing temporary composition alone does not close the remaining gap.
Browser timings compare Rust A/B, not browser execution against Torch.

All captured terminal predictions, both input VJPs, both raw parameter VJPs and
updated weights are compared to an independent eager Torch reference. Maximum
absolute difference across learner timing captures is `1.7695128917694092e-8`.
Per-workspace state fingerprints also remain identical across reset repetitions.
Every interval is checked live; retained full states are per lane/cadence, not
separate full parameter dumps for every update.

## Correctness and Ordinary Control

The optimized frozen run passes all 29 verification steps. Native Metal and
browser WebGPU each complete the existing 12-condition, 64-update custom-objective
fixture, including both gain policies, fusion choices and bit-identical JSON
checkpoint/resume. Eight guard/reuse scenarios cover late parameter overflow,
product/sum overflow on both sides of a four-source boundary, zero-weight failure,
stale identities, parameterless graphs and 1/2/3/4/5/17/256/1/5-term reuse across
a 257-element parameter. The new reuse fixture's reference difference is zero.

Torch 2.12.1 CPU/MPS replay checks 120 cases and 17,136 comparisons including
older fixtures. The learner subset has 5,904 comparisons, maximum absolute error
`5.960464477539063e-8`; overall max is `5.7220458984375e-6`, within the declared
`atol=2e-5, rtol=2e-4` gate. Contracts 22, NN 734, backend 114, integration 6,
resident Python 38, public WASM learner/autograd/fusion/forward clients, types,
formatting, CPU-only builds and 13 benchmark-admission tests pass.

One additional ordinary MSE/SGD control round uses the same frozen binaries but
the original workload, including per-step loss reads. Its geomeans are native
1.012 immediate / 0.985 deferred, browser 1.000 / 1.029. **Individual native
delays remain unresolved**: minima are 0.863 / 0.829. This control does not prove
universal regression freedom, nor resolve earlier recorded delays. The control's
full-state max error is `1.4901161193847656e-8`.

Native hardware is Apple M4 Metal/MPS, with Torch fallback disabled. Browser
workspaces report BrowserWebGpu, not exact physical-adapter identity. Owned GPU
jobs are serial; host contention is UNKNOWN. The earlier parallel GPU VJP one-off
remains UNKNOWN. No whole-repository CI-green claim is made.

[summary.json](summary.json) separates both learner rounds from the ordinary
control and retains per-round minima/worst configurations. [manifest.json](manifest.json)
binds raw/compressed hashes and sizes for fixtures, logs, receipts and reproduction
drivers. All compressed streams are decompressed and verified. The three rounds
preserve 2,700 intervals (2,160 retained), representing 21,600 timed updates
including warmup intervals. Immutable generated products remain at receipt paths.
