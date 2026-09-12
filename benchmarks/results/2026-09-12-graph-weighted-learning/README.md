# Weighted Resident Graph Learning

## Change

Frozen implementation: `c23591e53bcca00d7570005c3bcb7503b1a64d5a`.
Supported NN graphs now have an opt-in loss-independent resident learner:
forward, exact arbitrary-cotangent VJPs, weighted composition, explicit gain
normalization and atomic SGD. Gradients and weights do not need CPU readback.
Rust owns these semantics; Python and WASM expose the same owning handles.
See [the API and example](../../../docs/resident_graph_learner.md).

An owning batch accepts 1..=256 contributions from the same current forward.
The ordered weighted sum is checked at every intermediate. Even a zero-weight
bad source rejects the whole update; cancellation cannot erase overflow. All
parameter candidates are decided together before any commit. A submitted update
invalidates the tape, including zero-rate and rejected attempts. Saved receipts
retain their own acceptance/identity across subsequent work and destruction.
Checkpoints export only weights/plan; update policy must be selected on resume.

This is not an automatic migration of `ModuleTrainer`, its hypergrad/sync/band
policies, generic autograd, GNN or host `pure::Tensor`. Only the supported resident
graph operations are covered. The new path creates temporary composition buffers
and still performs GPU copies/dispatches; its throughput is **not** benchmarked
here. No Adam, momentum, cross-forward accumulation or mixed precision is added.
No CUDA/Furnace run, default routing change, push, merge or wheel release occurred.

## Validation

The adopted `verified` run passed all 29 recorded steps, bound to the source
above and 19 immutable generated-product hashes.

- Native Metal and browser WebGPU each completed 768 nonzero-rate updates in
  12 conditions: three seed/shape pairs, ranks 1/2/3, exact/module-compatible
  policy, fused/unfused plans. This is not an independent crossed seed/shape sweep.
- Two GPU-generated cotangents represent a weighted quadratic/quartic objective.
  All 64 updates are enqueued before mapped observations. Five checkpoints per
  condition check predictions, both raw VJPs and updated parameters against
  ordinary Rust NN. Every attempted update has a captured acceptance receipt.
- All conditions reduce the fixture objective. For example, shape `[4]` drops
  from `0.0443995893` to `0.0001333519`; shape `[2,129,4]`, exact policy, drops
  from `0.0386026204` to `0.0210493412`. These are bounded learning fixtures, not
  evidence of language-model quality or a Z-space advantage.
- Weight-only JSON checkpoint/resume produces bit-identical next-update
  parameters in every condition. Five core guard scenarios cover a late bias
  overflow, weighted-product/sum cancellation, invalid identities/arguments,
  zero-weight invalid gradients, receipt ownership/recovery and no-parameter graphs.
- Independent Torch 2.12.1 CPU/MPS replay: 120 cases, 17,136 comparisons including
  earlier fixtures. The new learner accounts for 48 replays of 64 updates and
  5,904 comparisons, maximum absolute error `5.960464477539063e-8`. Overall max
  error is `5.7220458984375e-6`, within `atol=2e-5, rtol=2e-4`.
- Contracts 22, NN 734, WGPU backend 114 and integration 6 tests pass. Resident
  Python tests: existing 32, autograd 3, learner 3; no skips. Public WASM clients
  verify both policies, 32-update learning, checkpoint/resume, 256-term admission,
  signed weights, stale tapes, zero-weight rollback and snapshot consumption.
  Earlier autograd/fusion/forward clients, generated/shipped type checks,
  formatting, CPU-only builds, benchmark/profile admission also pass.

Native and browser max differences from the Rust Module learning reference are
both `5.960464477539063e-8`. Native device is Apple M4 Metal/MPS; the browser
reports BrowserWebGpu without exact physical-adapter attestation. Torch fallback
is disabled. Owned GPU work is serial, not an assertion of exclusive host use.
The earlier parallel GPU VJP one-off remains UNKNOWN; this is not an all-CI claim.

Initial exploratory Python assertions used the wrong consumed-snapshot exception
type and rejection wording; the old TypeScript checker also assumed every NN
class had a private constructor. These tests were corrected without weakening
runtime guards: only the gradient batch has a public constructor. Failed draft
logs and later passing checks are retained, separate from the frozen adopted run.

## Ordinary-Path Performance

Two standard-matrix rounds compare frozen `16d41e64` products to `c23591e5`.
Both timing lanes run the **existing MSE/SGD route, fusion disabled**. This tests
shared-code regressions, not new-learner throughput. Each round contains nine
shape/depth/seed conditions, eight nonzero-rate updates per interval, two warmups
and eight retained blocks per observation cadence. All losses are read; compile,
reset and zero-rate probes are outside timing.

Ratios are geomeans of old median / new median; above 1 favors the new build.
Both rounds are shown without removing slower conditions.

| Runtime | Immediate Reads | Deferred Reads |
| --- | --- | --- |
| Native Metal | 0.976 / 1.019 | 1.002 / 1.000 |
| Chrome WebGPU | 0.998 / 1.008 | 1.004 / 1.011 |

Aggregate browser measurements are near unchanged. Native individual delays
remain **unresolved**: minimum immediate ratios are `0.682 / 0.935`; deferred
minima are `0.803 / 0.915`, at different conditions between rounds. Scheduling/
host contention is UNKNOWN. No universal speedup or regression-free claim follows.

Eager Torch MPS / candidate geomeans span `0.815..0.883`: Torch is still faster in
aggregate. This is not fastest-PyTorch/torch.compile comparison. SpiralTorch's
finite guards/transactional rollback have no identical counterpart in this eager
Torch finite-fixture reference. Browser timings are Rust A/B, not browser/Torch.

All 1,800 raw intervals, 1,440 retained intervals and 14,400 timed updates are
preserved. Full benchmark states agree with reference within `1.49e-8` maximum
absolute error. Per-condition extrema are in [summary.json](summary.json).
[manifest.json](manifest.json) binds the compressed logs, fixtures, receipts and
reproduction drivers with raw/compressed SHA-256 and sizes; every stream is
decompressed and verified. Generated binaries remain at the paths in receipts.
