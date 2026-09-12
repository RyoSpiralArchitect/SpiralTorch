# Loss-Independent Resident Graph VJPs

## Change

Frozen implementation: `16d41e643c260592b8e246cfbf73f399599dc066`.
`InferencePlan::compile_graph_autograd_wgpu` compiles supported NN graphs into
separate resident forward and arbitrary-cotangent backward phases. Python and
WASM expose the same Rust-owned kernels, tape identity, N-D layouts and guards.
There is no implicit MSE, gain row average, accumulation, clipping or SGD step.
See [the API and examples](../../../docs/resident_graph_autograd.md).

An opaque token identifies the latest forward of exactly one workspace. Its
prediction survives reuse/drop, while stale or foreign tokens cannot reuse the
tape. Each backward returns independent resident input and parameter gradients.
Forward failure cannot be masked by a zero cotangent; a failed backward does not
poison later valid VJPs of a valid forward. Every gradient retains whole-backward
guards, including a late parameter reduction failure. Acceptance is established
by guarded reads, not submission counters.

Parameters remain frozen. This is **not** a migration of ordinary ModuleTrainer,
generic autograd, GNN, or host-backed `pure::Tensor`. No default routing, CUDA,
remote Furnace run, wheel release, push or merge is claimed. GPU freeze operations
still cost dispatches/copies; the new API's throughput has not been benchmarked.

## Validation

The adopted run is `verified-b`; it passed all 27 recorded verification steps.

- Native Metal and browser WebGPU each ran six arbitrary-cotangent conditions:
  ranks 1/2/3, fusion on/off, four independent seeds per forward, offset/strided/
  broadcast views and a seed formed from the resident prediction.
- Eleven guard cases per runtime cover masked forward failure, adjoint overflow,
  gain reduction overflow, dense failures, wrong/stale tokens, invalid input/seed
  shape, separate GPU contexts, inherited failures, recovery, owning outputs and
  a parameterless repeated-residual VJP.
- Torch 2.12.1 CPU/MPS replay: 72 cases and 11,232 comparisons including existing
  training fixtures. Arbitrary-cotangent VJPs account for 24 cases, 96 replays and
  864 comparisons. Maximum absolute difference is `5.7220458984375e-6`, within
  the recorded `atol=2e-5, rtol=2e-4` gate. Native/browser differences from the
  ordinary Rust Module reference are at most `9.54e-7` / `1.20e-6`.
- Contracts 22, NN 734, WGPU backend 114 and integration 6 tests passed. Python
  existing 32 plus new 3 tests passed without skips. Public WASM autograd and
  fusion clients passed; existing forward clients passed 536 assertions over
  12 cases. Generated/shipped types, formatting, NN/Python/WASM CPU-only checks,
  11 benchmark-admission and four profile-admission tests passed.

The first full attempt, `verified`, stopped at the **old forward client setup**:
the runner omitted `forward-fixture.json`, causing ENOENT before that suite's GPU
execution. Its earlier suites passed. The runner now generates that fixture from
the same source; a complete rerun passed without changing the implementation.
Both attempts and the initial/corrected runners are preserved. Exploratory output
is retained separately and is not substituted for adopted source-bound output.

## Existing-Path Regression Comparison

Two complete standard-matrix rounds compare the old frozen `f3a4b271` products
with `16d41e64`. **Both run ordinary MSE/SGD, without pointwise fusion**: this
tests shared-code regression, not the speed of the new arbitrary-cotangent API.
Each round uses nine shape/depth/seed cases, eight nonzero-rate updates per
interval, two warmups and eight retained blocks per observation cadence.

Ratios are geometric means of **old median / new median** across all nine cases;
above 1 means the new build was faster. Columns show both rounds, not a selection.

| Runtime | Immediate Loss Reads | Deferred Loss Reads |
| --- | --- | --- |
| Native Metal | 0.973 / 1.015 | 1.041 / 1.023 |
| Chrome WebGPU | 1.004 / 0.996 | 1.001 / 0.997 |

Browser aggregate timings are close to unchanged. **Native individual regressions
remain unresolved**: worst immediate ratios are 0.756 / 0.822, despite the aggregate
changing direction between rounds. Worst browser immediate ratios are 0.976 /
0.957. Host contention is UNKNOWN; these results do not establish universal
performance preservation or a speedup. No slower cases were removed.

Eager Torch MPS / candidate geomeans are 0.830--0.885 across these rounds/cadences:
Torch remains faster in aggregate. This is not a fastest-PyTorch or torch.compile
comparison. Rust checks intermediates and transactional updates; the eager Torch
finite-fixture reference does not implement identical guard/rollback overhead.
Native hardware is Apple M4 Metal/MPS; browser reports BrowserWebGpu without exact
physical-adapter attestation. Owned GPU work was serial; the older parallel GPU
VJP one-off remains UNKNOWN. This is not a whole-repository/CI-green claim.

All 1,800 timing intervals, 1,440 retained intervals and 14,400 nonzero-rate updates
are preserved. Full benchmark states match the reference within `1.49e-8` maximum
absolute error. Every loss is read. Reset, compilation and zero-rate probes are
outside the timer. Browser timing is Rust A/B, not browser-vs-Torch performance.

[`summary.json`](summary.json) records per-round extrema and test counts.
[`manifest.json`](manifest.json) binds 101 artifacts: 17,239,088 compressed bytes,
336,480,715 raw bytes. Each compressed stream was decompressed and its length and
SHA-256 verified. Raw logs and immutable products remain in the external run
directory named in the receipts. The production API contract is in the source,
not reconstructed from these reports.
