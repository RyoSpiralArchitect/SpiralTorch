# Checked Descriptor Assembly: Fewer Allocations, Mixed Timing

Measured source: `d65b87f622c38a7dc3034fce90efda124d3ee18c`;
tree `bcc8542610f3d49ac2d76be42765d0a6a0494f92`.
Baseline products are from verified source
`542c744593dedf86e949aa82772e435acf35db04`, the shared-stage-guard version.
Verification passed **50 serial steps** plus **12 replication steps**.
All 17 frozen products per version were checked before and after measurement.
No source changed during a run.

**This is not an established inference-speed improvement.** The allocation
reduction is verified, but browser middle-size bursts and native large-size
bursts regress. `status: passed` in the evidence means correctness and measurement
admission passed, not a performance acceptance gate. The candidate is retained
locally for separate descriptor-assembly/comparison ablations, not promoted as
the new fastest path or published in a release.

See [API](../../../docs/module_resident_forward.md), [summary](summary.json),
[manifest](manifest.json), and [baseline evidence](../2026-09-12-module-shared-guard/README.md).

## Rust Change

Built-in Modules append their checked inference descriptors into one vector.
Nested Sequential modules no longer construct a vector for every leaf. The
optional `Module::append_inference_ops` companion preserves custom modules that
only implement `inference_ops` through its default bridge. Standard Sequential
restores the caller's existing prefix after a child fails. Descriptor ordering,
parameter handles, layouts and Linear finite-value validation are unchanged.

The resident cache uses exact byte-slice equality for parameter comparison,
without temporary arrays or hashes. Mutable/foreign pointer identity is still
insufficient: every forward compares parameter contents against protected
snapshots. Signed zero and layout changes remain significant. This is still
O(parameter values) CPU work, not parameter versioning or a zero-cost check.

Only the original NN Module's CPU preparation changes. GPU arithmetic, shaders,
guard indexing, output-slot ownership and queue scheduling are unchanged. The
same Rust implementation serves native Python and browser WASM. Existing host
Tensor, generic autograd and ModuleTrainer routing are not automatically migrated.

## Allocation And Correctness Checks

- A dedicated native integration-test allocator measures the same 64 checked
  descriptors: **69 allocations with the former assembly algorithm, 1 with the
  new original Module assembly**. This is an algorithm-level control using the
  current checked leaf descriptors, not instrumentation of the old binary.
- Nested assembly with caller-provided capacity makes **0 allocations in each
  of 20 calls**. It still clones parameter handles. Ordinary resident forwarding
  allocates its final vector; this is not globally allocation-free inference.
- Both CPU-only and WGPU-enabled builds pass the four allocation/compatibility
  tests. The production allocator and GPU timing are not instrumented.
- New cache tests cover odd lengths, signed zero, NaN payload bit distinctions,
  and external DLPack writes to the first/last weight and last bias. NaN weights
  are rejected before graph execution; restoring them resumes valid reuse.
  Previously returned output remains unchanged. Foreign writes are serialized,
  not concurrent unsynchronized access.
- Rust: contracts 23; CPU tensor 440; WGPU-enabled tensor 488; NN 752; backend
  130; integration 6; CPU handoff 7. One existing tensor test remains ignored.
  These are separate configurations, not unique-test totals.
- Python: 57 tests in the GPU-enabled build, including surface/host tests;
  six selected CPU-only tests. Three admission tests and the independent
  rectangular Torch CPU/MPS reference test also pass.
- Browser: 111 original-module assertions and 536 explicit graph-client
  assertions. Learner, training, autograd, pointwise, handoff, generated/shipped
  TypeScript and CPU-only Python/WASM checks pass.
- Independent Torch replay: 936 original-model browser output comparisons,
  maximum absolute error `1.1920928955078125e-7`; existing training/VJP replay
  120 cases / 17,136 comparisons, maximum `5.7220458984375e-6`.
- The four paired browser matrices match the admitted CPU reference for 371,664
  output comparisons, maximum absolute error `2.384185791015625e-7`.

## All Four Matrices

The fixed fixture uses seeds 17/29/43, three shapes, and
Scaler -> Linear -> GELU -> ReLU blocks. Each route has three warmups and nine
retained rotated sample blocks. A burst is eight independent forwards with fixed
resident input and one completed terminal host read, not recurrent decoding or
enqueue-only timing. Cold compilation is separate; per-call parameter checks
and descriptor assembly are included.

Native orders are baseline/candidate, candidate/baseline, baseline/candidate,
candidate/baseline. Each matrix also includes same-page browser A/B. All four
matrices remain in the record, with **12 paired seed-run medians per group**.
Times are medians of per-seed medians in ms/forward. Ratios are medians/ranges of
paired ratios, not ratios of grouped time medians.

| Shape | Blocks | Python Old | Python New | New / Old Median | Ratio Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| [2,3,7] | 2 | 0.0917 | 0.0908 | 0.996 | 0.801-1.380 |
| [2,8,64] | 8 | 0.1802 | 0.1834 | 1.012 | 0.963-1.072 |
| [4,8,128] | 16 | 0.5220 | 0.5349 | 1.017 | 1.007-1.105 |

| Shape | Browser Old | Browser New | New / Old Median | Ratio Range |
| --- | ---: | ---: | ---: | ---: |
| [2,3,7] | 0.0750 | 0.0750 | 1.000 | 0.833-1.200 |
| [2,8,64] | 0.1500 | 0.1625 | 1.083 | 0.923-1.083 |
| [4,8,128] | 0.4375 | 0.4375 | 1.000 | 1.000-1.029 |

Native device admission reports Apple M4; Torch 2.12.1 uses MPS with fallback
disabled. Chrome 152.0.7977.84 reports BrowserWebGpu; the exact physical browser
adapter is **UNKNOWN**. Native versions run in separate processes. Browser A/B
uses separate WASM/device instances with serial rotated routes in one page.
Host exclusivity is unknown and browser timing is visibly quantized.

Against eager Torch MPS in the candidate process, the module burst ratio medians
are 0.726/0.535/0.792 for small/middle/large, with ranges 0.606-0.807,
0.489-0.576 and 0.670-0.812. This does not show that the candidate improves on
SpiralTorch's immediate baseline. It is not fastest PyTorch, `torch.compile`,
a training-quality result or an isolated GPU-phase measurement.

## Regressions And Decision

- Native bursts regress in 22/36 pairs: 4 small, 6 middle and **all 12 large**.
  Large burst median regression is 1.7%, with individual regressions up to 10.5%.
  This cannot be dismissed solely because allocation counts improved.
- Browser bursts regress in 17/36 pairs: 5 small, 7 middle and 5 large.
  Middle paired-ratio median is 1.083; small/large medians are 1.0. Quantization
  limits precision, but it is not grounds for excluding the adverse results.
- Native single-call d2h medians are 1.001/0.981/0.945. All 12 large single-call
  pairs improve, unlike bursts. Browser d2h medians are approximately 1.0 across
  sizes. The CPU changes are not yet isolated from each other or from scheduling.
- Explicit graph and eager Torch implementations are unchanged controls in
  this slice. Their native paired-ratio medians are 1.004/0.998/0.999 and
  0.993/1.005/1.017 respectively. Small-case ranges are wide: 0.974-1.663 for
  explicit graph, 0.883-1.286 for Torch. Browser explicit-graph medians are 1.0.
  No control ratio normalizes away drift or erases a regression.
- The next acceptance step is to isolate descriptor assembly and bulk parameter
  comparison against the retained baseline. The current record establishes
  fewer descriptor allocations and preserved behavior, **not a burst-speed win**.
- No retries, discarded matrices, after-the-fact tuning, CUDA/Furnace run,
  push, merge or release. Dirty-source preflight logs remain separate from the
  admitted frozen-source timing evidence.

`summary.json` retains all trials, controls and regressions. **107 records**
contain 73,505,249 raw bytes, compressed to 1,702,192 bytes. The manifest binds raw
and compressed SHA-256 values, with a round-trip check for every member.
Production binaries are identified by hashes, not committed.
