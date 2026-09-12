# Shared Stage Guards In Resident NN Inference

Measured source: `542c744593dedf86e949aa82772e435acf35db04`;
tree `54f6b5f0ed389a0fc5d3962962ca5bedb889747c`.
Baseline products are from verified source
`b04028476f9ec6ae2b8487471ae4f942926eb6dc`, the bounded-output-reuse version.
Verification passed **48 serial steps**, then **12 steps** for three
predetermined paired replications. All 17 frozen products per version were
hash-checked before and after measurement. No source changed during a run.

See [API](../../../docs/module_resident_forward.md), [summary](summary.json),
[manifest](manifest.json), and [baseline evidence](../2026-09-12-module-output-reuse/README.md).

## Backend Change

Pointwise inference stages now write their own indexed words directly into the
shared graph validation buffer, alongside the existing dense-stage guards.
The graph clears that buffer once and retains the upstream-input guard copy.
It no longer allocates, clears and copies a separate flag buffer for every
pointwise stage. Arithmetic, accumulation policy, logical error-stage numbering
and owning output captures are unchanged. Shader guard addressing and internal
pointwise metadata change; this is not a claim of byte-identical shaders.

For P pointwise stages this removes P clear commands, P copy commands and P
private flag buffers. In the admitted 2/8/16-block fixtures, P is 4/16/32, so
8/32/64 management commands per forward disappear. These are code/plan-derived
counts, not GPU hardware counters or isolated phase timings. View packing,
parameter checks, command encoding, the final owning guard pass and explicit
terminal readback remain. The prior four-slot / 32 MiB retained-output budget
and its ownership rules are unchanged.

The original Rust Module, Python `model(WgpuTensor)`, browser
`Sequential.forward`, and explicit mixed inference graphs share this backend.
Standalone pointwise/training consumers retain flag slot zero. There is no new
Python/WASM arithmetic implementation, automatic host Tensor migration, or
generic autograd/ModuleTrainer routing change.

## Verification

- Rust: contracts 23; CPU tensor 440; WGPU-enabled tensor 488; NN 750;
  backend 130; integration 6; CPU handoff 7. These are separate configurations,
  not unique-test counts. One existing fractional-GL adapter test remains ignored.
- New GPU tests inspect all 25 guard words across early/late dense and pointwise
  overflow, both direct and explicit dispatch, pending snapshots and later valid
  reuse. Indexed guards at slots 0/3/31 preserve other words; scalar, empty and
  N-D inputs test inherited errors. An out-of-range slot is rejected.
- Python: 57 tests in the GPU-enabled build, including surface/host tests;
  six selected CPU-only tests. Three admission tests and an independent
  rectangular Torch CPU/MPS reference test also pass.
- Browser: 111 original-module assertions, including late-stage error numbering
  and reuse; 536 explicit graph-client assertions. Existing learner, training,
  autograd, pointwise, handoff, generated/shipped TypeScript and CPU-only
  Python/WASM checks pass.
- Independent Torch replay: 936 original-model browser output comparisons,
  maximum absolute error `1.1920928955078125e-7`; existing training/VJP replay
  120 cases / 17,136 comparisons, maximum `5.7220458984375e-6`.
- Four paired browser matrices match the admitted CPU reference for 371,664
  output comparisons, maximum absolute error `2.384185791015625e-7`.

## All Four Matrices

Unchanged fixture: seeds 17/29/43, three shapes, Scaler -> Linear -> GELU -> ReLU
blocks, three warmups and nine retained rotated sample blocks. Each burst runs
eight independent forwards of fixed resident input plus one completed host read.
It is not recurrent decoding or enqueue-only timing. Compilation is separate;
per-call parameter comparison and owning output handling are included.

Native order is baseline/candidate, candidate/baseline, baseline/candidate,
candidate/baseline. Each matrix also includes same-page browser A/B. All four
matrices remain in the record. Each group contains **12 paired seed-run medians**,
not twelve devices. Times below are medians of per-seed medians in ms/forward;
ratios are medians/ranges of paired ratios, not ratios of grouped time medians.

| Shape | Blocks | Python Old | Python New | New / Old Median | Ratio Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| [2,3,7] | 2 | 0.1096 | 0.0913 | 0.830 | 0.688-1.271 |
| [2,8,64] | 8 | 0.1955 | 0.1812 | 0.925 | 0.855-0.947 |
| [4,8,128] | 16 | 0.5524 | 0.5276 | 0.957 | 0.871-0.998 |

| Shape | Browser Old | Browser New | New / Old Median | Ratio Range |
| --- | ---: | ---: | ---: | ---: |
| [2,3,7] | 0.1000 | 0.0750 | 0.708 | 0.625-0.778 |
| [2,8,64] | 0.1875 | 0.1625 | 0.867 | 0.800-0.867 |
| [4,8,128] | 0.4875 | 0.4375 | 0.897 | 0.875-0.923 |

Native hardware reports Apple M4 Metal/MPS; Torch 2.12.1, MPS fallback disabled.
Chrome 152.0.7977.84 reports BrowserWebGpu, exact physical adapter **UNKNOWN**.
Native versions run in separate processes. Browser A/B uses separate WASM/device
instances with serial rotated routes in one page. Host exclusivity is unknown;
browser samples are visibly quantized.

Against eager Torch MPS in the candidate process, module burst ratio medians
are 0.739/0.537/0.781 for small/middle/large; ranges are 0.517-0.767,
0.526-0.550 and 0.733-0.822. This is not fastest PyTorch, `torch.compile`,
universal speedup, a training-quality comparison or isolated GPU-phase proof.

The explicit scalar graph also receives this patch. Its native paired ratio
medians are 0.811/0.927/0.960; browser medians are 0.774/0.857/0.944.
These routes are **not unchanged controls**. Eager Torch remains the unchanged
reference implementation. No reference ratio is used to normalize away drift.

## Regressions And Limits

- All 24 middle/large native burst pairs and all 36 browser burst pairs improve
  against the immediate baseline. This does not identify the cause of the
  previous output-reuse regression, or prove recovery against its earlier parent.
- Two small native burst pairs (seed 17, replications 1/2) regress by 6.2% and
  27.1%. Their d2h pairs also regress by 14.6% and 8.4%. They are retained.
- Small native Torch control ratios span 0.708-1.358 across processes. This
  large first-case drift limits attribution even after counterbalancing.
  Middle/large Torch controls span 0.944-1.023 and 0.920-1.042 respectively.
- Native single-call d2h paired ratio medians are 0.904/0.966/0.965. Browser
  d2h medians are approximately 1 for every size, so burst improvements do not
  establish a browser single-call gain. A strict greater-than-one browser entry
  is about `1.000000006`, below useful timer resolution; its raw value remains.
- No failed adopted pipeline, retries, discarded conditions or after-the-fact
  tuning. Dirty-source preflight is retained separately, not as timing evidence.
  Production allocation counters and GPU timestamps were not enabled.
- No CUDA/Furnace, release, push or merge in this slice. Broader model support,
  parameter-check costs and general training integration remain open.

`summary.json` retains each trial, raw ratios, explicit-graph/reference boundaries
and every regression. **104 records** contain 73,395,192 raw bytes, compressed
to 1,699,600 bytes. Raw and compressed SHA-256 values and round-trip checks bind
each archive member. Production binaries are identified by hashes, not committed.
