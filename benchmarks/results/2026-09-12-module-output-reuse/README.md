# Bounded Output Reuse Behind The Original Module

Measured source: `b04028476f9ec6ae2b8487471ae4f942926eb6dc`;
tree `0754c02d4fd7ef5bf43d2e9f072eecbb4fe5bdc1`.
Baseline products are from verified source
`e8bc5754e01b4deb170836a67ebae857d89155bb`, the direct-I/O implementation.
The complete verification passed **48 serial steps**, followed by **12 steps**
for three predetermined paired replications. All 17 frozen products per version
were hash-checked before and after measurement. No source changed during a run.

See [API](../../../docs/module_resident_forward.md), [summary](summary.json),
[manifest](manifest.json), and [baseline evidence](../2026-09-12-module-direct-io/README.md).

## What Changed

The original Rust Module, Python `model(WgpuTensor)`, and browser
`Sequential.forward` share the same `ResidentGraph::forward_tensor` path.
It now reuses an owning output slot only when neither its values nor its guard
have another strong or weak storage owner. Views, bound consumers and current
graph state retain ownership; old outputs are never overwritten while observable.

Each graph retains at most **four slots / 32 MiB of output values and guards**.
Busy or oversized outputs allocate separately, without waiting or CPU fallback.
This limits retained output data, not total physical GPU/model/scratch/binding
memory or externally held outputs. Native test-only allocation counters confirm
two allocations for twenty fixed-input forwards, three for recurrent forwarding,
and four with a retained early view/consumer. Counters are absent from the
production artifacts used for timing.

The retained final-stage and guard bindings are reused with the output slot.
Multi-stage first-input bindings are reused only for identical packed storage;
successful explicit upload/set-input invalidates that key. Singleton and changed
input boundaries still rebind. Parameter checks, shaders, accumulation policy,
one-submission schedule and deferred-error semantics are unchanged. This is not
globally allocation-free execution, automatic host Tensor migration, or a new
autograd/ModuleTrainer route.

## Verification

- Rust: contracts 23; CPU tensor 440; WGPU-enabled tensor 488; NN 750;
  backend 128; integration 6; CPU handoff 7. These are separate configurations,
  not unique-test counts. One existing fractional-GL adapter test remains ignored.
- Added checks cover pool saturation, actual 32 MiB GPU allocation, views,
  shared guards, pending snapshots, failed-output guard reuse, another executor
  and another thread retaining an old version. Existing graph-pattern/layout/
  kernel checks remain enabled. The cross-thread consumer reads only after the
  producer finishes; this is not a concurrent GPU-throughput benchmark.
- Python: 56 tests in the GPU-enabled build, including surface/host tests;
  six selected CPU-only tests. Three admission tests and the independent
  rectangular Torch CPU/MPS reference test also pass.
- Browser: 93 original-module assertions, including output reuse; 536 explicit
  graph-client assertions. Existing learner, training/autograd, pointwise, handoff,
  generated/shipped TypeScript and CPU-only Python/WASM checks pass.
- Independent Torch replay: 936 original-model browser output comparisons,
  maximum absolute error `1.1920928955078125e-7`; existing training/VJP replay
  120 cases / 17,136 comparisons, maximum `5.7220458984375e-6`.
- Four paired browser matrices match the admitted CPU reference for 371,664
  output comparisons, maximum absolute error `2.384185791015625e-7`.

## All Four Matrices

The fixture is unchanged: three seeds (17, 29, 43), three shapes, and blocks of
Scaler -> Linear -> GELU -> ReLU. Three warmups and nine retained rotated sample
blocks per condition. Each burst performs eight independent fixed-input forwards
and one final completed host read, not recurrent decoding or enqueue-only timing.
Compilation is separate; parameter comparisons and output ownership are timed.

Native version order is baseline/candidate, candidate/baseline,
baseline/candidate, candidate/baseline. Each matrix also runs the same-page
browser A/B. All four matrices are retained, including the initial run. Every
group below contains **12 paired seed-run medians**, not twelve independent
devices. Times are medians of those per-seed medians in ms/forward; ratios use
paired medians and are not ratios of the grouped time medians.

| Shape | Blocks | Python Old | Python New | New / Old Median | Ratio Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| [2,3,7] | 2 | 0.1391 | 0.1090 | 0.793 | 0.743-0.929 |
| [2,8,64] | 8 | 0.1940 | 0.1943 | 0.996 | 0.944-1.029 |
| [4,8,128] | 16 | 0.5312 | 0.5516 | 1.039 | 0.973-1.051 |

| Shape | Browser Old | Browser New | New / Old Median | Ratio Range |
| --- | ---: | ---: | ---: | ---: |
| [2,3,7] | 0.1125 | 0.1000 | 0.889 | 0.778-1.000 |
| [2,8,64] | 0.1875 | 0.1875 | 1.000 | 1.000-1.000 |
| [4,8,128] | 0.4875 | 0.4875 | 1.000 | 0.975-1.026 |

Native hardware reports Apple M4 Metal/MPS; Torch 2.12.1, MPS fallback disabled.
Chrome 152.0.7977.84 reports BrowserWebGpu, exact physical adapter **UNKNOWN**.
Native versions run in separate processes; browser versions use separate
WASM/device instances with serial rotated routes in one page. Host exclusivity
is not established, and browser samples are visibly quantized.

Against eager Torch MPS within each candidate process, the burst ratio medians
are 0.874 (small), 0.574 (middle), 0.826 (large); ranges are 0.695-0.908,
0.529-0.607 and 0.675-0.844 respectively. This is a bounded eager-reference
comparison, not fastest PyTorch, `torch.compile`, universal speedup, isolated
GPU-phase attribution or evidence about learning quality.

## Regressions And Limits

- Small native bursts improve in all twelve pairs. Middle bursts are essentially
  unchanged overall; large native bursts regress by 3.9% at the paired median,
  with 11/12 pairs slower and a worst ratio of 1.051. Resource reuse alone does
  not establish a throughput win for a deeper graph.
- Browser middle/large burst medians are unchanged. Two large pairs regress by
  2.6%; some small pairs are unchanged. Browser resolution limits interpretation
  of these small differences.
- Single-call d2h is a separate boundary: native paired ratio medians are
  0.811/0.847/0.964, but one small pair regresses by 6.8%. Quantized browser
  d2h has three slower pairs, including a 33.3% small-case regression. None are
  replaced by burst results or dropped from the archive.
- Initial small native controls drift by +26.7% (scalar) and -23.1% (Torch);
  the first replication has Torch drift of +10.4% and +11.8% in two cases.
  All controls are preserved. Counterbalancing does not prove exclusivity,
  remove warm-start effects, or establish inferential significance.
- No failed adopted pipeline, retries, discarded seeds or after-the-fact tuning.
  Dirty-source preflight logs are retained only as preflight, not timing evidence.
  Production allocation counters and GPU timestamps were not enabled.
- No CUDA/Furnace, release, push or merge in this slice. The native large-burst
  regression and remaining parameter/encoding costs remain open for follow-up.

`summary.json` includes each trial, all grouped ratios, control drift and
explicit regressions. The archive retains **106 records**, 73,623,269 raw bytes
compressed to 1,707,228 bytes. Each has raw and compressed SHA-256 values and
was round-trip checked. Production binaries are hash-identified, not committed.
