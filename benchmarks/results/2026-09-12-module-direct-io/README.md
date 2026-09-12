# Direct GPU I/O Behind The Original Module

Measured source: `e8bc5754e01b4deb170836a67ebae857d89155bb`;
tree `d25f3cb7894119765d30457a0d2a351d3e067c82`.
The first complete `verified-a` run passed **48 serial steps**, including
17 frozen product checks. Baseline products come from verified source
`c5349790398d406d1a5a39566cf5d3d22db20181`, not a reconstructed old checkout.
The two versions were freshly measured on the same nine input/weight fixtures.

See [API](../../../docs/module_resident_forward.md), [summary](summary.json),
[manifest](manifest.json), and [baseline evidence](../2026-09-12-module-resident-forward/README.md).

## What Changed

The cached original Rust Module now uses `ResidentGraph::forward_tensor`.
Contiguous offset-zero inputs are read directly; the final NN stage writes
directly into a fresh owning GPU tensor. Optional view packing, NN evaluation
and frozen error flags share **one queue submission**. The previous packed-input
module path used three submissions plus full-sized input/output bridge copies.
These are code-level execution boundaries, not GPU timestamp measurements.

Python `model(WgpuTensor)` and browser `Sequential.forward` inherit the change.
Explicit graphs expose `forward_tensor` / `forwardTensor`. Legacy upload,
set-input, dispatch and snapshot APIs still interoperate: dispatch repeats the
current input, and old outputs survive workspace reuse/drop. The NN math,
kernel choices, validation thresholds and parameter comparison are unchanged.

This is not allocation-free execution. Fresh outputs, boundary bindings, small
stage-flag copies and a guard compute pass remain. Strided/offset inputs may
need packing. Ordinary host-backed `pure::Tensor`, generic autograd and
`ModuleTrainer` are not silently migrated to this route.

## Verification

- Rust: contracts 23; CPU tensor 440; WGPU-enabled tensor 488; NN 750;
  backend 121; integration 6; CPU handoff 7. These are separate configurations,
  not unique-test counts. One existing fractional-GL adapter test remains ignored.
- New backend checks cover eight graph patterns, five layouts and six dense
  kernel/accumulation combinations, including singleton/rectangular boundaries,
  direct/explicit transitions, failed calls, invalid-value masking and ownership.
- Python: 55 tests in the GPU-enabled build, including surface/host tests;
  six selected CPU-only tests. Three CPU-only benchmark admission tests and the
  explicit rectangular Torch CPU/MPS reference test also pass.
- Browser: 62 original-module/direct-I/O assertions; 536 explicit graph-client
  assertions. Existing graph training, learner, autograd, pointwise, handoff and
  generated/shipped TypeScript checks pass. CPU-only Python/WASM builds pass.
- Independent Torch replay: 936 original-model browser output comparisons,
  maximum absolute error `1.1920928955078125e-7`; existing training/VJP replay
  120 cases / 17,136 comparisons, maximum `5.7220458984375e-6`.
- Paired browser timing captures match the admitted CPU reference for 92,916
  output comparisons, maximum absolute error `2.384185791015625e-7`.

## Matched Timings

Each block is Scaler -> Linear -> GELU -> ReLU. Three seeds, three warmups and
nine retained rotated sample blocks per condition. Each burst performs eight
independent forwards of fixed resident input and includes one final completed
host read. It is not recurrent decoding or asynchronous enqueue-only timing.

Each table cell is the median across three per-seed medians, in ms/forward.
Ratios use paired per-seed medians; below 1 is faster. Medians of ratios and
ratios of group medians are not interchangeable.

| Shape | Blocks | Python Old | Python New | New / Old Range |
| --- | ---: | ---: | ---: | ---: |
| [2,3,7] | 2 | 0.1795 | 0.1378 | 0.673-0.785 |
| [2,8,64] | 8 | 0.2150 | 0.1913 | 0.884-0.936 |
| [4,8,128] | 16 | 0.5221 | 0.5308 | 0.988-1.031 |

| Shape | Browser Old | Browser New | New / Old Range |
| --- | ---: | ---: | ---: |
| [2,3,7] | 0.1500 | 0.1000 | 0.667-0.714 |
| [2,8,64] | 0.1875 | 0.2000 | 1.000-1.067 |
| [4,8,128] | 0.4875 | 0.4875 | 1.000-1.000 |

Browser A/B routes rotate in the same page, using separate WASM/device instances.
Native versions run serially in separate processes, each with eager Torch
controls. Apple M4 Metal/MPS, Torch 2.12.1, MPS fallback disabled; Chrome
152.0.7977.84 reports BrowserWebGpu, exact physical adapter **UNKNOWN**.
Host exclusivity is not established; browser samples are visibly quantized.

Against eager Torch MPS in the candidate run, module burst ratios are
1.104-1.133 (small), 0.566-0.575 (middle), 0.780-0.796 (large).
The small module is still slower than Torch. This is not a fastest-PyTorch,
`torch.compile`, universal speedup, phase-isolation or training-quality claim.

## Regressions And Limits

- Browser middle-size burst seed 29 is 6.7% slower. Native large-size seeds
  29/43 are about 3% slower. Large browser burst medians are unchanged.
- Single-call d2h does not uniformly improve: native middle/large cases are
  up to 5.5% slower; some quantized browser cases are 10-20% slower. These
  captures are retained separately, not conflated with burst improvements.
- In the first/small native case, the cross-process Torch control changes by
  about -36% and the scalar control by +10%. Do not attribute the entire native
  difference to this patch. The same-page browser comparison independently
  shows the small-burst improvement; it does not identify an isolated GPU phase.
- No failed adopted pipeline, discarded seeds or timing retries. Preliminary
  dirty-source test logs are retained only as preflight, not timing evidence.
- No CUDA/Furnace, release, push or merge in this slice. Source and evidence
  are local commits. Kernel/allocator/parameter-check optimization remains open.

The archive retains **78 records**, 50,565,152 raw bytes compressed to 932,464
bytes. Each record has raw and compressed SHA-256 values and was round-trip
checked. Production binaries are identified by hashes, not committed here.
