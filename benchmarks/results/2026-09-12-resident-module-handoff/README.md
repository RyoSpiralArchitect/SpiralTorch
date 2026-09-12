# Checked Resident Module Handoff

Adopted runtime source: `a0428e6ccd83e9eba47bb4c3b16e4ef8ab75c2ce`,
tree `815415c147d5ec5397e2e41ab42a32d0bdbec174`, run `verified-b`.
All 37 verification steps completed. Fifteen frozen products were checked by
hash. This directory contains numerical/integration evidence, **not timings**.

## What Connects

An original Rust/Python Module lowers to the resident graph, learns on WebGPU,
then receives its updated weights through the Rust-owned baseline/ownership
checks. WASM exports the same portable plan rather than recreating handoff
semantics. See the [API guide](../../../docs/resident_module_handoff.md).

The `[2, 2, 2]` fixture has two Linear layers, GELU, ReLU and two gains. It runs
16 updates under `exact` and `module_compatible`, each with and without graph
fusion. Tests intentionally read each loss/acceptance and export one final
weight snapshot. This does not introduce per-operation activation readbacks.

All six parameters transferred to the original model in every condition.
Browser predictions and the resumed host forward pass had **maximum absolute
error 0** for this fixture, with both GPU-enabled and CPU-only Python products.
Each then completed a fresh `ModuleTrainer` phase with finite changed weights.
This is weight transfer, not continuation of the old optimizer/trainer state.

## Verification

- Rust configurations: contracts 23; Tensor CPU 440; Tensor WGPU 488; CPU
  handoff 7; NN WGPU 742; backend 117; integration 6 passed. Counts include
  repetitions across configurations. One existing fractional GL adapter test
  remains marked ignored; the new WGPU transpose test ran and rejected fallback.
- GPU-enabled Python: 48 tests passed. CPU-only Python: four selected host
  handoff tests passed; build metadata confirmed no WGPU/MPS/CUDA/HIP feature.
- Public WASM forward, fusion, autograd, learner and pointwise clients passed,
  as did generated/shipped TypeScript checks and the CPU-only WASM build check.
- Existing learning/autograd fixtures passed the independent PyTorch CPU/MPS
  replay: 120 cases, 17,136 comparisons, maximum absolute error
  `5.7220458984375e-6`. This is not a throughput comparison.

The native GPU was Apple M4 Metal. Chrome reported `BrowserWebGpu` with an
unspecified physical adapter. Owned GPU jobs were serial; host contention is
`UNKNOWN`. Prior mixed timings and the older parallel GPU VJP failure remain
unresolved. No CUDA/Furnace run, wheel release or automatic routing change is
part of this result.

## Failures Retained

The first CPU handoff test exposed a real column-major Tensor transpose bug:
forward matched, but `Linear.backward` input gradients did not. The fix retains
the row-major path and adds logical-layout, packed-transpose and DLPack result
isolation regressions. Protected snapshots can share; mutable results cannot
export a writable alias to the input. DLPack itself remains row-major-only.

Initial source-bound run `verified-a` stopped on Python rejection exception
types. The binding now preserves the InferencePlan `ValueError` contract. A new
DLPack test initially assumed unsupported column-major export; it was corrected
to exercise the real public transpose-result export path. These failed logs
are preserved separately, not counted as adopted passes.

`summary.json` distinguishes scope, cases and remaining boundaries.
`manifest.json` binds 82 compressed raw artifacts to both original and compressed
hashes. All were decompressed and verified at collection: 18,361,763 original
bytes, 810,116 compressed bytes. Frozen binaries are retained locally and
identified in the archived receipt; they are not committed here.
