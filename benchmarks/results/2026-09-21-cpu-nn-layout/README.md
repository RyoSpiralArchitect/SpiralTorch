# CPU NN Layout Preparation And Safe Faer Boundaries

Baseline: `9847fb7fc992efd82fcb000236314319557adc0a`.
Selected source: `d5fa45129429675370c3570ef534051c24c6a9ef`.
This follows the [ordinary-kernel comparison](../2026-09-21-cpu-row-major/README.md),
but measures a different boundary: actual `Sequential`/`Linear`/`Gelu` forwards,
their parameter-cache refreshes and layout preparation. It does not turn the
previous explicit CPU-kernel gap into a claim about every NN call.

## Implementation

- Share one bit-preserving CPU transpose across row-major packing, row/column
  layout conversion and Tensor transpose. Use 32x32 blocks for larger buffers;
  retain a simple loop at 4096 elements or fewer and a copy for vectors.
- Pack a column-major transpose directly, without an intermediate Tensor/copy.
  The public packed layout, parameter ownership and cache-invalidation rules do
  not change. This internal packing no longer emits an intermediate Tensor
  transpose event.
- Validate all Faer operand/output dimensions and exact input lengths before
  mutation or allocating output. The public safe entry points previously formed
  raw-pointer matrix views without checking the input slices. Safe slice-based
  views replace those unchecked constructors. Malformed calls return errors;
  valid numerical semantics and backend routing remain unchanged.
- Remove Faer's redundant destination-zeroing pass for nonempty `Accum::Replace`
  products. Tests start with NaN destinations and cover both layouts on both sides.

Native prepacked Auto chooses Faer for the fixed shapes in this CPU-only build.
Explicit Faer and CPU-SIMD modes are measured separately. No high-level routing
policy, public packed representation, GPU kernel or dependency was changed.

## Measurements

Apple M4, Rust 1.98.0 release, four configured Rayon threads. Host exclusivity and
thermal state are unknown. Native calls include output creation/destruction;
model/input construction is outside timing. There are three warmups, nine
untrimmed intervals and two calls per interval. Fixed-kernel measurements remove
HOME and the autotune-store path; validation separately exercises a local tuning
store. Thread configuration does not establish physical utilization.

Six fixed `(batch, input, hidden)` shapes include tiny, tail and larger Linear
workloads. The 768x3072 weight shape appears at both batch 8 and batch 32: these
are distinct NN workloads, not six distinct preparation sizes. An MLP is
Linear(input, hidden) -> tanh-approximate GELU -> Linear(hidden, input).
Each model has cached and invalidated-parameter conditions in all three backend
modes. Invalidation refreshes the actual finite/pack caches without changing
weights; **this is not an optimizer update or complete training benchmark**.

The selected native run has 90 conditions per process, two AB/BA rounds: **360
measured condition-runs**, plus 180 recorded full-grid preconditioning conditions.
All outputs pass independently constructed f64 references at atol=rtol=1e-4;
layout-only operations also require exact float32 reference bits. A warmed
allocation sample is recorded separately for every condition.

Geometric means of baseline/candidate median times; greater than one favors the
candidate:

| Boundary | Ratio |
| --- | ---: |
| Native ordinary weight pack | 2.610x |
| Native Tensor transpose | 2.992x |
| Native column-major transpose pack | 2.009x |
| Auto Linear, cache invalidation + forward | 1.780x |
| Auto MLP, cache invalidation + forward | 1.390x |
| Auto Linear, cached forward | 1.201x |
| Auto MLP, cached forward | 1.067x |
| WASM/Node weight pack | 2.099x |
| WASM/Node Tensor transpose | 1.911x |

All 36 native preparation comparisons are favorable. Column-major transpose-pack
allocations fall **5 -> 2** throughout the grid, and allocated bytes approximately
halve (768x3072: 18,874,488 -> 9,437,224). These are allocation requests/bytes,
not process RSS or peak live memory.

**Cached forwards are mixed, not a uniform win.** Explicit-Faer cached Linear/MLP
ratios are 0.930/0.897; explicit CPU-SIMD ratios are 0.983/1.000. Auto has individual
regressions too. Earlier repeats of the same first candidate also vary in both
directions. Do not attribute these variations to one cause or select only Auto's
favorable aggregate. The repeatable target of this change is preparation and
cache refresh, not a universal faster warmed matmul.

WASM uses real CPU `AutogradTensor.prepackRhs()`/`transpose()` calls with gradients
disabled. Six matrix shapes give **48 measured condition-runs** plus 24 recorded
preconditioning conditions. Every transpose and an untimed packed matmul check
match exact dyadic references; output hashes agree across module/round pairs.
Preparation and free are timed, while input creation, output export and the
correctness matmul are not. All 24 selected comparisons are favorable. The initial
blocked-only implementation regressed at 64x64; the simple small-buffer path
changes its pack ratios from 0.656/0.736 to 1.071/1.067, and transpose from
0.962/0.817 to 1.109/1.086. **Node CPU WASM is not browser or WebGPU performance.**

## PyTorch Comparison

`torch.json` retains 12 eager CPU `nn.Linear`/`nn.GELU(approximate="tanh")`
conditions on PyTorch 2.12.1, with the same float32 input/weight/bias formulas,
four intra-op threads and one inter-op thread. All pass an independently written
NumPy f64 reference. Torch weights are contiguous `(output, input)` tensors;
Rust's weights are stored `(input, output)` and its cached pack is column-major.
Input/model construction is excluded; output creation/destruction is included.

The geometric means of **Torch time / SpiralTorch time** are 0.725 (Auto cached
Linear) and 0.961 (Auto cached MLP). Only 4/12 shape/round comparisons favor
SpiralTorch in each group. Individual ratios range 0.193-9.002 / 0.399-6.240.
Tiny-case Python entry overhead contributes to these aggregates: this is not
kernel parity or an overall victory over PyTorch.

One Torch run is reused for both Rust rounds and cache states. Rust's explicit
cache invalidation and per-call validation have no Torch equivalent, so those
additional ratios in `torch-comparison.json` are not isolated kernel comparisons.
All conditions, intervals and previous Torch measurements remain published.

## Validation And Replay

The selected source passes 486 Tensor CPU tests; all 715 NN library tests; a new
Linear forward/gradient/cache-refresh integration test across both weight layouts,
three backend modes and two parameter revisions; scoped strict native/WASM Clippy;
six actual native Python-extension autograd tests; 20 WASM forward/backward cases
and eight numerical-boundary cases; seven WGPU-feature tests including a mandatory
real-GPU probe; and eight prepacked Tensor tests in a WGPU-enabled build. The
shared transpose test checks 99 shape combinations with NaN payloads, signed zero,
infinities, subnormals and output guards. This is not all-workspace/all-device CI.

`preflight/` retains both blocked-only native measurements, the initial WASM
regression, earlier Torch data and all compilation/validation logs. The first
candidate rebuilt byte-identically before the small-buffer correction. The final
worker is a distinct build at the selected commit, with matching-root compilation
and source hashes. Native binaries, Python extensions and WASM products remain
local; paths and hashes are in `provenance.json`. No extra deleted source or cache
was restored in this follow-up.

```bash
python3 -B -I verify.py
python3 -B -I test_verify.py
```

These check archive bytes, fixed condition grids, aggregations, source/product
bindings and recorded gates. **They do not perform fresh numerical execution.**

For fresh runs, use clean checkouts at the two pinned revisions. The new example
is archived as `cpu_nn_layout.rs`; put that same example in the baseline's
`crates/st-bench/examples/` before building. Freeze each worker immediately. The
`build.py` driver forces matching-root compilation and records hashes to prevent
reuse of another checkout's stale Cargo products. Then run into new directories:

```bash
python3 -B -I build.py /absolute/baseline /absolute/target /absolute/new-baseline
python3 -B -I build.py /absolute/candidate /absolute/target /absolute/new-candidate
python3 -B -I measure.py /absolute/new-native \
  /absolute/new-baseline/worker /absolute/new-candidate/worker
python3 -B -I check.py /absolute/candidate /absolute/target /absolute/new-check
python3 -B -I measure.py /absolute/new-wasm \
  /absolute/baseline/spiraltorch_wasm.js /absolute/new-check/wasm/spiraltorch_wasm.js \
  --wasm-harness /absolute/cpu_layout_bench.cjs
python3 -B -I -S torch_compare.py /absolute/torch/site-packages
```

Build the baseline WASM module using the same recorded release/bindgen commands
as the candidate. Adapt the machine-local Python-loader and bindgen paths in
`check.py` (the loader is archived as `python-client.py`). Do not overlap builds
with timing, change the fixture grid, or reinterpret these forwards as FT quality.
