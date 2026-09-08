# Resident NN Training: Native, Browser, And PyTorch

Measured implementation: `2321c6fd518dcedbc228b8323e4de0ccb5f0f90c`
(`c20ba4913e450a147619d3a2cfc425976c6c8d4a` tree), built from a clean checkout.
The preceding `72f35450` commit corrects Linear/LoRA VJP reduction. This directory
is a later evidence-only commit; development failures are retained separately
from the source-bound final runs.

## What Ran

The existing Rust `Sequential(Linear(4,7) -> Gelu -> Linear(7,3))` lowers through
`InferencePlan` into persistent GPU forward, mean-MSE, backward and plain SGD.
The native and browser fixtures compile the same Rust fixture and backend
shaders. Neither JavaScript nor Python implements the production training rules.

- Native: Apple M4 / Metal, Rust 1.98.0, release build.
- Browser: Chrome 152.0.7977.77 / WebGPU. A separate browser adapter probe reports
  Apple, Metal-3, non-fallback. Rust's browser adapter name is empty, so this is
  not an exact cross-client device-identity attestation.
- Independent reference: PyTorch 2.12.1, CPU and MPS, f32, tanh-approximate GELU,
  mean-MSE, plain SGD, MPS CPU fallback disabled.
- VJP: shapes `[4]`, `[3,4]`, `[2,47,4]`, scalar/register-2x2 kernels, and
  sequential/tiled/compensated accumulation. All 18 cases pass on each target.
  Non-square matrices and an MSE partial group are included.
- Learning: three seeds, `[2,16,4]`, 128 nonzero-rate updates at 0.2, plus initial
  and final zero-rate derivative probes. The update loop has no intermediate
  host readbacks or parameter reuploads; the small rate uniform is written per
  step. Transfer counts describe the code path, not hardware counters.
- Final weights are exported into a new immutable plan and verified through
  ordinary resident inference. The original plan remains unchanged.

## Results

Native and browser MSE agree for these fixtures:

| Seed | Initial MSE | After 128 Updates | Reduction |
| --- | ---: | ---: | ---: |
| 17 | 0.14218813 | 0.013794778 | 90.30% |
| 29 | 0.13536862 | 0.005683854 | 95.80% |
| 43 | 0.09850213 | 0.007045498 | 92.85% |

The PyTorch comparison includes loss, predictions, input gradients, every
parameter gradient and post-update parameters, not just a falling loss curve.
Across both execution targets and both PyTorch devices, the largest absolute
difference is `1.1920929e-7`. Against Rust's CPU reference, the largest retained
learning-state difference is `1.4901161e-7`. Tolerances were fixed at
`atol=1e-5, rtol=1e-4`. Replaying a seed through multiple clients is not a new seed.

Guard tests prove bitwise all-layer rollback when a later layer's SGD candidate
overflows, recovery before reading the earlier failed snapshot, and owned
snapshots across reuse/drop. Input-gradient overflow, MSE square overflow,
four GELU intermediate-overflow cases, four finite GELU saturation cases, ten
host validation cases and repeated-batch factors 1/2/5 also pass.

Related validation: 739 st-nn unit tests, 3 resident inference tests, the resident
training fixture and 4 finite-difference/reduction tests pass (747 total).
Backend tests pass (101 unit + 1 shader integration). Native and WASM backend
Clippy pass with warnings denied; workspace formatting passes. Existing browser
inference also passes six cases and four overflow guards.

### Inference Diagnostic

The existing source-bound inference benchmark was rerun: nine cases, three
seeds, two warmups and twelve retained samples per case. Native eager/resident
controls alternate; PyTorch is measured afterward. All numerical comparisons
pass. The six deep cases are **3.65-4.55x faster than the legacy SpiralTorch
host-readback path**, but **8/9 cases remain slower than eager PyTorch MPS**.
Every case and raw timing is retained, including the losses.

This is one diagnostic run with unknown macOS GPU contention, not a matched
pre-change performance study. It does not prove no performance regression,
fastest-PyTorch parity, or training throughput superiority.

## Failures And Boundaries

Development `live-training-v2.log` and `live-training-v3.log` retain the real
finite-input GPU GELU failure (`stage 0, mask 0x7f61`). Large finite `tanh`
arguments produced NaN on the native GPU. The checked forward and backward now
bound only the already-saturated f32 region; earlier square/cubic guards still
reject real overflow. Initial fixture type/mutability errors and native/WASM
Clippy failures are also retained, not erased by successful reruns. These logs
come from a changing development tree, not the frozen final build.

The VJP correction intentionally changes multirow Linear/LoRA parameter
gradients: the old extra row division is gone. Old learning rates may need
retuning. See the [training contract](../../../docs/resident_nn_training.md).

This is bounded synthetic learning, not an LLM, geometric-training ablation,
or general N-D autograd result. Public Python/JS training clients, CUDA training,
optimizer state/resume, arbitrary operators and training speed comparisons are
not part of this study. Step counters count submissions, not accepted updates;
capture each attempt whose individual validation history must be retained.

## Evidence And Replay

`summary.json` contains all compact results, source identity and product hashes.
`raw-logs.tar.xz` contains final native/browser reports, independent PyTorch
states, the complete inference timing report, development failures and build/test
logs. Generated binaries/WASM packages are retained locally, not in the archive.
`SHA256SUMS` binds the public summary and raw archive.

After extracting the archive into a fresh directory, replay the final native
and browser JSON files with `tools/validate_resident_training_vs_torch.py`, using
an exclusive new output path. Full build/browser commands are in the training
contract. On machines without MPS, select `--devices cpu`. No installed
SpiralTorch Python wheel is required for this independent reference replay.
