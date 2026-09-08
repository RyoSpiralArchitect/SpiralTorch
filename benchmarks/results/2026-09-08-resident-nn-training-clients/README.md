# Public NN Resident Training Clients

Measured implementation: `5c38855aa527c45f65ca5ea5674d2ea8ba98eeaf`, built from
a clean checkout with Rust 1.98.0. The installed Python extension bytes were
matched to both GPU and CPU-only source-built wheels. Browser reports hash the
generated JS/WASM assets and the exact Python fixture served to them. This is
source/build-log attribution, not embedded Git attestation or reproducible-build
proof. These are dev-profile packages, not a new PyPI release.

## What Changed

The existing `Sequential(Linear(4,7) -> Gelu -> Linear(7,3))` can now use
`plan.compile_training_wgpu()` in Python and `plan.compileTrainingWebGpu()` in
JavaScript. Both expose the Rust training implementation from
[the preceding study](../2026-09-08-resident-nn-training/README.md): mean-MSE,
VJP and transactional plain SGD, with weights and intermediate operations on
the device. Neither binding implements optimizer or derivative math.

Owning single-use snapshots expose loss, predictions, input/parameter gradients
and post-update weights. Rust validates dense parameter export against the source
graph. Invalid batches do not partially replace the current batch; numerical
rejections preserve every layer and retain structured error fields even when
read after a later valid step. Zero-rate probes now preserve signed-zero
parameter bits while still rejecting nonfinite intermediates/gradients.

## Results

- Python native: Apple M4 / Metal. Browser: Chrome 152.0.7977.77 / WebGPU.
  The Rust browser adapter report is anonymous (`Other`, empty name), so this
  is not an exact cross-client physical-device identity attestation.
- Each client runs 18 VJP cases: shapes `[4]`, `[3,4]`, `[2,47,4]`, two matrix
  kernels and three accumulation policies. Python and browser states are
  independently replayed by PyTorch 2.12.1 CPU/MPS with CPU fallback disabled.
- The maximum absolute difference across all VJP and learning comparisons is
  `1.1920928955078125e-7`, within fixed `atol=1e-5, rtol=1e-4`. Comparisons include
  loss, predictions, input gradients, every parameter gradient and updated weights.
- Three seeds run 128 Python updates. At update 64, each exported plan is also
  transferred to a fresh browser workspace for 64 additional updates. The final
  browser states numerically match uninterrupted Python states (max difference
  `0` in this fixture). Reimporting browser-trained plans for Python native
  resident inference also has max difference `0` against browser predictions.
- This is **weight-only handoff**, not optimizer checkpoint/resume. New workspaces
  start with fresh counters and require a batch upload. Replicated clients are
  not independent extra seeds.

| Seed | Initial MSE | After 128 Updates |
| --- | ---: | ---: |
| 17 | 0.1421881318 | 0.0137947779 |
| 29 | 0.1353686154 | 0.0056838538 |
| 43 | 0.0985021293 | 0.0070454977 |

All 25 frozen-source verification commands passed. This includes 749 st-nn
tests, 102 backend tests, 13 GPU Python NN/client tests, 3 existing Python matmul
tests, and 37 smoke tests on each wheel. CPU-only NN checks pass five surface
tests and intentionally skip eight GPU tests. Four replay-admission tests pass.
Generated/shipped TypeScript contracts agree. Browser training exercises 115
rejected input/state cases, including six nonfinite numerical cases; existing
browser inference and CPU-only transport tests also pass. WGPU backend strict
Clippy and workspace formatting pass.

## Failures And Limits

`zero-probe-red.log` preserves the real pre-fix `-0.0 -> +0.0` mutation under a
zero-rate derivative probe. The shader now skips only the parameter commit for
zero rate; the full finite-validation path remains active. The retained failing
log comes from the development tree, not the frozen final build.

Expanded strict Clippy checks were also tried: Python has 7 diagnostics, WASM
19 and st-nn 22, all in files unchanged from the parent commit. They remain
failures, not waived passes or separately reproduced baseline results. They are
outside the existing strict WGPU backend gate and are preserved in the archive.

These are small synthetic correctness/learning fixtures, **not training-speed,
LLM quality, or geometric-ablation results**. GPU contention is unmeasured;
CUDA/Furnace was not run. Ordinary `pure::Tensor` and arbitrary NN operators do
not automatically become device-resident; the supported explicit graph remains
Linear/GELU with contiguous N-D leading-axis metadata, mean-MSE and plain SGD.

## Evidence And Replay

`summary.json` records product hashes, source identity and compact outcomes.
`raw-logs.tar.xz` retains raw Python/browser/PyTorch states, final build/test
commands, dev results, failures and collection scripts. Wheels/WASM packages
remain local; the archive does not contain binaries or model downloads.
`SHA256SUMS` binds this README, summary and archive.

After extracting into a fresh directory, run
`tools/validate_resident_training_clients_vs_torch.py` with the retained
`python-training-final.json` and `browser-training-final.json`, specifying an
exclusive new output path. That tool needs PyTorch, not SpiralTorch. Use
`--devices cpu` without MPS. Full public-client build/run instructions are in
the [training guide](../../../docs/resident_nn_training.md#public-client-roundtrip).
