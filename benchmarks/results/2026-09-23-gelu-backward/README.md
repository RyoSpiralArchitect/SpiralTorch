# GELU Backward Liveness, 2026-09-23

Measured implementation: `0f7a03f49f3808344a1b1a1c15776be83b4fe3fb`.
Base: `62dced674158234726cfb2521c09d378e8ec3b6f`.

**Ordinary Tensor GELU backward now computes only its live gradient output.**
It no longer computes, allocates or reads discarded residual/bias outputs.
The full three-output helper remains available and batches terminal readback.
Both paths use backend-owned, filesystem-free kernels with the same derivative
as resident execution. CPU routing, finite guards, layout conversion and
fallback policy remain at the existing Tensor boundary.

## Results

Three runtime-order-rotated serial rounds, six shapes, two output contracts,
bursts 1/4, nine measured blocks after three warm-ups: **5,184 intervals**.
Each WGPU route occupies each execution position exactly three times per
condition. All runs use the same clean implementation commit.

| Live outputs | Native combined ratio | Browser combined ratio |
| --- | ---: | ---: |
| Gradient only | 3.044x | 2.755x |
| Gradient, residual and bias | 1.925x | 2.121x |

Ratios are geometric means of twelve cell-level median paired legacy/candidate
ratios, not ratios of global medians. All twelve combined cells per
contract/runtime exceed 1 in this run. The complete conditions, intervals,
slower factors and errors are in [results.json](results.json).

The one-output comparison separates observation from computation: selecting
only the gradient read from the identical legacy fused shader gives 2.081x
native / 2.507x browser; replacing that fused work with the plain derivative
gives a further 1.444x / 1.070x. The browser's latter factor exceeds 1 in only
five of twelve cells, with a coarse application clock. For three outputs,
the shared derivative alone gives 0.996x / 0.995x; batching its observation
gives 1.937x / 2.136x. Shared math alone is not a speedup claim.

The prepared interval includes encoding/bindings, per-operation submission,
per-operation GPU residual reset when needed, and terminal owning CPU
completion. Fixed uploads, allocation, compilation, checks and JSON are outside.
Bursts repeat the same operation and observe its last result, not a training
trajectory. Separate and batched reads use the same Rust snapshot decoder;
this is not a byte-identical benchmark of the old native read_buffer helper.
Strict high-level Tensor tests establish routing/correctness, not a measured
whole-Tensor-API multiplier.

Torch 2.12.1 uses actual ATen tanh-GELU backward with preallocated packed
outputs and one owning CPU copy, CPU/MPS, four intra-op/one inter-op threads,
no fallback and no torch.compile. At 128x1025, gradient-only/burst=1, median
intervals are 0.311 ms native plain, 0.400 ms browser plain, 0.213 ms Torch CPU
and 0.783 ms Torch MPS. CPU remains faster in some cells. These are shared-M4
application intervals, not GPU timestamps, end-to-end training or universal
SpiralTorch-over-PyTorch results.

## Correctness And Repaired Paths

- Independent scalar f64 and Torch CPU f64 references pass the frozen
  abs=2e-6 + rel=1e-5*abs(reference) gate, with the bias sum's absolute allowance
  scaled only by row count. Maximum absolute error is 3.0865490564e-6 and maximum
  scaled error is 0.365175425, below the limit of 1. No bitwise-equivalence claim.
- Twenty-four native/browser fixtures cover tails, padded rows, finite
  extremes and residual_seed + gZ. The high-level GPU tests additionally
  cover RowMajor/ColMajor/Chimera, empty tensors, nonfinite input/seed,
  overflow results and pre-device shape rejection.
- Twenty-three clean-source stages pass: format, four protocol and two archive
  tests, backend/Tensor native/WASM strict lint, real-GPU backend tests
  (192 library + one WGSL + two example tests), Tensor tests (16 autograd +
  three GELU), twelve NN layout tests, 1,001 core tests, builds and all nine
  runtime rounds. Test logs and commands are included, not inferred from a
  successful build.

The original public filesystem loader could not compile an override-sized
workgroup array on WGPU 0.20. Its handled error (process exit zero, not success)
is retained in [legacy-loader-result.json](legacy-loader-result.json) and
[legacy-loader-stderr.log](legacy-loader-stderr.log). The exact same probe after
repair is in [repaired-loader-result.json](repaired-loader-result.json).
Their source identities/commands are the baseline-probe-run and loader-fixed
attempts in [exploration.json](exploration.json).

Separately, compiling the historical backend shader with only override syntax
repaired produces four nonfinite gradient values among fifteen finite extreme
inputs, in every native/browser round. That diagnostic's exact output strings
are preserved in results.json; it is not a passing correctness gate or a claim
that the old loader ran. The canonical saturated derivative removes that
failure, while the legacy Tensor shader remains the valid timing control.

Plans reject invalid shapes, short strides, storage/index overflow and
unsupported dispatch geometry before Tensor buffers are allocated. The fused
32/16-byte uniform ABI and bias addition order are unchanged. Low-level callers
still own matching uniforms, buffers, nonaliasing, device identity and finite
policy. The plain kernel is contiguous; padded low-level fixtures use the
fused kernel. Batching may retain more staging memory than sequential reads;
per-buffer limits are not a total-memory budget.

## Evidence And Replay

The separate complete 5,184-interval screening and all 25 recorded
pre-publication top-level attempts remain in exploration.json. They are not
pooled with the clean-source study. No failed attempt was deleted; the loader
failure is represented by its explicit result, not a nonzero process code.

Public records contain all conditions, intervals, validations, source hashes
and replay commands. Arrays, executables and generated WASM/JS stay local:
[local-raw-manifest.json](local-raw-manifest.json) lists **548 files /
2,174,642,477 bytes**. Native identifies Apple M4/Metal. Browser reports
BrowserWebGpu, Chrome 153.0.8010.53, with a separate non-fallback Apple adapter
probe that does not attest the Rust device. Browser page, assets and all three
twelve-case streams have recorded hashes.

See [the protocol guide](../../gelu-backward/README.md) for numerical replay.
Verify public archive fixity:

```sh
python3 -I -B benchmarks/gelu-backward/evidence.py verify benchmarks/results/2026-09-23-gelu-backward
```

Add --raw-root and --source-root to rehash local raw evidence, recompute all
summaries and compare the measured source. Verification is not GPU reexecution.
No CUDA/Furnace run, model-quality gain, whole-model multiplier or universal
device/geometry guarantee is included in this milestone.

Post-publication CI outcomes and their resolution are tracked separately in
[CI-FOLLOWUP.md](CI-FOLLOWUP.md); the original local measurements remain intact.
