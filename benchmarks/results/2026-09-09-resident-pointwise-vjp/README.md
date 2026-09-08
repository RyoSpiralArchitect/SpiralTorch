# Resident Pointwise VJP: Native, Browser, And Torch

Measured source: `8165b5b6b4a075ef101fb4f0d337d3f4032f2025`.
Tree: `a979c3905557ca2119045f55d8d798d03008ab9f`.
Native and WASM binaries were rebuilt from that clean commit, copied to
read-only files, source-checked, and hash-checked again after execution.
The publication commit adds evidence/documentation only.

## What Is Connected

Checked pointwise reverse mode now supports immutable N-D input slots,
original-input residuals, ReLU/tanh-GELU, logical strided inputs/cotangents,
and summed broadcast gradients. The existing resident Linear/GELU trainer's
pre-update input gradient can feed this VJP entirely on GPU.

The NN fixture uses a zero-rate probe, not a separately committed external
gain update. This is **not** whole-graph Scaler/ReLU lowering, general view
adjoints, implicit autograd, or a joint NN/gain optimizer transaction.
See the [API and boundaries](../../../docs/resident_pointwise.md).

## Numeric Evidence

- Native Metal and Chrome WebGPU each pass six VJP recipes plus an NN backward
  bridge: scalar, empty, channel/middle-axis broadcast, 513-row and
  65,537-row reductions. GPU/CPU fixtures agree.
- Independent Torch 2.12.1 CPU/MPS replay: **26 matching cases, 2 explicit
  reference gaps**, maximum absolute difference `2.2649765014648438e-5`.
  Acceptance is `atol=2e-5, rtol=2e-4`, not an absolute-only threshold.
- Both gaps are the same empty tensor's Torch MPS backward internal assertion,
  one per client. Torch CPU handles both. No CPU fallback is enabled.
  Status is `passed_with_reference_gaps`, not a full pass.
- The first strict run failed on this assertion and remains in `initial/`,
  including its frozen fixtures, source/build receipt and exact error.
  Only the explicit `--allow-mps-empty-reference-gap` option admits that
  specific error; nonempty errors and all numeric mismatches still fail.
- The previous forward/NN/eight-step SGD suite also matches Torch in all
  **48** replays, maximum absolute difference `1.4901161193847656e-8`.

Guards cover zero-seed masking of bad forward values, derivative overflow,
late reduction invalidating every gradient, empty inherited failures, foreign
devices/queues, mixed host inputs, strided cotangents, immutable captures, and
a rejected NN step poisoning upstream gradients.

## Bounded Performance

Apple M4, Metal versus eager Torch 2.12.1 MPS, float32, no fallback.
Each lane starts with resident inputs and a prepared non-contiguous logical
view. Eight residual blocks plus ReLU make 33 forward operations; each sample
computes forward and an explicit-cotangent VJP to input, channel gain and
scalar scale. Terminal host output **and all three gradients** are included.
Uploads, view construction, plan compilation and JSON transport are excluded.

| Logical Shape | Seed | Rust Median ms | Torch Median ms | Torch / Rust |
| --- | ---: | ---: | ---: | ---: |
| 15 x 8 x 64 | 17 | 8.2223 | 11.6012 | 1.4110 |
| 15 x 8 x 64 | 29 | 9.3575 | 16.2094 | 1.7322 |
| 15 x 8 x 64 | 43 | 5.2700 | 9.0952 | 1.7259 |
| 31 x 16 x 128 | 17 | 6.7700 | 8.3937 | 1.2398 |
| 31 x 16 x 128 | 29 | 7.4029 | 8.4280 | 1.1385 |
| 31 x 16 x 128 | 43 | 8.1784 | 8.2407 | 1.0076 |

Two warmups and eight retained samples per recipe, alternating lane order.
All raw intervals and full captures are retained, including the near-parity
large case. Maximum captured error is `2.384185791015625e-6`.
The residual recipe avoids relying on a vanishing output alone.

Rust recomputes forward for VJP and checks intermediates; Torch retains an
autograd tape without equivalent finite guards. OS/GPU contention is unknown.
The observed 1.01--1.73x ratio is diagnostic, not a universal win, confidence
interval, NN training-throughput result, browser timing, CUDA, or
`torch.compile` comparison. In particular the largest near-parity case does
not establish a meaningful speedup.

## Reproduce And Audit

`manifest.json` binds executable hashes, source identity and every captured
file's compressed and decoded hash/length. Gzip files contain complete JSON,
not excerpts. `SHA256SUMS` covers all publication files.
`checks/` retains the 17 shared-contract, 109 backend, 438 CPU-tensor and
3 resident-NN tests, strict backend/native/WASM lint checks, example checks,
and initial failures. The initial contract test had an incorrect expectation:
zero through ReLU with a finite MAX seed is zero, not overflow. It was corrected
and a genuinely overflowing positive-input case added. The initial new-loop
lint finding was fixed before the successful runs. Existing vendor warnings
and 22 pre-existing NN-library lint warnings are not claimed fixed.

```bash
gzip -dc vjp-bench.json.gz > /absolute/new-vjp-bench.json
python -I /absolute/SpiralTorch/tools/bench_pointwise_vjp_vs_torch.py \
  --verify /absolute/new-vjp-bench.json
shasum -a 256 -c SHA256SUMS
```

For a fresh GPU run, build and freeze `resident_pointwise_vjp_bench` from
clean source, then use its controller's `--binary`, `--torch-device mps`
and a new `--output`. Do not compare a stale executable with a new checkout.
