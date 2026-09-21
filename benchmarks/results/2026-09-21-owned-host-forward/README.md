# Owned Host Forward

This follows PR #2107. Baseline runtime and measurement workers:
`b6dcfb45e8d55fe408b1891a627b9d47e8bca268`; candidate:
`d17fdcdbee71a351f5521fa4e626e9948399b01e`.
The baseline already includes the unchanged chain benchmark. Its first added
regression-test attempt did not compile because `Tensor::ones` does not exist.
Test-only correction `afa83a24d06e4ccf2497ef1102b800935bfec64a` passed the seven
baseline contracts. Both attempts are retained; baseline runtime/harness hashes
match across the correction. No failed attempt is relabeled as passing.

## Change And Ownership

`Module::forward_owned(Tensor)` defaults to the existing borrowed `forward`,
preserving custom modules without requiring overrides. `Sequential` transfers
intermediate ownership to the next layer. Its borrowed public entry still clones
the input handle, protecting the caller's tensor. Backward retains the original
borrowed recomputation and saved-activation behavior, including nested models.

`Gelu::forward_owned` calls `Tensor::try_into_gelu`. It reuses a buffer only when
the tensor is row-major, tracked, privately owned, and both storage Arc levels
are unique. Shared tensors, snapshots, foreign buffers, writable exports (even
after the export handle is dropped) and non-row-major layouts use a fresh output.
Mutation revokes weak content stamps without copying the unique values buffer.
Input validation, error precedence, outlier checks and scalar arithmetic match
`try_gelu`. A failed consumed unique input is discarded; surviving aliases are
not modified. No production unsafe code, new heuristic, backend-policy override,
resident/autograd kernel or dependency change is introduced.

The Python Sequential already owns these Rust modules, so it inherits the host
path without a second Python implementation. Public WASM Sequential remains a
resident WebGPU path; this host optimization is measured via an explicitly
test-only WASM adapter, not presented as a new deployed browser API.

## Measurements

Recorded Apple M4 host, four Rayon threads. Host exclusivity and thermal state
are unknown. Precondition both workers, then baseline/candidate and
candidate/baseline. All conditions and intervals, including unfavorable ones,
are retained. Ratios are geometric means of per-condition median-time ratios,
baseline/candidate; greater than one favors the candidate.

| Borrowed-input GELU chain | Depth 1 | Depth 4 | Depth 16 |
| --- | ---: | ---: | ---: |
| Native, more than 64 elements | 1.003x | 1.052x | 1.079x |
| Node CPU WASM shim, more than 64 elements | 1.000x | 1.075x | 1.098x |
| Native, at most 64 elements | 1.008x | 1.889x | 2.458x |
| Node CPU WASM shim, at most 64 elements | 1.002x | 1.251x | 1.518x |

Both targets favor the candidate in all 18 paired comparisons at each of depths
4 and 16. Depth 1 is a mixed, near-neutral control. Tiny native intervals use only
eight calls, so timer/allocator noise and dispatch cost limit their interpretation.
This repeated-activation chain is an allocation stress control, not a realistic
model or a training-throughput claim.

Native allocation requests change from 3/12/48 at depths 1/4/16 to **3/3/3**.
At 32x3072, requested bytes for depth 16 change from 6,292,736 to 393,296.
These are cumulative allocation requests for one warmed call, not peak RSS.

The unchanged 90-condition grid uses real Linear and Linear-GELU-Linear modules,
Auto/Faer/CpuSimd, cached and parameter-cache-invalidated calls, plus packing and
transpose controls. All 72 paired MLP observations save three allocation requests
and one activation buffer plus 80 bytes. Warmed MLP ratios for Auto/Faer/CpuSimd
are 1.040x/1.015x/1.008x, but only 6/12, 4/12 and 7/12 observations favor the
candidate. Invalidated MLP is 0.991x/1.008x/0.998x. Unchanged Auto Linear is 0.993x
with a 0.730-1.184 range. The allocation reduction is consistent; whole-model
latency is mixed. Cache invalidation is not an optimizer step.

Torch 2.12.1 CPU runs identical float32 fixtures and tanh-GELU chains in eager
no-grad mode, four intra-op and one inter-op thread. For non-tiny inputs,
Torch/candidate is **0.651x/0.715x/0.762x** at depths 1/4/16: Torch remains faster.
Python entry costs, guards and implementation differ; this is not compiled/fused
Torch, an equivalent-dispatch comparison or a fastest-PyTorch claim.

The full cohorts contain 630 measured and 315 preconditioning conditions:
native 108+54, WASM shim 108+54, NN 360+180, Torch 54+27. Chains use three warmups,
15 intervals and eight calls per interval (WASM uses 512 for tiny inputs).
NN uses three warmups, nine intervals and two calls. Allocation/free is timed;
fixture setup and output exports are excluded. Independent float64 checks pass
throughout. Old/new chain output-bit hashes match within each target. The shim
also checks exact old/new forward and backward hashes across 24 combinations of
shape, nesting, depth and RowMajor/ColMajor/Chimera layout, plus errors, empty
shapes and signed zero. Whole-NN checks are float64 tolerance checks, not a
published old/new tensor-bit replay.

## Validation And Replay

The 13 local stages cover formatting; 486 tensor tests; 715 NN unit tests plus
11 ownership/GELU and one Linear integration tests; strict scoped native/WASM
Clippy; actual WASM build and binding generation; four public WASM GELU and 20
matmul cases; Python build plus three GELU/six autograd tests; and 12 opted-in
WGPU integration tests, including nine strict-GPU layout pairs. Exact commands,
source identities and unmodified logs are in `verification`. Hosted CI and
external review status are tracked separately on the pull request.

The standalone shim is measurement-only. It calls real Rust `Sequential` through
wasm-bindgen; JS invocation and owning-result free are timed, exported arrays are
not. No browser or GPU is timed. Its lockfile preserves repository dependency
versions, adding only the shim package. Fresh offline lock generation initially
rejected a yanked pinned GEMM dependency; offline metadata using a copy of the
existing repository lockfile retained that pin. No dependency was upgraded.

`manifest.json` binds the exact published file inventory. Run:

```sh
python3 -B -I verify.py .
python3 -B -I test_verify.py .
```

This verifies bytes, identities, cohorts, allocation observations, numerical
hash agreement and derived ratios. It **does not execute kernels** or establish
hardware performance from hashes. Raw arrays and compiled workers remain at
`provenance.json`'s local root; only their hashes are published.

For numerical/performance replay, use separate clean checkouts of the two runtime
commits. Build `cpu_gelu_chain` and `cpu_nn_layout` with Cargo 1.98.0,
`--locked --release`, four build jobs, and freeze each worker before building the
other version. For the baseline regression tests use the test-only corrected
commit above. Copy `reproduction/wasm-shim` into a temporary directory and point
its repository paths at the selected checkout without changing dependency
versions. Build its wasm32 release with `--locked`, and generate Node bindings
using wasm-bindgen 0.2.104. Freeze the pair in `baseline-build/wasm` or
`candidate-build/wasm`, beside the corresponding native executables.

Place the reproduction drivers and both worker directories below a fresh log
root. Run `measure.py LOG native`, `measure.py LOG wasm`, and
`nn_measure.py LOG/nn LOG/baseline-build/cpu_nn_layout LOG/candidate-build/cpu_nn_layout`.
Run `torch_chain.py` separately with Torch 2.12.1. Do not overlap builds/timing.
The recorded `check.py` preserves original local paths; on another machine point
its extension loader to the included `python-client.py`. Drivers refuse to
overwrite existing cohorts. Another machine/compiler produces a new observation,
not identical timings or an automatic extension of this performance claim.
