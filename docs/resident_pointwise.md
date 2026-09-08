# Prepared Resident Pointwise Chains

The [N-D tensor bridge](resident_nd_tensor.md) removes intermediate host
round trips. `st_tensor::NdPointwisePlan` also removes per-operation queue
submission and, in fused mode, intermediate tensor allocations for a checked,
shape-preserving chain.

This is explicit lowering, not an automatic optimizer for arbitrary
`Module::forward` graphs. It uses the existing `WgpuTensorDevice` and returns an
ordinary immutable `NdTensor`, ready for `set_input_tensor`,
`upload_batch_tensors`, or further resident operations.

## Rust Usage

Enable `st-tensor/wgpu_dense` for WGPU. The same API also evaluates host tensors
in CPU-only builds, using the shared finite-value contract.

```rust
use st_tensor::{
    ElementwiseOp, NdPointwisePlan, NdTensor, PointwiseChain, PointwiseExecution,
    PointwiseStep,
};

fn prepare(x: &NdTensor, bias: &NdTensor, gain: &NdTensor)
    -> Result<NdPointwisePlan, st_tensor::NdTensorError>
{
    NdPointwisePlan::new(
        PointwiseChain::new(3, vec![
            PointwiseStep { op: ElementwiseOp::Add, rhs: Some(1) },
            PointwiseStep { op: ElementwiseOp::Multiply, rhs: Some(2) },
            PointwiseStep { op: ElementwiseOp::Gelu, rhs: None },
        ])?,
        &[x, bias, gain],
    )
}

fn evaluate(plan: &NdPointwisePlan, x: &NdTensor, bias: &NdTensor, gain: &NdTensor)
    -> Result<NdTensor, st_tensor::NdTensorError>
{
    plan.run(&[x, bias, gain], PointwiseExecution::Fused)
}
```

Input zero starts the accumulator. Each binary step's `rhs` names an original
input slot, not a previous intermediate; `Some(0)` adds/multiplies the original
input, allowing residual chains. Inputs can change between calls without
recompiling, but their shapes, strides, offsets, and actual device/queue must
match preparation. The plan never retains input values.

## Scheduling And Failure Semantics

- `Sequential`: one submission and checked storage result per step.
- `Batched`: the same kernels and intermediate storage, submitted together.
- `Fused`: one generated pointwise dispatch, one output allocation and guard;
  original inputs stay resident and only their small validity flags are copied.
- CPU plans evaluate the checked steps for all three modes; they do not silently
  choose a GPU, and GPU plans never fall back to CPU.

Compilation and metadata preparation happen once. Each call owns fresh output
storage. Outputs are submitted before being returned, so there is no exposed
unsubmitted tensor and no alias invalidated by the next execution.

The initial bound is 1..=256 steps and 1..=16 input slots, further limited by the
device's storage binding budget (inputs + four bindings). All input slots must
be used. RHS broadcasting is allowed only **to the first input's existing
shape**. A scalar-to-vector or scalar-to-empty shape expansion is rejected at
preparation, rather than changing the eager validation domain.

Strided, narrowed, scalar, broadcast and empty inputs are supported within that
fixed domain. Empty outputs still inherit prior failures. Every evaluated
intermediate keeps the same finite checks, including GELU's square/cubic/inner
checks before saturation: a later ReLU cannot hide overflow. Neither this
pointwise API nor its NN bridge adds general N-D autograd.

## Verification And Benchmarking

The native/browser `resident_nd_tensor` fixture includes each execution mode
at zero, one and twenty preprocessing iterations, followed by the existing
Linear/GELU graph and eight resident SGD steps. Dedicated guards cover masked
overflow, empty domains, output reuse, signed zero, layout and device rejection.

The source-bound worker/controller support a four-lane diagnostic:

```bash
cargo build --locked --release -p st-nn --no-default-features --features wgpu --example resident_nd_bench
python -I tools/bench_nd_tensor_vs_torch.py --pointwise \
  --binary /absolute/frozen/resident_nd_bench --torch-device mps \
  --output /absolute/new-pointwise-bench.json
python -I tools/validate_nd_tensor_bench.py /absolute/new-pointwise-bench.json
```

Commit/freeze the source and verify the embedded binary identity before timing.
All lanes start resident, include view setup and terminal host readback, and
exclude upload, compilation and JSON transport. Two warmups and eight retained
samples rotate/reverse lane order; every raw interval and one full capture per
lane are retained. Rust checks intermediate finiteness; eager Torch is not
given matching guards. This is not a general speedup, browser timing, CUDA,
`torch.compile`, or training-throughput claim.
