# Ordered Tensor Means

`st_tensor::mean_tensors_scaled(&partials, scale)` is the portable numerical
reducer used by GoldenRetriever. It is also available in source-built Python
as `spiraltorch.mean_tensors_scaled` and in WASM as `TensorMeanBatch.meanScaled`.
The native Golden worker pool remains in `st-core`/`st-nn`; browser clients do
not import that thread pool to aggregate worker gradients or model updates.

## Numerical Contract

This is a signed-vector arithmetic mean, not the probability/KL
`z_space_barycenter` solver. Each output coordinate sums finite f32 inputs in
input order, starting at positive zero in f64, divides by the number of inputs,
multiplies by the f32 scale promoted to f64, then rounds to f32. Non-finite
inputs, scales, or final outputs are errors. Scale zero does not bypass input
validation. Empty input lists are invalid; matching zero-volume tensors work.

The reducer supports mixed row-major/column-major inputs without converting
them to full row-major buffers. Other supported layouts retain Tensor's existing
conversion path. It uses at most 1024 f64 scratch values (8 KiB), independent
of the output size, plus output storage and input handles. Row-major batches
use linear blocks; mixed batches use 16x64 tiles to keep both source orders local.
Inputs remain immutable and the result is a row-major value Tensor, not an
autograd graph node. Execution is native CPU or WASM CPU, never a GPU fallback.

Golden still owns rank-plan validation and the guard. It passes `1.0 + guard`
computed in f32 to this reducer. Its planning receipt and independent output
validator remain unchanged; `plan_executed` is still false. This work does not
turn the guard's rank plan into an executed rank kernel.

## Use

```rust
use st_tensor::{mean_tensors_scaled, Tensor};

let partials = [
    Tensor::from_vec(1, 2, vec![1.0, -2.0])?,
    Tensor::from_vec(1, 2, vec![3.0, 4.0])?,
];
let mean = mean_tensors_scaled(&partials, 1.25)?;
assert_eq!(mean.data(), &[2.5, 1.25]);
```

See the [Python examples](../../bindings/st-py/README.md) and
[WASM example](../../bindings/st-wasm/README.md#portable-tensor-means-source-builds).
WASM batches load partial-major flattened Float32Array data once, then produce
independent output arrays on repeated calls. Call `free()` when finished.

## Compare

`crates/st-tensor/examples/mean_tensors_bench.rs` alternates the shared reducer
with the previous Golden reduction body, validates all output bits, and records
three warmups plus 16 samples for 54 layout/shape/count/seed cases. This measures
the reduction body, not the entire Golden worker runtime or planner.

Build from clean source, record its commit/build flags, copy the executable to
a new immutable path, and hash it before timing. Do not benchmark a mutable
shared Cargo target while another build can replace it. Keep builds and other
owned benchmarks out of the measurement window.

```sh
cargo build -p st-tensor --no-default-features --example mean_tensors_bench --release
install -m 555 target/release/examples/mean_tensors_bench NEW_BINARY
shasum -a 256 NEW_BINARY
NEW_BINARY > native.json
python -I tools/bench_tensor_mean.py --native-results native.json --output python.json
node tools/test_resident_browser.cjs MODULE_DIR CHROME NEW_OUTPUT '' '' '' '' tensor-mean
node bindings/st-wasm/tests/tensor_mean_types.cjs MODULE_DIR/spiraltorch_wasm.js
```

The Python comparison uses input-order f64 Torch operations, not reassociated
`stack.mean` or an f32 reduction. Inputs are prepared outside timing, outputs
are new on each call, and Torch runs with one CPU thread. Torch input validation
is outside timing; SpiralTorch retains validation inside its timed call.
`--device cuda --torch-only` measures PyTorch CUDA with synchronization and no
transfer time. It is not a SpiralTorch CUDA kernel. Inspect Furnace availability
before launching; the runner also rejects foreign GPU processes at point checks.

The [2026-09-07 results](../../benchmarks/results/2026-09-07-golden-portable-reduction/README.md)
retain failed candidates, browser fixture failures, native/Python/WASM/CUDA
numerical checks, and preexisting strict-lint failures. These bounded component
measurements do not establish model quality, training speed, or general framework
superiority.
