# N-D Tensors On Shared Rust Storage

`st_tensor::NdTensor` joins the existing host `Tensor` snapshot owner, shared
`NdLayout`, and native/browser WGPU storage. It is an explicit migration path,
not a replacement device router or an automatic change to ordinary
`Module::forward`. The same resident data can now enter and leave an existing
`InferencePlan` without intermediate CPU tensor readback.

```text
Tensor / NdTensor -> explicit upload -> immutable GPU storage
                                       |
                reshape / permute / narrow / broadcast views
                                       |
                         add / mul / ReLU / GELU
                                       |
                  existing NN plan: fused Linear + GELU
                                       |
                  more N-D operations or resident SGD
                                       |
                            explicit terminal read
```

## Rust Use

Enable `st-tensor/wgpu_dense` (or `wgpu`) and `st-nn/wgpu`. Reuse one
`WgpuTensorDevice` on the **existing** runtime so that elementwise operations
share their compiled pipeline. CPU-only builds retain host N-D operations.

```rust
use st_backend_wgpu::runtime;
use st_nn::{layers::Gelu, resident::InferencePlan, Linear, Sequential};
use st_tensor::{NdLayout, NdTensor, Tensor, WgpuTensorDevice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (runtime, _) = runtime::ensure_default_runtime_blocking("my.nd.nn")?;
    let device = WgpuTensorDevice::new(runtime.clone())?;
    let x = Tensor::from_vec(6, 4, vec![0.25; 24])?
        .into_nd()?.reshape(&[2, 3, 4])?.to_wgpu(&device)?;
    let bias = NdTensor::from_vec(&[4], vec![0.1; 4])?.to_wgpu(&device)?;
    let input = x.permute(&[1, 0, 2])?.narrow(0, 1, 2)?.add(&bias)?.gelu()?;

    let mut model = Sequential::new();
    model.push(Linear::new("up", 4, 7)?);
    model.push(Gelu::new());
    model.push(Linear::new("down", 7, 3)?);
    let plan = InferencePlan::from_module(&model, NdLayout::contiguous(input.shape())?)?;
    let mut nn = plan.compile_wgpu(runtime.clone())?;
    nn.set_input_tensor(input.as_wgpu().unwrap())?;
    nn.dispatch()?;
    let output = NdTensor::from_wgpu(nn.tensor_snapshot(&device)?).relu()?;
    assert_eq!(output.shape(), &[2, 2, 3]);
    let values = output.read_values()?; // first result transfer to the host
    assert_eq!(values.len(), 12);

    let target = NdTensor::from_vec(&[2, 2, 3], vec![0.1; 12])?.to_wgpu(&device)?;
    let mut training = plan.compile_training_wgpu(runtime)?;
    training.upload_batch_tensors(input.as_wgpu().unwrap(), target.as_wgpu().unwrap())?;
    training.step(0.01)?;
    let prediction = NdTensor::from_wgpu(training.prediction_tensor(&device)?);
    let _pre_update_values = prediction.read_values()?;
    Ok(())
}
```

On WASM, request the runtime asynchronously and use `read_values_async().await`.
The browser fixture compiles this same Rust core; it does not reimplement the
layout, activation, derivative, optimizer, or rejection logic in JavaScript.

## Ownership And Numerical Contract

- Host `Tensor::into_nd()` reuses uniquely owned native storage, and already
  protected snapshots share storage. `to_nd()` freezes a borrowed tensor.
  Mutable/foreign aliases, including preexisting writable DLPack exports, use
  the existing `Tensor::into_snapshot` isolation contract rather than an unsafe
  zero-copy shortcut. Unsupported tiled/Chimera host layouts are rejected.
- Scalars, empty axes, right-aligned broadcasting and positive-stride views
  preserve logical shape. `reshape` requires a contiguous view; explicit
  `contiguous()` packs a strided view on its current device. Views share storage;
  arithmetic creates an immutable result. Broadcast views are not writable.
- Transfers and final reads are explicit. Mixed CPU/GPU operands and different
  device/queue handles are rejected rather than silently copied or routed.
- NN input bridges copy **on the GPU** into mutable workspace buffers. Strided
  inputs may first need a GPU packing operation. NN output bridges freeze a
  GPU-owned result and its guards before workspace reuse. These bridges avoid
  CPU round trips; they are **not zero-copy**.
- Host arithmetic rejects non-finite inputs/intermediates/results immediately.
  GPU arithmetic propagates a deferred validity flag. An overflow followed by
  ReLU, or even an empty view, cannot erase the failure. GELU checks its
  intermediates before applying the shared clamped-tanh saturation rule.
- Resident training incorporates upstream input/target failures into its
  existing all-layer finite decision. `0x80000000` denotes an upstream tensor
  failure: input enters forward stage zero, target enters the loss stage.
  A rejected attempt changes **no weights**, including when subsequent values
  look finite. A valid host upload clears the previous source association.
- Shape/device errors leave the prior NN batch and generation unchanged.
  `prediction_tensor()` contains the pre-update prediction but is guarded by
  the **whole step's** acceptance. Generic tensor outputs aggregate failures;
  use the existing NN snapshots for detailed stage diagnostics.

This first slice does **not** add general N-D autograd, batched `NdTensor`
matmul/reductions, negative strides, mutable broadcast views, or arbitrary
`ModuleTrainer` graph lowering. In the fixture, preprocessing is outside the
gradient tape; NN VJPs start at the prepared input. The legacy 2-D `Tensor` API
is unchanged. New public Python/JavaScript `NdTensor` classes and a PyPI release
are separate work, not implied by successful binding compilation.

## Reproduce

Run checks with runtime tests explicitly enabled; a software adapter is not
accepted as GPU evidence:

```bash
cargo test --locked -p st-kernel-contracts
cargo test --locked -p st-tensor --no-default-features --features cpu --lib nd::tests
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-backend-wgpu resident_tensor -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-nn --no-default-features --features wgpu --test resident_nd_tensor -- --test-threads=1
cargo run --locked --release -p st-nn --no-default-features --features wgpu --example resident_nd_tensor > native-nd.json
cargo build --locked --release -p st-nn --no-default-features --features wgpu --target wasm32-unknown-unknown --example resident_nd_tensor_browser
wasm-bindgen --target web --out-dir nd-pkg --out-name spiraltorch_wasm target/wasm32-unknown-unknown/release/examples/resident_nd_tensor_browser.wasm
node tools/test_resident_browser.cjs nd-pkg /path/to/chrome browser-nd.json '' '' '' '' nd-tensor
python tools/validate_nd_tensor_vs_torch.py --reports native-nd.json browser-nd.json --devices cpu mps --output nd-replay.json
cargo build --locked --release -p st-nn --no-default-features --features wgpu --example resident_nd_bench
python tools/bench_nd_tensor_vs_torch.py --binary target/release/examples/resident_nd_bench --torch-device mps --output nd-bench.json
```

Use a `wasm-bindgen` CLI version matching the lockfile, a WebGPU-enabled browser,
and the existing Playwright runner dependency. Choose fresh report paths;
reports are not overwritten. Adapt artifact paths if using `CARGO_TARGET_DIR`.

The shared fixture covers three shapes/seeds, including a strided NN input,
60 chained elementwise operations, inference output reuse, eight SGD steps,
transactional rejection, device mismatch, and snapshot lifetime. Independent
Torch replay compares preprocessing, predictions, loss, VJPs and trained weights.

The separate benchmark alternates execution order for six fixed recipes, keeps
all eight post-warmup intervals, excludes upload/pipeline construction/JSON, and
includes terminal host readback for both implementations. Rust checks every
intermediate's validity; eager Torch does not, so the cost contracts differ.
Keep failures and slower cases. One device and this bounded workload do not
establish a general performance advantage or a training-quality improvement.

The [first measured baseline](../benchmarks/results/2026-09-09-resident-nd-tensor/README.md)
passes native/browser numeric and SGD replay, but this unfused 60-operation
elementwise chain is still 3.4-5.3 times slower than eager Torch MPS on the
tested M4. Full compressed records and a no-selection revalidator are included;
GPU residency is not itself a measured speedup.
