# Existing NN Modules, GPU-Resident Training

The same Rust `Sequential(Linear -> Gelu -> Linear)` and `InferencePlan` used by
[resident inference](resident_nn_inference.md) now support an explicit mutable
GPU training workspace. It implements **mean-MSE, exact vector-Jacobian products,
and transactional plain SGD**, on native WGPU and browser WebGPU. This is not a
new model definition API or automatic acceleration of every `ModuleTrainer`.

```text
existing Sequential -> frozen plan -> mutable GPU copy
                                      |
upload input + targets once            |
          GPU [fused forward -> loss -> VJP -> candidate SGD]
                                      |
                  all layers finite? --+-- no: keep all weights
                                      |
                                     yes: commit all layers
                                      |
              next step reuses GPU parameters and activations
```

## Rust Use

Enable `st-nn/wgpu`. The native example below uses the existing module graph:

```rust
use st_backend_wgpu::runtime;
use st_nn::{layers::Gelu, resident::InferencePlan, Linear, Sequential};
use st_tensor::{NdLayout, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Sequential::new();
    model.push(Linear::new("up", 4, 7)?);
    model.push(Gelu::new());
    model.push(Linear::new("down", 7, 3)?);
    let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 5, 4])?)?;
    let (runtime, _) = runtime::ensure_default_runtime_blocking("my.training")?;
    let mut training = plan.compile_training_wgpu(runtime.clone())?;
    training.upload_batch(&[0.25; 40], &[0.1; 30])?;
    training.step(0.01)?;
    let snapshot = training.state_snapshot()?;
    let state = snapshot.read()?;
    assert_eq!(state.output_layout.shape(), &[2, 5, 3]);
    println!("pre-update MSE: {}", state.loss);

    let parameters = state.parameters.into_iter().map(|layer| {
        Ok((Tensor::from_vec(layer.inner, layer.cols, layer.weights)?,
            Tensor::from_vec(1, layer.cols, layer.bias)?))
    }).collect::<st_tensor::PureResult<Vec<_>>>()?;
    let trained_plan = plan.with_parameters(parameters)?;
    let _inference = trained_plan.compile_wgpu(runtime)?;
    Ok(())
}
```

In WASM Rust, request `WgpuRuntime::request_headless(...).await` and use
`snapshot.read_async().await`. A dedicated browser example runs this exact core;
the public JavaScript/Python `InferencePlan` clients currently expose inference,
not these new training methods. No new wheel is required for Rust or the
dedicated browser example; do not expect an installed Python wheel to expose
`compile_training_wgpu` yet.

## Execution And Ownership

- Parameters are uploaded once at construction. A batch upload changes input and
  target buffers, not the weights. Logical N-D leading axes become batch rows;
  Linear acts on the last axis. Only nonempty contiguous, offset-zero inputs are
  supported. There is no general broadcasting or strided GPU autograd yet.
- Each `step(rate)` enqueues one command buffer containing forward, loss,
  backward, and update passes. Saved preactivations support GELU's derivative.
  Both matrix VJPs reuse the canonical GEMM, with transpose addressing instead of
  materializing transposed matrices. Scalar/register-2x2 and all three existing
  accumulation policies are available through `compile_training_wgpu_with_options`.
- Activations, gradients, parameters and SGD candidates remain device-resident.
  A small rate-uniform write occurs per step. There is no intermediate host
  tensor readback. These are code-path boundaries, not hardware-counter claims.
- All layers prepare candidate parameters before a separate global finite check.
  Only then can any layer commit. An invalid loss, activation, derivative,
  gradient, scaled update, or candidate rejects the entire step. This protects
  numerical failures, not arbitrary device loss or hardware failure.
- `step()` returns a **submitted attempt number**, not proof of acceptance.
  `loss_snapshot().read()` validates all flags and reads only loss plus flags.
  `state_snapshot()` additionally captures pre-update predictions/gradients and
  **post-update parameters**. Zero rate is a derivative probe; invalid gradients
  still reject. Snapshots own their data across further steps and workspace drop.
- Capture a snapshot for each attempt whose individual acceptance matters. A
  later valid step does not certify an earlier uncaptured step; flags are per
  attempt, not a persistent training-history ledger. Captured failures survive
  a subsequent successful step, and rejected numeric steps can be retried with
  a new batch/rate without reuploading parameters.
- `parameter_snapshot()` exports current parameters before training or after a
  rejected step. `with_parameters()` creates a new frozen plan and preserves the
  graph. Neither the source module nor the original plan is mutated. Full-state
  and parameter snapshots concatenate their data in a staging buffer and may
  hit a device's buffer-size limit before the resident computation itself does.
- This version is plain SGD only: no momentum, Adam, clipping, gradient
  accumulation, checkpoint/resume format, LoRA lowering, or automatic host-model
  synchronization. Existing `pure::Tensor` remains host-backed and 2D.

## Gradient Reduction Correction

`Linear::backward` and `LoraLinear::backward` previously divided parameter
gradients by the number of rows, even when the incoming loss cotangent had
already been averaged. Their input gradients did not get this extra division.
Those parameter gradients are now true VJPs: `dW = X^T dY`, `db = sum_rows(dY)`;
mean-MSE alone owns its `2 / element_count` reduction.

**This intentionally changes multirow training behavior.** For the same supplied
cotangent, parameter gradients are now `rows` times the old values. Existing
learning rates tuned around the extra averaging may need retuning; single-row
behavior is unchanged. Do not reinterpret old training results as if they used
the corrected derivative. Finite differences cover Linear, nonzero LoRA A/B,
and Sequential, including input gradients. Batch-duplication tests ensure that
repeating identical samples leaves a mean-loss SGD update unchanged.

The shared checked fused GELU also bounds the `tanh` argument in its already
saturated f32 region. This avoids a native GPU NaN for large finite arguments;
square/cubic overflow checks remain before saturation and still reject.

## Reproduce

```bash
cargo test -p st-nn --no-default-features --test training_gradient_reduction
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test -p st-nn --no-default-features \
  --features wgpu --test resident_training --test resident_inference -- --test-threads=1
cargo run --release -p st-nn --no-default-features --features wgpu \
  --example resident_training > /tmp/new-native-training.json

cargo build --release -p st-nn --no-default-features --features wgpu \
  --example resident_training_browser --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir /tmp/training-web --out-name spiraltorch_wasm \
  target/wasm32-unknown-unknown/release/examples/resident_training_browser.wasm
node tools/test_resident_browser.cjs /tmp/training-web /path/to/chromium \
  /tmp/new-browser-training.json '' '' '' '' nn-training

python -I tools/validate_resident_training_vs_torch.py \
  --reports /tmp/new-native-training.json /tmp/new-browser-training.json \
  --devices cpu mps --output /tmp/new-training-torch.json
```

Use a `wasm-bindgen` CLI matching Cargo.lock and Playwright in the Node environment.
The browser runner launches a fresh isolated headless profile. The PyTorch
validator needs PyTorch, not an installed SpiralTorch wheel; omit `mps` on
machines without it. MPS CPU fallback must be disabled. Output report paths are
exclusive; use fresh names for reruns and retain failures.

The shared fixture covers 18 VJP cases (three shapes, two kernels, three
accumulation policies), all-layer rollback, recovery, GELU saturation/overflow,
and 128-step synthetic fits over three seeds. It verifies parameter and input
gradients, post-update weights, leading-axis shapes, owned snapshots, mean-loss
batch duplication and export back to resident inference. PyTorch independently
replays the retained batches with tanh-GELU, mean-MSE and plain SGD. These are
bounded correctness/learning tests, **not throughput or language-model quality
claims**; CUDA and broad NN operator training remain unmeasured here.
