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
use st_tensor::NdLayout;

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

    let trained_plan = plan.with_dense_parameters(state.parameters)?;
    let _inference = trained_plan.compile_wgpu(runtime)?;
    Ok(())
}
```

In WASM Rust, request `WgpuRuntime::request_headless(...).await` and use
`snapshot.read_async().await`. Python and JavaScript expose this same core;
their bindings do not implement a second loss, derivative, or optimizer.

During workspace construction, Rust specializes the known forward activation,
transpose and validation attributes for each training matrix stage. Matching layers reuse the
same pipeline, with only required forward variants compiled. The generic
host-tensor shader path is unchanged. Reuse a compiled training workspace across
steps: pipeline preparation can dominate a one-shot call. See the
[measured stage-specialization results and limits](resident_nn_training_stage_specialization.md).

## Python And Browser Clients

Build a current-source Python wheel with `wgpu` (enabled by default), or a
WASM package with `--features webgpu`. Older installed/PyPI wheels may not have
these methods. This source change alone is not a PyPI release. CPU-only builds
still import the plan/classes but explicitly reject GPU compilation.

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Linear(4, 7, name="up"))
model.add(st.nn.Gelu())
model.add(st.nn.Linear(7, 3, name="down"))
plan = model.inference_plan([2, 5, 4])
training = plan.compile_training_wgpu()
training.upload_batch_values([0.25] * 40, [0.1] * 30)
receipts = []
for _ in range(64):
    training.step(0.01)
    receipts.append(training.loss_snapshot())
losses = [receipt.read() for receipt in receipts]
trained = training.parameter_snapshot().read_plan()
payload = trained.to_json()  # send these bytes to the browser
```

An uploaded Python `Tensor` must have shape `(product(leading_axes), last_axis)`;
use `upload_batch(input, target)` for that form, or flat `upload_batch_values`.
Logical shape stays N-D in the workspace and full snapshot. Each successful
batch upload invalidates the current step until the next `step`; invalid uploads
leave both the prior batch and counters unchanged.

```javascript
import init, {InferencePlan} from "./pkg/spiraltorch_wasm.js";
await init();
const plan = InferencePlan.fromJson(payload);
const pending = plan.compileTrainingWebGpu(undefined, "register_2x2", "compensated");
plan.free(); // the pending compilation owns its plan
const training = await pending;
training.uploadBatch(new Float32Array(40).fill(0.25), new Float32Array(30).fill(0.1));
training.step(0.01);
const snapshot = training.stateSnapshot();
const statePromise = snapshot.readState();
snapshot.free();
training.free(); // the pending read owns its GPU buffers
const state = await statePromise;
console.log(state.loss, state.inputShape, state.weightGradientValues(0));
const trained = state.toPlan();
const returnedPayload = trained.toJson();
trained.free(); state.free();
```

Python uses `state_snapshot().read_state()` and `state.to_plan()`, with
`weight(i)`, `bias(i)`, `weight_gradient(i)`, `bias_gradient(i)` returning host
Tensors. JS accessors return fresh `Float32Array`s. Snapshots are single-use,
including failed reads; metadata remains available after consumption. Numeric
rejections carry `code="training_step_rejected"`, `stage`, and `flags` on the
Python `ValueError` / JS `Error`. Runtime/shape errors do not claim numeric
rejection. JS step/generation counters are `bigint`, Python counters are integers.

This is **weight-only handoff**, not optimizer checkpoint/resume: a newly compiled
workspace starts fresh counters and no batch. Only stateless SGD is implemented.
Keep canonical Rust plan JSON as a string when transporting it; parsing and
reserializing with JS can lose signed-zero parameter bits.

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
  still reject, but accepted probes leave parameter bits unchanged, including
  negative zero. Snapshots own their data across further steps and workspace drop.
- Capture a snapshot for each attempt whose individual acceptance matters. A
  later valid step does not certify an earlier uncaptured step; flags are per
  attempt, not a persistent training-history ledger. Captured failures survive
  a subsequent successful step, and rejected numeric steps can be retried with
  a new batch/rate without reuploading parameters.
- `parameter_snapshot()` exports current parameters before training or after a
  rejected step. `with_dense_parameters()` creates a new frozen plan and validates
  stage count, shapes, activation identity, lengths and finite values, preserving the
  graph. Neither the source module nor the original plan is mutated. Full-state
  and parameter snapshots concatenate their data in a staging buffer and may
  hit a device's buffer-size limit before the resident computation itself does.
- This version is plain SGD only: no momentum, Adam, clipping, gradient
  accumulation, checkpoint/resume format, LoRA lowering, or automatic host-model
  synchronization. Existing `pure::Tensor` remains host-backed and 2D.

## Measured Pass Encoding

The resident path also has a source-bound training benchmark against its prior
implementation and eager PyTorch. Native Metal groups one step's dispatches into
one compute pass; BrowserWebGpu and unmeasured backends keep one pass per dispatch.
This changes encoding overhead, not the math, dispatch order, finite guards, or
snapshot acceptance contract. Browser measurements were mixed, so the native
optimization is deliberately not applied everywhere.

See [the measured workload, results, and reproduction commands](resident_nn_training_benchmarks.md).

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

### Public Client Roundtrip

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I bindings/st-py/tests/test_nn_resident_training.py -v
python -I bindings/st-py/examples/resident_nn_training.py --output /tmp/new-python-training.json
OUT_DIR=/tmp/new-training-clients EXAMPLES_DIR=/tmp/no-example-copy \
  bash scripts/build_wasm_web.sh --dev -- --features webgpu
cp /tmp/new-python-training.json /tmp/new-training-clients/training-fixture.json
node tools/test_resident_browser.cjs /tmp/new-training-clients /path/to/chromium \
  /tmp/new-browser-clients.json '' '' '' '' nn-training-clients
python -I bindings/st-py/examples/resident_nn_training.py \
  --browser-report /tmp/new-browser-clients.json --output /tmp/new-python-return.json
python -I tools/validate_resident_training_clients_vs_torch.py \
  --python-report /tmp/new-python-training.json --browser-report /tmp/new-browser-clients.json \
  --devices cpu mps --output /tmp/new-client-torch.json
```

The Python example uses installed bindings, not a source-tree import. The
browser fixture replays all 18 VJP cases and continues three Python 64-update
exports for 64 more updates. It checks these against uninterrupted Python
128-update runs; the separate PyTorch tool independently replays both clients.
The Python return phase imports the browser's trained plan and runs native
resident inference. Each update's acceptance is captured, without requiring a
CPU wait between updates. CPU-only WASM uses `--features nn` and fixture mode
`nn-training-clients-cpu`; it verifies transport and explicit unavailable errors,
not training. Build the CPU-only wheel with
`maturin build --manifest-path bindings/st-py/Cargo.toml --no-default-features --features python-default`.
