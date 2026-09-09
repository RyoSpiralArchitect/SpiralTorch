# Mixed NN Graphs Without A Training Tape

`InferencePlan::compile_graph_wgpu` connects existing Rust `Linear`, `Gelu`,
`Relu`, `Scaler` and nested `Sequential` modules to a forward-only resident
executor. It accepts both v1 dense plans and v2 mixed plans, including
parameter snapshots exported by resident graph training. There is no target,
MSE, backward, gradient-policy selection or optimizer allocation in this path.

The dense-only `compile_wgpu` remains available. Both inference executors
share the checked, non-taped matmul/bias/GELU kernel; mixed graph pointwise
stages reuse the existing Rust `PointwisePlan`, not a second implementation of
the operations. Unsupported modules still fail at lowering without fallback.

## GPU Composition

Enable `st-nn/wgpu` and `st-backend-wgpu`:

```rust
use st_backend_wgpu::runtime;
use st_nn::{layers::{Gelu, Relu, Scaler}, resident::InferencePlan, Linear, Sequential};
use st_tensor::NdLayout;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Sequential::new();
    model.push(Scaler::new("scale", 4)?);
    model.push(Linear::new("up", 4, 7)?);
    model.push(Gelu::new());
    model.push(Relu::new());
    model.push(Linear::new("down", 7, 3)?);

    let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 5, 4])?)?;
    let (runtime, _) = runtime::ensure_default_runtime_blocking("my.graph")?;
    let mut graph = plan.compile_graph_wgpu(runtime.clone())?;
    let device = graph.tensor_device().clone();
    let input = device.upload(&[2, 5, 4], &[0.25; 40])?;
    let gains = device.upload(&[4], &[1., 0.5, 0.75, 1.25])?;
    let preprocessed = input.mul(&gains)?.relu()?;
    graph.set_input_tensor(&preprocessed)?;
    graph.dispatch()?;
    let result = graph.output_tensor()?;
    let shifted = result.add(&device.upload(&[3], &[0.1, 0.2, 0.3])?)?;

    let next = InferencePlan::from_module(&Relu::new(), graph.output_layout().clone())?;
    let mut next = next.compile_graph_wgpu(runtime)?;
    next.set_input_tensor(&shifted)?;
    next.dispatch()?;
    let snapshot = next.snapshot()?;
    drop(graph);
    drop(next);
    let values = snapshot.read()?; // first CPU observation of graph results
    assert_eq!(values.len(), 30);
    Ok(())
}
```

On WASM use asynchronous runtime discovery and `snapshot.read_async().await`.
The same Rust compiler and executor run in the browser. Python and WASM expose
this compiler through owning handles; neither client reconstructs the graph's
operations, strides, broadcasting or error propagation.

## Python And Browser Clients

Use a current-source wheel with `nn` and `wgpu` (both are default features):

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Scaler.from_gain("scale", st.Tensor(1, 4, [1., .5, .75, 1.25])))
model.add(st.nn.Linear(4, 7, name="up"))
model.add(st.nn.Gelu())
model.add(st.nn.Relu())
model.add(st.nn.Linear(7, 3, name="down"))
plan = model.inference_plan([2, 5, 4])
gpu = plan.compile_graph_wgpu()
device = gpu.tensor_device()
x = device.upload([2, 5, 4], [.25] * 40)
gpu.set_input_tensor(x.relu())
gpu.dispatch()
y = gpu.output_tensor().add(device.upload([3], [.1, .2, .3]))
del gpu
snapshot = y.snapshot()
values = snapshot.read_values()  # explicit first host observation
payload = plan.to_json()        # the same Rust-owned plan can go to a browser
```

The independent factory `st.WgpuTensorDevice.create()` uses the same default
device/queue as NN compilation. `st.WgpuTensor`, `st.WgpuTensorDevice` and
`st.WgpuTensorSnapshot` are also available under `spiraltorch.wgpu`. Only the
factory/upload/operation methods create handles; direct constructors are not
part of the API. Ordinary host-backed `st.Tensor` is a separate type.

Build the production WASM package with `webgpu`, then use the exact plan JSON:

```javascript
import init, { InferencePlan } from "./pkg/spiraltorch_wasm.js";
await init();
const plan = InferencePlan.fromJson(payload);
const pending = plan.compileGraphWebGpu();
plan.free(); // the compilation promise already owns the Rust plan
const gpu = await pending;
const device = gpu.tensorDevice();
const x = device.upload([2, 5, 4], new Float32Array(40).fill(.25));
gpu.setInputTensor(x);
gpu.dispatch();
const y = gpu.outputTensor();
gpu.free(); x.free(); device.free();
const snapshot = y.snapshot();
y.free();
const shape = snapshot.shape; // Uint32Array; independent metadata
const reading = snapshot.readValues();
snapshot.free(); // the pending read owns its buffers, not this JS wrapper
const values = await reading;
```

`WgpuTensorDevice.create()` is an asynchronous browser factory. Tensor methods
`reshape`, `permute`, `narrow`, `broadcast_to`/`broadcastTo`, `contiguous`, `add`,
`mul`, `relu`, `gelu` and `snapshot` all use the same Rust implementations.
Shape/axes/index arguments must be integers, not booleans or fractional values;
browser shape arguments are `number[]`, and data arguments are `Float32Array`.
Shape/stride metadata is copied out, with strides measured in elements.
Standalone tensors support scalar and empty layouts; NN plans still require
nonempty last-axis inputs.

The handle also connects existing public executors:

- Dense inference accepts `set_input_tensor` / `setInputTensor`; its
  `tensor_snapshot(device)` / `tensorSnapshot(device)` returns a GPU tensor.
- Graph training accepts `upload_batch_tensors` / `uploadBatchTensors`, and
  returns `prediction_tensor` / `predictionTensor` and `input_gradient_tensor`
  / `inputGradientTensor`. Both are pre-update captures with the complete
  training guard, not proof that an enqueued SGD step was accepted.
- Mixed inference uses `output_tensor` / `outputTensor`; any of these results
  can feed another compatible executor without intermediate host readback.

Mixed `dispatch()` returns the submitted-dispatch counter; snapshot metadata
retains input generation and dispatch separately. The legacy dense executor's
return value remains its input generation. Snapshot reads consume the handle
once, including a rejected read. An invalid upstream value stays invalid after
ReLU, reshape or an NN pass; errors are deferred to explicit observation.

CPU-only Python builds retain the class names but reject device creation and
graph compilation with `NotImplementedError`. An `nn`-only WASM build transports
plans and rejects `compileGraphWebGpu`; it does not export `WgpuTensor*` classes.
Neither path substitutes host Tensor operations when WebGPU is unavailable.

## Transfer And Ownership Contract

- `upload` copies finite host values into a stable input buffer. Invalid lengths
  or nonfinite values leave the prior input/output generation intact.
- `set_input_tensor` accepts exact logical N-D shapes on the same device and
  queue. Narrow, permuted and broadcast views are packed on-device when needed,
  then copied to stable graph input storage. This is not zero-copy admission.
- Parameters, bindings and activation buffers are prepared once. `dispatch`
  uses one command submission, with no per-dispatch buffer/binding allocation
  or host readback. Intermediate activation buffers are shared between stages.
  Pointwise validation flags use small GPU-only copies into the graph guard.
- `output_tensor` freezes output and all guards into an immutable GPU tensor.
  It performs an on-device identity/capture pass, not a CPU readback or mutable
  workspace alias. Subsequent dispatches cannot invalidate the returned tensor.
- `snapshot` freezes output and stage flags into an owned readback buffer.
  Reading is explicit and consumes the snapshot. Graph drop/reuse is safe.
- `dispatch` returns the submitted-dispatch counter, **not** a success receipt.
  `generation` counts accepted inputs independently. Errors detected on the
  GPU are deferred until observation, including earlier overflow masked by a
  later ReLU and invalid upstream tensors. Graph readback error stage
  `stage_count` denotes the upstream input guard; `0..stage_count` are NN stages.
- Frozen plans do not track live model updates. Ordinary `Module::forward`,
  host-backed 2D `pure::Tensor`, automatic autograd and `ModuleTrainer` routing
  are unchanged. The N-D graph applies Linear along the last axis; it is not a
  generic attention/convolution/arbitrary-module compiler.

These are structural transfer boundaries, not hardware-counter measurements
or a claim that this path is faster than PyTorch.

## Public Client Validation

The production-client fixture imports a frozen core fixture, preserves its
input values and plan contents, and executes 12 recipes through the public handles:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I bindings/st-py/tests/test_wgpu_tensor.py -v
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I bindings/st-py/tests/test_nn_resident_graph_forward.py -v
python -I bindings/st-py/examples/resident_graph_forward.py \
  --fixture native.json --output /tmp/new-python-forward.json
# Build the production webgpu package, not the dedicated Rust fixture cdylib.
OUT_DIR=/tmp/new-public-module EXAMPLES_DIR=/tmp/no-example-sync \
  bash scripts/build_wasm_web.sh --features webgpu
cp native.json /tmp/new-public-module/forward-fixture.json
node tools/test_resident_browser.cjs /tmp/new-public-module /path/to/chromium \
  /tmp/new-browser-forward.json '' '' '' '' nn-forward-clients
PYTORCH_ENABLE_MPS_FALLBACK=0 /path/to/torch-python -I tools/verify_resident_graph_forward_torch.py \
  native.json /tmp/new-python-forward.json /tmp/new-browser-forward.json \
  --output /tmp/new-client-torch.json
```

The verifier requires the hash-matching core JSON alongside client reports and
rejects input/plan/recipe drift before importing Torch. Browser asset hashes
must also match the declared fixture lineage. Client ownership checks are
reported separately from the core fixture's cross-device rejection checks;
the public clients intentionally share one default runtime.

## Reproduce

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-nn \
  --no-default-features --features wgpu --test resident_graph_forward
cargo run --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_graph_forward > native.json
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --target wasm32-unknown-unknown --example resident_graph_forward_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_graph_forward_browser.wasm \
  --target web --out-dir /tmp/new-forward-module --out-name spiraltorch_wasm
node tools/test_resident_browser.cjs /tmp/new-forward-module /path/to/chromium \
  /tmp/new-forward-browser.json '' '' '' '' nn-graph-forward
PYTORCH_ENABLE_MPS_FALLBACK=0 /path/to/torch-python -I tools/verify_resident_graph_forward_torch.py \
  native.json /tmp/new-forward-browser.json --output /tmp/new-forward-torch.json
```

Use the `wasm-bindgen` version matching Cargo.lock and supply Playwright in the
Node environment. The isolated browser fixture executes 12 mixed-model recipes
and nine guard/compatibility groups, including repeated dispatch, strided and
broadcast inputs, GPU preprocessing/postprocessing, graph-to-graph composition,
snapshot lifetime, recovery and both plan versions. Torch checks all six output
captures per recipe on each explicitly requested device, without SpiralTorch
math. This fixture measures correctness and ownership, not throughput.
