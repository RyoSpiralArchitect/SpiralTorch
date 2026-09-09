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
The same Rust compiler and executor run in the browser. This addition is a Rust
API and an executable browser validation fixture, **not yet a public Python/JS
mixed-inference handle**. Existing public dense-inference and mixed-training
clients are unchanged; those clients can already transport the v2 plan.

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
