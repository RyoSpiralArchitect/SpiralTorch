# Loss-Independent Resident Graph Autograd

`InferencePlan` can compile supported NN modules into separate GPU-resident
forward and backward phases. This reuses the existing dense and pointwise VJP
kernels, without coupling differentiation to mean-MSE or SGD. Rust owns tape
identity, shape/device checks, derivatives and finite guards. Python and WASM
only expose owning handles.

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Scaler("gain", 4))
model.add(st.nn.Linear(4, 3, name="projection"))
model.add(st.nn.Gelu())
plan = model.inference_plan([2, 5, 4]).fuse_pointwise()
graph = plan.compile_graph_autograd_wgpu()
device = graph.tensor_device()
graph.set_input_tensor(device.upload([2, 5, 4], [0.1] * 40))
forward = graph.forward()
prediction = forward.prediction_tensor()
# Example: derivative of sum(0.25 * prediction**2), formed on the GPU.
cotangent = prediction.mul(device.upload([], [0.5]))
gradients = graph.backward(forward, cotangent)
dx = gradients.input_gradient_tensor()
dgain = gradients.parameter_gradient_tensor(0)
# Only this explicit observation reads back to the CPU.
print(dgain.snapshot().read_values())
```

Rust: `plan.compile_graph_autograd_wgpu(runtime)?`, `graph.forward()?`, then
`graph.backward(&forward, &cotangent)?`. `GraphForward::prediction()` and
`GraphGradients::{input_gradient, parameter_gradients}` return resident tensors.
WASM: `await plan.compileGraphAutogradWebGpu()`, `graph.forward()`, then
`graph.backward(forward, cotangent)`. Its handles use `predictionTensor()`,
`inputGradientTensor()`, `parameterGradientTensor(index)` and asynchronous
`snapshot().readValues()`. The same N-D tensor/view API applies to both clients.
All compilers also have explicit tile/kernel/accumulation options.

## Contract

- Cotangents must have the exact output shape and share the same WGPU context.
  Strided, broadcast and offset views are packed on the GPU. No CPU fallback.
- Backward computes an **exact mathematical VJP**. There is no implicit mean,
  gain row average, clipping, band weighting, accumulation or optimizer step.
  Multiple cotangents may reuse one forward; each result owns independent
  tensors. Parameter gradients follow the portable plan's IDs and shapes.
- Parameters are frozen at compilation and never updated by this API. New input
  or another forward invalidates the old token for backward. A token from another
  workspace is rejected even when its counters match. Bad host input/seed shape
  checks leave the previous valid tape and counters unchanged.
- Predictions and returned gradients survive workspace reuse/drop. A prediction
  carries only its forward guards. Every gradient carries **all** forward and
  backward guards, including late parameter reductions. A zero seed cannot mask
  an invalid forward. A bad backward does not poison later valid VJPs of a valid
  forward. Submission counters do not prove acceptance; guarded reads do.
- This route allocates forward/adjoint tape and raw gradients, but not full-sized
  target/loss partials or SGD candidate/effective-gradient buffers. Output and
  gradient handles are frozen by GPU operations in their respective submission.
  That avoids host round trips, not all GPU copies or dispatch overhead.

## Scope

Supports the same `Linear`, `Gelu`, `Scaler`, `Relu`, nested `Sequential` and
portable pointwise graph contract as [resident training](resident_graph_training.md).
For inference only, use [the forward-only compiler](resident_graph_forward.md)
to avoid backward tape entirely. Existing MSE/SGD training and its explicit
`ModuleCompatible` policy remain unchanged.

This is a composable prerequisite for richer losses and band replays, **not** an
automatic migration of `ModuleTrainer`, generic autograd, GNN, or `pure::Tensor`.
Their existing learning policies must be connected explicitly rather than
silently replaced. The source Module and published wheels are not changed by
compiling a resident graph. Performance is unclaimed until separately measured.

The shared `resident_graph_training` native/browser fixture includes arbitrary
cotangents, GPU-derived seeds, ranks 1/2/3, fusion on/off, retained snapshots,
overflow, stale/wrong-workspace tokens and recovery. The Torch validator replays
these against `torch.autograd.grad(..., grad_outputs=cotangent)` on explicitly
selected devices; this is numerical validation, not a throughput comparison.
