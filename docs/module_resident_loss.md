# Ordinary Losses In Resident Learning

The existing Rust `MeanSquaredError` can now return a loss and its exact
prediction cotangent as owning GPU tensors. Python and WASM expose that same
`Loss::evaluate_resident` contract; neither reconstructs the objective.

The operation reuses the existing native/browser training MSE shader entry
points and parameter layout. Pipelines are initialized lazily per TensorDevice.
Predictions and targets must have identical logical shapes on the same device.
N-D offset, permuted and broadcast views are packed on GPU when necessary.
There is no implicit upload, host read, optimizer or fallback.

## Python

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Scaler.from_gain("gain", st.Tensor(1, 1, [2.])))
model.add(st.nn.Relu())
baseline = model.inference_plan([2, 2, 1])
learner = baseline.compile_graph_learner_wgpu(gradient_policy="exact")
device = learner.tensor_device()
x = device.upload([2, 2, 1], [1.] * 4)
target = device.upload([2, 2, 1], [1.] * 4)
learner.set_input_tensor(x)
objective = st.nn.MeanSquaredError()

for _ in range(32):
    forward = learner.forward()
    loss = objective.evaluate_resident(forward.prediction_tensor(), target)
    gradient = learner.backward(forward, loss.prediction_gradient_tensor())
    learner.sgd(gradient, 0.1)
    # No value, gradient or parameter readback inside this loop.

learner.update_snapshot().read()  # Acceptance of the last attempted update.
updated = learner.parameter_snapshot().read_plan()
baseline.apply_parameters_to(model, updated)  # Explicit weight-only handoff.
```

To observe the pre-update loss, explicitly read
`loss.loss_tensor().snapshot().read_values()`. A `ResidentLoss` and either of
its output handles survive later operations and dropped producer wrappers.
Capture each update receipt if every intermediate acceptance must be checked:
the last receipt does not certify earlier attempts.

The parameter handoff retains the existing topology/value/optimizer checks.
It does not migrate optimizer state or synchronize the original host Module on
every step. While the learner owns updated GPU parameters, the original Module
continues to contain its old values until the explicit handoff.

## Rust And WASM

In Rust, import `st_nn::Loss` and call
`MeanSquaredError::new().evaluate_resident(&prediction, &target)?`.
The returned `ResidentLoss` has `value()` and `prediction_gradient()`.
Pass the latter directly to a resident autograd graph or learner.

In WASM, `new MeanSquaredError().evaluateResident(prediction, target)` returns
`ResidentLoss`, with `lossTensor()` and `predictionGradientTensor()`.
Read through `snapshot().readValues()` only when needed and free the ordinary
WASM handles after use. The browser uses the same Rust loss and update kernels.

## Mathematical And Failure Contract

The value is `sum((prediction-target)^2) / numel`, with shape `[1,1]`.
The cotangent is `2 * (prediction-target) / numel`, retaining the input's logical
N-D shape. Empty tensors follow the existing SpiralTorch MSE convention: zero
loss and an empty cotangent, unlike PyTorch's mean-of-empty NaN.

Both outputs share one whole-loss validity guard. Input failures, overflowing
differences/squares/seeds and invalid reductions invalidate both. Thus a loss
overflow cannot leave an apparently usable finite seed. This joint evaluation
has a stronger acceptance condition than calling the ordinary backward alone.
A learner seeded by an invalid pair rejects the whole parameter transaction,
including zero-rate updates. A later valid loss does not clear a retained
invalid pair, and an earlier failure does not poison a fresh valid evaluation.

The loss derivative is exact; `module_compatible` gain averaging, when explicitly
chosen, applies only in the learner's update policy. Ordinary CPU
`forward/backward` behavior is unchanged. Unsupported Rust losses reject the
resident method by default. Committed tensor execution plans cannot be bypassed.
CPU-only Python builds expose a clear unsupported-feature error.

This connects a real high-level Loss to GPU learning, not generic
`ModuleTrainer::train_epoch` migration. Hypergrad, band replay, distributed
accumulators and trainer policies are not silently replaced by plain SGD.
No throughput improvement or fastest-PyTorch claim follows from this API alone.
