# Reusable Resident Pointwise Plans

Python and WASM expose Rust's existing shape-specialized `PointwisePlan` as
`WgpuPointwiseInputs` and `WgpuPointwisePlan`. They use the same kernels as Rust
and N-D tensor chains, not Python/JavaScript arithmetic. This is an explicit
execution surface, not a change to `Tensor`, `ModuleTrainer` or default routing.

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Scaler.from_gain("gain", st.Tensor(1, 1, [2.0])))
gpu = model.inference_plan([2, 1]).compile_graph_learner_wgpu(
    gradient_policy="exact"
)
gpu.upload_values([1.0, 2.0])
device = gpu.tensor_device()
inputs = st.WgpuPointwiseInputs()
inputs.add(gpu.forward().prediction_tensor())
inputs.add(device.upload([], [0.5]))
# Explicit derivative of mean(prediction**4) / 4: prediction**3 / N.
cube = inputs.compile([("multiply", 0), ("multiply", 0), ("multiply", 1)])
for step in range(8):
    forward = gpu.forward()
    inputs.set(0, forward.prediction_tensor())
    gradient = gpu.backward(forward, cube.run(inputs, execution="fused"))
    gpu.sgd(gradient, 0.01)
    print(gpu.update_snapshot().read())  # Acceptance, not an invented loss value.
```

Prepare once, then replace values using `inputs.set(slot, tensor)`. `add` and
`set` clone immutable handles without uploading, copying GPU values, submitting
work or reading data. `set` requires the slot's exact shape, strides, offset and
device/queue; failed replacement leaves the previous binding intact. Construct
another collection for a different layout. Plans retain layouts and the device,
not the original values. Run them against any matching input collection.

## Operations and Scheduling

Each step is `(operation, rhs)`. Operations are `identity`, `add`, `multiply`,
`relu`, and `gelu`. Unary steps require `None`/`null`; binary steps require an
original input slot. **Slot zero means the original first input, not the current
intermediate.** The running left operand starts at input zero. All declared
inputs must be used, with 1..16 inputs and 1..256 steps; actual device binding
limits can be stricter and fail explicitly. No silent fallback occurs.

All inputs broadcast into the first input's fixed logical domain. This includes
scalar, empty, offset and strided layouts; a program cannot change that domain
mid-chain to hide invalid nonempty work. Device, shape and arity checks are Rust
owned. Python rejects boolean indices; WASM checks integers before narrowing.

- `sequential` submits the existing individual tensor operations.
- `batched` encodes individual operations in one submission, retaining temporary tensors.
- `fused` runs the prepared chain in one dispatch, without intermediate value buffers.

Every mode preserves finite-input/intermediate checks and inherited failures,
including overflow masked by ReLU or an empty view. Outputs are immutable and
survive collection replacement, plan destruction and parent-object release.
This exposes forward elementwise execution; the Rust pointwise VJP API itself is
not newly bound here. The caller still defines the correct loss cotangent.

## Browser and Rust

WASM uses a JSON array for the same operation/slot pairs and requires an explicit
execution string. Original tensor objects are borrowed, not consumed by `add`.

```javascript
import init, {WgpuTensorDevice, WgpuPointwiseInputs} from "spiraltorch-wasm";
await init();
const device = await WgpuTensorDevice.create();
const x = device.upload([2, 1], new Float32Array([2, 4]));
const norm = device.upload([], new Float32Array([0.5]));
const inputs = new WgpuPointwiseInputs();
inputs.add(x); inputs.add(norm);
const cube = inputs.compile(JSON.stringify([
  ["multiply", 0], ["multiply", 0], ["multiply", 1]
]));
const seed = cube.run(inputs, "fused");
const snapshot = seed.snapshot();
for (const h of [cube, inputs, x, norm, seed, device]) h.free();
console.log(await snapshot.readValues()); // [4, 32]
snapshot.free();
```

Rust callers may keep using `PointwisePlan::new(...).run(&inputs, execution)` or
the new `resident_tensor::pointwise::PointwiseInputs` owning collection. Both
lead to the same plan. Python CPU-only wheels reject collection construction;
the WASM classes require `webgpu`. Existing `NdPointwisePlan` host execution is
unchanged, and a published wheel is not updated by a local source build.

The learner benchmark accepts `--graph --learner --fuse-learner-seeds` to select
only candidate cubic-cotangent fusion, without changing the model's graph fusion,
weights, data, objective or optimizer. Its browser counterpart selects workload
`learner` and optimization `fuse-learner-seeds`. Compilation stays outside timing;
two VJPs, weighted SGD and every update acceptance remain inside. This measures
an explicit caller choice, not an automatic backend speedup for all learning.
