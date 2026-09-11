# The Original NN Model On Resident GPU Inputs

The high-level model now owns and reuses the existing Rust resident graph.
It does not require a second hand-built GPU model:

```text
explicit upload -> model.forward_resident -> more GPU operations -> explicit read
                        Linear + GELU
                      Scaler / ReLU / ...
```

Enable `st-nn/wgpu` in Rust, `wgpu` in the Python build, or `webgpu` in WASM.
These are current-source APIs, not a claim about older published wheels.

## Python

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Linear("up", 4, 7))
model.add(st.nn.Gelu())
model.add(st.nn.Scaler("gain", 7))
model.add(st.nn.Relu())
model.add(st.nn.Linear("down", 7, 4))

device = st.WgpuTensorDevice.create()
x = device.upload([2, 5, 4], [0.25] * 40)
for _ in range(20):
    x = model(x)  # same model, owning GPU output, no host readback
print(model.resident_cache_info())
print(x.snapshot().read_values())  # explicit terminal observation
```

`model.forward(x)` and `model(x)` select by input type, not a hidden fallback:
host `Tensor` keeps its existing host-returning route; `WgpuTensor` uses the
input's device and returns `WgpuTensor`. Supported modules are `Linear`,
`Scaler`, `Gelu`, `Relu`, and compositions of them in `Sequential`.
Unsupported layers reject before executing their host implementation.

## Rust And Browser

In Rust import `st_nn::Module` and call
`model.forward_resident(&input)?` with a
`st_backend_wgpu::resident_tensor::ResidentTensor`. The same
`resident_forward_stats()` and `clear_resident_forward_cache()` methods are
available. Custom modules may explicitly opt into the shared
`ResidentForwardCache::forward(operations, input)`; descriptors must faithfully
represent their ordinary forward semantics.

The browser owns a real `st_nn::Sequential`, not a JavaScript implementation:

```javascript
import init, {Sequential, WgpuTensorDevice} from "./spiraltorch_wasm.js";
await init();
const model = new Sequential();
model.addLinear("up", 4, 7);
model.addGelu();
model.addScaler("gain", new Float32Array(7).fill(1));
model.addRelu();
model.addLinear("down", 7, 4);
const device = await WgpuTensorDevice.create();
const input = device.upload([2, 5, 4], new Float32Array(40).fill(0.25));
const output = model.forward(input);
const stats = model.residentCacheInfo(); // bigint counts
console.log(stats.compilations);
stats.free();
model.clearResidentCache();
model.free(); // output still owns its values
const snapshot = output.snapshot();
console.log(await snapshot.readValues());
snapshot.free();
output.free();
input.free();
device.free();
```

`model.inferencePlan(shape)` still exports a fixed portable plan. After resident
training, `baseline.applyParametersTo(model, updatedPlan)` performs the same
checked, baseline-matching weight handoff as Python's
`baseline.apply_parameters_to(model, updated_plan)`. The next forward follows
the changed weights. See [handoff rules](resident_module_handoff.md); this is not
optimizer-state resume.

## Reuse And Boundaries

- Each parameterized Module holds one bounded, replaceable graph, not a cache
  growing with every shape. Shape, device/queue, operation sequence, layout or
  parameter-bit changes rebuild it. Identical values reuse it.
- Mutable/foreign parameters are compared by bits on every call. This deliberately
  includes externally shared DLPack writes; pointer identity is not sufficient.
  Comparison is O(parameter values) CPU work, not an unmeasured zero-cost claim.
- Input packing and owning output capture copy on the GPU. There is no per-layer
  host readback, but this is not zero-copy and does not promise one queue submission
  for the entire public call.
- The existing Linear/bias/GELU fusion is reused. Broader pointwise fusion is not
  enabled implicitly. Returned tensors survive cache reuse, clear and model drop.
  Invalid-input flags remain visible at explicit readback even after zero gains
  or ReLU could otherwise hide them.
- Cache counters describe compilation/selection/submission, not successful GPU
  completion or numerical validation. An empty Sequential returns its input
  handle without dispatch and leaves the counters at zero.
- This forward path has no backward tape or optimizer. Generic `ModuleTrainer`
  is not automatically moved to the GPU. An active committed tensor execution
  plan is rejected because it does not yet describe this composite resident
  route; uncommitted scopes do not override an explicitly supplied GPU input.

For timing, `tools/bench_graph_forward_paths.py --include-module` adds ordinary
`model(WgpuTensor)` calls to the existing matched fixture and PyTorch controls.
The module d2h route excludes input upload; compare it separately from h2h.
The fixed-input burst routes each perform eight independent forwards and one
terminal host read. Per-call parameter comparison and GPU copies are timed;
cold compilation is recorded separately.

The [source-bound first record](../benchmarks/results/2026-09-12-module-resident-forward/README.md)
includes the small-model slowdown as well as the deeper-model wins, CPU-only
build checks, independent Torch replay and the rejected exploratory attempts.
