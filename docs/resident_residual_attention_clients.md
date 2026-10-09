# Residual attention training from Python and WASM

These are **unreleased source APIs after 0.4.29**, not additions to the published
0.4.29 wheel. Both clients call the same
[Rust residual block](resident_residual_attention.md):

```text
u = pre(x)
y = x + attention(u, z_bias, pair_bias)
output = y + feed_forward(y)
```

The two branch plans can include LayerNorm, Linear, GELU, Scaler and the existing
ToposResonator. A single Rust parameter owner updates every block parameter
atomically. Python and JavaScript do not reconstruct normalization, VJPs,
geometry, guard propagation or optimizer decisions.

## Python

Build the GPU-enabled source wheel with the default binding features. This
complete small example constructs real Rust Modules before freezing plans:

```python
import spiraltorch as st

shape = (2, 3, 4)
pre = st.nn.Sequential()
pre.add(st.nn.LayerNorm("pre", 4, -1.0, 1e-5))

feed = st.nn.Sequential()
feed.add(st.nn.LayerNorm("post", 4, -1.0, 1e-5))
feed.add(st.nn.Linear("up", 4, 7))
feed.add(st.nn.Gelu())
kernel = st.ToposResonatorKernel(
    coupling=0.2, iterations=4, saturation=0.12, porosity=0.3, max_values=42
)
feed.add_topos_resonator("gate", st.Tensor(1, 7, [0.8] * 7), kernel)
feed.add(st.nn.Linear("down", 7, 4))

projections = [
    st.nn.Linear(name, 4, 4).inference_plan(shape)
    for name in ("query", "key", "value", "output")
]
attention = st.nn.AttentionInferencePlan.from_projection_plans(
    *projections, heads=2, causal_offset=0
)
source = st.nn.ResidualAttentionPlan.from_plans(
    pre.inference_plan(shape), attention, feed.inference_plan(shape)
)
model = source.compile_training_wgpu()
device = model.tensor_device()
x = device.upload(shape, [i * 0.05 for i in range(24)])
target = device.upload(shape, [0.0] * 24)
forward = model.forward(x)
loss = st.nn.MeanSquaredError().evaluate_resident(
    forward.prediction_tensor(), target
)
gradients = model.backward(forward, loss.prediction_gradient_tensor())
update = model.sgd(gradients, 0.01)
assert update.read() == 1  # Explicit GPU acceptance readback.
assert len(model.parameter_tensors()) == 13
```

`forward(x, z_bias=..., pair_bias=...)` accepts the existing logical attention
bias tensors. Their derivatives are available as
`z_bias_gradient_tensor()` and `pair_bias_gradient_tensor()`; the caller owns
their updates. Parameter order is pre-graph, fused QKV weight/bias, output
weight/bias, then feed-forward graph. A usual block has 12 tensors; this
Topos recipe has 13. The names are exported from `spiraltorch.nn`, including
the opaque plan, trainer, forward and gradient classes.

CPU-only source builds use
`--no-default-features --features python-default`. Plan composition still
works; `compile_training_wgpu()` explicitly raises `NotImplementedError`.
It never silently trains a different CPU model.

## Browser

The `webgpu` build exposes `ResidualAttentionPlan.fromPlans(pre, attention,
feedForward)` and `compileTrainingWebGpu()`. Existing `InferencePlan` JSON
v3/v5 handles LayerNorm/Topos through Rust validation; no client-side schema
interpretation or math is added. Alternatively, construct real Modules with
`Sequential.addLayerNorm(name, features, curvature, epsilon)`,
`addLinear`, `addGelu` and `addToposResonator`.

```js
const source = st.ResidualAttentionPlan.fromPlans(pre, attention, feedForward);
const pending = source.compileTrainingWebGpu();
source.free(); // The pending compilation owns a frozen Rust plan.
const model = await pending;
const biases = new st.WgpuAttentionBiases();
const forward = model.forward(input, biases);
const prediction = forward.predictionTensor();
const loss = new st.MeanSquaredError();
const evaluated = loss.evaluateResident(prediction, target);
const seed = evaluated.predictionGradientTensor();
const gradients = model.backward(forward, seed);
const update = model.sgd(gradients, 0.01);
const accepted = update.read();
update.free(); // The read promise owns its capture.
if (await accepted !== 1n) throw Error("Unexpected update revision");
for (const handle of [gradients, seed, evaluated, loss, prediction,
                      forward, biases, model]) handle.free();
```

Here `pre`, `attention`, `feedForward`, `input` and `target` are previously
constructed owning handles. Free them separately after their last use. Returned
tensors survive destruction of the trainer/forward/gradient wrapper. Only the
latest forward token is reusable for backward; attempted updates invalidate old
gradients even when numerical rejection preserves all weights.
`attemptedUpdates` is not an acceptance receipt.

## Verification and Boundaries

The unchanged 60-case Torch fixture and two 32-update recipes run through both
public clients, including optional bias gradients and terminal residual overflow.
The update loops have no intermediate readback. Browser tests additionally
exercise freeing the plan during async compilation, freeing a receipt during
readback, strict LayerNorm construction and real LayerNorm/Topos module assembly.
CPU-only Python and Node/WASM tests cover plan composition and the explicit
GPU feature gate. The shipped Python/TypeScript declarations accompany the API.
The TypeScript contract test requires `tsc` (CI pins TypeScript 5.9.2). It checks
both declaration files for syntax errors before the API-shape assertions and
verifies that an intentionally unclosed class is rejected. `skipLibCheck` avoids
unrelated library type checking; this is not a full downstream application check.

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B \
  bindings/st-py/tests/test_nn_residual_attention_training.py -v
node bindings/st-wasm/tests/residual_attention_plan.cjs /path/to/cpu/spiraltorch_wasm.js
node bindings/st-wasm/tests/residual_attention_training_types.cjs /path/to/web/spiraltorch_wasm.js
```

Serve the repository on loopback and open
`bindings/st-wasm/tests/residual_attention_training_clients.html?module=/path/to/web/spiraltorch_wasm.js`.
Use a generated web module served from the same origin.
[Recorded results](../benchmarks/results/2026-10-09-resident-residual-attention-clients/README.md)
separate compile checks, native runtime, browser runtime and feature gates.

This remains one synthetic training block, not a complete language model,
checkpoint, KV-cache or global multi-block optimizer. Original host Modules stay
unchanged. Separate block owners must not be presented as one atomic full-model
update. No speed or language-quality improvement is claimed.
