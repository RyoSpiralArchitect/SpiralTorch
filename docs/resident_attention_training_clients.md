# Attention training from Python and WebAssembly

Source addition **after 0.4.29**, not an API in the published 0.4.29 wheel.
Both clients call [the same Rust projection training core](resident_attention_projection_training.md).
They add no separate attention math, loss normalization, optimizer or route policy.
Build Python with the default WGPU features, or WASM with `--features webgpu`.

## Compose existing projections

`AttentionInferencePlan.from_projection_plans` (JS: `fromProjectionPlans`)
accepts four existing `InferencePlan` snapshots in Q/K/V/output order. Each must
be exactly one Linear without GELU or other operations. Their logical input
layouts must match `[batch, tokens, input_width]` for Q/K/V and
`[batch, tokens, heads * head_dim]` for output. The shared Rust constructor rejects
richer plans rather than discarding operations. Existing plan JSON is sufficient;
there is no new weight serialization format.

The head count is a positive integer dividing Q/K/V width. Omit `causal_offset`
for unmasked attention; zero masks future keys. Shapes and offsets are checked.
Host plans work without GPU features, but compiling them for training fails
explicitly when WGPU/WebGPU is not built in.

~~~python
import spiraltorch as st

shape = [2, 4, 8]
projections = [st.nn.Linear(8, 8).inference_plan(shape) for _ in range(4)]
plan = st.nn.AttentionInferencePlan.from_projection_plans(
    *projections, heads=2, causal_offset=0)
model = plan.compile_training_wgpu()
device = model.tensor_device()
x = device.upload(shape, [0.1] * 64)
target = device.upload(shape, [0.0] * 64)
loss_fn = st.nn.MeanSquaredError()

forward = model.forward(x)  # Optional z_bias= and pair_bias= resident tensors.
loss = loss_fn.evaluate_resident(forward.prediction_tensor(), target)
gradients = model.backward(forward, loss.prediction_gradient_tensor())
update = model.sgd(gradients, 0.01)
print(update.read())  # Explicit GPU acceptance readback, not just submission.
~~~

The browser uses the same sequence with owning WASM handles:

~~~javascript
// st is the initialized spiraltorch_wasm module; plans are four InferencePlans.
const plan = st.AttentionInferencePlan.fromProjectionPlans(...plans, 2, 0);
const compiling = plan.compileTrainingWebGpu();
plan.free(); // The Rust plan was cloned before the promise was returned.
const model = await compiling;
const device = model.tensorDevice();
const x = device.upload([2, 4, 8], new Float32Array(64).fill(0.1));
const target = device.upload([2, 4, 8], new Float32Array(64));
const biases = new st.WgpuAttentionBiases();
const lossFn = new st.MeanSquaredError();
const forward = model.forward(x, biases);
const prediction = forward.predictionTensor();
const loss = lossFn.evaluateResident(prediction, target);
const seed = loss.predictionGradientTensor();
const gradients = model.backward(forward, seed);
const update = model.sgd(gradients, 0.01);
console.log(await update.read()); // bigint; a rejected update throws.
for (const handle of [update, gradients, seed, loss, prediction, forward,
                      lossFn, biases, target, x, device, model]) handle.free();
~~~

Both compile methods also expose existing `tile_mnk`, `kernel` and
`accumulation` options, with unchanged defaults.

## Ownership and explicit boundaries

- Parameters are four immutable resident handles: fused QKV weight/bias and
  output weight/bias, in that order. `parameter_tensors()` /
  `parameterTensors()` owns snapshots, not writable aliases or host copies.
- Backward requires this owner's latest forward at the current parameter
  revision. Another forward invalidates the old tape, not its retained prediction.
  Repeated cotangents return independently owned gradients.
- `sgd` submits one all-or-none update. `attempted_updates` /
  `attemptedUpdates` and the receipt's attempted revision are **not** acceptance.
  `ResidentParameterUpdate.read()` reads frozen flags and rejects failed updates.
  A rejected numerical attempt advances the revision while preserving all weights.
- Runtime biases are caller-owned `[B,H,T]` and `[B,H,T,T]` tensors.
  Their logical gradients are returned by `z_bias_gradient_tensor()` /
  `zBiasGradientTensor()` and the corresponding pair-bias method. They are not
  silently optimized. Callers also own view/broadcast adjoints.
- Upload and readback stay explicit. Original Modules/plans do not change.
  There is no automatic module synchronization, checkpoint/resume, Adam, dropout,
  KV-cache or complete language model in this client API.

## Reproduce the public-client checks

After installing a freshly built default-feature wheel in an isolated environment:

~~~bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
  python -I -B bindings/st-py/tests/test_nn_attention_training.py -v
~~~

For WASM, build the complete package rather than the standalone Rust probe:

~~~bash
cargo build --locked --release -p spiraltorch-wasm --features webgpu \
  --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm \
  --target web --out-dir target/attention-training-clients-web
node bindings/st-wasm/tests/attention_training_types.cjs \
  target/attention-training-clients-web/spiraltorch_wasm.js
python3 -I -S -m http.server 8770 --bind 127.0.0.1
~~~

Use wasm-bindgen CLI 0.2.129 to match Cargo.lock. Open
`http://127.0.0.1:8770/bindings/st-wasm/tests/resident_attention_training_clients.html?module=/target/attention-training-clients-web/spiraltorch_wasm.js`.
Require `passed: true`, all 30 conditions, 16 MSE/SGD steps and the rejection/
lifetime guards. Download the result before closing the page. CI checks the
native Python client and generated/shipped types; browser execution is a separate
runtime check. These synthetic correctness checks are not performance or language
quality evidence.
