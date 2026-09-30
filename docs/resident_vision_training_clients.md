# Resident Vision Training Clients

Python and browser WebGPU wrap the same Rust `ResidentConvNeXtClassifier`.
They do not reimplement the model, loss, VJP, SGD decision, or checkpoint format.
The classifier owns both backbone and head; there is no separate head optimizer.

## Python

Requires a wheel built with `nn,wgpu` (both are default features). This is a
small synthetic example, not a quality benchmark or pretrained model:

```python
import json
import spiraltorch as st

kind = st.vision.ResidentConvNeXtClassifier
config = json.loads(kind.default_config_json())
config.update(input_channels=1, input_hw=[8, 8], stage_dims=[2, 4],
              stage_depths=[1, 1], patch_size=[2, 2], epsilon=1e-3)
device = st.WgpuTensorDevice.create()
model = kind.create(device, json.dumps(config), num_classes=2, batch_size=2, seed=17)

pipeline = st.TransformPipeline(seed=77)
pipeline.add_normalize([0.5], [0.25])
pipeline.add_resize(8, 8)
pipeline.enable_wgpu()
images = [st.ImageTensor(1, 12, 14,
          [((i * 31 + n * 17) % 257) / 256 for i in range(168)]) for n in range(2)]
x = pipeline.apply_resident_batch(images, device)
y = device.upload([2, 1], [0.0, 1.0])
objective = st.nn.CrossEntropyWithLogits()
updates = []
for _ in range(4):
    forward = model.forward(x)
    loss = objective.evaluate_resident(forward.prediction_tensor(), y)
    gradients = model.backward(forward, loss.prediction_gradient_tensor())
    updates.append(model.sgd(gradients, 0.01))

# Explicit observation, outside the resident loop. Rejection raises an exception.
print([update.read() for update in updates])
payload = model.checkpoint_snapshot().read_json()
resumed = kind.from_checkpoint_json(device, payload)
host_model = kind.host_from_checkpoint_json(payload)  # Ordinary VisionModel inference.
```

The [resident DataLoader](resident_vision_input.md) supplies the same inputs via
`batch.images()` and `batch.upload_targets()`. The compiled model has a fixed
NCHW batch shape; an incompatible tail batch is rejected, not dropped or padded
silently. The input pipeline and caller choose how to handle that tail.

## JavaScript

Build `spiraltorch-wasm` with `webgpu`. After module initialization, import
`ResidentConvNeXtClassifier`, `WgpuTensorDevice`, `VisionTransformPipeline` and
`CrossEntropyWithLogits` from the generated module. The config JSON is exactly
the Rust/Python `ConvNeXtConfig` record, including all fields.

```javascript
const device = await WgpuTensorDevice.create();
const config = JSON.parse(ResidentConvNeXtClassifier.defaultConfigJson());
Object.assign(config, {input_channels: 1, input_hw: [8, 8], stage_dims: [2, 4],
  stage_depths: [1, 1], patch_size: [2, 2], epsilon: 0.001});
const model = ResidentConvNeXtClassifier.create(device, JSON.stringify(config), 2, 2, 17n);
const pipeline = await VisionTransformPipeline.createGpu(77);
pipeline.addNormalize(new Float32Array([0.5]), new Float32Array([0.25]));
const x = pipeline.applyResidentBatch(2, 1, 8, 8, new Float32Array(128).fill(0.75));
const y = device.upload([2, 1], new Float32Array([0, 1]));
const objective = new CrossEntropyWithLogits("mean", -100n, 0);
const forward = model.forward(x);
const prediction = forward.predictionTensor();
const loss = objective.evaluateResident(prediction, y);
const cotangent = loss.predictionGradientTensor();
const gradients = model.backward(forward, cotangent);
const update = model.sgd(gradients, 0.01);
console.log(await update.read());  // Explicit acceptance readback, returns bigint.
const snapshot = model.checkpointSnapshot();
const payload = await snapshot.readJson();
const resumed = ResidentConvNeXtClassifier.fromCheckpointJson(device, payload);
// Release WASM handles when no longer needed; retained handles own their buffers.
for (const handle of [snapshot, update, gradients, cotangent, loss, prediction, forward]) handle.free();
```

Readbacks are asynchronous in browsers. Seeds and attempted revisions are u64
values represented as `bigint`; shape/class counts must be unsigned integers.
Device creation is explicit, and a model can return its device with `tensorDevice()`.

## Ownership And Restart

- `forward` returns an owning prediction and versioned token. Only the latest
  successful forward of that model is valid for `backward`.
- `backward` consumes an explicit resident loss cotangent. It exposes input and
  parameter gradients without an implicit update or host transfer.
- `sgd` returns a frozen update handle without mapping. `attempted_updates` /
  `attemptedUpdates` counts attempts, including numerical rejections, not successes.
  Only explicit `update.read()` confirms that every parameter was accepted.
- Rejected updates retain all old weight values but invalidate old derivative
  tokens. Submit a fresh forward/backward for a valid retry.
- `parameter_tensors()` / `parameterTensors()` returns immutable-version handles;
  retained predictions, weights, receipts and checkpoints survive later updates
  or destruction of their model owner.
- `checkpoint_snapshot()` / `checkpointSnapshot()` captures weights and model
  metadata immediately. Its one-shot JSON read maps that frozen capture even if
  training has continued. Restore creates a fresh owner, never reviving old tokens.
- Checkpoints contain plain-SGD model state and attempted revision, not the
  DataLoader cursor, augmentation RNG, rate schedule or ModuleTrainer policies.
  Those are still caller-owned. Cross-runtime numerical equivalence is distinct
  from within-runtime bitwise continuation.

Native resident pooling synchronizes its backward-plan cache rather than making
the Python model thread-affine. Python borrowing still rejects overlapping mutable
calls on one model; use explicit orchestration rather than racing training steps.

## Replay The Public Client Checks

With a freshly built and installed `nn,wgpu` wheel, run:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -m unittest discover \
  -s bindings/st-py/tests -p test_vision_resident_training.py
python -I tools/test_vision_classifier_handoff.py
```

Build the `webgpu` WASM package and generate web bindings as described in the
[WASM guide](../bindings/st-wasm/README.md). With Playwright available to Node,
run the real-browser fixture and then replay its checkpoint in Python:

```bash
node tools/test_resident_browser.cjs "$WASM_WEB_DIR" "$CHROME" \
  "$BROWSER_REPORT" '' '' '' '' convnext-classifier-clients
node bindings/st-wasm/tests/resident_types.cjs "$WASM_WEB_DIR/spiraltorch_wasm.js"
python -I tools/verify_vision_classifier_handoff.py \
  --browser-report "$BROWSER_REPORT" --output "$HANDOFF_REPORT"
```

Choose new output paths; these commands do not overwrite evidence. The browser
checks multiple updates, frozen snapshots, bitwise within-runtime restart,
invalid-label rejection preserving every weight, valid retry and stale/foreign
token rejection. The Python replay loads the browser checkpoint at revision 6,
checks host and resident inference, and compares one additional CE/VJP/SGD step
with the browser's revision 7, including every parameter. Cross-runtime checks
use `abs(actual - expected) / (1 + abs(expected)) < 2e-4`, not a general bitwise
promise. Both consume the recorded normalized input: no data/RNG replay is implied.

The [bounded result](../benchmarks/results/2026-10-01-vision-training-clients/README.md)
records the measured errors, validation scope and remaining gates.
