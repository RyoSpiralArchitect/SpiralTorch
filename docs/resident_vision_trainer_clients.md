# Resident Vision Trainer Clients

Python `spiraltorch.vision.ResidentVisionTrainer` (also exported at the package
root) and browser `ResidentVisionTrainer` are thin clients of the same
`st_vision::resident_trainer::ResidentVisionTrainer`. They own the classifier,
input order/cursor, shuffle/transform RNGs, accepted/rejected counts and schedule
together. They are not another Python/JavaScript implementation of training.
Python requires `nn,wgpu`; WASM requires `webgpu`.

## Ownership And Configuration

Both clients use Rust `ResidentVisionTrainerConfig` JSON. Start with
`default_config_json()` / `defaultConfigJson()` and supply the desired ConvNeXt
architecture, class count, batch size, shuffle and learning-rate settings.
`model_seed` and `shuffle_seed` are **decimal strings**, including values above
JavaScript's safe-integer limit. Numbers, signed strings and leading zeros are
rejected rather than silently rounded. `learning_rate` is either
`{"kind":"constant","rate":0.001}` or `{"kind":"warmup_cosine","state":
{"base_lr":0.002,"min_lr":0.0001,"warmup_steps":10,"total_steps":100,"step":0}}`.
A new owner requires a fresh schedule; resume uses the checkpoint instead.

The clients freeze a copy of the supplied in-memory dataset and clone the
optional pipeline. Subsequent changes to those caller objects cannot change
the active trainer. The common Rust constructor attaches transforms to the
provided tensor device; no separate pipeline device or implicit fallback is
chosen in either binding. CPU-created transform templates are accepted.

The caller supplies the SHA256 identifying the immutable dataset's pixels,
targets and ordering. The API validates the ID format, length and saved ID,
but does not scan or authenticate the data. Reusing an ID for changed contents
violates the caller's contract. Only nonempty, full fixed-size batches are
supported; tails are not silently dropped or padded.

## Python

```python
import hashlib
import json
import struct
import spiraltorch as st

dataset = st.vision.TensorVisionDataset("CIFAR10")
digest = hashlib.sha256()
for index in range(20):
    pixels = [((index * 13 + j * 7) % 101) / 128 for j in range(48)]
    target = index % 2
    digest.update(struct.pack("<49f", *pixels, target))
    dataset.push(st.ImageTensor(3, 4, 4, pixels), target=st.Tensor(1, 1, [target]))

pipeline = st.vision.TransformPipeline(seed=29)
pipeline.add_horizontal_flip(0.5)
pipeline.add_normalize([0.5], [0.25])
config = json.loads(st.ResidentVisionTrainer.default_config_json())
config.update(num_classes=2, batch_size=2, model_seed="43", shuffle_seed="17")
config["model"].update(input_channels=3, input_hw=[4, 4], stage_dims=[2, 3],
                       stage_depths=[1, 1], patch_size=[2, 2], epsilon=0.001)
device = st.WgpuTensorDevice.create()
trainer = st.ResidentVisionTrainer.create(
    device, dataset, digest.hexdigest(), json.dumps(config), pipeline)
submitted = trainer.submit_next()  # images/loss stay resident
outcome = trainer.settle()         # explicitly observe GPU acceptance
assert outcome.accepted
print(submitted.loss_tensor().snapshot().read_values())  # optional mapping
payload = trainer.checkpoint_snapshot().read_json()
resumed = st.ResidentVisionTrainer.from_checkpoint_json(
    device, dataset, digest.hexdigest(), payload, pipeline)
```

`state_json()` returns the Rust observation state, not a mutable control object.
`restore_checkpoint_json(payload)` restores an existing settled owner atomically.
All failed validation leaves the live owner unchanged. Frozen snapshot handles
remain usable after later training or owner release; `read_json()` consumes the
snapshot once. Retained submission image/loss handles do not own training state.

The [Z-space control entry](resident_vision_zspace_control.md) accepts complete
Rust meta-optimizer reports at settled boundaries, without client-side rate math.
Uncontrolled checkpoints stay v1; applied control and its replay clock use v2.

## Browser

```javascript
// dataset is a TensorVisionDataset filled with push(c,h,w,Float32Array,classId,label?).
// configJson is the same Rust configuration JSON; datasetSha256 is caller-owned.
const trainer = ResidentVisionTrainer.createWithPipeline(
  device, dataset, datasetSha256, configJson, pipeline);
const submitted = trainer.submitNext();
const outcome = await trainer.settle();
if (outcome.accepted) {
  const snapshot = trainer.checkpointSnapshot();
  const payload = await snapshot.readJson();
  snapshot.free();
  // Retain payload outside the WASM instance, then recreate dataset/pipeline/device.
  const resumed = ResidentVisionTrainer.fromCheckpointJsonWithPipeline(
    device, dataset, datasetSha256, payload, pipeline);
}
submitted.free();
outcome.free();
```

For an untransformed stream use `create(...)` and `fromCheckpointJson(...)`
without a pipeline. The distinction between no pipeline and an empty pipeline
is preserved in checkpoints. Browser class IDs must be nonnegative u32 values
exactly representable in f32; image dimensions must be nonnegative u32 integers,
and pixel arrays must really be `Float32Array`. Rust validates their shape and
classification range; an out-of-range class causes update rejection at settlement.

`settle()` holds a mutable WASM owner borrow until its Promise resolves. Do not
concurrently reuse or free that owner. Await its result, then inspect state or
submit again. An unresolved update blocks submission, restoration and capture;
submission never means acceptance. Unknown mapping failures remain pending so
settlement can be retried. A numerical rejection consumes the input batch but
preserves every parameter and does not advance the accepted-update schedule.

## Portability And Replay

The scheduler now calls the same pinned Rust `libm::cosf` on native and wasm32.
The first cross-runtime test exposed a one-ULP rate difference from platform
`f32::cos` at accepted step 62 (attempt 69). The shared arithmetic fixes that
without Python/JS rate calculations or loosening the rate-bit comparison.
The warmup/cosine formula is unchanged, but historical native builds can produce
slightly different rates. Old payloads still load; reproducing the historical
bitwise trajectory requires its original build. Do not infer cross-version or
cross-device bitwise weight equivalence from checkpoint portability.

Build and install a fresh wheel, then emit a synthetic native replay fixture:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
SPIRALTORCH_VISION_TRAINER_HANDOFF=/tmp/vision-trainer-python.json \
python -I bindings/st-py/tests/test_vision_trainer_clients.py -v
```

Build `spiraltorch-wasm --release --features webgpu --target wasm32-unknown-unknown`
and run `wasm-bindgen --target web --out-name spiraltorch_wasm` into `MODULE_DIR`.
Then serve the actual generated module, not a mock:

```bash
node bindings/st-wasm/tests/resident_types.cjs MODULE_DIR/spiraltorch_wasm.js
node tools/serve_vision_trainer_fixture.cjs MODULE_DIR \
  /tmp/vision-trainer-python.json /tmp/vision-trainer-browser-new 8768
```

Open `http://127.0.0.1:8768/?schedule=constant&mode=control`, then run `prefix`.
Close that tab before opening `mode=resume` in a new tab/WASM instance. Run
`mode=python` to continue the native prefix in the browser. Repeat these four
phases with `schedule=cosine`. Each page persists its result on the loopback
server and shows either success or the actual failure. The server creates a
new output directory and refuses to overwrite phase records. It never reads
browser storage or launches a browser. Native-process restart is exercised by
the Python test; browser restart is a fresh document/instance, not an OS browser
process crash or a filesystem durability test.

Finally continue the browser checkpoints in native Python:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
SPIRALTORCH_VISION_TRAINER_BROWSER_DIR=/tmp/vision-trainer-browser-new \
python -I bindings/st-py/tests/test_vision_trainer_clients.py \
  Gpu.test_browser_checkpoints_continue_in_native -v
```

All cases use the same small 24-parameter-tensor / 456-value architecture,
20 synthetic images, batch 2, shuffled horizontal flips and normalization.
Each completed 100-attempt trajectory has 90 accepted and 10 rejected updates.
Within-runtime restart checks the entire final checkpoint exactly. Cross-runtime
continuation checks all input/rate bits and clocks exactly, and every weight
with a scaled f32 bound of `2e-4`. This is correctness/restart evidence, not
real-image accuracy, transfer-inclusive throughput, or a Z-space-policy benefit.
CI runs CPU-safe Python surface checks and generated/shipped TypeScript checks;
the actual browser and Python-GPU runs require the explicit procedure above.

The [bounded result and pre-fix failure](../benchmarks/results/2026-10-01-vision-trainer-clients/README.md)
record all measured conditions without turning this fixture into a performance claim.
