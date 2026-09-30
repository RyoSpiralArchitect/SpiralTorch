# Resident Vision Input

`st-vision` owns image preparation for Rust, Python and browser WebGPU. A
homogeneous image batch uploads once, then normalization, resize, crop and
sampled flips operate on the same resident NCHW tensor consumed by NN kernels.
There is no implicit image readback between these stages.

## Contract

- Normalize evaluates checked `(x - mean) / std`, not a precomputed reciprocal.
  Statistics must be finite; std must be positive. Supply one mean/std pair
  for broadcasting, or exactly one per image channel. Incorrect lengths no
  longer repeat the last statistic silently.
- Normalize may appear before, after or between geometry operations. The
  fused pointwise plan and resident statistics are cached per layout/device.
  Subtraction preserves f32 rounding; extreme division uses integer
  significands to preserve subnormal operands/results without shader-f64.
- Geometry inherits the upstream validity guard and checks every stage. A
  later crop or ReLU cannot hide an invalid normalization outside its output.
- Flips consume random choices in image-major order, matching repeated CPU
  application. Homogeneous NCHW batches and strided resident inputs are
  supported. Mixed image sizes and resident ColorJitter fail explicitly.
- These operations transform images only. Boxes/masks remain caller-owned
  annotation metadata, not silently transformed detection/segmentation targets.
- `TransformDispatcher::from_runtime` reuses an existing device/queue and
  embedded shaders. The default native route also uses embedded shaders, so
  an installed wheel does not require the build checkout's WGSL files.

## Python

Requires a WGPU-enabled wheel. Normalization is also available on CPU through
the same `TransformPipeline`, without enabling a GPU dispatcher.

```python
import spiraltorch as st

dataset = st.TensorVisionDataset("CIFAR10")
for label in range(2):
    values = [((i * 7 + label * 11) % 53) / 52 for i in range(3 * 16 * 16)]
    dataset.push(st.ImageTensor(3, 16, 16, values),
                 target=st.Tensor(1, 1, [float(label)]), label=str(label))

pipeline = st.TransformPipeline(seed=19)
pipeline.add_normalize([0.5], [0.25])
pipeline.add_horizontal_flip(0.5)
pipeline.add_center_crop(12, 12)
pipeline.enable_wgpu()
device = st.WgpuTensorDevice.create()
loader = dataset.dataloader(2, seed=7, pipeline=pipeline)
batch = loader.next_resident_batch(device)
x = batch.images()          # WgpuTensor [2, 3, 12, 12], no readback
y = batch.upload_targets()  # WgpuTensor [2, 1], explicit target upload
print(x.shape, y.shape, batch.labels())
```

`ResidentVisionBatch` retains target/annotation metadata but not the source
host image copies. `upload_targets()` requires every target to have the same
`(1, K)` shape; it never derives class IDs from string labels. `None` denotes
loader exhaustion, as in `next_batch()`.

`pipeline.apply_resident(image, device)` returns CHW;
`apply_resident_batch(images, device)` returns NCHW;
`apply_from_resident(x)` continues an existing NCHW tensor without uploading.
The ordinary `apply(image)` returns a host image and still performs Normalize
on CPU. Enabling WGPU is not a promise that this host-returning API stays resident.
The [resident classifier client](resident_vision_training_clients.md) consumes
these tensors through the same Rust model; it does not reimplement training in Python.

## Browser

Build `spiraltorch-wasm` with `webgpu`. The hand-written TypeScript declarations
and wasm-bindgen surface expose the same Rust pipeline:

```javascript
const pipeline = await VisionTransformPipeline.createGpu(19);
pipeline.addNormalize(new Float32Array([0.5]), new Float32Array([0.25]));
pipeline.addRandomHorizontalFlip(0.5);
pipeline.addCenterCrop(12, 12);
const pixels = new Float32Array(2 * 3 * 16 * 16).fill(0.5);
const x = pipeline.applyResidentBatch(2, 3, 16, 16, pixels);
// Pass x into resident NN operations; map only an explicit output snapshot.
```

`applyFromResident(x)` continues a resident batch. `apply(...)` explicitly
returns a host `VisionImage` asynchronously; `createCpu(seed)` is the Rust CPU
reference. Browser callers supply their data batches; there is no JS DataLoader
binding. The separate [classifier client](resident_vision_training_clients.md)
exposes model training and asynchronous checkpoint mapping.

## Failure And Restart Boundaries

CPU `apply` commits neither the image nor RNG on failure, and CPU `next_batch`
commits neither cursor nor transform RNG when a later sample fails. Resident
submission validates shapes/statistics/supported operations before committing
RNG or cursor, but it does not read back GPU validity. **A successfully submitted
batch advances cursor/RNG even if a downstream GPU guard rejects that batch.**
Its loss/update receipt determines numerical acceptance, not the returned
tensor handle. Rust/browser `apply_gpu_async`/`apply` wait for validity and keep
image/RNG unchanged on mapping/validation failure.

Model checkpoints still do not capture data cursor, augmentation RNG or trainer
policy. The separate [input checkpoint](vision_input_checkpoint.md) now captures
cursor/order and augmentation RNG; callers must pair it with the model at a
settled boundary and retain trainer policy state. The shared native/browser fixture
checks four normalized DataLoader-to-ConvNeXt CE/VJP/SGD steps against CPU, and
checks that invalid normalization rejects all classifier weights. This is
synthetic correctness evidence, not real-data accuracy or throughput evidence.

The shared pointwise vocabulary includes `subtract` and `divide`, with matching
VJPs. Portable graph plans containing these operations use inference-plan v4;
v1/v2/v3 exports keep their previous meanings and reject a downgraded v4 program.
