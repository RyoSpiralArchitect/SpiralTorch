# Matched Vision Learning

`tools/run_vision_matched_learning.py` compares the public Python resident
classifier with an independent eager PyTorch reference on real CIFAR-10 images.
SpiralTorch's model, normalization, loss, VJP and updates execute in Rust/WGPU;
the Python harness selects identical samples and orchestrates the two arms.
This is a correctness/learning harness, **not a throughput benchmark**.

## Match The Architecture, Not The Name

The reference in `tools/vision_convnext_torch_reference.py` is not torchvision's
ConvNeXt. It matches this repository's actual Rust model:

- NCHW patch convolution, channel-wise 7x7 depthwise blocks, per-token affine
  LayerNorm, two Linear layers with tanh-approximate GELU and a residual branch.
- A plain stride-2 convolution between stages.
- A final affine LayerNorm over flattened NCHW features, then spatial average
  pooling that keeps channels separate, then a Linear classification head.
- The effective f32 LayerNorm epsilon includes the Rust curvature factor
  `epsilon * (1 + 0.1 * sqrt(-curvature))`.

Both arms load the **same Rust-created checkpoint**, including every named
parameter; matching random seeds across unrelated initializers is not sufficient.
The fixed recipe uses 3x32x32 images, stage dimensions `[8,16]`, depths `[1,1]`,
patch `[4,4]`, ten classes, curvature `-1`, epsilon `0.001`, mean CE and plain
SGD at `0.01`. There is no pretrained model, augmentation, momentum, weight
decay, mixed precision or Z-space policy ablation in this slice.

## Replay

Use a wheel built from this source with `nn,wgpu` and an environment containing
NumPy, PyTorch and torchvision. The recorded environment uses PyTorch 2.12.1
and torchvision 0.27.1. The dataset is obtained through torchvision's official
CIFAR-10 loader, with its archive and member integrity checks. Download is
opt-in; subsequent runs should omit `--download`.

```bash
python -I tools/test_vision_matched_learning.py

python -I tools/run_vision_matched_learning.py \
  --data-root "$CIFAR_ROOT" --download --torch-device cpu \
  --seeds 17 29 43 --train-per-class 128 --test-per-class 32 \
  --batch-size 16 --epochs 5 --output "$CPU_REPORT"

PYTORCH_ENABLE_MPS_FALLBACK=0 python -I tools/run_vision_matched_learning.py \
  --data-root "$CIFAR_ROOT" --torch-device mps \
  --seeds 17 29 43 --train-per-class 128 --test-per-class 32 \
  --batch-size 16 --epochs 5 --output "$MPS_REPORT"
```

MPS requires a real available device and explicit disabled fallback. Native
WGPU rejects CPU fallback adapters. PyTorch uses f32, one CPU thread, highest
matmul precision and deterministic algorithms. `--epochs 0` performs admission
and initial evaluation only; it must not be reported as a learning run.

For the larger matched run, use `--train-per-class 1000 --test-per-class 160`
with the same remaining recipe. Counts must be exactly divisible by batch size;
there is no silent tail dropping or padding. Start with new report paths:
reports and the sibling `.checkpoints` directory never overwrite existing data.

## What Is Checked

Before learning, each pair executes one disposable update on real images and
compares normalized input, logits, CE, input VJP, **every named parameter VJP**,
every updated weight and the next prediction. The bound is
`abs(actual - expected) / (1 + abs(expected)) < 2e-4`. Missing or non-finite
values and mismatched roles/counts fail admission. Both models are then restored
to the untouched initial checkpoint, so admission is not an extra training step.

Training uses the same per-epoch index permutation and updates in both arms.
Every SpiralTorch update is explicitly accepted before continuing. Reports retain
all epoch losses/accuracies, accepted counts, permutation hashes, sample indices,
pixel/label hashes and per-parameter admission errors. Initial and final Rust
checkpoints are saved locally, with byte counts/hashes in the report. The harness
also records its source hashes and the installed native binding's binary hash.

The held-out subset comes from the official test split and is inspected every
epoch. Treat it as a development comparison, not an untouched final test set for
hyperparameter selection. No sample is silently mixed into the training split.
Metric scoring is shared CPU code applied to each model's own logits; model
updates are independent after the shared initialization.

## Interpretation And Next Gates

Agreement shows that this Rust/WGPU execution path can reproduce a matched
PyTorch learning trajectory on the recorded conditions. It does **not** show
that Z-space improves learning, that standard torchvision ConvNeXt is equivalent,
that full-dataset quality is adequate, or that WGPU is faster.

Admission maps all gradients, and the training loop maps acceptance/loss each
step. The two frameworks run sequentially in one process, so this harness must
not be used to infer transfer-inclusive throughput or peak memory. A separate
timing protocol must equalize observation boundaries, warm up both arms, alternate
their order, synchronize completion, and report native/browser environments
separately. Browser real-image learning, full training-state restart and
optimizer-policy ablations remain open gates.

Dataset source: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), described
in Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*
(2009). Dataset files and model checkpoint payloads are kept local; results and
verification metadata can be published without redistributing the images.
