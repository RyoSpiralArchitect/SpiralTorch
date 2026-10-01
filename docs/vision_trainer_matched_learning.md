# Real-Image Learning Through The Shared Trainer

`tools/run_vision_trainer_matched_learning.py` carries the matched CIFAR-10
comparison into the public Rust-owned `ResidentVisionTrainer`. The older
[classifier-only comparison](vision_matched_learning.md) remains unchanged.

## Ownership And Independence

Rust owns the model, shuffled input order/cursor, horizontal-flip RNG,
normalization, learning-rate schedule, accepted-update clock and combined
checkpoint. Python supplies an immutable in-memory dataset and explicitly
observes submissions; it does not reconstruct the sampler or scheduler.

The independent PyTorch model reuses the established eager reference and loads
the same initial Rust weights. Every training input is checked against the
original CIFAR pixels with independent f32 normalization, permitting only an
entire-image horizontal flip when enabled. Torch receives that independently
reconstructed input, the labels selected by Rust and the submitted learning
rate. This compares model learning under shared input/control decisions, not
independent sampler/scheduler implementations or torchvision's ConvNeXt.

The architecture stays at two stages `[8,16]`, depths `[1,1]`, patches `[4,4]`,
ten classes and 5,530 f32 values. Both arms use plain SGD; the optional shared
warmup/cosine schedule is not a Z-space policy intervention. A disposable
admission step first compares normalization, prediction, CE, input VJP, every
parameter VJP, every updated weight and next prediction. It never advances the
training owner or the measured Torch model.

## Actual Process Restart

Each seed launches three separate isolated Python processes, sequentially:

1. `control`: train the shared trainer and the independent Torch model to the
   requested end, evaluate each epoch, and capture initial/split/final states.
2. `prefix`: create a fresh trainer, run only to the split and exit normally.
3. `resume`: recreate dataset/pipeline/device, restore the prefix checkpoint
   through the public trainer API and run to the same end.

The checker compares every step's sample IDs, whole-batch normalized-pixel hash,
observed flip choices, learning-rate bits, loss bits, acceptance and clocks.
Every full control epoch must visit every selected training sample exactly once.
All initial/prefix/restore/final checkpoint bytes are checked, including their
local file hashes, not only final scores. Final weights from the matched control
are compared with every independent Torch parameter at the existing scaled f32
bound `2e-4`. The exact resumed checkpoint therefore also matches that control.

This is clean-process continuation on one recorded native adapter, not a
power-loss durability, cross-version or cross-device bitwise guarantee. The
Torch reference runs the uninterrupted control; this does not test restoring a
Torch optimizer. The Rust checkpoint is never used to overwrite trained Torch
weights and hide drift.

## Run

Use a current `nn,wgpu` wheel with NumPy, Torch and torchvision. The dataset must
already exist at `CIFAR_ROOT`; this runner never downloads it. Use the opt-in
download command in the classifier guide if needed. Data selection and integrity
checks reuse that guide's official torchvision CIFAR-10 loader.

```bash
python -I tools/test_vision_trainer_matched_learning.py
python -I tools/test_vision_matched_learning.py

PYTORCH_ENABLE_MPS_FALLBACK=0 python -I tools/run_vision_trainer_matched_learning.py \
  --data-root "$CIFAR_ROOT" --output "$NEW_FIXED_RUN_DIR" \
  --torch-device mps --seeds 17 29 43 --train-per-class 128 \
  --test-per-class 32 --batch-size 16 --epochs 5 --restart-at 37

PYTORCH_ENABLE_MPS_FALLBACK=0 python -I tools/run_vision_trainer_matched_learning.py \
  --data-root "$CIFAR_ROOT" --output "$NEW_AUGMENTED_RUN_DIR" \
  --torch-device mps --seeds 17 29 43 --train-per-class 128 \
  --test-per-class 32 --batch-size 16 --epochs 5 --restart-at 37 \
  --horizontal-flip --schedule cosine
```

`--torch-device cpu` selects the independent CPU reference, while SpiralTorch
still requires a real native WGPU adapter. MPS fallback is rejected. The fixed
case uses rate `0.01`; cosine warms up for ten accepted updates, then decays to
one tenth of that rate over the requested trajectory. No tail batches are
dropped. Outputs are exclusive: reusing an existing run directory is an error.

Each local seed directory retains its recipe, raw phase records and combined
checkpoints. `summary.json` excludes machine paths and process IDs while keeping
source/native-binary hashes, environment and input identity, all epoch scores,
parameter comparisons, raw-record/checkpoint hashes and replay results. Retain
the local originals; publish the summary, verification and method without
redistributing dataset images. The official test subset is evaluated every
epoch and is development evidence, not an untouched final test.

## Scope

CI runs the stdlib-only replay checker's negative tests, not these real-data GPU
jobs. The comparison maps every input and loss and settles every update. The
frameworks run sequentially, and evaluation explicitly snapshots the model.
It must not be used to claim throughput, peak-memory reduction, asynchronous
overlap, browser real-image support or Z-space learning advantage. Those need
separate matched measurements; do not remove observations and silently relabel
this correctness protocol as a speed benchmark.
