# Resident Vision Training Timing

`tools/bench_vision_trainer_vs_torch.py` measures the Rust-owned resident trainer
through the public Python client against the existing independent eager Torch
reference. This is the small SpiralTorch ConvNeXt architecture, not torchvision
ConvNeXt or a comparison with an optimized/compiled production Torch model.

## Measurement Boundary

Each worker uses cached CIFAR-10 only: 1,280 training and 320 unused development
images. Both frameworks start from the same seeded checkpoint and use its first
shuffled batches. The interval cannot cross an epoch boundary. Raw float32 CHW
images have already been decoded and divided by 255 on the host. Timings include
batch selection, image/target upload, normalization, forward, CE, backward, SGD,
and observed update completion. Dataset construction, model creation, warmup,
admission reads and final checkpoint/parameter reads are outside the interval.
Device pipelines are warmed, but the timed model is a freshly restored owner;
this is not an indefinitely hot, steady-state run. No augmentation is used.

Rust retains all-parameter guarded acceptance. Torch uses eager SGD with
`foreach=False, fused=False` and explicit MPS synchronization after every update;
it does **not** implement the same rejection transaction. Feedback mode enables
the real Rust default loss gate with an identity proposal, which must preserve
the constant rate. Torch observes its scalar loss but does not reimplement the
gate. Thus this tests observation overhead, not a useful Z-space policy.

Disposable admission compares normalization, prediction, loss, input/parameter
gradients, updates and next prediction. After every measured interval, all
parameters and the last loss must match Torch within the existing scaled f32
bound. Baseline/candidate pairs additionally require exactly identical final
Rust model/input/trainer checkpoints, not just close losses. Repetition order
alternates runtime and framework ordering; each worker is a separate process.
Do not run other GPU workloads or builds concurrently with measured intervals.

`--profile` adds per-step host phase clocks and is diagnostic only. The
settlement phase includes GPU computation waiting, not just mapping overhead.
The verifier rejects mixing instrumented and ordinary intervals. The tool does
not establish a peak-memory, browser, full-dataset quality or universal speed
claim. GPU memory and epoch-boundary/input-stream scaling remain open gates.

## Run

Use two separately retained wheel environments with the same Python/Torch
dependencies. The tool records native binary and harness hashes; retain their
build source/feature receipts alongside the results. It never downloads data.

```bash
python -I tools/bench_vision_trainer_vs_torch.py \
  --data-root "$CIFAR_ROOT" --output "$NEW_RESULT_DIR" \
  --baseline-python "$BASELINE_PYTHON" --candidate-python "$CANDIDATE_PYTHON" \
  --seeds 17 29 43 --batches 1 16 64 --steps 16 --warmup 3 --repeats 5
python -I tools/test_vision_training_timing.py
```

Omit the candidate interpreter for an initial baseline or diagnostic profile.
All output directories and files are exclusive-create. Keep images and raw
weights local; publish all conditions, timings, verification and provenance
hashes rather than selecting favorable runs or republishing the dataset.
