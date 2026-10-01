# Real-Image Loss-Gate Ablation

This extends the [shared-trainer comparison](vision_trainer_matched_learning.md)
without adding a Python policy. The same Rust model, sampler, plain SGD owner,
rate-control projection and optional loss-feedback gate perform every update.
The initial test isolates one prescribed rate proposal and the existing gate.
It does not measure geometric parameter updates or an adaptive proposal producer.

The [completed three-seed result](../benchmarks/results/2026-10-01-vision-feedback-ablation/README.md)
has 400 updates per arm and exact fresh-process continuation for all 12 arms.
The gate is active but does not establish an advantage over nominal SGD or the
integrated-rate-matched control. Keep this negative result as the control for
future Rust observation/policy changes, rather than tuning until one seed wins.

## Four Matched Arms

| Arm | Nominal rate | Proposal | Rust loss gate |
| --- | --- | --- | --- |
| `baseline` | 0.01 | None | Disabled |
| `fixed_proposal` | 0.01 | Fixed 0.5 multiplier | Disabled |
| `loss_feedback` | 0.01 | The same fixed 0.5 multiplier | Core defaults |
| `dose_matched` | Mean of that seed's realized feedback rates | None | Disabled |

The last arm is a retrospective mechanism control: it uses information from the
completed feedback trajectory and is not an online/deployable policy. Its total
learning rate matches the feedback arm within f32 rounding, not necessarily
bit-for-bit. The actual sum and discrepancy are recorded. Comparing it with
feedback tests timing effects, rather than mistaking a lower total rate for a
benefit of the gate. These controls do not replace an independently tuned SGD
baseline or establish a generally better optimizer.

The control report comes from the public Rust meta-optimizer API, using an
explicitly prescribed gradient/hint, not a gradient measured from the classifier.
It is applied once before the initial checkpoint. Resume restores the consumer
and gate from that checkpoint; it never resets or reapplies the proposal.

## Protocol

Each seed uses the same initial weights, CIFAR-10 subset, image order and
normalization in every arm. Augmentation and cosine scheduling are disabled
here to isolate rate control. Each arm independently runs uninterrupted,
prefix and resumed processes, using the established all-batch/all-checkpoint
verification. PyTorch trains its own weights and gradients under the same
Rust-selected batches and applied rates. It is a numerical learning reference,
not a separate implementation of the feedback policy.

All gate observations, rates and pre-update losses are retained. The native
model checkpoint, input checkpoint and initial scores must agree across arms;
each batch identity, normalized-input hash and acceptance decision must match.
Saved Torch weights are compared independently in every arm. An inactive gate
is a valid experimental finding, not a reason to tune settings until it wins.

Use a fresh `nn,wgpu` wheel containing the
[CE rounding correction](../benchmarks/results/2026-10-01-vision-ce-rounding/README.md),
NumPy, Torch and torchvision. Dataset downloads are never automatic:

```bash
python -I tools/test_vision_trainer_matched_learning.py
python -I tools/test_vision_feedback_ablation.py
PYTORCH_ENABLE_MPS_FALLBACK=0 python -I tools/run_vision_feedback_ablation.py \
  --data-root "$CIFAR_ROOT" --output "$NEW_ABLATION_DIR" \
  --torch-device mps --seeds 17 29 43 --train-per-class 128 \
  --test-per-class 32 --batch-size 16 --epochs 5 --restart-at 37
```

After completion, independently recheck retained records without importing
NumPy, Torch or SpiralTorch. `SOURCE_REVISION` identifies the measured runner:

```bash
python -I tools/run_vision_feedback_ablation.py --verify "$NEW_ABLATION_DIR" \
  --source-ref "$SOURCE_REVISION" --output "$NEW_VERIFICATION_JSON"
```

Every output is exclusive; failed or earlier runs are not overwritten. Keep
images and raw checkpoints local, and publish summaries, verification and hashes.
Per-epoch official test-subset evaluation is development evidence, not an
untouched final test. Per-step readbacks and sequential frameworks rule out
throughput or peak-memory claims. The gate guards the external proposal: closing
it returns to the nominal rate, which increases the rate for a proposal below one.
