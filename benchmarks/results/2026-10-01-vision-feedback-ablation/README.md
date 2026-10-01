# Real-Image Shared Feedback Ablation

The existing Rust loss gate is active and exactly restartable on these runs,
but **does not demonstrate a learning advantage over nominal SGD or an
integrated-rate-matched control**. This is a bounded mechanism experiment, not
a general optimizer, geometric-update, throughput or final-test claim.

## Protocol

Three seeds (17, 29, 43), 1,280 CIFAR-10 training images and 320 development
evaluation images per seed, batch 16, five epochs and 400 accepted updates per
arm. The shared two-stage ConvNeXt has 5,530 f32 parameters in 24 tensors.
Images, starting weights, Rust-selected batch order and normalized input hashes
match across all four arms. There is no augmentation or cosine schedule.

| Arm | Applied rule |
| --- | --- |
| `baseline` | Constant nominal SGD rate 0.01 |
| `fixed_proposal` | A prescribed Rust-projected 0.5 rate multiplier |
| `loss_feedback` | The same proposal guarded by the existing Rust gate, default configuration |
| `dose_matched` | Constant rate equal to the f32-rounded mean realized feedback rate for that seed |

The last arm is retrospective: it is a mechanism control, not an online policy.
The prescribed report does not contain gradients measured from the classifier;
this does not test an adaptive producer or geometric parameter updates.
PyTorch independently computes gradients and updates under the same Rust-selected
inputs and rates; it does not independently implement the feedback policy.

SpiralTorch uses WGPU/Metal and PyTorch uses MPS on the same Apple M4. The
isolated Python 3.12.6 environment has SpiralTorch 0.4.27, NumPy 2.5.3,
PyTorch 2.12.1 and torchvision 0.27.1; implicit MPS fallback is disabled.
Original arm summaries retain the exact architecture, dataset indices/hashes,
normalized-input contract, policy configuration, source and binary hashes.

## Results

Final development accuracy (%); all epoch evaluation accuracies equal the
independent PyTorch reference in every arm:

| Seed | Baseline | Fixed proposal | Feedback | Dose matched |
| --- | ---: | ---: | ---: | ---: |
| 17 | 21.5625 | 19.3750 | 21.8750 | 23.1250 |
| 29 | 24.0625 | 19.0625 | 24.0625 | 24.0625 |
| 43 | 29.6875 | 26.5625 | 29.0625 | 29.3750 |
| Mean | 25.1042 | 21.6667 | 25.0000 | 25.5208 |

Mean final cross entropy is 2.034476 (baseline), 2.102065 (fixed proposal),
2.037988 (feedback) and 2.037596 (dose matched). Per-seed losses and every
epoch are in [the original summary](summary.json), not just selected endpoints.
These three seeds and small repeatedly inspected evaluation subsets are not
an untouched test or a statistical claim of superiority/equivalence.

| Seed | Open observations | Halted observations | Updates with a changed rate | Max gate | Adjacent batch-loss increases |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 152 / 400 | 194 / 400 | 152 / 400 | 0.625 | 194 / 399 |
| 29 | 116 / 400 | 240 / 400 | 115 / 400 | 0.750 | 207 / 399 |
| 43 | 143 / 400 | 201 / 400 | 143 / 400 | 0.625 | 208 / 399 |

Closing the gate disables the external half-rate proposal: it returns toward
nominal SGD rather than stopping training or lowering its rate further. Feedback
recovers much of the fixed proposal's deficit but does not beat the controls.
Epoch-mean training losses fall throughout all five epochs, while roughly half
the adjacent batch losses rise. This motivates examining how Rust separates
observation noise from regression; it does not prove noise caused each closure.
No defaults were tuned using these results.

The retrospective control matches total applied rate within f32 rounding:
the largest absolute rate-sum difference is `9.46e-8`. Its constant rate is
approximately 0.00962344, 0.00972344 and 0.00963906 for seeds 17, 29 and 43.

## Verification And Provenance

Every arm runs uninterrupted, prefix and resume in separate processes, split
at update 37/400. All 400 batch records, rates, losses, complete control/feedback
state and bound checkpoints match uninterrupted execution exactly. All 12 arms
have zero rejected updates. Final independently saved Torch weights also pass:
the largest scaled error across all arms is `1.70e-7`.

The primary source capture is `4fbd71f95e0a126406d53ae9d2afe9b7f7b3cac7`,
with the second-order small-tail CE kernel from `4d5125ef`. Independent offline
[verification](verification.json) recomputes the result from retained raw
records and checks measured runner source hashes. Each `seed-N/ARM/` directory
contains its original summary and verification, copied byte-for-byte.

The earlier complete run used the protected-rounding kernel from `c24743dd`.
Its [summary](prior/summary.json), [verification](prior/verification.json) and
[build comparison](build-comparison.json) are retained. The comparison checks
9,600 individual records and 84 checkpoint files across all 12 arms, finding
identical recorded trajectories, checkpoints and epoch metrics between builds.
This is bounded observed equivalence, not equivalence of the kernels in general.

Both measured builds **precede** the subsequent cubic small-tail CE follow-up
and Rust 1.99 compatibility changes. A separate macOS CI runtime exposed small
loss errors outside the original approximation range. These learning results
do not certify that device or later binaries; see the
[portability follow-up](../../../docs/resident_vision_feedback.md).

All raw records and weights from both complete runs remain local. The
[provenance manifest](provenance.json) hashes all 341 retained files; no images
or checkpoints are republished here. `SHA256SUMS` covers the public files.
Per-update host observations, framework sequencing and checkpoint mapping mean
this is not a training-throughput or peak-memory measurement.

## Reproduce

Use the recorded source capture and build features in `provenance.json`, an
isolated environment, and an already available CIFAR-10 dataset. Downloading is
not automatic. Each output path must be new; old results are never overwritten.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -I tools/run_vision_feedback_ablation.py \
  --data-root "$CIFAR_ROOT" --output "$NEW_ABLATION_DIR" \
  --torch-device mps --seeds 17 29 43 --train-per-class 128 \
  --test-per-class 32 --batch-size 16 --epochs 5 --restart-at 37
python -I tools/run_vision_feedback_ablation.py --verify "$NEW_ABLATION_DIR" \
  --source-ref 4fbd71f95e0a126406d53ae9d2afe9b7f7b3cac7 \
  --output "$NEW_VERIFICATION_JSON"
```

The verifier does not import ML libraries. Review the
[protocol and evidence boundaries](../../../docs/vision_feedback_ablation.md)
before comparing a different dataset, producer, model or learning-rate recipe.
