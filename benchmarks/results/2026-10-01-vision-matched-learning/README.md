# Matched Real-Image ConvNeXt Learning

The public Python resident classifier completed matched CIFAR-10 learning
against an independent PyTorch implementation. Model, normalization, CE, VJP
and updates run in Rust/WGPU. This is learning/correctness evidence, not a
throughput benchmark or evidence of a Z-space learning advantage.

## Conditions And Results

All runs use the actual Rust two-stage architecture, not torchvision ConvNeXt:
24 parameter tensors / 5,530 values, shared initial weights, batch 16, plain
SGD at 0.01, f32, normalization, no augmentation or policy intervention.
PyTorch is 2.12.1; SpiralTorch's native adapter is Apple M4 through Metal.
See [the full recipe and replay guide](../../../docs/vision_matched_learning.md).

| Report | Train / evaluation images | Seeds | Epochs | Accepted updates per seed |
| --- | --- | --- | --- | --- |
| `admission-cpu-report.json` | 160 / 80 | 17 | 0 | 0; disposable admission only |
| `pilot-cpu-report.json` | 1,280 / 320 | 17, 29, 43 | 5 | 400 |
| `pilot-mps-report.json` | 1,280 / 320 | 17, 29, 43 | 5 | 400 |
| `expanded-mps-report.json` | 10,000 / 1,600 | 17, 29, 43 | 5 | 3,125 |
| `count-guard-cpu-report.json` | 160 / 80 | 17 | 0 | 0; post-hardening admission |
| `count-guard-mps-report.json` | 160 / 80 | 17 | 0 | 0; post-hardening admission |

Both small pilots finish at 22.8125%, 24.6875%, and 28.125% accuracy for
seeds 17, 29 and 43 respectively, identical between the matched arms.
The expanded MPS comparison completes 9,375 updates per framework:

| Seed | Initial accuracy | Final accuracy, both arms | Rust/WGPU final CE | PyTorch MPS final CE |
| --- | --- | --- | --- | --- |
| 17 | 9.8125% | 35.1875% | 1.747702545 | 1.747702494 |
| 29 | 8.1250% | 34.3750% | 1.764769775 | 1.764769737 |
| 43 | 10.5000% | 37.1875% | 1.701238604 | 1.701238581 |

Every initial/epoch evaluation in the expanded run has equal accuracy;
the largest absolute evaluation CE difference is 5.961e-8. These aggregate
metrics do not prove long-run weight identity. Every weight and gradient is
compared numerically in the disposable admission step only.
The official test subset is evaluated each epoch: it is development evidence,
not an untouched final test for tuning. No full-dataset quality claim is made.

## Evidence And Provenance

The report JSON files are byte-for-byte original records, with all epoch
metrics, admission errors, sample indices, permutation and pixel/label hashes.
Dataset images, raw logs, native binaries and checkpoint payloads stay local.
The expanded run retains six Rust checkpoints (initial/final for each seed);
their hashes and byte counts are public. The early pilots retained checkpoint
hashes but not payloads or exact source/binary hashes. No Torch final weights
were saved. Do not imply otherwise from matched scores.

The expanded run's source hashes match commit
`1001cbfff5d15fc4773c6c637861a0892195a885`. After that measurement, admission
was hardened to reject missing/extra updated parameter tensors rather than
silently truncating `zip`. Historical records contain all 24 roles/5,530
values, checked independently; CPU and MPS admission were repeated after the
guard change. The original learning reports were not rewritten or rerun under
a falsely attributed source version.

`expanded-mps-verification.json` records independent coverage checks plus
local verification of all six checkpoint files, the measured source revision
and the installed native binary. This validates records and byte identity,
not execution by itself. Recheck public coverage offline, without ML imports:

```bash
python -I tools/test_verify_vision_matched_learning.py
python -I tools/verify_vision_matched_learning.py \
  benchmarks/results/2026-10-01-vision-matched-learning/expanded-mps-report.json
```

For the local artifact checks, add `--checkpoint-dir "$CHECKPOINT_DIR"`,
`--source-ref 1001cbfff5d15fc4773c6c637861a0892195a885`, and
`--native-binary "$NATIVE_BINARY"`. SHA256SUMS covers the public reports and
verification record. Negative tests reject missing roles, samples, seeds,
epochs, wrong update counts, non-finite metrics, admission-bound failures and
altered learning-rate, batch, preprocessing or dataset recipe fields.

## Remaining Gates

Each learning step explicitly reads acceptance/loss; the frameworks run
sequentially in one process. Timing this harness does not establish speed or
peak-memory comparisons. Browser real-image learning, transfer-inclusive
performance, full input/trainer-state restart, and policy-on/off quality
remain open. Model checkpoints alone do not close the restart gate.
