# Real-Image Shared-Trainer Learning And Restart

The public Rust-owned `ResidentVisionTrainer` now completes real CIFAR-10
learning against an independent PyTorch model and resumes in a fresh native
process without changing any recorded batch/update or final bound checkpoint.
This is correctness/learning evidence, not a speed or Z-space advantage claim.

## Conditions

The measured adapter is Apple M4 / Metal. The two-stage Rust architecture has
24 parameter tensors / 5,530 f32 values, stages `[8,16]`, depths `[1,1]`, patches
`[4,4]` and ten classes. Batch size is 16. Torch loads the same initial weights
but computes its own model, gradients and updates. Rust owns sampling, flips,
normalization and rates; Torch receives independently reconstructed versions of
those selected inputs. This is not torchvision ConvNeXt or an independent
sampler/scheduler comparison. MPS fallback was explicitly disabled.

| Original report | Train / evaluation | Seeds | Epochs | Updates per matched arm / seed | Restart |
| --- | --- | --- | --- | --- | --- |
| `smoke-cpu.json` | 160 / 80 | 17 | 1 | 10 | 3 / 7 |
| `fixed-mps-initial.json` | 1,280 / 320 | 17,29,43 | 5 | 400 | 37 / 363 |
| `fixed-mps-retained.json` | 1,280 / 320 | 17,29,43 | 5 | 400 | 37 / 363 |
| `flip-cosine-mps-retained.json` | 1,280 / 320 | 17,29,43 | 5 | 400 | 37 / 363 |

Every row passed. The initial probes precede independent Torch-weight retention;
their original reports are preserved, not rewritten as if those weights existed.
The retained fixed run repeats the initial MPS condition, not three new seeds.
The two retained conditions are the six primary matched trajectories: 2,400
accepted updates per framework, plus 2,400 Rust prefix/resume updates. Disposable
admission updates are separate and never advance the measured training owners.

## Results

Final development accuracy is identical between SpiralTorch and Torch in every
row below. All initial and per-epoch accuracy comparisons also match.

| Seed | Initial accuracy | Fixed SGD | Flip + cosine |
| --- | --- | --- | --- |
| 17 | 10.625% | 21.5625% | 20.3125% |
| 29 | 8.4375% | 24.0625% | 19.6875% |
| 43 | 8.75% | 29.6875% | 24.6875% |

The largest absolute evaluation CE difference across the retained conditions is
`5.961e-8`. Independent comparisons of both saved final-weight artifacts cover
every value; their largest scaled error is `1.700e-7`, below the declared
`2e-4` bound. None of these cross-framework results assert bitwise equivalence.

For native restart, each seed runs control, prefix and resume in three distinct
processes. All 400 sample-ID batches, transformed-image hashes, observed flips,
rate/loss bits, acceptance decisions and clocks match exactly. Initial, split,
restored and final combined checkpoint bytes match exactly. Every epoch covers
the selected training IDs once. Both retained conditions have zero rejections;
the separate synthetic client fixture covers rejection semantics.

Augmentation is not merely configured: the flip runs observe 3,182 / 3,166 /
3,204 flipped images out of 6,400 presentations for seeds 17 / 29 / 43. Cosine
produces 400 distinct recorded rate values; fixed SGD produces one. The
flip-plus-cosine condition has lower final accuracy here. Because two factors
change together, this is a wiring/restart stress test, not a causal attribution
to either augmentation or scheduling, and not evidence of an improvement.

## Provenance And Verification

Runtime versions are Python 3.12.6, NumPy 2.5.3, Torch 2.12.1, torchvision 0.27.1,
and a source-built SpiralTorch 0.4.27 `nn,wgpu` wheel. The wheel is the portable
client build recorded in [the client result](../2026-10-01-vision-trainer-clients/README.md),
whose Rust source is captured by `7cacf291c18c883f539690b9514cbbf6eaf52220`.
The initial runner source is captured by `c069783b285462fc4cf44ab2edddd3a790c2d0cd`;
the retained runs use `2d767c0c2169fb78788ccf3f22bb8eedc409b28c`. Reports contain
the actual runner/reference source hashes and installed native-binary hash.

Report JSONs are byte-for-byte original summaries, containing selected sample
IDs, input/configuration identity, every epoch score, per-parameter admission
and final comparisons, and phase/checkpoint hashes. Dataset images, raw phase
records and both frameworks' final weights stay local. The two `verification`
records independently recheck those raw files, complete trajectories and saved
weight values. SHA256SUMS covers the public reports and verification records.

See [the execution recipe](../../../docs/vision_trainer_matched_learning.md).
With retained local originals, revalidate without NumPy/Torch/GPU imports:

```bash
python -I tools/test_vision_trainer_matched_learning.py
python -I tools/verify_vision_trainer_replay.py "$RETAINED_RUN_DIR" \
  --source-ref 2d767c0c2169fb78788ccf3f22bb8eedc409b28c \
  --output "$NEW_VERIFICATION_FILE"
```

The test split is inspected every epoch and is development evidence, not an
untouched final test. Each step maps input/loss and settles acceptance; the
frameworks run sequentially. Do not infer throughput, peak memory, browser
real-image execution, power-loss durability or cross-device replay from this
result. Shared resident optimizer/Z-space policy and a separately designed
transfer-inclusive benchmark remain the next work.
