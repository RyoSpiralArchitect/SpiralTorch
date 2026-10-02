# Matched Resident Vision Training Throughput

## Result

The 180-worker sweep completed: three seeds, batches 1/16/64, plain and
identity-feedback modes, five repetitions, and baseline/candidate runtimes.
Every worker times both SpiralTorch and the independent eager Torch reference,
for 360 measured intervals. All final parameters and losses pass the existing
scaled f32 comparison; all 90 paired Rust checkpoints are exactly identical.
The independent retained-file verifier checks every one of the 5,530 parameter
values per worker. The largest observed scaled parameter error is
`7.891782990102535e-8`, below the unchanged `2e-4` bound.

The candidate captures update acceptance and the feedback scalar in one
readback batch instead of separate readbacks. This preserves the transaction
and feedback rules, but is **not a demonstrated general throughput win**.
The plain path is unchanged and serves as a timing control.

| Batch | Mode | Baseline examples/s | Candidate examples/s | Torch examples/s | Paired candidate/baseline median (min, max) |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Plain | 130.1 | 134.1 | 373.4 | 1.023 (0.920, 1.067) |
| 1 | Feedback | 126.5 | 129.2 | 380.2 | 1.030 (0.890, 1.193) |
| 16 | Plain | 799.1 | 797.4 | 5937.9 | 0.997 (0.975, 1.073) |
| 16 | Feedback | 784.6 | 798.0 | 5838.0 | 1.019 (0.998, 1.040) |
| 64 | Plain | 1046.7 | 1048.5 | 21836.5 | 1.000 (0.997, 1.009) |
| 64 | Feedback | 1045.0 | 1045.7 | 21767.5 | 1.001 (0.994, 1.010) |

Throughput cells are medians across seeds and repetitions (15 measurements per
Rust variant, 30 for Torch). Ratios are computed within each seed/repetition
pair before aggregation, not by dividing the displayed medians. These are
observed ranges, not confidence intervals. The large-batch gap remains around
21x on this workload. Do not attribute a small timing shift to the changed
readback path when the unchanged control also shifts.

## Boundary

- Apple M4, native WGPU/Metal versus Torch MPS with fallback disabled; Python
  3.12.6, NumPy 2.5.3, Torch 2.12.1 and torchvision 0.27.1. Measured workers
  run with `-I` and record `sitecustomize=false`.
- The shared small ConvNeXt architecture has stage widths 8/16, depths 1/1,
  4x4 patches, and 24 parameter tensors. This is not torchvision ConvNeXt or
  optimized/compiled production Torch.
- Cached CIFAR-10 only, 1,280 training images and 320 unused development images.
  Each interval uses the first 16 shuffled batches from the same seeded
  checkpoint, after three disposable warmup steps; it does not cross an epoch.
- Timed work includes host batch selection, normalization/upload, forward,
  CE, VJP, SGD and completion. Images were already decoded and divided by 255.
  Model creation, admission, warmup and final parameter reads are excluded.
  The timed model owner is freshly restored, not indefinitely steady-state.
- Rust keeps all-parameter guarded acceptance. Torch uses ordinary eager SGD,
  `foreach=False, fused=False`, and an explicit completion synchronization per
  step, not the same rejection transaction. Feedback uses an identity proposal
  with the default Rust gate; Torch only observes loss. No useful policy or
  model-quality advantage is tested.

The separate instrumented diagnostic records host submission and settlement.
Settlement includes outstanding GPU work, not just host mapping overhead.
It must not be mixed into ordinary throughput measurements. Source inspection
shows serial batch/spatial reductions in convolution weight and bias VJPs;
the increasing batch-size gap motivates GPU pass measurements, but does not
yet establish those kernels as the dominant cost. Peak GPU memory, real-image
browser training, epoch-boundary scaling and other devices remain open.

## Verification And Provenance

`conditions.json` records the requested grid, dataset selection/fixity, runtime
versions, binary hashes and representative contract. Per-interval recipes and
model configurations are in `measurements.json`; `table.json` preserves the
unrounded table. `verification.json` is the independent offline replay of
coverage, raw-file fixity, all-weight parity and paired checkpoint identity.
It does not attest hardware or reproduce elapsed time.

Candidate Rust source and captured harness:
`a8dcac827d570ea27b3075561354a3c25dbd6cf6`. Baseline Rust source:
`dc957fdbea85181cf0058da3ce6a2bad9bc9f13a`. Both wheels use Rust 1.98.0 and
`--no-default-features --features extension-module,nn,wgpu`. The deleted
temporary baseline environment was recovered from a cached package; its native
binary SHA matches the earlier diagnostic exactly. Its repacked wheel archive
hash is not asserted to be the original wheel archive hash.

Focused native validation passed 17 resident-trainer tests and five parameter
receipt tests. The actual browser parameter fixture passed 19 named checks
plus 16 Conv2d updates, including one-map scalar capture, view offsets,
non-finite guards, rejection precedence and retained snapshot lifetime.
`browser-parameters.json` records that fixture separately from native timing.
The full WASM trainer was also rebuilt from the candidate: both constant and
cosine schedules pass 100-attempt control, 37-attempt prefix, fresh-document
63-attempt restart and Python-to-browser continuation. Browser restart is
bitwise identical; Python-to-browser parameter error is zero on all 24 tensors
(456 values). Browser-to-Python continuation and exact feedback-state replay
also pass. The Python suite has five passes and one skipped prior external
fixture. `browser-restart.json` records all eight cases and retained-file hashes.
This is a synthetic correctness regression, not browser real-image learning.
Strict no-dependency Clippy passed for the changed backend/vision libraries;
the broader dependency-inclusive check remains blocked by 24 existing
`st-nn` lints, whose log hash is retained rather than suppressed.

`provenance.json` hashes retained raw records, checkpoints, logs and binaries,
including the failed initial diagnostic whose disposable input pipeline lacked
its GPU dispatcher. Raw weights, images and local paths are not published.
Public hashes check these published records, not access to private raw files.

## Reproduce

Follow [the measurement contract](../../../docs/vision_training_timing.md) with
two retained wheel environments and an already cached dataset. Build the
baseline and candidate from the source revisions above with the listed features
and identical Python dependencies, then run:

```bash
"$CANDIDATE_PYTHON" -I tools/bench_vision_trainer_vs_torch.py \
  --data-root "$CIFAR_ROOT" --output "$NEW_RESULT_DIR" \
  --baseline-python "$BASELINE_PYTHON" --candidate-python "$CANDIDATE_PYTHON" \
  --seeds 17 29 43 --batches 1 16 64 --steps 16 --warmup 3 --repeats 5
python3 -I tools/verify_vision_training_timing.py "$NEW_RESULT_DIR" \
  --source-ref a8dcac827d570ea27b3075561354a3c25dbd6cf6 \
  --output "$NEW_VERIFICATION_JSON"
python3 -I tools/test_vision_training_timing.py
```

Do not run other GPU workloads or builds during measurement. Output paths must
be new. The tool does not download the dataset. Verify this publication with
`shasum -a 256 -c SHA256SUMS` from this directory.
