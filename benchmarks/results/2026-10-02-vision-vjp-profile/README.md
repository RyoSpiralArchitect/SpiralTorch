# Actual Convolution VJP Pass Diagnostic

## Result

All 54 workers complete: seeds 17/29/43, batches 1/16/64, plain and
identity-feedback modes, three repetitions. Both routes' full final checkpoints
match the retained, independently Torch-checked training reference byte for
byte: 108 comparisons. The 864 profiled steps cover 3,456 convolution VJPs and
10,368 existing compute passes.

The convolution-only bottleneck hypothesis is not supported by these timings.
At batch 64, the four convolution VJPs together take about 3.2 ms inside a
profiled host step of about 63 ms. This does **not** identify the remainder as
GPU compute, NN backward, or readback time. Forward, packing, non-convolution
gradients, updates, dispatch and host waits remain unattributed.

| Batch | Mode | Control host step ms | Profiled host step ms | Convolution VJP ms |
| --- | --- | ---: | ---: | ---: |
| 1 | Plain | 7.412 | 7.918 | 0.132 |
| 1 | Feedback | 7.588 | 7.817 | 0.130 |
| 16 | Plain | 20.304 | 20.345 | 0.819 |
| 16 | Feedback | 20.722 | 20.523 | 0.813 |
| 64 | Plain | 65.779 | 63.018 | 3.155 |
| 64 | Feedback | 67.006 | 62.868 | 3.167 |

Each cell is the median of nine per-worker means, each from 16 steps. The
unrounded [table](table.json) includes convolution ranges. [Measurements](measurements.json)
retain all conditions' per-step host and per-pass GPU durations. Twelve pass
durations per profiled step follow the operation geometry and input/weight/bias
order recorded in that case. An empty pass list is the unprofiled control.

Profile query resolution and mapping occur between steps, outside the host
step clock. This observation changes scheduling. Alternating route order is
not balanced within three repetitions, and these diagnostic timings must not
be interpreted as a profiling speedup or replace the separate
[transfer-inclusive throughput sweep](../2026-10-02-vision-training-throughput/README.md).

## Scope And Validation

The workload is the same small 5,530-parameter ConvNeXt, cached CIFAR-10 input,
normalization, initial owner and 16-batch order as the prior native comparison.
Each route restores a fresh owner after three disposable warmup steps.
Execution is native WGPU/Metal on Apple M4, Rust 1.98.0; Python only prepares
the retained case. This sweep does not run Torch again: it compares all Rust
checkpoint bytes to the already Torch-checked reference. No new images or
models are downloaded. This is not full-dataset quality, peak GPU memory,
browser real-image throughput, or a Z-space policy benefit.

The capture uses an explicitly requested timestamp-enabled runtime. It records
the original convolution passes without changing their shader arithmetic or
pass boundaries. It shares one bounded query set per capture, checks immutable
VJP guards before returning success, resolves only written queries, and releases
scope state after errors, cancellation and unwinding. Default runtimes remain
unprofiled. Four outstanding captures and 128 VJPs per capture are the local
bounds, not a guarantee against all device-resource exhaustion.

Validation covers 253 native backend tests, 113 vision tests and one explicitly
ignored manual stage diagnostic; strict no-dependency Clippy and the repository's
nightly formatter also pass. The shared fixture passes nine checks natively and
in an actual BrowserWebGpu document, including unsupported/nested capture,
context clones, dense/depthwise parity, views, invalid guards, retained/cancelled
results and initialized-prefix reads. The browser result is correctness evidence,
not a browser timing benchmark. Seven offline-verifier tests include corrupt
timestamps, input order, source, checkpoint, pixels and incomplete coverage.

[Verification](verification.json) independently reads the retained worker files,
checks the full requested grid, all input identities and all checkpoint bytes,
then recomputes durations from absolute query ticks. It is not hardware
attestation. [Validation](validation.json) records build/test log and binary
hashes, the actual browser fixture, and the earlier exploratory pilot separately.
Private images, weights and local paths are not published; hashes alone do not
make those raw files publicly replayable.

Captured runtime and launcher source:
`891f6d9b8e531a7b2757f006c59bc29ffc33b92c`. The independent verifier and this
publication follow that source capture; no runtime math changed between them.

## Reproduce And Handoff

Use the existing dataset and retained timing cases; see
[the profiling guide](../../../docs/resident_convolution_profiling.md).

```bash
cargo +1.98.0 build --locked --offline --release -p st-vision \
  --features nn,wgpu --example vision_trainer_gpu_profile
"$PYTHON" -I tools/profile_vision_trainer_gpu.py \
  --data-root "$CIFAR_ROOT" --timing-root "$RETAINED_TIMING_DIR" \
  --binary target/release/examples/vision_trainer_gpu_profile \
  --output "$NEW_PROFILE_DIR"
python -I tools/verify_vision_gpu_profile.py "$NEW_PROFILE_DIR" \
  --timing-root "$RETAINED_TIMING_DIR" \
  --source-ref 891f6d9b8e531a7b2757f006c59bc29ffc33b92c \
  --output "$NEW_VERIFICATION_DIR"
python -I tools/test_vision_gpu_profile.py
```

Do not run other GPU work or builds during capture. Reuse no output directory.
Verify the public leaf with `shasum -a 256 -c SHA256SUMS`.

This closes the bounded vision diagnostic, not its performance gap. Do not
launch a convolution-specific rewrite on this evidence. Shared NN graph VJPs
and execution waits are the next profiling candidates and can be examined on
the returning language-model workload. Browser real-image learning/restart and
peak-memory measurement remain explicitly deferred, not silently completed.
