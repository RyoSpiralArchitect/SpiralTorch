# Vision resident geometry sequence: bounded native result

`st-vision` now groups adjacent resize, crop, and sampled horizontal-flip
stages into one WGPU geometry sequence. The prior path uploaded and read back
the image at every stage. Python can explicitly select this native path with
`TransformPipeline.enable_wgpu()`; the CPU path remains the default.

The measured source was `db1f3e2c0266f85488e6ede42ecfaec67ed2272d`
against base `9e9e10a62dc7a0bd2cb410b4234dad79e8c60cf2`. A subsequent
test-only commit gated the pre-existing backbone suite on `nn` for CPU-only
builds; it does not change the measured transform code. The adapter was
`Apple M4`, `IntegratedGpu`, `Metal`, on macOS 26.4.1.

The benchmark runs the same 3x512x512 input through resize to 384x384, flip,
center crop to 320x320, and flip. It compares four separate WGPU calls with
one resident sequence. Each run alternates route order, warms up twice,
measures seven pairs, and verifies finite outputs within `1e-5` absolute
error. Five sequential runs all had a shorter resident route median. The
median of run medians was 7.044 ms separate versus 2.143 ms resident; paired
gains ranged from 31.87% to 82.14%. Per-interval timings vary substantially,
so this is a single-host, single-shape latency result, not a general
TorchVision or training-throughput claim. All run medians and hashes are in
[`measurements.json`](measurements.json).

The complete `st-vision` WGPU suite passed 82 tests; the CPU-only suite passed
53. A first CPU-only attempt failed because `tests/backbones.rs` compiled
without its required `nn` feature; that log is retained and the corrected
suite passes. Both default and CPU-only Python binding checks passed. A fresh
0.4.27 wheel installed into an isolated venv passed the Python opt-in,
three-frame CPU/WGPU parity, and disable test.

The original logs, wheel, and executable stay under
`~/Library/Logs/SpiralTorch/vision-resident-sequence-v1/`; their hashes are
listed in [`raw-SHA256SUMS`](raw-SHA256SUMS). From that directory, verify
with `shasum -a 256 -c /path/to/raw-SHA256SUMS`. Verify these published
files with `shasum -a 256 -c SHA256SUMS`.

Replay the benchmark from the measured source with:

```sh
cargo run --release -p st-vision --features wgpu \
  --example transform_sequence_bench
```

The current synchronous host-visible transform dispatcher does not make
browser WebGPU vision preprocessing available. The ecosystem roadmap records
an async WASM path and matched browser fixture as the next admission gate.
