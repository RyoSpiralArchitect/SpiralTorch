# Resident Convolution VJP Profiling

`TensorDevice::profile_convolution_vjps(|| operation())` records the existing
input, weight and bias gradient compute passes while the real Rust/NN execution
path runs. Dense and depthwise convolutions use their ordinary shaders, pass
boundaries and submission order. No graph reconstruction or second derivative
implementation is used.

Request a separate runtime with `WgpuRuntime::request_profiled_headless` (or its
native blocking form). Unsupported timestamp devices fail before the closure
runs; there is no CPU clock substitute and the default runtime is not replaced.
The caller must exclusively own execution and error scopes on that context
during capture. Captures follow cloned runtime contexts, including fresh
TensorDevice wrappers constructed by input preparation and NN graphs. A new
`WgpuContext::new` is a separate capture context even if it wraps the same handles.

## Ownership And Limits

Each synchronous closure has one query set for at most 128 VJPs. This avoids
Metal's counter-sample-buffer exhaustion from one allocation per VJP. At most
four owning results can be outstanding per cloned context. Nested captures and
foreign-thread VJPs fail before their dispatch; ordinary execution allocates no
queries. Capture errors and unwinding clear the scoped registration. They do
not roll back operations the closure has already submitted.

The returned owning `ConvolutionVjpProfileReadback` holds the initialized query
prefix and immutable numerical guards. Read it with `read()` natively or
`read_async().await` in WASM. Query resolution occurs only after submitted work
completes, avoiding Metal's unwritten final-pass samples. Invalid VJP guards or
timestamp data fail rather than produce a successful profile. Retaining or
discarding a result does not change model values, parameter ownership, or
trainer acceptance. A profile is not an optimizer acceptance receipt.

Durations cover only the three convolution gradient passes. Input packing,
forward, other NN gradients, output capture, parameter updates, query
resolve/maps and host work are not included. Zero/quantized durations are valid.
The JSON form preserves absolute ticks as strings. Gaps between samples are
not automatically GPU compute time, and subtracting their sum from host wall
time does not attribute the remainder to a particular kernel.

## Real Trainer Diagnostic

The driver consumes cached CIFAR-10 and the retained inputs/checkpoints from
[the matched training timing tool](vision_training_timing.md). It never downloads
data or publishes images. The native example recomputes the input identity,
runs the ordinary trainer with and without capture, and requires both full final
checkpoints to equal the prior Torch-checked reference exactly.

```bash
cargo build --locked --release -p st-vision --features nn,wgpu \
  --example vision_trainer_gpu_profile
"$PYTHON" -I tools/profile_vision_trainer_gpu.py \
  --data-root "$CIFAR_ROOT" --timing-root "$RETAINED_TIMING_DIR" \
  --binary target/release/examples/vision_trainer_gpu_profile \
  --output "$NEW_PROFILE_DIR"
```

Defaults cover seeds 17/29/43, batches 1/16/64, plain/identity-feedback and three
repetitions. The two routes alternate ordering. Each starts a fresh owner after
disposable warmup, then executes the same 16 batches. Profile reads happen after
each settled step and outside its host clock. This inter-step observation is
diagnostic, not a replacement for uninstrumented throughput. All output paths
are exclusive-create; retain failed attempts rather than overwrite them.

The native/browser shared fixture is
`crates/st-backend-wgpu/examples/support/convolution_profile_checks.rs`.
It checks default-device refusal, context-clone propagation, dense/depthwise
value parity, views, nested capture, non-finite guards, retained/cancelled
results and initialized-prefix reads. The browser page is
`bindings/st-wasm/tests/convolution_profile.html`.
