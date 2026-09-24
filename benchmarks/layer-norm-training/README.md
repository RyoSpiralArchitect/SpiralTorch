# Resident LayerNorm training comparison

This is a matched, **exploratory** 32-step affine LayerNorm + MSE + SGD
training comparison. It exercises the explicit resident WGPU backend rather
than changing ordinary `Tensor`, autograd, or default training dispatch. The
fixed f32 input and target bits, update rule, epsilon, and final loss,
parameters, and affine gradients are checked across Rust CPU, native WGPU,
PyTorch CPU, and PyTorch MPS. The PyTorch control requests all three VJPs;
the installed MPS runtime has a known affine-only gradient-mask issue on a
separate fixture.

Six shapes are tested: 2x3, 8x257, 32x256, 64x768, 128x1025, and 256x256.
Each route has three warmups and nine measured intervals in rotating order.
The timed 32-step path returns CPU-owned final loss, gamma, beta, dgamma,
and dbeta. WGPU `host_to_host_all` includes initial uploads and terminal
readback; `preloaded_all` excludes initial uploads but retains terminal
readback; `preloaded_affine_only` omits the unused input gradient and is
**not work-matched** to all-gradient controls. PyTorch MPS includes initial
upload and final CPU copies. PyTorch CPU starts with host-owned inputs. All
numbers include host-side dispatch, allocation, and framework overhead; none
are device-only kernel timings.

The reported loss and affine gradients are from the 32nd forward/backward,
just before its SGD update; gamma/beta are from just after that update. This
is a step-log convention, not a post-update evaluation pass.

The seven separate stage probes each end with one snapshot: readback-only,
forward, MSE, backward-all, backward-input-only, backward-affine-only, and
parameter update. They diagnose where time is spent but are not additive
components of the 32-step run. The affine/update probes also snapshot the
input to retain a comparable terminal observation.

The native example defaults to the original `sequential` SGD update: gamma
and beta each execute multiply, then add. Set
`SPIRALTORCH_LAYER_NORM_UPDATE_EXECUTION=batched` or `fused` to prepare two
layout-specialized pointwise chains before timing. Each parameter then uses
one GPU submission instead of two. `batched` keeps two dispatches per
parameter; `fused` uses one dispatch. All routes retain the same update
equation and final-output checks. This opt-in benchmark path does not change
the default `Tensor`, NN trainer, or browser dispatch.

On an Apple Metal host with PyTorch MPS available, replay from the measured
source commit and keep the full JSON reports outside the repository:

```sh
RAW="$HOME/Library/Logs/SpiralTorch/layernorm-training-replay-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RAW"
SPIRALTORCH_STRICT_GPU=1 CARGO_BUILD_JOBS=4 cargo +1.98.0 run --release -q \
  -p st-tensor --example layer_norm_training_residency_bench \
  --features wgpu_dense > "$RAW/native-run1.json"
PYTORCH_ENABLE_MPS_FALLBACK=0 python3 -I -B \
  benchmarks/layer-norm-training/torch_bench.py > "$RAW/torch-run1.json"
python3 -I -B benchmarks/layer-norm-training/compare.py \
  "$RAW/native-run1.json" "$RAW/torch-run1.json" > "$RAW/comparison-run1.json"
python3 -I -B -m unittest discover -s benchmarks/layer-norm-training \
  -p test_compare.py -v
```

Use a new local `RAW` directory for each replay. Repeat the three commands
for `run2` and `run3`; do not run the GPU routes concurrently. A successful
comparison validates every shape, route, interval, final numerical output,
decreasing loss, and input/target digest, and includes SHA-256 hashes of both
raw inputs. The public fixed-source results are in
`benchmarks/results/2026-09-24-layernorm-training/`. They publish every
condition's median and validation record, not the larger raw JSON reports.

Separate Rust and PyTorch host stacks and different GPU implementations make
these measurements unsuitable for a universal speed claim. No browser WebGPU
performance result is implied by the native Metal run.

The subsequent medium-row affine workgroup optimization and its native/browser
validation are in `benchmarks/results/2026-09-24-layernorm-affine-workgroups/`.
The prepared pointwise update comparison, including the separately rejected
grouped-submission experiment, is in
`benchmarks/results/2026-09-24-layernorm-update-dispatch/`.
