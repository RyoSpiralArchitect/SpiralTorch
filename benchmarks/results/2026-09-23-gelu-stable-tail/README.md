# Stable GELU Derivative Tails, 2026-09-23

Measured implementation: `e97bba6454d374e7165632bbe906c9730edeaffc`.
Original optimization base: `62dced674158234726cfb2521c09d378e8ec3b6f`.
This is a new complete measurement, not a relabeling of the earlier result.

## Why Remeasure

The first liveness implementation passed locally, but strict GPU CI caught
a residual-gradient mismatch at z=9.999, upstream=-0.5, residual_seed=0.375.
Its observed gradient was -0.5000034 and residual -0.1250034 instead of -0.125.
The diagnostic-only follow-up reproduced this. Both failures, exact output,
log hashes and the inference about tail amplification remain in the
[original investigation](../2026-09-23-gelu-backward/CI-FOLLOWUP.md).

The shared WGSL derivative now evaluates the same tanh-GELU derivative using
q=exp(-2*abs(inner)) and inverse=1/(1+q). The approximate CDF is inverse for
nonnegative inner and q*inverse otherwise; sech-squared is 4*q*inverse^2.
This avoids amplifying the rounding of tanh near +/-1 through `1-tanh^2`.
The abs(x)>=10 limiting branch is unchanged. No tolerance or test was relaxed.
The correction reaches plain/fused GELU and resident VJP/training together.
CPU code and high-level finite/layout/fallback policy are unchanged; GPU/CPU
rounding is not claimed bitwise identical over the entire finite-f32 domain.

## Results

The original frozen six shapes, one/three-output contracts, bursts 1/4,
three warm-ups and nine balanced measured blocks were repeated in three
serial, runtime-order-rotated rounds: **5,184 final intervals**.

| Live outputs | Native legacy/candidate | Browser legacy/candidate |
| --- | ---: | ---: |
| Gradient only | 2.908x | 2.684x |
| Gradient, residual and bias | 1.918x | 2.189x |

These are geometric means of twelve cell-level median paired ratios. Every
combined cell exceeds 1 in this run. For one output, eliminating unused reads
gives 2.108x native / 2.561x browser, and eliminating unused fused computation
gives 1.367x / 1.054x. The browser's latter factor exceeds 1 in only six of
twelve cells and is coarse-clock limited. For three outputs, changing only the
formula gives 0.996x / 0.989x, not a speedup; batching its reads gives
1.927x / 2.147x. All intervals and slower factors remain in results.json.

At 128x1025, gradient-only/burst=1, median intervals are 0.237 ms native plain,
0.400 ms browser plain, 0.204 ms Torch CPU and 0.791 ms Torch MPS. CPU remains
faster in some cells. Torch 2.12.1 uses the matched ATen tanh backward operation,
preallocated packed outputs and one owning CPU copy, with four intra-op and
one inter-op threads, no MPS fallback and no torch.compile.

Timing includes encoding/bindings, per-operation submissions and GPU residual
reset where needed, and terminal owning CPU completion. It excludes allocation,
fixed uploads, compilation, checks and JSON. Bursts repeat the same operation
before observing its last result. Both WGPU readback controls use the same
snapshot decoder, not a byte-identical old native read_buffer helper.
This is not a whole-Tensor/model multiplier, a GPU timestamp study, a paired
cross-study estimate of the formula's cost, or universal superiority to PyTorch.

## Validation And Provenance

All 23 clean-source stages pass, including native/WASM strict lint, four
protocol and two archive tests, 192 backend library + one WGSL + two example
tests, 16 autograd + three GELU tests, twelve NN layout tests, 1,001 core tests,
builds and all nine runtime rounds. Twenty-four domain fixtures cover finite
extremes, tails, padding and residual accumulation. Maximum absolute error is
2.9010104843e-6; maximum scaled error is 0.365060258 < 1 under the unchanged
abs=2e-6, rel=1e-5 gate (only bias absolute allowance scales with row count).

exploration.json retains fourteen earlier stages and the separate complete
5,184-interval screening. Neither that screening nor the earlier implementation's
measurements are pooled into the final result. The old backend formula's four
nonfinite outputs after syntax-only repair remain a separate diagnostic.

The raw snapshot lists **212 files / 2,170,779,102 bytes**. Large arrays,
executables and generated WASM/JS remain local. Public records retain all
conditions, intervals, source/asset/stream hashes, commands and test logs.
Native identifies Apple M4/Metal; browser reports BrowserWebGpu and Chrome
153.0.8010.53 with a separate non-fallback Apple adapter probe, not Rust-device
attestation. Low-level buffer/uniform ownership requirements and possible
batched-readback staging-memory growth remain unchanged.

See [the protocol guide](../../gelu-backward/README.md) for numerical replay.

```sh
python3 -I -B benchmarks/gelu-backward/evidence.py verify benchmarks/results/2026-09-23-gelu-stable-tail
```

Use --raw-root and --source-root to rehash raw artifacts, recompute summaries
and check measured source. This is not GPU reexecution. Final CI/review outcomes
are recorded on [PR #2118](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2118),
separately from these local measurements. No CUDA/Furnace or model-quality claim.
