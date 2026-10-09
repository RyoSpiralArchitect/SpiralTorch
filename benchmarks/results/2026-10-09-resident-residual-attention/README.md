# Resident residual attention: correctness, not a speed benchmark

Post-0.4.29 source validation of one Rust-owned training block:

```text
u = pre(x)
y = x + attention(u, z_bias, pair_bias)
output = y + feed_forward(y)
```

The pre-norm recipe joins LayerNorm, fused QKV, attention, output projection,
both residuals and LayerNorm/MLP under one all-or-none SGD owner. An optional
ToposResonator between GELU and the final MLP Linear participates in the same
VJP and update. This is not a complete decoder or a new published wheel.

## Matched Numerical Results

The frozen independent PyTorch 2.12.1 CPU-f32, one-thread fixture covers 60
conditions: three shapes, causal/unmasked attention, none/Z/pair/both/zero score
biases, and Topos off/on. Both routes use the same fixed
`3e-6 + 5e-5 * abs(reference)` tolerance. Parameter gradients are packed in the
same order as the Rust owner, including the learned Topos gate.

| Maximum absolute error | Native WGPU | Browser WebGPU |
| --- | ---: | ---: |
| Forward | 1.1921e-7 | 1.1921e-7 |
| Input VJP | 5.9605e-8 | 5.9605e-8 |
| Parameter VJP | 1.1921e-7 | 1.1921e-7 |
| Z-bias VJP | 8.3819e-9 | 8.3819e-9 |
| Pair-bias VJP | 7.4506e-9 | 7.4506e-9 |
| Final parameters, both 32-update recipes | 2.9803e-8 | 2.9803e-8 |

Both routes completed 32 plain-block and 32 Topos-block SGD updates without
intermediate host readback, plus 12 guard checks. Native used Metal on Apple M4.
The browser reported `BrowserWebGpu` / `Other` and an empty adapter name; no
hardware identity is inferred from that. Per-condition results and both update
trajectories are in [native.json](native.json) and [browser.json](browser.json).

These are two correctness recipes, **not** a matched Topos quality ablation:
their initialization draws differ. No throughput, language-quality, full-model
training or CPU/MPS/CUDA performance equivalence is claimed.

An [additional integration check](integration-validation.json) compiled the
existing default Python extension and the WASM `webgpu` binding against the
committed Rust change. This is build compatibility, not a new wheel/runtime test.
Offline inspection of the frozen Topos recipe finds a maximum gate change of
`0.00345635` after 32 updates; the recorded native and browser final gate errors
are both zero. The gate is active in this recipe, without implying quality gains.

## Regressions and Review

- Full `st-nn` library: 874 passed; existing attention integrations: 2 passed;
  new residual integration: 1 passed.
- Backend tensor tests: 93 passed, one pre-existing ignored profiling experiment;
  resident training: 47 passed; resident graph: 27 passed.
- CPU-only composition and WASM release compilation passed. Browser execution
  was separate live validation, not inferred from a successful WASM build.
- Native tests inject overflow into each of the 13 parameter candidates and
  require bitwise preservation of all parameters after rejection.
- Independent read-only review found two real defects: terminal prediction
  overflow and terminal input-gradient overflow could fail to reject the update.
  Both were reproduced before the fix. Frozen shared guards now propagate each
  failure into every returned derivative, including optional Z/pair gradients.
- Follow-up review found no remaining actionable findings. The independent
  reviewer inspected source only; the parent ran all runtime checks.
- A cache regression initially assumed every strided dense input stayed direct.
  Some legitimately pack on GPU. The test now distinguishes those paths while
  keeping all bad-output, recovery and retained-value assertions unconditional.
  Cached graph bindings already reload current alias flags on every submission;
  no production cache change was needed.

No numerical threshold was relaxed. Earlier failed probes remain in private
logs, whose hashes are included in [validation.json](validation.json). Compiled
binaries and raw logs are kept locally; this directory publishes results and
verification provenance, not duplicate build artifacts or machine-local paths.

## Reproduce

See [the API and commands](../../../docs/resident_residual_attention.md). The
committed Torch fixture is immutable reference data; do not regenerate it as a
substitute for checking the current implementation. Its generator is standalone
Torch code with no SpiralTorch import. `validation.json` pins the fixture,
changed source files, tested artifacts and logs by SHA-256.

Python/WASM public residual-block wrappers, a full language-model path,
checkpoint/optimizer-history handling and matched quality evaluation remain
follow-up work. Optional geometry biases receive gradients but remain
caller-owned rather than being silently optimized by this block.
