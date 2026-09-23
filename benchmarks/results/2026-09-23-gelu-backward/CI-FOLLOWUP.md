# CI Follow-Up

## Initial CI Failure

The first PR #2118 GPU job failed after the local clean-source study.
This is not a passing cross-device claim or a reason to relax the numerical gate.

- Head: `b7ed219e5252309b1a7ae1796b4e74b2bd11829f`.
- [Failed job](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/35821939252/job/107055466626).
- Runner image: macos-26-arm64 20260907.0351.1; macOS 26.6.2 / 25G83.
- The real-GPU backend WGSL/domain example passed. Tensor's residual output
  comparison failed at `wgpu_gelu_liveness.rs:67`; the other two Tensor tests passed.
- The original assertion did not identify the shape, index or actual value.
  The follow-up adds those diagnostics without changing inputs or tolerances.
- The identical parallel Tensor test command passed locally; twenty further
  parallel test-binary runs also passed. This does not resolve the CI failure.

The complete failed job log remains local as
`post-publication/ci-failed-wgpu.log`, SHA-256
`d0b6909df09040185ab8cf0cf6b1759126f634370c2752d93e04884acb4a1804`.
The local reproduction records are in post-publication/ci-reproduce-default
and post-publication/ci-reproduce-20 under the same raw root.

The published timings, raw snapshot and measured source hashes are unchanged.
Test-diagnostic revisions are not remeasured kernel revisions: a source-root
verification must use the recorded measured commit, not assume every later test
file has the same bytes. Merge remains gated on resolving the failed CI.

## Diagnostic Reproduction And Stable Tail

The diagnostic-only commit `353c42ddefe132270e6f2d067812cf8f48721995`
[reproduced the failure](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/35823105752/job/107058992898):

```text
residual 17x31 index=101 accumulated=true z=9.999 seed=-0.5 base=0.375 gz=-0.5000034 actual=-0.1250034 expected=-0.125
```

The observed residual is consistent with adding the seed to the already shifted
gradient, not with a mismatched readback segment. The approximately 3.4e-6
gradient error passes its relative tolerance but exceeds the residual's tighter
absolute-plus-relative allowance after cancellation. The old `1-tanh(inner)^2`
form amplifies a tiny tail error near tanh=1 by the large cubic slope.
The internal tanh value was not captured; this interpretation is an inference
from the observed outputs and formula, not a measured driver implementation.

The shared WGSL derivative now uses q=exp(-2*abs(inner)), inverse=1/(1+q),
the corresponding signed CDF, and sech-squared=4*q*inverse^2. This is the same
tanh-GELU derivative without subtracting nearly equal tail values. It retains
the exact abs(x)>=10 limiting branch. No input, tolerance, finite guard, CPU
routing or test enablement was changed. Resident VJP/training, plain backward
and fused backward all receive the shared numerical correction.

The complete second failed job log remains local as
`post-publication/ci-diagnostic-wgpu.log`, SHA-256
`71562d5399006ae6310978e5c3fc92091a168674552c888abc8dd4af7bf2a91a`.
The stable-tail revision is remeasured separately in
[the stable-tail archive](../2026-09-23-gelu-stable-tail/README.md); the first
study is preserved rather than relabeled as a measurement of the correction.

This file records the investigation, not live CI status. Final CI and review
completion are recorded on [PR #2118](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2118).
