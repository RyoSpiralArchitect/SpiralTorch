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
