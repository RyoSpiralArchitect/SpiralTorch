# SpiralTorch Backend Feature Matrix

SpiralTorch targets a unified runtime that can dispatch to multiple accelerators without rewriting kernels. The table below summarizes the current expectations and caveats for each supported backend so contributors can prioritize validation and fixes.

| Capability | CPU (default) | WGPU | MPS | CUDA | HIP / ROCm |
|------------|---------------|------|-----|------|------------|
| Build flag | _none_ | `--features wgpu` | `--features mps` | `--features cuda` | `--features "hip,st-backend-hip/hip-real"` |
| Min toolchain | Stable Rust | Stable Rust + system WebGPU drivers | Stable Rust + macOS 14 SDK | Stable Rust + CUDA 12 Toolkit & NVRTC | Stable Rust + ROCm 6 toolchain |
| Tensor ops | ✅ Full | ✅ Full (verify image/texture paths) | ✅ Full | ✅ Full | ⚠️ Incomplete complex kernels |
| Autodiff / hypergrad | ✅ | ✅ | ✅ | ✅ | ⚠️ Requires additional testing |
| Planner & scheduler | ✅ | ✅ | ✅ | ✅ | ⚠️ Needs async queue profiling |
| Telemetry | ✅ Structured logging | ✅ GPU timelines | ✅ Instruments via macOS unified logging | ✅ CUPTI hooks planned | ⚠️ Pending counter wiring |
| Python wheel support | ✅ | ✅ (default build) | ✅ | ✅ | ⚠️ Needs wheel audit |

## Usage Notes
- **Feature flags are additive.** Combine multiple backend features during development to compile shared traits, but prefer single-backend release builds for predictable binaries.
- **Driver hygiene matters.** WGPU relies on system graphics drivers. For reproducible CI, pin to Dawn or Vulkan SDK releases and snapshot supported adapters in documentation.
- **Telemetry parity.** Each backend should expose comparable metrics: execution traces, memory transfers, and planner decisions. Track gaps in the issue tracker with the `telemetry` label.
- **ONNX export validation.** Prioritize ONNX parity for CPU and CUDA first. Once the operator parity score exceeds 0.9 for those targets, extend the test suite to WGPU and MPS.
- **Testing cadence.** Keep nightly smoke tests for CPU and CUDA. Run weekly matrix jobs across WGPU, MPS, and HIP until HIP achieves parity with CUDA kernels.

## Future Work
- Capture backend-specific quirks (e.g., alignment constraints, shader compilation limits) in per-backend subpages.
- Publish CI artifacts that include perf regressions, kernel autotune summaries, and binary size trends for each backend.

