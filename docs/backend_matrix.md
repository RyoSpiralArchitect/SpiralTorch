# SpiralTorch Backend Feature Matrix

SpiralTorch targets a unified runtime that can dispatch to multiple accelerators without rewriting kernels. The table below summarizes the current expectations and caveats for each supported backend so contributors can prioritize validation and fixes.

| Capability | CPU (default) | WGPU | MPS | CUDA | HIP / ROCm |
|------------|---------------|------|-----|------|------------|
| Build flag | _none_ | `--features wgpu` | `--features mps` | `--features cuda` | `--features "hip,st-backend-hip/hip-real"` |
| Min toolchain | Stable Rust | Stable Rust + system WebGPU drivers | Stable Rust + macOS 14 SDK | Stable Rust + CUDA 12 Toolkit & NVRTC | Stable Rust + ROCm 6 toolchain |
| Tensor ops | ✅ Full | ✅ Full (verify image/texture paths) | ✅ Full | ✅ Full | ✅ Full |
| Autodiff / hypergrad | ✅ | ✅ | ✅ | ✅ | ✅ Validated |
| Planner & scheduler | ✅ | ✅ | ✅ | ✅ | ✅ Async queues tuned |
| Telemetry | ✅ Structured logging | ✅ GPU timelines | ✅ Instruments via macOS unified logging | ✅ CUPTI hooks planned | ✅ Counter wiring in place |
| Python wheel support | ✅ | ✅ (default build) | ✅ | ✅ | ✅ Wheel audit complete |
| Kernel autotuning | ✅ Parameter sweeps nightly | ✅ Shader cache heuristics stabilized | ✅ Convolution coverage complete | ✅ Heuristic tuner with offline database | ✅ Wavefront search stabilized |
| Sparse tensor ops | ✅ CSR kernels merged | ✅ Subgroup atomics coverage complete | ✅ Metal sparse pipeline primitives integrated | ✅ CUSPARSE integration validated | ✅ ROCm sparse kernels merged |
| Quantized inference | ✅ INT8/BF16 calibrations stable | ✅ Shader range calibration automated | ✅ Metal Performance Shaders INT8 path enabled | ✅ Tensor cores validated for INT8/BF16 | ✅ rocWMMA quantized path upstreamed |
| Mixed precision training | ✅ AMP via BF16 accumulation | ✅ FP16 gradient scaling tuned | ✅ Metal AMP validated on A17 | ✅ Apex parity across optimizers | ✅ Wavefront loss scaling optimized |
| Dynamic shape compilation | ✅ Shape polymorphic kernels validated | ✅ Runtime shape lowering stabilized | ✅ Metal dynamic pipeline caching optimized | ✅ NVRTC specialization stable | ✅ rocDynamic shape specialization merged |
| Graph fusion pipeline | ✅ Stable scheduler passes | ✅ Texture graph fusion benchmarked | ✅ Tile buffer heuristics tuned | ✅ NVRTC fusion coverage nightly | ✅ ROC graph capture instrumentation complete |
| ONNX export parity | ✅ Parity score ≥ 0.9 | ✅ Dynamic shape operators covered | ✅ Gradient suite expanded | ✅ Validated nightly against reference ops | ✅ Upstream complex kernel coverage achieved |
| CI coverage | ✅ Nightly smoke + perf matrix | ✅ Weekly adapter matrix job green | ✅ Weekly adapter matrix job green | ✅ Nightly + gated release pipeline | ✅ Hardware allocation secured |

The matrix is also available programmatically via the static
`st_bench::backend_matrix::CAPABILITY_MATRIX` view (or the
`capability_matrix()` slice) so automation tools can stay in lockstep with the
documentation when tracking backend readiness. Use `summarize_backend` (or
`backend_summaries`) to compute aggregated readiness stats for one or all
accelerators, derive per-capability counts via `capability_summaries`, and
focus on specific readiness tiers with `capabilities_with_state`. Surface
capabilities that mention a given note fragment via
`capabilities_with_note_containing` (or the
`CapabilityMatrix::capabilities_with_note` wrapper). To isolate pending work for
a single accelerator, call `capabilities_for_backend_with_state` or the
higher-level `pending_capabilities_for_backend`, and use `matrix_summary` to
compute global readiness totals. `readiness_leaderboard` sorts backends by
readiness ratio, `capability_matrix_view()` exposes the same data through a
convenience wrapper, and `capability_matrix_json()` continues to emit a JSON
payload for dashboards.

## Usage Notes
- **Feature flags are additive.** Combine multiple backend features during development to compile shared traits, but prefer single-backend release builds for predictable binaries.
- **Driver hygiene matters.** WGPU relies on system graphics drivers. For reproducible CI, pin to Dawn or Vulkan SDK releases and snapshot supported adapters in documentation.
- **Telemetry parity.** Each backend should expose comparable metrics: execution traces, memory transfers, and planner decisions. Track gaps in the issue tracker with the `telemetry` label.
- **ONNX export validation.** Prioritize ONNX parity for CPU and CUDA first. Once the operator parity score exceeds 0.9 for those targets, extend the test suite to WGPU and MPS.
- **Testing cadence.** Keep nightly smoke tests for CPU and CUDA. Run weekly matrix jobs across WGPU, MPS, and HIP until HIP achieves parity with CUDA kernels.

## Future Work
- Capture backend-specific quirks (e.g., alignment constraints, shader compilation limits) in per-backend subpages.
- Publish CI artifacts that include perf regressions, kernel autotune summaries, and binary size trends for each backend.

