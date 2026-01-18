# SpiralTorch Backend Feature Matrix

SpiralTorch targets a unified runtime that can dispatch to multiple accelerators without rewriting kernels. The table below summarizes the current expectations and caveats for each supported backend so contributors can prioritize validation and fixes.

<!-- AUTOGEN:BEGIN backend-matrix -->
| Capability | CPU (default) | WGPU | MPS | CUDA | HIP / ROCm |
| --- | --- | --- | --- | --- | --- |
| Build flag | _none_ | `--features wgpu` | `--features mps` | `--features cuda` | `--features "hip,st-backend-hip/hip-real"` |
| Min toolchain | Stable Rust | Stable Rust + system WebGPU drivers | Stable Rust + macOS 14 SDK | Stable Rust + CUDA 12 Toolkit & NVRTC | Stable Rust + ROCm 6 toolchain |
| Tensor ops | ✅ Full (cpu/faer) | ✅ WGPU dense + frac kernels | ❌ Feature placeholder (no kernels wired) | ❌ Feature placeholder (no kernels wired) | ⚠️ hip GEMM (matmul); extend op coverage |
| Autodiff / hypergrad | ✅ Ready | ⚠️ Validate tapes with WGPU execution | ❌ Backend placeholder | ❌ Backend placeholder | ⚠️ Validate tapes with HIP execution |
| Planner & scheduler | ✅ Ready (backend-agnostic) | ✅ Ready (backend-agnostic) | ✅ Ready (backend-agnostic) | ✅ Ready (backend-agnostic) | ✅ Ready (backend-agnostic) |
| Telemetry | ✅ Tracing + structured logging | ⚠️ GPU timing hooks planned | ❌ Backend placeholder | ⚠️ CUPTI hooks not wired | ⚠️ ROCm counters pending |
| Python wheel support | ✅ Ready | ✅ Ready (default build) | ❌ Feature placeholder | ⚠️ Requires CUDA toolchain build | ⚠️ Requires ROCm toolchain build |
| Kernel autotuning | ⚠️ CPU tiling heuristics (faer + autotune) | ⚠️ Shader cache heuristics | ❌ Backend placeholder | ❌ Backend placeholder | ❌ Backend placeholder |
| Sparse tensor ops | ❌ Not implemented | ❌ Not implemented | ❌ Not implemented | ❌ Not implemented | ❌ Not implemented |
| Quantized inference | ⚠️ i8 matmul path present; validate end-to-end | ⚠️ int8 kernels present; validate end-to-end | ❌ Backend placeholder | ❌ Backend placeholder | ❌ Backend placeholder |
| Mixed precision training | ⚠️ BF16/FP16 roadmap | ⚠️ wgpu_f16 feature (validate) | ❌ Backend placeholder | ❌ Backend placeholder | ❌ Backend placeholder |
| Dynamic shape compilation | ⚠️ Planned | ⚠️ Planned | ❌ Backend placeholder | ❌ Backend placeholder | ❌ Backend placeholder |
| Graph fusion pipeline | ⚠️ Planned | ⚠️ Planned | ❌ Backend placeholder | ❌ Backend placeholder | ❌ Backend placeholder |
| ONNX export parity | ⚠️ Export scaffolding (JSON artefacts); ONNX pending | ❌ Not implemented | ❌ Not implemented | ❌ Not implemented | ❌ Not implemented |
| CI coverage | ⚠️ Unit tests + docs checks | ⚠️ GPU CI planned | ❌ No CI | ❌ No CI | ❌ No CI |
<!-- AUTOGEN:END backend-matrix -->


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
