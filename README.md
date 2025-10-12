# SpiralTorch — WGPU-first Unison + CUDA Scaffolds + MidK + Ameba Autograd (v1.4.6)

**Focus of v1.4.6**  
- Favor **WGPU-first**: unify heuristics; CUDA path remains as resource kernels but is not required.
- **WGPU SUBGROUPS K-way** pass polished (no fixed-size local arrays; mark-taken with -inf; repeated subgroup max).
- **Compaction kernels** updated: a simple parallel row-scan + scatter (portable WGSL).
- **Ameba** grew: unrolled/implicit hypergrad helpers; `gather_real` backward; soft-shape L0 surrogate + integerization.
- **MidK** API unchanged (CPU fallback), GPU entry points stable for drop-in later.
- **Deprecation path**: CUDA-first can be turned off via env; WGPU becomes the primary auto device if available.

CPU-only builds still work out of the box. GPU kernels are shipped as resources you can plug into your runtime.
See `HOWTO_INTEGRATE.md` and `DEPRECATION_NOTES.md`.
