# SpiralTorch v1.7.6 Overlay

This drop-in extends **v1.7.5** with **real kernel sources** and small plumbing:
- **WGPU** WGSL kernels: `warp‑heap (subgroup)` for TopK, and compaction scan for Mid/Bottom.
- **CUDA** kernels: `warp‑heap` + `warp‑bitonic` rowwise TopK (row-major).
- **HIP** kernels: `shared‑heap` + `warp‑heap` rowwise TopK (wavefront=64).
- **Tuner** table format can carry `mkd/ctile/two_ce_hint` (already added in v1.7.5).
- **Self‑Rewrite** staged tightening (optional): 0.60 → 0.65 → 0.70.
  - `ST_REWRITE_ESCALATE=1` enables auto-escalation using `~/.spiraltorch/rewrite_state.json`.

## Quick use
```bash
unzip -o spiraltorch-overlay-v1_7_6.zip
# (WGPU) Put WGSL into your pipeline builder, or keep exec stubs and wire later.
# (CUDA) Build the .cu to PTX in your build.rs or out-of-tree and load at runtime.
# (HIP) Build the .hip.cpp with hipcc to HSACO and load via your HIP runtime.
```

## Files
- `crates/st-core/src/backend/wgpu_kernels_rankk.wgsl`  — WGPU kernels (TopK / MidK / BottomK paths)
- `crates/st-core/src/backend/cuda_topk_rankk.cu`       — CUDA warp-heap/warp-bitonic TopK (rowwise)
- `crates/st-core/src/backend/hip_topk_rankk.hip.cpp`   — HIP shared-heap/warp-heap TopK (rowwise)
- `crates/st-core/src/backend/wgpu_exec.rs`             — Strategy dispatch stubs (mk/mkd/ctile aware)
- `crates/st-core/src/backend/cuda_exec.rs`             — Strategy dispatch stubs
- `crates/st-core/src/backend/hip_exec.rs`              — Strategy dispatch stubs
- `crates/st-core/src/backend/wgpu_heuristics.rs`       — Choice extended with `two_ce_hint`
- `crates/st-core/src/ability/self_rewrite.rs`          — staged escalation (0.65/0.70) optional

> NOTE: Exec stubs still return `Ok(())` so the repo remains buildable. Swap the stub calls with
> your backend runtime invocations (compile pipelines / load PTX / load HSACO) when ready.
