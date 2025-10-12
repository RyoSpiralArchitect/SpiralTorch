# SpiralTorch v1.6.8 Overlay
This overlay delivers:
1) **True keep‑k kernels (HIP)**: shared‑heap & warp‑cooperative (shfl) variants with the *same interface* as bitonic.
2) **WGPU WGSL keep‑k** implementations (subgroup / workgroup), plus a Rust driver to select pipelines.
3) **SpiralK/SoftLogic 'mk' (merge_kind)** heuristics end‑to‑end wiring (0=bitonic, 1=shared, 2=warp).
4) **Orchestrator** updated to auto‑pick bitonic/shared/warp by size & env (`TOPK_KERNEL`), with tile→final preserved.
5) **Absorb hook points** exposed (engine & optimizer) — no‑op safe and ready for ROCm 1‑bit / ZeRO injection.

Apply this over v1.6.6 (or v1.6.5 + 1.6.6 overlay).

## Environment
- `TOPK_KERNEL=auto|bitonic|shared|warp` (default: auto)
- `HIP_SHARED_LIMIT_BYTES` (default: 98304)
- `SPIRAL_HEUR_K` may include `mk:` or `soft(mk, val, w, cond)` where `val ∈ {0,1,2}`

## Build
- HIP: `cargo build -p st-core --features hip,st-backend-hip/hip-real --release`
- WGPU: `cargo build -p st-core --features wgpu --release`
