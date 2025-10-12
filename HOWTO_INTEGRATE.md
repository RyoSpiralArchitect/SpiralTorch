# HOWTO Integrate v1.4.6

1) Copy this workspace into your repo (or cherry-pick files). CPU build works out of the box.
2) Unified heuristics (backend-agnostic) — *single source of truth*:
   ```rust
   use st_core::backend::heuristics as heur;
   let ch = heur::choose(rows, cols, k, subgroup_supported);
   // use ch.use_2ce/ch.wg/ch.kl/ch.ch to configure kernels
   ```
3) CUDA: load `backend/cuda_topk_wmma_pass1.cu` & `backend/cuda_topk_kway_pass2.cu` via NVRTC/PTX and wire as Pass1/Pass2.
4) WGPU: create shader modules for `backend/wgpu_topk_subgroup.wgsl` and `backend/wgpu_compaction.wgsl` as needed, only if
   `adapter.features()` reports SUBGROUPS; otherwise keep your existing portable WGSL path.
5) MidK: call `st_core::ops::topk_midk_gpu::midk2d(x, k, Device::Auto)` — currently CPU fallback; swap internals to GPU when ready.
6) SpiralK (runtime heuristics) example:
   ```bash
   export ST_MEM_BUDGET_MB=512
   export ST_ELEM_BYTES=4
   export SPIRAL_HEUR_SOFT=1
   export SPIRAL_HEUR_K='
   u2:(c>32768)||(k>128);
   wg:sel(sg,256,128);
   kl:sel(k>=64,32,sel(k>=16,16,8));
   ch:sel(c>16384,8192,0);
   soft(wg,256,0.30,sg);
   penalty(kl,32,0.25,!sg);
   penalty_if(mem()>67108864, 0.25);
   '
   ```

## Device priority (auto selection)
We recommend preferring **WGPU** over CUDA while transitioning:
```bash
export ST_DEVICE_AUTO_PRIORITIES="wgpu,cuda,mps,cpu"   # or "wgpu,cpu"
```
Your host can read this and pick the first available backend.
