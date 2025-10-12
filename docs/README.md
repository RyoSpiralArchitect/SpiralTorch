# SpiralTorch Docs

## WASM → WGPU Heuristics
- The WASM tuner exports `wgpu_heuristics.rs` as a **nearest-cluster table** (k-means centers).
- Place the file in `crates/st-core/src/backend/` to override the fallback.

## SpiralK (K-like DSL)
- Optional feature: `--features kdsl`
- Runtime override via environment variable `SPIRAL_HEUR_K`:
  ```text
  u2:(c>32768)||(k>128);
  wg:sel(c<4096,128,256);
  kl:max(8, sel(k>=32,32, sel(k>=16,16,8)));
  ch:sel(c>16384,8192,0)
  ```
- Added `min(x,y)` / `max(x,y)` primitives, for minimax-style rules:
  ```text
  # example: trade-off using pseudo cost function
  wg:sel( (log2(c)*0.5 + log2(k)) > 24, 256, 128 );
  kl:max(8, sel(k>4096,32, sel(k>512,16,8)));
  ```

## Safari / WebGPU Notes
- We request **downlevel defaults** for limits and avoid subgroup ops by default.
- Workgroup size 128/256 both present; older Safari may perform better with 128.
- If your Safari lacks WebGPU, enable *Develop → Experimental Features → WebGPU* or use Chrome/Edge.

## CUDA NVRTC
- Set `ST_NVRTC_ARCH` (e.g., `--gpu-architecture=compute_80`) if auto default is not ideal.
- To use half2 path in candidate generation (bandwidth save), set `ST_CUDA_HALF=1`.
  Falls back to f32 path if not available.
