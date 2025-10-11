
# Runtime Knobs (v1.3.18)
- CUDA
  - `ST_CUDA_WMMA=1`
  - `ST_CUDA_WARPS_PER_BLOCK=4` (4-warp kernel)
- WGPU
  - TopK unified: single CE for K picks
- MPS
  - `ST_MPS_POOL_AUTOTUNE=1`
  - window/range: see PoolAutoTune API
