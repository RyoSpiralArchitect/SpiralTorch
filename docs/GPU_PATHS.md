
# GPU Paths

| Op     | Preferred | Secondary | Fallback |
|--------|-----------|-----------|----------|
| matmul | CUDA WMMA / MPSMatrix / WGPU tiled | CUDA tiled / WGPU compute | CPU |
| softmax/logsumexp | WGPU f16 storage | WGPU f32 | CPU |
| topk   | WGPU unified (single CE) | older WGPU | CPU |
| where  | WGPU/CUDA/MPS |  | CPU |
