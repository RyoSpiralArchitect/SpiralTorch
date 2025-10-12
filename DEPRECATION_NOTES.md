# Deprecation Notes — CUDA-first → WGPU-first

- The unified heuristics (`backend::heuristics::choose`) now targets **WGPU-first** by design.
- CUDA resource kernels remain available but are **optional**. You can disable CUDA probing in your host by:
  - honoring `ST_DEVICE_AUTO_PRIORITIES=wgpu,cpu` (example) or your own config file.
- As WGPU subgroup paths become complete (TopK/BottomK/MidK, compaction, softmax family),
  CUDA can be removed entirely from your product build without losing functionality.
- Ameba Autograd techniques (real-index, soft-shape) are backend-agnostic and work on CPU/WGPU.
