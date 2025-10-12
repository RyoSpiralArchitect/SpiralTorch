# SpiralTorch (v1.6.5, HIP/WGPU/CUDA-ready skeleton)

This is a **buildable minimal workspace** capturing the latest distributed/HIP + heuristics stack:
- RCCL communicator bootstrap (**env/file + TTL + retry**)
- 3-stage TopK orchestrator with **device allgather** and **HIP Pass2 bitonic merge** (replaceable by shared/warp heap)
- Probabilistic parameter consensus (**mean/median**) via **HIP allgather** or **Redis**
- Self‑Rewrite knobs (threshold/min_samples/cooldown) via env
- CI for wheels (opt-in matrix & tag-driven release)

> This repo is intentionally minimal and feature-gated so you can drop it into your project or use it standalone to validate the GPU communication/heuristics pipeline.

## Build (CPU-only)
```bash
cargo build -p st-core
```

## Build HIP real (ROCm)
```bash
export ROCM_PATH=/opt/rocm
export HIPCC=$ROCM_PATH/bin/hipcc

# Distributed env
export WORLD_SIZE=4
export RANK=$LOCAL_RANK
export RCCL_UNIQUE_ID_FILE=/tmp/rccl_uid.bin   # or set RCCL_UNIQUE_ID_B64

cargo build -p st-core --features hip,st-backend-hip/hip-real --release
```

See **README-RCCL.md** for RCCL bootstrap details.
