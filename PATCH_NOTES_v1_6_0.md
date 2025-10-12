# v1.6.0 Overlay — Apply Guide

This ZIP adds:
- `crates/st-backend-hip` (HIP backend). Build with `-p st-core --features hip` to include the crate. Use `st-backend-hip/hip-real` to link amdhip64+RCCL.
- Distributed orchestrator stubs: `crates/st-core/src/distributed/topk3_stage.rs`, `prob_params.rs`.
- Optional GPU wheels workflow: `.github/workflows/wheels_gpu.yml`.

## 1) Cargo wiring (st-core)
Edit `crates/st-core/Cargo.toml`:
```toml
[features]
hip = ["dep:st-backend-hip"]
[dependencies]
st-backend-hip = { path = "../st-backend-hip", optional = true }
```

## 2) Module registration
Edit `crates/st-core/src/distributed/mod.rs` and add:
```rust
pub mod topk3_stage;
pub mod prob_params;
```

## 3) HIP real build
Install ROCm, set envs:
```bash
export ROCM_PATH=/opt/rocm
export HIPCC=$ROCM_PATH/bin/hipcc
cargo build -p st-core --features hip --release
```

## 4) Next steps
- Implement communicator lifecycle (RCCL unique id exchange) per cluster stack.
- Replace Pass2 CPU fallback with HIP device K-way + RCCL allgather.
- Connect SpiralK/SoftKanren weights to `prob_params` sampling and do HIP allreduce for consensus.
