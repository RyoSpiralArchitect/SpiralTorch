# SpiralTorch (v1.3.104)

Rust-first, Torch-like tensor & ops, with optional GPU backends (WGPU / MPS / CUDA) and Python wheels.

## What's new (v1.3.104)
- **SpiralK (K×Lisp-inspired DSL)** crate (`st-kdsl 0.1.0`) added. Runtime override via env `SPIRAL_HEUR_K`:
  **SpiralK > WASM-generated table > Fallback** precedence.
- **Heuristics unification**: WGPU/MPS/CUDA TopK all consume the same `backend::wgpu_heuristics::Choice`.
- **Build hook**: if `backend/wgpu_heuristics_generated.rs` is missing, `build.rs` writes a stub (so you can commit the tuner output later).
- **Safari/WebGPU**: portable WGSL kept as default; subgroup path is optional and feature-gated at runtime.
- **CUDA TopK**: continues to reuse the same heuristics; half2 candidate path retained (WMMA-ready skeleton).

## Quickstart (Rust)
```bash
# CPU
cargo build -p st-core

# WGPU (portable defaults; Safari-safe)
cargo build -p st-core --features wgpu

# MPS (macOS)
cargo build -p st-core --features mps

# CUDA (NVRTC); arch via ST_NVRTC_ARCH (e.g., compute_80)
ST_NVRTC_ARCH=--gpu-architecture=compute_80 cargo build -p st-core --features cuda

# SpiralK DSL override
SPIRAL_HEUR_K='u2:(c>32768)||(k>128); wg:sel(c<4096,128,256); kl:sel(k>=32,32,sel(k>=16,16,8)); ch:sel(c>16384,8192,0)' cargo build -p st-core --features wgpu,kdsl
```

## Python
```bash
pip install maturin
maturin build -m bindings/st-py/Cargo.toml --release               # CPU
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu
maturin build -m bindings/st-py/Cargo.toml --release --features mps
maturin build -m bindings/st-py/Cargo.toml --release --features cuda
```

## DSL Examples
```text
# subgroup-aware tuning
u2:(c>32768)||(k>128);
wg:sel(sg,256,128);
kl:sel(k>=64,32,sel(k>=16,16,8));
ch:sel(c>32768,8192,sel(c>16384,4096,0))
```
