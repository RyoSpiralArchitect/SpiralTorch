# SpiralTorch (v1.3.98)

**Rust-first, Torch-like tensor & ops**, with optional GPU backends (WGPU / MPS / CUDA) and Python wheels.

## What's new (v1.3.98)
- **CUDA TopK**: Adds a **WMMA-best** path (currently half2/shared fused candidate generation via NVRTC; falls back automatically).  
  Reuses **`wgpu_heuristics.rs`** for **`KLANE`** and **`ctas_per_row`** mapping when available, keeping strategies aligned across backends.
- **MPS TopK**: Adds a Metal compute implementation (portable 1CE / 2CE-like) to bring parity with WGPU path.
- **WGPU TopK**: Subgroup-aware entrypoints compiled **only if feature is present**; Safari-safe defaults remain the baseline.
- **SpiralK (K-like DSL)**: Extends with `min(x,y)` / `max(x,y)`, enabling minimax-style scoring expressions.  
  Runtime override via `SPIRAL_HEUR_K` (feature `kdsl`). 
- **Docs**: Safari/WebGPU compatibility notes; tuner workflow → **k-means table** (and threshold-regression hybrid suggestion).

> Target: Aggregate heuristics from **WASM measurements** → ship a table or **SpiralK** program → reuse on **WGPU/MPS/CUDA**.

## Quick Start

### Rust (CPU / WGPU / MPS / CUDA)
```bash
cargo build -p st-core                          # CPU
cargo build -p st-core --features wgpu          # WGPU
cargo build -p st-core --features mps           # MPS (macOS)
cargo build -p st-core --features cuda          # CUDA (NVRTC; arch via ST_NVRTC_ARCH or compute_80 default)

# Optional K-like DSL (SpiralK)
SPIRAL_HEUR_K='u2:(c>32768)||(k>128); wg:sel(c<4096,128,256); kl:max(8,sel(k>=32,32,sel(k>=16,16,8))); ch:sel(c>16384,8192,0)' cargo build -p st-core --features wgpu,kdsl
```

### Python Wheels
```bash
pip install maturin
maturin build -m bindings/st-py/Cargo.toml --release               # CPU
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu
maturin build -m bindings/st-py/Cargo.toml --release --features mps
maturin build -m bindings/st-py/Cargo.toml --release --features cuda
```

### WASM Tuner
```bash
rustup target add wasm32-unknown-unknown
wasm-pack build --release --target web examples/wasm-webgpu-topk-where
python -m http.server 8000
# http://localhost:8000/examples/wasm-webgpu-topk-where/www/
# 1) Grid Scan → 2) k-means → 3) Export `wgpu_heuristics.rs` (nearest-cluster table)
# (Hybrid: you may maintain both a table and an analytic form via SpiralK)
```

## License
AGPL-3.0-or-later
