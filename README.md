# SpiralTorch (v1.3.90)

**Rust-first, Torch-like tensor & ops**, with optional GPU backends (WGPU / MPS / CUDA) and Python wheels.

## What's new (v1.3.90)
- **CUDA TopK**: Adds **KLANE=128**, **multi-CTA per row** (2-stage K-way candidates→merge), **double-buffered ILP** (float4 prefetch).
- **WASM Tuner**: Grid scan + **k-means clustering** to synthesize `wgpu_heuristics.rs` as a **nearest-cluster table**, not nested if/else.
- **K-like DSL (SpiralK)**: New optional crate `st-kdsl` to describe heuristic rules in a tiny **K-like expression language** (feature: `kdsl`).
- **WGPU TopK**: Subgroup-aware heuristic hook preserved. Kernels remain portable (no subgroup ops required).

> Goal: establish a pipeline where **browser measurements → heuristic table (or SpiralK) → native** builds, plus a path to real K/q integration later.

## Quick Start

### Rust (CPU / WGPU / MPS / CUDA)
```bash
cargo build -p st-core                          # CPU
cargo build -p st-core --features wgpu          # WGPU
cargo build -p st-core --features mps           # MPS (macOS)
cargo build -p st-core --features cuda          # CUDA (NVRTC)
# Optional K-like DSL to override heuristics:
cargo build -p st-core --features wgpu,kdsl
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
# open http://localhost:8000/examples/wasm-webgpu-topk-where/www/
# Run grid scan → k-means → Export `wgpu_heuristics.rs`
# Put into: crates/st-core/src/backend/
```

## License
AGPL-3.0-or-later
