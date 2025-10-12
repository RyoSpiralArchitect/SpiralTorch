# SpiralTorch (v1.3.72)

**Rust-first, Torch-like tensor & ops**, with optional GPU backends (WGPU / MPS / CUDA) and Python wheels.

## What's new (v1.3.72)
- **CUDA TopK**: Adds **KLANE=128** variant and **2-stage K-way** (pass1 candidates → pass2 merge), plus **double-buffered ILP** in the scan loop (`float4` prefetch).
- **WGPU TopK 2CE**: Keeps 1CE/2CE pipelines; adds **subgroup-aware heuristic hook** (uses adapter features to bias params). Kernels remain portable (no subgroup-only ops required).
- **where_nd (Direct‑Strided)**: Single large upload per tensor, kernel computes offsets from base+strides (bytes). Fewer queue writes vs. segmented path.
- **WASM tuner**: Browser demo now measures TopK (rows/cols/K) search space and **generates `wgpu_heuristics.rs`** you can drop into `st-core/src/backend/` (or let `build.rs` create a stub).

## Quick Start

### Rust (CPU / WGPU / MPS / CUDA)
```bash
cargo build -p st-core                          # CPU
cargo build -p st-core --features wgpu          # WGPU
cargo build -p st-core --features mps           # MPS (macOS)
cargo build -p st-core --features cuda          # CUDA (NVRTC needed)
```

### Python Wheels
```bash
pip install maturin
maturin build -m bindings/st-py/Cargo.toml --release               # CPU
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu
maturin build -m bindings/st-py/Cargo.toml --release --features mps
maturin build -m bindings/st-py/Cargo.toml --release --features cuda
```

### Python Usage
```python
import numpy as np, spiraltorch as st

# TopK (auto selects CUDA > MPS > WGPU > CPU)
x = np.random.randn(8, 65536).astype(np.float32)
vals, idx = st.topk2d(x, k=4096, device="auto")
print(vals.shape, idx.shape)

# where_nd (non-contiguous safe; direct-strided path)
cond = (np.random.rand(2,3,4,5) > 0.5)
xt = np.random.randn(2,3,4,5).astype(np.float32)[:, :, ::-1, :]
out = st.where_nd(cond, xt, np.zeros_like(xt), device="auto")
print(out.shape)
```

### WASM Tuner (WebGPU in-browser)
```bash
rustup target add wasm32-unknown-unknown
wasm-pack build --release --target web examples/wasm-webgpu-topk-where
# serve examples/wasm-webgpu-topk-where/www + generated pkg/
python -m http.server 8000
# open: http://localhost:8000/examples/wasm-webgpu-topk-where/www/
# run "Run Tuner" → download 'wgpu_heuristics.rs' → place into crates/st-core/src/backend/
# rebuild: cargo build -p st-core --features wgpu
```

## License
AGPL-3.0-or-later
