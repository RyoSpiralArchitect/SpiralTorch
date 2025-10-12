# SpiralTorch (v1.3.56)

**Rust-first, Torch-like tensor & ops**, with optional GPU backends (WGPU / MPS / CUDA) and Python wheels.

## What's new in v1.3.56
- **CUDA TopK K-way (real kernel)**: warp4 + shared K-way merge + vectorized loads (`float4` / `ld.global.v4`) via NVRTC (feature: `cuda`).
- **WGPU TopK**: autotune search space shrunk by rows/cols/K heuristics + optional **2CE** path (candidates → final merge).
- **where_nd (Python)**: zero-copy STRIDED multi-segment enumeration (no CPU repack); negative strides canonicalization.
- **WASM demo**: minimal browser UI to initialize WebGPU and load the wasm module skeleton.

## Quick Start

### Rust (CPU)
```bash
cargo build -p st-core
```

### Rust (WGPU)
```bash
cargo build -p st-core --features wgpu
```

### Rust (MPS, macOS)
```bash
cargo build -p st-core --features mps
```

### Rust (CUDA, optional)
```bash
cargo build -p st-core --features cuda
```

### Python wheels
```bash
pip install maturin
maturin build -m bindings/st-py/Cargo.toml --release               # CPU
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu
maturin build -m bindings/st-py/Cargo.toml --release --features mps
# (optional) CUDA requires a CUDA-capable runner and NVRTC
maturin build -m bindings/st-py/Cargo.toml --release --features cuda
```

### Python usage
```python
import numpy as np, spiraltorch as st

# TopK: device="auto" picks CUDA > MPS > WGPU > CPU (if available)
x = np.random.randn(8, 8192).astype(np.float32)
vals, idx = st.topk2d(x, k=32, device="auto")
print(vals.shape, idx.shape)

# where_nd: zero-copy STRIDED multi-segment path (non-contig safe; broadcast OK)
cond = (np.random.rand(2,3,4,5) > 0.5)
xt = np.random.randn(2,3,4,5).astype(np.float32)[:, :, ::-1, :]  # non-contiguous view example
yt = np.zeros_like(xt)
out = st.where_nd(cond, xt, yt, device="auto")
print(out.shape)
```

## License
AGPL-3.0-or-later
