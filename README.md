# SpiralTorch

**SpiralTorch v1.3.30** — Rust-based tensor/autograd + backends (CPU, WGPU, CUDA, MPS) with Python bindings.

## Build
- CPU: `cargo build -p st-core`
- GPU (feature-gated): `--features wgpu` / `--features cuda` / `--features mps`

## Python Wheels (maturin)
- CPU wheel: `maturin build -m bindings/st-py/Cargo.toml --release`
- WGPU wheel: `maturin build -m bindings/st-py/Cargo.toml --release --features wgpu`

## Device auto-selection (Python)
Order and capability-aware fallback:
| Op     | Preferred order (if supported)               | Fallback when not supported |
|--------|----------------------------------------------|-----------------------------|
| where  | CUDA/MPS/WGPU → CPU                          | CPU                         |
| TopK   | CUDA/MPS/WGPU → CPU                          | CPU                         |

> Current kernels in this repo implement `where_nd` and `TopK` on **WGPU**. CUDA/MPS auto-detection exists, but these ops route to WGPU if available; else CPU. CUDA/MPS routes are placeholders for future kernels.

## Notes
- `device="auto"` in Python chooses available backend with kernel support; otherwise CPU.
- Build banner (ASCII) is printed by `build.rs` on successful build.
