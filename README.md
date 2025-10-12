
# SpiralTorch (v1.3.52-r1)

Torch‑like tensors and ops with a Rust core and multi‑backend execution (CPU, WGPU, MPS, CUDA).  
License: **AGPL‑3.0‑or‑later** • Repo: https://github.com/RyoSpiralArchitect/spiraltorch

## Highlights
- **WGSL where_nd (real)**: broadcast + arbitrary strides, byte‑packed cond, base offsets.
- **Segments API (skeleton)**: Python enumerates hole‑skipping segments; backends to wire next.
- **WGPU autotune (scaffold)**: pick `k_lane` using adapter limits/vendor.

## Install (local)
```bash
# CPU
maturin build -m bindings/st-py/Cargo.toml --release
# WGPU
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu
# MPS
maturin build -m bindings/st-py/Cargo.toml --release --features mps
# CUDA
maturin build -m bindings/st-py/Cargo.toml --release --features cuda
```
> Successful build prints an ASCII banner (see `st-core/build.rs`).

## Quick Start (Python)
```python
import numpy as np, spiraltorch as st
x = np.random.randn(4,3,64,128).astype(np.float32)[:, :, ::-1, :]
y = np.zeros((1,3,1,1), dtype=np.float32)
cond = (np.random.rand(4,1,1,1) > 0.5)
z = st.where_nd(cond, x, y, device="auto")
X = np.random.randn(32, 4096).astype(np.float32)
vals, idx = st.topk(X, k=256, device="auto")
```

## Device Auto‑Selection
Default order: `cuda > mps > wgpu > cpu`. Override:
```python
import spiraltorch as st
st.set_device_order("cuda,mps,wgpu,cpu")    # or env: SPIRALTORCH_DEVICE_ORDER
```

## GPU Backend Fallback Order
| Priority | Backend | Notes |
|---|---|---|
| 1 | CUDA | NVIDIA PTX kernels |
| 2 | MPS  | Apple MPS |
| 3 | WGPU | WGSL compute |
| 4 | CPU  | Portable |

## Known Issues
- **TopK (WGPU)**: autotune scaffold only; CPU fallback used in this snapshot.  
- **MPS/CUDA where_nd**: stubs; real kernels will be wired in upcoming patches.  
- **Segments Upload**: Python generates segments; WGPU/MPS wiring is TBD in this snapshot.

