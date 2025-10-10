# SpiralTorch-rs (v1.3.2)
[![wheels](https://img.shields.io/github/actions/workflow/status/RyoSpiralArchitect/spiraltorch/wheels.yml?label=wheels&logo=github)](https://github.com/RyoSpiralArchitect/spiraltorch/actions/workflows/wheels.yml)
![license](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)
![pyversions](https://img.shields.io/pypi/pyversions/spiraltorch-rs.svg)

> **🚨 World's first PyTorch-like tensor library with full Python 3.14 support.**
> **🧠 Rust core. tensors & real autograd. Fast, small, readable. Fused ops. Wheels included.**
> **We started OCT 9, 2025. It already works.**
> **SpiralTorch-rs is a fast, clean Rust implementation of a Torch-like tensor engine with autograd, plus Python bindings via PyO3. ⚡ Apple Metal (MPS) on-device.**  


---

## ✨ TL;DR

- **Rust-first / Torch-like:** `ndarray` core, **real autograd** (multi-output, topo backward, broadcast/unbroadcast)
- **Device-first (MPS):** `matmul2d / batched` run via **MPSMatrix** on device;  
  **N-D reduce (sum)** completes on GPU (only the final 1 float is read back)
- **Buffer Pool:** power-of-two size classes + LRU cap (env tunable)
- **Wheels CI:** **universal2** (macOS arm64/x86_64) & **musllinux** (x86_64/aarch64) for Python 3.8–3.14

> Project started **2025-10-09**. It already runs. Results first, excuses later.

---

## What’s new in **v1.3.2**

- **MPSMatrix GEMM (forward):** `matmul2d` and **batched forward** on device (CPU fallback available)
- **N-D reduce (sum):** automatic 1-pass / 2-pass on device
- **Buffer Pool:** pow2 classes + LRU; tune with env vars
- **Python bindings** included (PyO3 / maturin) + **wheels CI** (universal2 & musllinux)

---

## Install (10 seconds)

**PyPI** (when published)
```bash
pip install -U spiraltorch-rs
```

**From source (today)**
```bash
# Rust core (CPU)
cargo build -p st-core

# macOS (MPS)
cargo build -p st-core --features mps

# Python bindings (venv recommended)
pip install -U maturin
maturin develop -m bindings/st-py/Cargo.toml                    # CPU
maturin develop -m bindings/st-py/Cargo.toml --features mps     # MPS
```

Optional pool tuning:
```bash
export SPIRALTORCH_MPS_POOL_MAX_MB=512
export SPIRALTORCH_MPS_POOL_MAX_PER_CLASS=64
```

---

## Quickstart (Python)

```python
import spiraltorch_rs as st, numpy as np

# MPS GEMM (on-device) / CPU fallback elsewhere
A = st.PyTensor.from_f32(np.random.randn(128,64).astype(np.float32), True).to("mps")
B = st.PyTensor.from_f32(np.random.randn(64,96).astype(np.float32),  True).to("mps")
Y = st.matmul2d(A, B)
st.sum_all(Y).backward()
print("device-grad?", getattr(A, "grad_device_available", lambda: False)())

# N-D reduce (multi-axis)
X = st.PyTensor.from_f32(np.random.randn(32,64,4096).astype(np.float32), True).to("mps")
S = st.sum_axes(X, [1,2], keepdim=True)
S.backward()
```

**einsum (DP planner + greedy fallback)**
```python
a = st.PyTensor.from_f32(np.arange(6,dtype=np.float32).reshape(2,3), True)
b = st.PyTensor.from_f32(np.arange(12,dtype=np.float32).reshape(3,4)/10, True)
y = st.einsum("ij,jk->ik", (a,b), True)
st.sum_all(y).backward()
print(a.grad().shape, b.grad().shape, y.shape())
```

---

## Feature set (core ops)

- **Autograd:** multi-output nodes, topological backward, NumPy-style broadcasting/unbroadcasting  
- **Generalized einsum:** **DP planning** (batch/broadcast-aware) + greedy fallback  
- **Segment ops:** `segment_{sum,mean,max,min}` / `unsorted_segment_*` / `ragged_segment_*` / `coalesce_indices`  
- **index_reduce:** `sum/mean/min/max/amin/amax/prod` (**`prod` has exact grads even with zeros**)  
- **logprod:** stable log-domain product → returns `(logabs, sign)`; grads flow through `logabs`  
- **Device-first autograd (GradBuf):** when ops support it, **grads stay on GPU end-to-end**

---

## Quick example (Rust)

```rust
use st_core::{tensor::Tensor, ops::{matmul, reductions}};
let a = Tensor::ones(&[4, 8]).requires_grad(true);
let b = Tensor::ones(&[8, 16]).requires_grad(true);
let y = matmul::matmul2d(&a, &b).unwrap();
reductions::sum_all(&y).unwrap().backward().unwrap();
assert!(a.grad().is_some() && b.grad().is_some());
```

---

## Wheels / CI / Release

- Tag `v*.*.*` → **wheels.yml** runs:
  - **macOS**: universal2 (arm64 / x86_64)
  - **musllinux**: x86_64 / aarch64
- Auto-publish to PyPI: set `PYPI_API_TOKEN` in repo secrets (username `__token__`)

**Compatibility**
| OS / Arch                 | Python  | Wheel           |
|---------------------------|---------|-----------------|
| Linux x86_64 / aarch64    | 3.8–3.14| manylinux2014 ✅ |
| macOS x86_64 / arm64      | 3.8–3.14| universal2 ✅    |
| Windows x86_64            | 3.8–3.14| ✔️              |
| abi3 (cp38-abi3, per-OS)  | 3.8+    | optional ✅      |

---

## Contributing

Early days. **Fork it, break it, tell us.**  
Rust 2021 / `cargo fmt` / `cargo clippy`. Python via `maturin develop`.

---

## License

**AGPL-3.0-or-later**  
© SpiralReality / Ryo (SpiralArchitect)

> *“The torch is just the beginning. The reality spirals out from here.”*
```

want it punchier/snarkier or more corporate? I can tune the voice in 30 seconds.
