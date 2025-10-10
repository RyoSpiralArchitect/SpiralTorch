SpiralTorch-rs v1.0.1 — Rust Tensor & Autograd Core + Python bindings

🚨 World's first PyTorch-like tensor library with full Python 3.14 support.
🧠 Rust core. Autograd-capable. Fused ops. Wheels included. PyTorch is still preparing for Python 3.14.
We started OCT 9, 2025. It already works.
SpiralTorch-rs is a fast, clean Rust implementation of a Torch-like tensor engine with autograd, plus Python bindings via PyO3.

Highlights

Generalized einsum with DP optimization (batch/broadcast aware) + greedy fallback
Segment ops: segment_{sum, mean, max, min}, unsorted_segment_* (by index semantics), ragged_segment_* (via row_splits), and coalesce_indices
logprod (stable log-domain product): returns (logabs, sign), gradient flows through logabs
Exact gradients for index_reduce(..., reduce="prod") even with zeros (base/src) and include_self
Multi-output autograd node support
Out-of-place ops (v1 policy), NumPy-like broadcasting
Quickstart

Rust

cargo test -p st-core
Python (editable install)

pip install -U pip maturin
cd bindings/st-py
maturin develop -m pyproject.toml

python - <<'PY'
import spiraltorch_rs as st, numpy as np
a = st.PyTensor.from_f32(np.arange(6,dtype=np.float32).reshape(2,3), True)
b = st.PyTensor.from_f32(np.arange(12,dtype=np.float32).reshape(3,4)/10, True)
y = st.einsum("ij,jk->ik", (a,b), True)
st.sum(y).backward()
print("OK:", a.grad().shape, b.grad().shape, y.shape())
PY
Wheels & PyPI

GitHub Actions builds wheels for Python 3.8–3.14 on Linux/macOS/Windows, plus manylinux2014 aarch64.
abi3 wheels (cp38-abi3) are built per-OS so new Python minors are usable immediately.
Tag a release (v1.0.1) and set PYPI_API_TOKEN to auto-publish.
See QUICKSTART.md for more details.

SpiralTorch-rs is a lightweight, fast, Torch-inspired tensor engine written in Rust,
with full Python bindings via PyO3 + maturin. It supports dynamic ND tensors, autograd,
backward graph construction, and a minimal, readable API.

Yes — it already supports Python 3.14.
Unlike certain large corporate libraries that shall remain unnamed.

✨ Features

Tensor with f32, i32, and bool storage
Autograd with .backward() (topological traversal, multi-output ops)
Broadcasting + unbroadcasting, dynamic shapes
Pure Rust core (ndarray) with PyO3 bindings
CI builds: manylinux2014, aarch64 (QEMU), macOS (x86_64/arm64), Windows
Python 3.8–3.14 wheels, plus optional abi3 wheels
MIT OR Apache-2.0 dual license
🐍 Install (Python)

From PyPI (recommended once released)

pip install -U spiraltorch-rs
From source (today, including 3.14)

pip install -U pip maturin
git clone https://github.com/RyoSpiralArchitect/spiraltorch.git
cd spiraltorch/bindings/st-py
python3.14 -m maturin develop -m pyproject.toml
🚀 Quickstart (Python)

import spiraltorch_rs as st
import numpy as np

# Create two trainable tensors
a = st.PyTensor.from_f32(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), True)
b = st.PyTensor.from_f32(np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32), True)

# Einsum + reduction, then backward
y = st.einsum("ij,ij->", (a, b), True)
s = st.sum(y)
s.backward()

print("y:", y.data())
print("a.grad shape:", a.grad().shape, "b.grad shape:", b.grad().shape)
🦀 Core (Rust)

cd crates/st-core
cargo test
📦 Wheels CI (tag and ship)

Push a tag to build wheels for 3.8–3.14 (incl. aarch64 + abi3) and publish to PyPI:

git tag v1.0.1
git push origin v1.0.1
Set PYPI_API_TOKEN (scoped token) in repo secrets.
Username is __token__ (already wired in the workflow).
✅ Compatibility Matrix

OS / Arch	Python	Wheel
Linux x86_64 / aarch64	3.8 – 3.14	✔️ manylinux2014
macOS x86_64 / arm64	3.8 – 3.14	✔️
Windows x86_64	3.8 – 3.14	✔️
abi3 (cp38-abi3)	3.8+ (per-OS)	✔️ optional
🧠 Why this exists

Run Torch-like code on Python 3.14 today
Readable core, hackable ops, no CMake nightmares
Minimal surface area with real autograd semantics
🤝 Contributing

Early days. Fork it, break it, file issues.
PRs welcome once the public API stabilizes.

📜 License

AGPL-3.0-or-later

🌀 Author

Ryo ∴ SpiralArchitect and SpiralReality

“The torch is just the beginning. The reality spirals out from here.”


# SpiralTorch-rs (v1.3.2)

Rust-first Torch-like tensor/autograd core with optional **Apple Metal (MPS)** acceleration.

## Highlights
- **Device-backed autograd** (GradBuf): keep gradients on device end-to-end.
- **MPSMatrix GEMM**: 2D forward and **batched forward** run on-device.
- **ND reduce (sum) on-device**: `reduce_nd_sum_auto` selects 1‑pass or 2‑pass.
- **Buffer Pool**: size classes (power-of-two) with LRU cap (env tunable).

## Build
CPU only:
```bash
cargo build -p st-core
```

MPS (macOS 13+):
```bash
cargo build -p st-core --features mps
```

## Quick Rust example
```rust
use st_core::{tensor::Tensor, ops::{matmul, reductions}};
let a = Tensor::ones(&[4, 8]).requires_grad(true);
let b = Tensor::ones(&[8, 16]).requires_grad(true);
let y = matmul::matmul2d(&a, &b).unwrap();
reductions::sum_all(&y).unwrap().backward().unwrap();
assert!(a.grad().is_some() && b.grad().is_some());
```

## License
AGPL-3.0-or-later


今はこれ。。。。いい感じにくっつけれないかな？！
