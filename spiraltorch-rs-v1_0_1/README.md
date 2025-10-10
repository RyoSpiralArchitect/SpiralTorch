# SpiralTorch-rs v1.0.1 — Rust Tensor & Autograd Core + Python bindings

**SpiralTorch-rs** is a fast, clean Rust implementation of a Torch-like tensor engine with autograd, plus Python bindings via PyO3.

## Highlights
- **Generalized `einsum`** with DP optimization (batch/broadcast aware) + greedy fallback
- **Segment ops**: `segment_{sum, mean, max, min}`, `unsorted_segment_*` (by index semantics), `ragged_segment_*` (via row_splits), and `coalesce_indices`
- **`logprod`** (stable log-domain product): returns `(logabs, sign)`, gradient flows through `logabs`
- **Exact gradients for `index_reduce(..., reduce="prod")`** even with zeros (base/src) and include_self
- **Multi-output autograd** node support
- Out-of-place ops (v1 policy), NumPy-like broadcasting

## Quickstart

### Rust
```bash
cargo test -p st-core
```

### Python (editable install)
```bash
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
```

## Wheels & PyPI
- **GitHub Actions** builds wheels for **Python 3.8–3.14** on Linux/macOS/Windows, plus **manylinux2014 aarch64**.
- **abi3 wheels (cp38-abi3)** are built per-OS so new Python minors are usable immediately.
- Tag a release (`v1.0.1`) and set `PYPI_API_TOKEN` to auto-publish.

See `QUICKSTART.md` for more details.
