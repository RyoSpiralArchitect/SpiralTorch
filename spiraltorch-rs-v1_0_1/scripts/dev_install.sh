#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip maturin
cd "$(dirname "$0")/../bindings/st-py"
maturin develop -m pyproject.toml
python - <<'PY'
import spiraltorch_rs as st, numpy as np
a = st.PyTensor.from_f32(np.arange(6,dtype=np.float32).reshape(2,3), True)
b = st.PyTensor.from_f32(np.arange(12,dtype=np.float32).reshape(3,4)/10, True)
y = st.einsum("ij,jk->ik", (a,b), True)
st.sum(y).backward()
print("OK:", a.grad().shape, b.grad().shape, y.shape())
PY
