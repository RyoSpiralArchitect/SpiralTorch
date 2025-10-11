
# SpiralTorch

**SpiralTorch v1.3.18** — a lean PyTorch-like tensor/autograd stack written in Rust, with optional GPU backends (WGPU / CUDA / MPS) and a Python binding via PyO3.

> License: AGPL-3.0-or-later

## Highlights (v1.3.18)
- **CUDA**: 4‑warp/CTA fused bwd kernel slot with coalesced load/prefetch hooks; true in‑place gradient accumulation via `into(...)` APIs.
- **WGPU**: **TopK unified** (single compute pass, K picks) with **single readback**; **f16 storage** `logsumexp` (fwd) and `CE/NLL` (fwd/bwd) fully device-side.
- **MPS**: Memory pool auto‑tuning with moving windows and adaptive exploration range; runtime selection (Matrix default / Compute fallback).
- **Python**: Wheels via maturin; unified `topk/where` API surface (WGPU live; CUDA/MPS exposed as kernels land).

## Install

### Python wheels (from GitHub Actions artifacts or PyPI when available)

```bash
pip install spiraltorch  # once published
```

### Build from source (CPU only)

```bash
# Rust
cargo build -p st-core

# Python (build wheel locally with maturin)
pip install maturin
maturin build -m bindings/st-py/Cargo.toml --release --features pyo3/extension-module
pip install target/wheels/spiraltorch-*.whl
```

### Optional GPU Features

- `--features cuda` (requires CUDA toolchain; kernels loaded via PTX)
- `--features wgpu` (cross-platform via WebGPU/WGSL)
- `--features mps` (macOS Metal)

```bash
cargo build -p st-core --features "wgpu"         # WGPU
cargo build -p st-core --features "mps"          # MPS
cargo build -p st-core --features "cuda"         # CUDA
```

> You can combine features, e.g., `--features "wgpu,cuda"`.

## Python Usage

```python
import spiraltorch as st

x = st.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
y = st.softmax(x, dim=-1)
loss = st.logsumexp(x, dim=-1).mean()
loss.backward()
print(x.grad().numpy())
```

> GPU devices are selected by constructing tensors on those backends (API to be expanded).

## Docs

- `docs/RELEASE_NOTES_v1_3_18.md`
- `docs/HOWTO_topk_ce_f16.md`
- `docs/RUNTIME_KNOBS.md`
- `docs/GPU_PATHS.md` (summary routes & fallbacks)

## Contributing

- Style: rustfmt + clippy; Python: black/ruff; Docstrings in Google style.
- Tests: add unit tests next to ops, and Python tests under `bindings/st-py/tests/`.
- All PRs must include unit tests derived from specs.

