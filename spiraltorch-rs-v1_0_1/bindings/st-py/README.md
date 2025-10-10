# SpiralTorch-rs (Python)

Python bindings for the SpiralTorch-rs core (Rust) via PyO3 + maturin.

- **Key features**: DP-optimized `einsum`, segment ops, `logprod`, `index_reduce`, and more
- **Python**: 3.8–3.14 (regular wheels + abi3)
- **OS**: Linux (x86_64/aarch64 manylinux2014), macOS (x86_64/arm64), Windows (x86_64)

## Dev install
```bash
pip install -U pip maturin
maturin develop -m pyproject.toml
```
