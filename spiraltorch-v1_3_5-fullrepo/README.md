# SpiralTorch (v1.3.5)

Rust-first, Torch-like tensors & autograd with **MPS (Metal)**, **WGPU (cross‑platform GPU)**, and **CUDA (skeleton)** backends.
AGPL‑3.0‑or‑later. Repo URL: https://github.com/RyoSpiralArchitect/spiraltorch

## Highlights (v1.3.5)
- **WGPU**: unified WGSL kernel file, consolidated PSO setup; shared transpose pipelines for 2D/batched.
- **WGPU**: 2D GEMM backward on-device using `transpose_2d` + tiled GEMM (`go@Bᵀ`, `Aᵀ@go`).
- **WGPU**: softmax backward on-device (rowwise dot + fused grad).
- **WGPU**: sum_axes multi-axis reduction path via flattened rows/cols (1/2‑pass auto switch).
- **MPS**: **β-accumulate** fused batched backward for matmul; temporary transpose buffers from a pool.
- **CUDA**: initial PTX (add_vec, transpose_2d) + cust loader; API mirrors MPS/WGPU for drop-in ops dispatch.
- **Autograd**: device-side grad accumulation via backend `add()` (no host roundtrips).

## Build banner
On successful build, Cargo will print an ASCII banner in warnings (visible in CI logs too).

## Install / Build (local)
```bash
# WGPU (Windows/Linux/macOS)
cargo build -p st-core --features wgpu

# MPS (macOS)
cargo build -p st-core --features mps

# CUDA (skeleton: add/transpose; extend as needed)
cargo build -p st-core --features cuda
```

## Python wheels
This repo includes **maturin** bindings under `bindings/st-py` and a GitHub Actions workflow to build wheels:

- **macOS**: universal2 (x86_64 + arm64)
- **Linux**: musllinux_1_2 (x86_64, aarch64)
- **Windows**: win_amd64
- Optional: manylinux if you switch the target in workflow

### Local (maturin)
```bash
pipx install maturin  # or pip install maturin
maturin develop -m bindings/st-py/pyproject.toml --features wgpu    # or --features mps
# wheel build:
maturin build   -m bindings/st-py/pyproject.toml --release --features wgpu
ls -1 target/wheels/*.whl
```

## Minimal Python usage
```python
import spiraltorch_rs as st

print(st.__version__)
print("available backends:", st.backends())
# (Bindings are a thin skeleton in this archive; extend to expose full Tensor API)
```

## CI (wheels + release)
- See `.github/workflows/release-wheels.yml`
- On pushing a tag like `v1.3.5`, wheels are built and uploaded as workflow artifacts (or as release assets if you enable the Release job).

## License
AGPL‑3.0‑or‑later
