# SpiralTorch (v1.3.7)

Rust-first, Torch-like tensors & autograd with **CPU (ndarray)**, **MPS (Metal)**, **WGPU (cross‑platform GPU)**, and **CUDA** backends.
AGPL‑3.0‑or‑later. Repo URL (example): https://github.com/RyoSpiralArchitect/spiraltorch

## Key Features (v1.3.7)
- **Autograd**: multi-output nodes, device-side grad accumulation path (design in place).
- **Ops**: matmul(2D/batched), reduce (sum over axes), elementwise (add/mul/relu), softmax backward (rowwise dot + fuse).
- **WGPU**: unified WGSL module (transpose / reduce (1/2‑pass) / tiled GEMM / softmax bwd). Mixed-precision façade (fp32 accumulate).
- **MPS**: batched matmul backward with **β‑accumulate**; BufferPool with **LRU + size classes** and trace.
- **CUDA**: shared‑memory **tiled GEMM (16×16)** PTX (+ add/transpose), bwd via tiled GEMM. WMMA stub path (Tensor Core) prepared.
- **Python bindings** (PyO3 + abi3) and **wheel CI** (universal2 macOS / musllinux (x86_64, aarch64) / Windows).

## Build banner
You’ll see an ASCII banner in build logs (local & CI) once the core crate builds.

## Build (local)
```bash
# CPU baseline
cargo build -p st-core

# WGPU (Windows/Linux/macOS)
cargo build -p st-core --features wgpu

# MPS (macOS only)
cargo build -p st-core --features mps

# CUDA (add/transpose/tiled gemm)
cargo build -p st-core --features cuda
```

## Python bindings (maturin; abi3)
```bash
pipx install maturin  # or: pip install maturin
# dev install
maturin develop -m bindings/st-py/pyproject.toml --features wgpu
# build wheels
maturin build   -m bindings/st-py/pyproject.toml --release --features wgpu
ls -1 target/wheels/*.whl
```

## Minimal Python usage
```python
import spiraltorch_rs as st
print(st.__version__)
print("available backends:", st.backends())
```

## CI
Push a tag like `v1.3.7` to trigger the wheel workflow. Artifacts appear in Actions or Release (if you enable attaching).
See `.github/workflows/release-wheels.yml`.

## Notes
- WMMA/Tensor Core: stub hook is present; real kernel can replace the stub to enable HMMA path (fp16 inputs, fp32 accum).
- WGPU f16/bf16: API façade for mixed precision is there; once WGSL f16 is enabled, map inputs to fp16 → accumulate in fp32.
- MPS BufferPool: enable tracing via `ST_MPS_POOL_TRACE=1`; override caps with `ST_MPS_POOL_MAX_BYTES`, `ST_MPS_POOL_MAX_PER_CLASS`.
