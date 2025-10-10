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
