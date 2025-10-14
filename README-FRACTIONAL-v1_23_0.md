
# Fractional & Learning Overlay (v1.23.0)

This overlay extends v1.22.0 with:
- **WGPU kernels**:
  - `wgpu_fracdiff_gl.wgsl`: 1D GL convolution along the last axis (Pad Zero/Reflect), subgroup-friendly.
  - `wgpu_frac_specmul.wgsl`: spectral multiply `|k|^{2s}` on interleaved complex.
  - Host wrappers in `st-core/src/backend/wgpu_frac.rs` (feature `wgpu`).
- **Autograd helper** for **alpha gradient** (central-diff) in `ops/frac_autograd.rs`.
- **CPU FFT** fractional Laplacian along last axis (`ops/frac_fft.rs`) using `rustfft`.
- **Bandit** module (`st-logic/src/bandit.rs`) and enhanced learning hooks.

## Build notes

Add (or ensure) Cargo dependencies:
```toml
# st-core/Cargo.toml (snippet)
[features]
wgpu = ["dep:wgpu", "dep:futures-intrusive", "dep:futures-lite", "dep:bytemuck"]

[dependencies]
ndarray = "0.15"
bytemuck = { version = "1", features=["derive"] }
futures-intrusive = { version = "0.5", optional = true }
futures-lite = { version = "2", optional = true }
wgpu = { version = "0.19", optional = true } # or your workspace's version
num-complex = "0.4"
rustfft = "6"
st-frac = { path = "../st-frac" }
```

## Quick smoke

```bash
# GL (CPU) example
cargo run --example fracdiff

# Fractional Laplacian (CPU): y = (-Δ)^s x along last axis
cargo run --example fracl ap
```

## WGPU use (host wrapper)

```rust
#[cfg(feature="wgpu")]
{
    use st_core::backend::wgpu_frac::wgpu_frac;
    let device: wgpu::Device = /* your WGPU device */ unimplemented!();
    let queue:  wgpu::Queue  = /* your queue */ unimplemented!();
    // GL fracdiff
    let coeff = st_frac::gl_coeffs(alpha, klen);
    let y_gpu = wgpu_frac::fracdiff_gl_wgpu(&device, &queue, &x, &coeff, 1.0, axis, /*pad_zero=*/true);
    // Spectral multiply (fractional Laplacian in freq domain)
    //  - Do FFT on CPU for now, upload interleaved (re,im), multiply on GPU, read back, IFFT on CPU.
}
```

## Autograd w.r.t alpha

Use `ops/frac_autograd::backward_with_alpha(..., AlphaGradMethod::CentralDiff{eps:1e-3})` to obtain `(gx, galpha)`.  
Clamp/regularize `alpha ∈ [0,1]` in your optimizer (or project to sigmoid).

## Online learning (Soft + Bandit)

- **Weights**: `~/.spiraltorch/soft_weights.json` (updated via `learn::commit_result`).
- **Bandit** (Exp3-like): `~/.spiraltorch/bandit.json` storing per-arm weights.  
  Use to choose among macro strategies (e.g., heap/bitonic, 1CE/2CE) with exploration rate `gamma`.
```rust
let mut b = st_logic::bandit::Bandit::load_or_default(0.07);
let arm = "topk.heap"; // your key
let p   = b.prob(arm);
b.update(arm, reward); // reward ∝ speedup
b.save();
```

## Roadmap

- WGPU: vectorized loads and subgroup reduce inside kernels; N-D indexer reuse from where_nd/scan.
- FFT on-device: add WGSL/CUDA/HIP FFT or bind to platform FFT and keep spectral multiply on GPU.
- α analytic gradient: digamma-based derivative of GL coefficients (or construction via log‑recurrence).
- Self‑Rewrite: unify with weights/bandit, guarded by Wilson interval, commit to `heur.kdsl`.
```

