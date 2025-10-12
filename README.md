# SpiralTorch v1.7.1 Overlay

This overlay advances the **unified WGPU-first plan** while absorbing HIP/CUDA know‑how:

1) **WGPU Subgroups 1CE (generalized)** — rows with large `cols` handled by tiled sweep:
   subgroup leaders compress → shared small pool → **final K selection** within the same CE.
2) **MidK 2CE mainline** — `scan_pass` → `apply_pass` kernels (WGSL, HIP). Single API: `ops::midk_compact`.
3) **Ameba Autograd** — `implicit(..., solve="cg")` with conjugate gradient using Jvp approximation.
4) **Self‑Rewrite × Unison** — Redis median (per bucket) injected as low‑weight Soft rules → local win‑rate CI guard → persistent `heur.kdsl`.
5) **Wheel/CI** — universal2 / musllinux matrices with optional WGPU/HIP/CUDA feature toggles.

> Drop‑in overlay for v1.7.0+ trees. Old paths fall back safely.

## Env knobs
- `TOPK_KERNEL=auto|bitonic|shared|warp`
- `SPIRAL_HEUR_SOFT=1`
- `SPIRAL_HEUR_K="u2:...; mk:...; soft(mk,...) ..."` (mk: 0=bitonic / 1=shared / 2=warp)
- `SPIRAL_SELF_REWRITE=1`, `SPIRAL_SELF_MIN_TRIALS=20`, `SPIRAL_SELF_ALPHA=0.05`, `SPIRAL_HEUR_PATH=~/.spiraltorch/heur.kdsl`
- `REDIS_URL=redis://127.0.0.1/` (optional; Unison mediation)

## Build quick
- WGPU: `cargo build -p st-core --features wgpu --release`
- HIP:  `cargo build -p st-core --features hip,st-backend-hip/hip-real --release`

## MidK usage
```python
mid_vals, mid_idx = st.midk_compact(x, lower=-tau, upper=+tau, device="wgpu")
```

## Hypergrad (CG)
```python
out = st.hypergrad.implicit(step_fn, w0, lr, data, val_loss_fn, solve="cg", iters=32)
```

CI workflow is included at `.github/workflows/wheels.yml` (optional; edit matrix as needed).
