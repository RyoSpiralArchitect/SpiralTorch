# Elliptic Directional Learning

The map stays unchanged. The immutable Rust `EllipticLearningBatch` now exposes
`jvp(direction)` alongside `vjp(upstream)`. It applies the stored differential
to three input-direction values per row and returns nine feature-direction
values per row. Accumulation uses f64 before checked f32 output, as in the VJP.
Empty snapshots accept only empty directions; incorrect sizes, nonfinite seeds
and nonfinite results fail rather than returning zeros.

The anchored snapshot accepts a joint direction `jvp(orientations, raw_mix)`:

```text
g = tanh(raw_gate)
dy = (1 - g) * Dphi(x) dx + (1 - g*g) * d_raw_gate * (anchor - phi(x))
```

`raw_mix` here is the gate's **direction**, not a new gate value. All primal
values come from the original snapshot. The fixed anchor is not differentiated
with respect to input coordinates. At a zero primal gate and zero gate direction,
the result exactly equals the local JVP. Finite input validation remains active
even if the gate saturates.

## Clients and Learning

Python snapshot methods and WASM `EllipticLearningBatch.jvp` /
`EllipticAnchoredLearningBatch.jvp` delegate to Rust. The pointwise and anchored
Torch bridges also implement
[`torch.autograd.forward_ad`](https://docs.pytorch.org/tutorials/intermediate/forward_ad_usage.html).
They retain the existing backward rule, host transport and float32 restriction.
This does **not** add `torch.func`, `vmap`, higher-order differentiation, causal
attention JVPs or resident GPU derivatives.

```python
import torch
import spiraltorch as st

warp = st.EllipticWarp(1.3, 3, 2)
x = torch.tensor([[1.0, 0.3, -0.4]])
direction = torch.tensor([[0.0, 0.2, 0.3]])
with torch.autograd.forward_ad.dual_level():
    dual = torch.autograd.forward_ad.make_dual(x, direction)
    y = st.elliptic_warp_autograd(warp, dual)
    features, feature_direction = torch.autograd.forward_ad.unpack_dual(y)

snapshot = warp.map_orientations_batch(x.flatten().tolist())
pullback_direction = snapshot.vjp(snapshot.jvp(direction.flatten().tolist()))
```

`J^T J` is the Euclidean feature-space pullback metric, not the Hessian of a
language-model loss. No inverse or preconditioner is silently substituted into
autograd. The small executable `bindings/st-py/examples/elliptic_pullback_fit.py`
demonstrates damped Gauss-Newton with bounded conjugate-gradient solves and an
actual-loss line search. It holds the first orientation coordinate fixed, thus
removing the redundant radial direction. It fits synthetic targets using native
JVP/VJP calls without materializing the full Jacobian or normal matrix.

## Training-Only Diagnosis

`bindings/st-py/examples/hf_elliptic_chart_probe.py` reads a **completed** anchored
study without changing its records. It captures 16 uniformly spaced training
blocks from the frozen model's adapter insertion point, then compares both maps
at identical initial, trained-tangent and trained-elliptic projections for all
three seeds. It neither trains nor evaluates either endpoint. Basis JVPs must
agree exactly with basis VJPs; all checkpoint, source, runtime and input hashes
are bound in the report. Corpus text, model weights and hidden tensors stay local.

The measurements concern the **bare local feature map**, before the trained
anchor gate and readout. `feature_covariance_effective_rank` is the participation
ratio of centered-feature covariance eigenvalues, not an algebraic rank or an
information-theoretic dimension. A small singular value by itself is not proof
of a vanishing complete-model gradient or a causal explanation of the CE gap.

See the [numeric verification record](../benchmarks/results/2026-10-03-elliptic-directional-learning/README.md).
Real LM training with a curvature-aware update is still a separate experiment;
the production adapters continue to use ordinary, unchanged reverse-mode gradients.
