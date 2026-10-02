# Elliptic Features In Real Learning

For the separate, full-context token-relational adapter, see
[causal elliptic learning](elliptic_causal_learning.md). The adapter below remains
pointwise; its existing checkpoint semantics are unchanged.

`EllipticResidualAdapter` connects the existing Rust elliptic/Lie feature map to
Torch/HF losses. It does not substitute a Python approximation of the geometry.
The Rust `EllipticLearningBatch` owns nine features per row and an immutable
Jacobian snapshot; `vjp` contracts that snapshot in Rust. Python and WASM are
clients of the same batch, including validation and empty-batch behavior.

## Numerical Repair

The old f32 `acos(z)` formulation lost small polar angles after normalization
rounded `z` to one. `[0.0001, 0.0002, 1]` produced zero polar and rotor features.
An unrelated denominator floor also attenuated the azimuth derivative near poles.
The shared Rust implementation now uses `atan2(hypot(x,y), z)` with its matching
derivative, and f64 norm intermediates preserve large finite orientations such as
`[1e20, 2e20, 3e20]`. The forward differential still returns f32 features/Jacobians.
VJP contraction accumulates in f64 and rejects nonfinite output.

The nine features and the existing mathematical map are preserved, apart from
these numerical corrections. Exact poles and the azimuth cut have no unique
chart derivative: learning rejects them instead of inventing a zero gradient.
Inputs whose extreme component ratios round onto a pole in f32 are also rejected.
Forward-only legacy telemetry can still describe those points.

## Torch/HF Adapter

```python
import torch
from spiraltorch import EllipticResidualAdapter

adapter = EllipticResidualAdapter(768, strength=0.1, curvature_radius=1.0,
                                  sheet_count=2, spin_harmonics=1)
hidden = torch.randn(2, 32, 768, requires_grad=True)
adapter(hidden).square().mean().backward()
```

The path is `F -> 2 learned coordinates -> (1,u,v) -> 9 Rust features -> F`.
The fixed positive first coordinate chooses a local hemisphere chart, avoiding
poles/the azimuth seam for finite coordinates. It is not a global spherical atlas.
The output projection starts at zero, so the residual begins as an identity.
The first update trains the output projection; later updates can train the input
projection through the Rust VJP. There are `11*F + 2` trainable parameters.
`strength=0` bypasses the adapter. Tokens are never mixed by this adapter.

Insert it explicitly after a **tensor-valued** HF block, as in the
[Topos placement guide](geometric_learning_bridge.md). It accepts f32, transports
geometry to CPU even for GPU tensors, and returns gradients on the source device.
The standard affine projections execute on the module's Torch device. No resident
WGPU, AMP, higher-order, sharded-model or compiler support is claimed.

The existing `elliptic_warp_autograd(warp, orientation)` now batches CPU transport
and invokes the Rust snapshot VJP, rather than transferring every row and
contracting a second derivative representation in Python. It is strictly f32,
has a 65,536-row limit before host materialization, handles empty/noncontiguous
batches, and rejects degenerate/nonfinite/chart-singular rows. This intentionally
replaces the old silent zero-feature fallback and silent dtype narrowing.
Native classes are resolved after facade initialization; last-telemetry state is
context-local, not shared between threads. A backward retains its original
configuration even if the warp is subsequently reconfigured.

Learning does not eagerly convert each row's telemetry into Python objects.
`return_telemetry=True` and `EllipticWarpFunction.last_telemetry()` materialize
and cache those objects on demand from the same immutable forward snapshot.
Repeated requests retain the same telemetry objects, including after backward
or warp reconfiguration. Rust still computes and stores the native telemetry;
this removes unnecessary Python conversion, not geometry computation or CPU
transport. The context retains at most its latest snapshot, and materialization
releases its batch reference (autograd independently retains what it needs).
No adapter configuration or checkpoint schema changes are required.

Python warp construction/configuration now rejects invalid radii and zero sheet
or harmonic counts instead of silently clamping. Radii must be at least `1e-6`
and have a finite f32 maximum geodesic. Rust callers can use `for_learning` for
the same validation; the old builder remains available for legacy telemetry.

Save the adapter's `state_dict` and optimizer separately from the base model.
The state includes the canonical warp recipe and strength. Recreate the same
placement before loading; this is not an HF/safetensors-native adapter format.

## WASM

```javascript
const kernel = new EllipticWarpKernel(1.0, 4, 2, 64);
const batch = kernel.forward(new Float32Array([1, 0.3, 0.4]));
kernel.free(); // The immutable snapshot owns everything needed for backward.
try {
  const features = batch.features;
  const gradient = batch.vjp(new Float32Array(9).fill(1));
} finally { batch.free(); }
```

This is f32 scalar WASM execution, not WebGPU. Configuration/count ingress is
validated without integer coercion. The browser fixture checks native agreement,
snapshot lifetime, invalid inputs, and a 100-update local-coordinate learning loop.

## Bounded Pretrained Experiment

`bindings/st-py/examples/hf_elliptic_learning.py` accepts a local HF model directory,
an explicit tensor-valued block path and feature width. It downloads nothing.
The controls are off, a tangent-linear approximation derived from the same Rust
Jacobian, and the full elliptic map. Both active controls have identical parameter
counts/initial affine weights and update schedules, but different compute cost.

The [recorded experiment](../benchmarks/results/2026-10-02-elliptic-learning-bridge/README.md)
uses pretrained GPT-2, three seeds and six Adam updates on two authored sentences.
Base weights stay frozen and hash-identical. Both projections learn and logits
change. The simpler tangent control does better on this tiny corpus; no geometric
quality advantage is claimed. Disjoint substantial data, continuation and an
explicit extra-cost accounting are still necessary before production FT adoption.
