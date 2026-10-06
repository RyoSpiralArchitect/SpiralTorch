# WaveGate Learning Clients And Pullbacks

`st_nn::WaveGate::vjp` exposes the current Rust forward map's first-order
input/gate/bias pullback without averaging, gradient rewriting or accumulation.
Python exposes the same method on `spiraltorch.nn.WaveGate` (also available as
`spiraltorch.WaveGate`). Both use ordinary SpiralTorch `Tensor` objects.

```python
import spiraltorch as st

layer = st.nn.WaveGate("wave", 2, -1.0, 0.5)
x = st.Tensor((2, 2), data=[0.2, -0.3, 0.4, 0.5])
dy = st.Tensor((2, 2), data=[0.1, -0.2, 0.3, 0.4])
y = layer(x)
dx, dgate, dbias = layer.vjp(x, dy)
```

`vjp` uses the **current** parameters; it is not a captured forward tape. Call it
before changing weights or infusing text. It returns parameter gradients of shape
`(1, features)` summed over rows, and an input gradient of the input shape. Empty
batches return zero parameter gradients without creating accumulators. Finite
inputs/upstream/parameters and finite resulting derivatives are checked before
accumulation. The active Rust NN execution policy selects CPU or WGPU; this
host-`Tensor` path still uploads/readbacks and is not resident GPU execution.

## Immutable Learning Clients

`st_nn::WaveGateKernel::forward` takes input rows and shared gate/bias vectors.
It returns an owned `WaveGateLearningBatch` containing the input, parameters,
output and recipe. Its reusable `vjp` reads that snapshot, not a live module.
Both calls explicitly select CPU and restore the caller's execution policy.
Python exposes the same kernel and snapshot; WASM exposes them with the `nn`
feature. Output/gradient getters return copies in both clients. Free WASM
snapshots and pullbacks after use; retained snapshots consume host memory.

The new kernel defaults to saturation 1.0 and porosity 0.05. This is an explicit
learning recipe, **not** the legacy module's inferred saturation 10000/topos
porosity. Match all configuration values when comparing the two interfaces.

```python
import torch
from spiraltorch import WaveGateAdapter

adapter = WaveGateAdapter(64, strength=0.1, curvature=-1.0)
x = torch.randn(2, 16, 64)
optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = (adapter(x) - 0.9 * x).square().mean()
loss.backward()
optimizer.step()
```

The adapter starts as the identity with zero shared gate/bias (2F parameters).
Both can receive gradients on the first step. Leading axes are independent
rows; projection mixes only the final feature axis, not tokens. Strength zero
is an explicit bypass. A `state_dict` stores learned vectors and the recipe;
optimizer state must also be saved for an identical next update.
`wave_gate_autograd(x, gate, bias, kernel=...)` exposes the non-residual operation.

Torch retains its normal in-place version checks. Replacing the adapter's
recipe after forward does not reinterpret an already-created backward.
Float32 inputs and first-order gradients only: no AMP, compile, double
backward, trainable curvature or implicit model discovery is promised.
Torch GPU tensors incur explicit CPU copies; WASM runs scalar Rust here.
The text encoder/infusion API remains on the native module and is not included
in this parameter-owned adapter.

### Bulk Host Transport

Python also exposes `forward_buffer`, `forward_with_log_radius_buffer`,
`output_buffer`, `vjp_buffer`, and `vjp_with_log_radius_buffer`. Inputs must export
C-contiguous, native-endian float32 buffers (`array('f')`, NumPy arrays or typed
memoryviews), not untyped bytes. Rust copies them before releasing the GIL;
output and gradient bytearrays are fresh owners, never aliases of the immutable
snapshot. The radius derivative is still a scalar. Empty batches are supported.

`wave_gate_autograd` and `WaveGateAdapter` select this transport when Torch/NumPy
interop is available, with the existing list route as the optional-dependency
fallback. Negative views and noncontiguous inputs are materialized explicitly.
WaveGate and fractional adapters share only these transport helpers, not their
mathematics. This reduces Python scalar boxing; it is **not** zero-copy, a new
learning rule, a selective VJP, or a resident GPU path. The radius kernel reuses
row scratch storage in Rust, including WASM, without changing reduction order.

`tools/benchmark_wave_gate_learning.py --native-profile release --log-radius
1.38629436112 --output result.json` compares an explicit list route, the public
bridge and an independent differentiable Torch reference. All request the same
input/gate/bias/radius gradients. Omit `--log-radius` to test the legacy map.
The report retains per-round timings, errors and tensor/native/source hashes;
cross-build comparisons require matching output and gradient bytes first.
CPU host-transport timings are not model-throughput or quality evidence.

For a cached HF model, the local-only
`bindings/st-py/examples/hf_elliptic_learning.py --geometry wave_gate` probe
compares off, the parameter-matched tangent map at zero, and WaveGate.
Its tiny authored corpus tests wiring, not generalization. Zero initialization
and full-batch deterministic updates mean multiple seed labels are not
independent experimental draws. No speed competition is attached to this probe.

## On-Demand Conditioning

`WaveGateLearningBatch::conditioning` describes the captured forward in Rust.
Python and WASM expose `conditioning_json()`; Torch can request the same data
with `wave_gate_autograd(..., return_conditioning=True)` or
`adapter.forward_with_conditioning(x)`. Neither stores global last-call state.
The latter returns `(x, None)` when strength is zero because no geometry executes.
Normal calls do not compute these statistics.

Counts distinguish parameter saturation from affine saturation. Row diagnostics
include `r = norm(S(affine)) / sqrt(-curvature)` and the projection Jacobian's
radial and tangential gains relative to its zero-norm slope: `1-tanh(r)^2`
and `tanh(r)/r`. Both limits are one at zero. Empty row aggregates are null,
not fabricated zeros. These are local projection gains, not full model gradient
norms or evidence that saturation caused a change in language-model quality.

The offline `hf_wave_gate_conditioning.py` example consumes a fixed configuration,
hash-bound local corpus and cached model. It pairs minibatch schedules across
off, tangent, WaveGate and wide-saturation arms; tracks actual parameter gradients;
and verifies an extra next update against an adapter/Adam checkpoint loaded from
disk. Those extra updates are validation only, not endpoint selection. Its novel
split excludes the trailing Gutenberg notice before partitioning. The development
partition is held out from adapter training, not necessarily from GPT-2 pretraining.

## Map And Chain Rule

### Optional Learnable Radius

`WaveGateKernel::forward_with_log_radius(..., log_radius)` captures an explicit
shared scalar alongside input/gate/bias. Its `vjp_with_log_radius` returns the
usual three gradients plus a sum-reduced log-radius derivative. Legacy `forward`
and `vjp` remain unchanged; requesting a radius derivative on a legacy snapshot
is an error. Python and WASM expose the same methods.

```python
adapter = WaveGateAdapter(768, learnable_radius=True, log_radius=0.0)
optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
```

With `R = exp(log_radius)`, `s = sqrt(-curvature)`, and `z = S(affine)`, the map is
`R * tanh(norm(z)/(s*R)) * z/norm(z)`. The limit at zero is zero and its input
Jacobian there is `I/s` for every radius, not an adjustable initial gain. The
radius derivative is zero at identity initialization; it can start learning
after gate/bias move. Rust uses f64 intermediates and a small-argument series
for the log-radius derivative to avoid subtractive cancellation.

`log_radius=...` without `learnable_radius=True` creates a fixed buffer, useful
for controls. Omitting both keeps legacy v1 checkpoints and 2F parameters.
Radius adapters use v2 extra state and require a matching fixed/learnable mode
when restoring; the scalar tensor and Adam state must both be saved. A learned
radius adds one scalar parameter. Rust rejects nonfinite log-radius values or
radii outside the positive normal f32 range rather than silently clipping them.

This optional CPU/WASM path changes the coordinate output radius; it is not a
claim about a fully covariant curvature-radius manifold implementation. It is
not yet wired into the legacy `Module::backward` policy or resident WGPU kernels.
The fixed-origin-gain design is a learning hypothesis, not evidence of an LLM
quality advantage. The paired radius study uses a separately fixed protocol.

### Existing Unit-Radius Map

Let `S` be the existing open-topos porous saturation:

```text
effective_gate = S(gate)
affine = input * effective_gate + bias
z = S(affine)
output = tanh(norm(z) / sqrt(-curvature)) * z / norm(z)
```

The existing unit-ball scaling convention is preserved; this is not a new claim
that the formula is the general curvature-radius exponential map. Row norms mix
features within a row, not tokens or batch rows. At zero, the continuous projection
Jacobian is `I / sqrt(-curvature)`.

The parameter pullback is `S'(gate) * sum_rows(grad_affine * input)`. Previously,
`S'(gate)` was omitted. The porous map's outside slope can be **negative**, so the
missing factor can reverse the update direction, not merely change its size.
At exactly the saturation boundary, the implementation selects the inside slope
of one; the mathematical piecewise map is not differentiable at that boundary.
The regression fixture gave analytic `+0.023985479` versus finite difference
`-0.0006646663` before the repair.

`Module::backward` retains its existing training policy: parameter gradients are
averaged over input rows and rewritten through the safety saturation. Sequence
wrappers can still override the parameter reduction scale. The corrected chain
rule is applied **before** that policy. `vjp` applies neither policy and mutates
neither accumulator, so an external loss/optimizer can own its reduction once.
Nonfinite pullback rejection no longer occurs after updating the accumulators.

## Numerical Scale And Shared Backend Repairs

- CPU projections and their VJP use f64 norm/intermediate arithmetic with f32
  results, avoiding squared-norm overflow/underflow and cubic denominators.
- WGPU shares a max-scaled norm calculation between direct projection, fused
  WaveGate forward and backward. It separates radial and tangential gradients.
- The derivative no longer switches to a linear rule just because the absolute
  norm is below f32 epsilon. Curvature sets the relevant dimensionless scale.
- Small-argument shader projection uses a stable `tanh(a)/a` series rather than
  relying on relative accuracy of the backend's `tanh` near zero. Its output is
  evaluated without multiplying two tiny quantities before dividing them.
- Porous saturation uses scale-safe ratios for its forward map and slope. The
  slope no longer vanishes solely because the saturation scale is tiny.

These are correctness changes, not speed claims. The extra WGPU reduction and
wider CPU intermediates need separate cost measurements before performance claims.
See the [recorded validation](../benchmarks/results/2026-10-02-wave-gate-vjp/README.md).
