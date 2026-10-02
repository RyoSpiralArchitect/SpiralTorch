# Learnable Elliptic Context Correction

The [paired causal study](elliptic_causal_study.md) did **not** show a geometry
advantage: tangent beat elliptic, and replacing local features with causal mixing
hurt both feature families on both reused evaluation sets, in all three seeds.
That result is preserved. It does not establish *why* mixing hurt.

This operator tests a narrower next hypothesis: keep the current token's local
features and let training select the contribution of context, rather than replace
local features unconditionally. No language-quality gain is claimed yet.

## Rust Contract

```text
f = phi(orientation)                        # existing nine elliptic/Lie features
a = causal_attention(f, f, f, scale=1/3)     # existing checked Rust attention
g = tanh(raw_mix)                           # one shared finite f32 parameter
y = (1-g)*f + g*a
```

This is a signed correction in **ambient feature space**, not a manifold
interpolation or a new geodesic attention metric. Negative g is allowed. Rust
owns the blend and its complete first-order VJP, including the local path, all
three tied Q/K/V paths, and `sum(upstream*(a-f))*(1-g*g)` for the raw gate. The
shared-parameter gradient is summed, never averaged. Gate/blend accumulation uses
f64 before checked f32 output; the derivative is of the algebra, not rounding.

At raw gate zero, features and orientation VJP are exactly the original local
map. The gate derivative can be nonzero. Attention must still run at zero to
compute that derivative, so initialization does not imply pointwise runtime cost.
At a single token the context equals the local feature, so the gate gradient is
zero. Large raw magnitudes saturate tanh and can stop gate learning.

Rust callers use `EllipticWarp::differentiate_gated_causal_batch(orientations,
[batch, sequence], raw_mix, max_rows, max_pairs)`. Its immutable snapshot exposes
`features()`, `mix()` and `vjp(upstream) -> EllipticGatedCausalGradients` with
`orientations` and `raw_mix`. Changing the original warp cannot change backward.

## Python Learning

```python
import torch
from spiraltorch import EllipticGatedCausalResidualAdapter

# Explicit placement in a tensor-valued block of a float32 frozen GPT-2.
model.eval().requires_grad_(False)
model.config.use_cache = False
adapter = EllipticGatedCausalResidualAdapter(model.config.n_embd, strength=0.1)
model.transformer.h[0].mlp = torch.nn.Sequential(model.transformer.h[0].mlp, adapter)
optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)

optimizer.zero_grad()
loss = model(batch, labels=batch, use_cache=False).loss  # full unpadded [B,T]
loss.backward()
optimizer.step()
```

There are `11*F+3` trainable parameters (8451 at F=768): `F -> 2` orientation,
zero-start `9 -> F` readout, and one raw gate starting at zero without an RNG draw.
The whole residual starts at identity; its first step trains the readout. Later
loss gradients can reach orientation and gate. A distinct checkpoint schema
rejects accidental interchange with local or unconditional-causal adapters.
Save the adapter and optimizer together for exact continuation.

The lower-level `elliptic_gated_causal_autograd(warp, orientations, raw_mix)`
accepts float32 `[B,T,3]` plus a scalar float32 tensor on the same device, returning
`[B,T,9]`. Both derivatives use the same native snapshot. Direct native
`warp.map_gated_causal_batch(..., batch_size=B, sequence_length=T, raw_mix=g)`
returns a snapshot whose `vjp` yields `(orientation_gradient, raw_mix_gradient)`.

The existing causal bounds and limitations still apply: nonempty full unpadded
contexts, at most 65536 rows and 1048576 potential score pairs by default. Use
`use_cache=False` for training **and generation**. There is no padding mask, KV
cache, AMP, higher-order differentiation, compilation or streaming-state support.
Geometry runs on Rust CPU with explicit host transport even for GPU tensors;
MPS transport correctness is not resident WGPU execution or a speed claim.

## WASM Learning

```javascript
const kernel = new EllipticWarpKernel(1, 3, 2, 64);
const batch = kernel.forwardGatedCausal(orientations, B, T, rawMix, B*T*T);
try {
  const gradients = batch.vjp(upstream);
  try {
    // These are the same Rust derivatives, not a JavaScript geometric formula.
    rawMix = Math.fround(rawMix - learningRate * gradients.rawMix);
    const orientationGradient = gradients.orientations;
  } finally { gradients.free(); }
} finally { batch.free(); kernel.free(); }
```

`bindings/st-wasm/tests/elliptic_gated_causal_learning.html` performs finite
differences at negative/zero/positive gates, exact zero checks, causal isolation,
snapshot lifetime/validation checks, and a 150-update gate-learning loop. Serve
the fixture with fresh wasm-bindgen web output at `/module/`.

## Evidence And Next Experiment

The [validation record](../benchmarks/results/2026-10-03-elliptic-gated-context/README.md)
covers Rust, Python/HF, and actual browser WASM. The Torch reference comparison
uses the same signed blend and attention mathematics, including the prior
experiment's `[2,128,3]` orientation shape. It is a correctness comparison, not
a throughput benchmark. Tiny random HF models establish gradient wiring only.

Before claiming useful LLM learning, pair this gated geometry with a **gated
ordinary tangent control**, including the extra scalar, initialization, optimizer,
batch order and budgets. Keep pointwise controls and report the learned gate
trajectory. Gate sign or movement alone is not evidence of improved language
quality. Reused Pride/Alice endpoints remain exploratory.
