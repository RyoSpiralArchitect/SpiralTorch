# Rust Anchored Elliptic Learning

The [frozen-checkpoint intervention](elliptic_context_ablation.md) found that a
fixed chart anchor improved the saved elliptic models relative to their learned
attention context. That was an inference intervention, not a training comparison.
This operator makes the same correction differentiable in the Rust core so that
we can test it as an actual learning mechanism with matched ordinary controls.

## Contract

```text
f = phi(orientation)             # existing nine elliptic/Lie features
a = phi(1, 0, 0)                 # same warp, fixed chart anchor
g = tanh(raw_mix)                # shared finite f32 scalar
y = (1-g)*f + g*a
dx = Dphi(x)^T ((1-g)*upstream)
draw_mix = sum(upstream * (a-f)) * (1-g*g)
```

Rust owns the feature map, anchor, blend and both first-order derivatives. The
raw-gate gradient is summed over all rows/features, not averaged. Gate/blend and
sum accumulation use f64 before checked f32 output. At zero gate, forward and
orientation VJP are bitwise the original local map, while the gate can learn.
At the chart anchor the gate derivative is zero. Saturated positive gates can
remove input dependence and stop gate learning; no artificial gradient is added.

This is an affine correction in ambient feature space, not geodesic interpolation,
a new manifold or extra attention. With a linear readout, some scaling can be
absorbed into its weights; parameterization and optimization effects must be
distinguished from expressivity. Negative gates extrapolate away from the anchor.

`EllipticWarp::differentiate_anchored_batch(orientations, raw_mix, max_rows)`
returns an immutable `EllipticAnchoredLearningBatch`. Its `features()`, `mix()`
and `vjp(upstream)` use owned snapshots; the VJP returns `orientations` and
`raw_mix`. Rows are independent, including across all leading tensor axes.
There is no quadratic attention allocation or score-pair budget. Empty batches
are allowed and return a zero shared-gate gradient. Existing chart/finite-value
checks remain active even for a saturated gate.

## Python And HF

```python
import torch
from spiraltorch import EllipticAnchoredResidualAdapter

model.eval().requires_grad_(False)
adapter = EllipticAnchoredResidualAdapter(model.config.n_embd, strength=0.1)
model.transformer.h[0].mlp = torch.nn.Sequential(model.transformer.h[0].mlp, adapter)
optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
optimizer.zero_grad()
loss = model(batch, labels=batch, use_cache=False).loss
loss.backward()
optimizer.step()
```

The adapter has `11*F+3` trainable parameters (8451 at F=768): an F-to-2 projection,
a zero-initialized 9-to-F readout and one zero-start raw gate. Adding the gate
consumes no random numbers. The entire residual starts at identity; initially
only the readout receives a nonzero gradient. The fixed `(1,u,v)` input chart,
anchor and warp configuration are not new trainable parameters. A distinct
checkpoint schema rejects accidental interchange with pointwise or causal
adapters; save the optimizer as well for exact continuation.

`elliptic_anchored_autograd(warp, orientation, raw_mix)` accepts float32 `[...,3]`
plus a scalar float32 gate on the same device, returning `[...,9]`. The direct
native API is `warp.map_anchored_batch(..., raw_mix=...)`, whose `vjp` returns
`(orientation_gradient, raw_mix_gradient)`. The default row limit is 65536.

The pointwise operator itself introduces no sequence or KV-cache state. Tiny-HF
tests verify cached token-by-token logits against full-context logits after
training, but this is not blanket compatibility evidence for every HF model,
padding/loss setup or generation stack. Float32, first-order CPU transport is the
supported bridge; GPU tensors explicitly travel through host memory. MPS parity
is not resident GPU execution. No AMP, higher-order autograd or compilation claim.

## Browser WASM

```javascript
const kernel = new EllipticWarpKernel(1, 3, 2, 64);
const batch = kernel.forwardAnchored(orientations, rawMix);
try {
  const gradients = batch.vjp(upstream);
  try {
    rawMix = Math.fround(rawMix - learningRate * gradients.rawMix);
    const dx = gradients.orientations;
  } finally { gradients.free(); }
} finally { batch.free(); kernel.free(); }
```

The browser runs the same Rust implementation. The fixture
`bindings/st-wasm/tests/elliptic_anchored_learning.html` checks signed-gate finite
differences, exact initialization, row isolation, invalid inputs, snapshot
lifetime and a 150-update synthetic gate-learning loop. Serve it with fresh
wasm-bindgen web output mounted at `/module/`.

## Evidence And Next Comparison

The [validation record](../benchmarks/results/2026-10-03-elliptic-anchored-learning/README.md)
covers native Rust, Python/HF and actual browser WASM. The Torch oracle uses the
same blend and Rust feature map, including the `[2,128,3]` training shape; it is
a correctness check, not a speed comparison. Tiny random HF models prove that
the real loss reaches orientation/readout/gate and that Adam continuation agrees.
Neither that nor the synthetic WASM loss proves a pretrained-model quality gain.

Next compare anchored elliptic with anchored ordinary tangent, paired at 8451
parameters and zero gate/readout with the same projection initialization, Adam,
data order and update budget. Keep the learned-attention pair as a separate
control. The original negative results and all losing conditions remain intact;
do not promote the fixed-weight intervention into a retrained quality claim.
