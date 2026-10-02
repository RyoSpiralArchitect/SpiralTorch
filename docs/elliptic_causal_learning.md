# Causal Learning On Elliptic Features

`EllipticCausalResidualAdapter` connects the existing Rust elliptic/Lie map to
token-to-token learning. It is not another Python geometric formula. Rust owns
the feature map, structural causal attention and their complete first-order VJP;
Python provides affine projections, optimizer integration and tensor transport.
WASM exposes the same Rust snapshot, including backward.

## Contract

For each unpadded sequence, let `f_i = phi(1, u_i, v_i)` be the existing nine
elliptic features. Queries, keys and values are tied to these features:

```text
p_ij = softmax_j(dot(f_i, f_j) / 3),  j <= i
y_i = sum_j p_ij * f_j
output_i = input_i + strength * readout(y_i)
```

The score is a dot product of elliptic features, **not a geodesic distance**.
There is one attention head, no dropout and no learned attention temperature.
The `F -> 2` orientation projection and zero-start `9 -> F` readout have the same
`11*F + 2` trainable parameters as the pointwise adapter (8450 at F=768).
Initially the residual is exactly identity. The first update trains the readout;
later loss gradients reach the orientation projection through native attention
and the chart Jacobian. The VJP sums all three tied Q/K/V contributions before
applying the existing elliptic VJP.

`st-kernel-contracts::attention::attention_vjp_reference` is the shared standard
attention derivative, including optional Z/pair biases and absolute causal query
offsets. The elliptic wrapper deliberately uses full sequences and no biases.
Scores use the existing checked f32 arithmetic. Gradient normalization and
accumulation use f64 before checked f32 output. This differentiates the algebraic
map, not floating-point rounding, and does not promise bitwise PyTorch parity.
The forward score routine is shared with the pre-existing attention reference.

## Python

```python
from spiraltorch import EllipticCausalResidualAdapter, elliptic_causal_autograd
import torch

# Example placement for a float32 CPU, frozen GPT-2 with hidden width 768.
# Choose the tensor-valued block explicitly for other architectures.
model.eval().requires_grad_(False)
model.config.use_cache = False
adapter = EllipticCausalResidualAdapter(768, strength=0.1)
model.transformer.h[0].mlp = torch.nn.Sequential(model.transformer.h[0].mlp, adapter)
optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)

# batch is an unpadded [B,T] token tensor; labels are not pre-shifted.
optimizer.zero_grad()
loss = model(batch, labels=batch, use_cache=False).loss
loss.backward()
optimizer.step()
```

The public lower-level helper accepts `[B,T,3]` orientations and returns `[B,T,9]`.
The adapter accepts `[B,T,F]`, requires nonempty B/T axes and rejects the wrong
feature width or dtype. At most 65536 rows and 1048576 potential score pairs
(`B*T*T`, conservatively including masked pairs) are admitted per adapter call.
This bounds computation before host materialization; it is not a score-buffer
allocation. Attention uses row scratch rather than a retained probability matrix.

Only full, **unpadded** contexts are supported. Packed sequences must have genuine
sequence boundaries; do not concatenate independent examples into a single row.
Use `use_cache=False` for both HF training and generation. Passing only the newest
token under a KV cache loses prior adapter context and is not equivalent. No
padding-mask support, cache integration, streaming state, AMP, compiler support
or higher-order derivatives are claimed. Generation quality is unmeasured.

Geometry/attention execute in Rust f32 CPU even for accelerator tensors; the
projections use the module's Torch device. This is explicit transfer, not resident
WGPU execution. The checkpoint schema distinguishes causal and pointwise adapters
and rejects accidental interchange. Save/load the adapter and optimizer together.

## Rust And WASM

Rust callers use `EllipticWarp::differentiate_causal_batch(orientations, batch,
sequence, max_rows, max_pairs)`. The immutable returned snapshot provides
`features()` and `vjp(upstream)`; later warp changes cannot affect backward.

```javascript
const kernel = new EllipticWarpKernel(1.0, 2, 1, 64);
const batch = kernel.forwardCausal(
  new Float32Array([1, 0.2, 0.3, 1, -0.4, 0.5]), 1, 2, 4
);
try {
  const output = batch.features;
  const orientationGradient = batch.vjp(new Float32Array(18).fill(1));
} finally { batch.free(); kernel.free(); }
```

Browser validation is `bindings/st-wasm/tests/elliptic_causal_learning.html`.
Serve it as `/` and fresh wasm-bindgen web output as `/module/`; its visible
`#result` JSON records finite differences, causal boundaries and an actual small
gradient-descent loop. The browser test does not require WebGPU.

## Evidence And Next Comparison

The [bridge validation record](../benchmarks/results/2026-10-02-elliptic-causal-learning/README.md)
covers Rust gradients, equivalent PyTorch forward/VJP, future/batch isolation,
prefix consistency, native snapshot lifetime, exact Adam continuation and actual
tiny-HF loss gradients. Browser WASM also learns using that same native VJP.

This establishes the new trainable mechanism, **not better LLM quality**. The
previous pointwise comparison favored tangent over elliptic. A next fixed-budget
comparison must distinguish causal mixing from geometry itself: include pointwise
and causal ordinary controls, pair initialization/data, and label reused evaluation
sets exploratory. Speed comparisons must use the same mathematics, not compare
this richer map to a different simpler model and call the difference overhead.

The [paired 2x2 study](elliptic_causal_study.md) implements that next comparison
with a fixed recipe, ordinary causal control and the shared restartable HF loop.

Its negative result motivates the optional [learnable context correction](elliptic_gated_context.md):
start from local features and learn a signed contextual contribution instead of
replacing them unconditionally. This next operator does not overturn that result.
