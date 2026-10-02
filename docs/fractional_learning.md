# Fractional Memory Learning

`st_frac::learning` makes the existing causal Grunwald-Letnikov (GL) operator
usable with an external loss and optimizer. It is a finite-history difference
operator, not an infinite-memory recurrent model or fractional integral. Rust
owns the forward map, input VJP and order derivative. Python and WASM transport
the same immutable forward snapshot rather than reconstructing the mathematics.

## The Learning Rule

On one chosen axis, with zero padding before its first element:

```text
c[0] = 1
c[k] = c[k-1] * (k - 1 - alpha) / k
y[t] = h^(-alpha) * sum(c[k] * x[t-k], k=0..min(t, kernel_len-1))
```

The shared positive scalar `alpha` has a true loss gradient, including the
derivative of `h^(-alpha)`. Integer alpha is supported: the coefficient
derivative must not be truncated merely because a coefficient becomes zero.
The input VJP is the transpose of this causal map; the joint JVP accepts both
an input tangent and an alpha tangent. The shared-alpha VJP **sums**, not
averages, contributions across every sequence and feature. Loss reduction is
the caller's responsibility.

The owned snapshot fixes the recipe and alpha derivative for that forward.
Subsequent caller input/parameter changes cannot silently change its VJP. No
normalization, optimizer, telemetry policy or gradient clipping is hidden in
the kernel. In Rust:

```rust
use st_frac::learning::{FractionalGlKernel, FractionalLearningError};

fn main() -> Result<(), FractionalLearningError> {
    let kernel = FractionalGlKernel::new(8, 0.7, 128, 1024)?;
    let saved = kernel.forward(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1], 1, 0.5)?;
    let gradients = saved.vjp(&[0.0, 0.0, 0.0, 1.0])?;
    let tangent = saved.jvp(&[1.0; 4], 0.1)?;
    assert!(gradients.alpha.is_finite() && tangent.iter().all(|v| v.is_finite()));
    Ok(())
}
```

## Python And HF

The native `FractionalGlKernel` and `FractionalGlLearningBatch` need no PyTorch.
The optional first-order AD transport is imported from `spiraltorch`:

```python
import torch
import spiraltorch as st

x = torch.randn(2, 16, 8, requires_grad=True)
alpha = torch.tensor(0.5, requires_grad=True)
y = st.fractional_gl_autograd(x, alpha, axis=1,
    kernel=st.FractionalGlKernel(kernel_len=8, step=0.7))
y.square().mean().backward()  # input and alpha gradients come from Rust
```

`FractionalMemoryAdapter(F)` is an identity-initialized residual:

```text
x + strength * tanh(gate[F]) * GL(x, alpha=exp(log_alpha), axis=time)
```

It has `F+1` trainable parameters. The ordinary Torch gate and positive scalar
reparameterization surround the Rust operator. The zero gate preserves the
initial function; alpha's gradient is initially zero and can become active
after the gate moves. There is no straight-through or artificial alpha update.

For an already-loaded GPT-2-style HF model, insert it at a **tensor-valued**
block boundary before constructing the optimizer:

```python
model.eval().requires_grad_(False)  # deterministic frozen base
model.config.use_cache = False
block = model.transformer.h[0]
adapter = st.FractionalMemoryAdapter(
    model.config.n_embd, initial_alpha=0.5, strength=0.1, kernel_len=32,
).to(next(model.parameters()).device)
block.mlp = torch.nn.Sequential(block.mlp, adapter)
optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

optimizer.zero_grad()
loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
loss.backward()
optimizer.step()
```

This is an explicit architecture-specific example, not a universal HF hook.
`tokens` must be equally sized, unpadded, single-document prefixes and the base
must produce float32 hidden states. Other architectures need an appropriate
tensor-valued boundary. Save `adapter.state_dict()` **and** the optimizer state;
the adapter's extra state preserves the GL recipe, dimensions and strength.
It does not save the base model, trainer cursor, RNG or data lineage.

## Independent Local And History Gates

The full-GL adapter ties its current-feature and past-feature contributions to
one gate. `FractionalHistoryAdapter` instead keeps an ordinary local gate and
a strictly-past GL gate independent:

```text
H_alpha(x)[t] = h^(-alpha) * sum(c[k] * x[t-k], k=1..min(t, kernel_len-1))
local = x + strength * tanh(local_gate[F]) * x
output = local + strength * tanh(gate[F]) * H_alpha(x)
```

Rust's `FractionalGlKernel::forward_history` removes the zero-lag coefficient
and its **scaled alpha derivative** before convolution. It does not subtract
two rounded tensors to obtain history. This retains small historical values
next to large current values, avoids overflow of an unused current term, and
accounts for the non-unit step derivative. The first time position, or every
position for `kernel_len=1`, has zero history and zero history derivatives.
Other lanes and future positions cannot influence it. Shape, finite-domain and
work-budget checks are shared with `forward`; this is not an unbounded filter.

Python exposes the same snapshot through `kernel.forward_history(...)` and
`st.fractional_gl_history_autograd(x, alpha, axis=1, kernel=kernel)`. In the HF
example above, replace the adapter constructor with:

```python
adapter = st.FractionalHistoryAdapter(
    model.config.n_embd, initial_alpha=0.5, strength=0.1, kernel_len=32,
).to(next(model.parameters()).device)
```

Construct it before inserting it and creating the optimizer. Both feature
gates start at zero, so the initial function is unchanged. `gate` is the
**history** gate; `local_gate` is the current-feature gate. With a zero history
gate, the finite output exactly recovers the ordinary pointwise gate, while
the history gate can still receive gradients. This is not a speed fast path:
the history snapshot is still evaluated. The alpha gradient is zero until
the history gate moves, regardless of the local gate. There is no surrogate
alpha gradient. The adapter has `2*F+1` parameters, not `F+1`; it is **not a
capacity-matched comparison** to the original adapter or a pointwise gate.

Its extra-state schema is `spiraltorch.fractional_history_adapter.v1`, distinct
from the full-GL adapter. Cross-loading those recipes raises an error instead
of silently changing the operator. Save the optimizer state as well as the
adapter; reconstruct the same adapter class and parameter order on resume.
All float32, full-prefix, padding and cache restrictions below still apply.

The history tests cover the independent Torch polynomial, joint input/order
derivatives, causality, non-unit step, cancellation/overflow and empty history.
The tiny-HF fixture exercises both gates and alpha under the actual causal
language loss with frozen base weights and exact Adam continuation. The WASM
fixture jointly fits local/history gates and order to a synthetic target using
the Rust VJP, rather than a JavaScript GL implementation:

```sh
node bindings/st-wasm/tests/fractional_history.mjs <node-bindgen-module>
```

The compiled-WASM fixture's 500 updates reduced MSE from `0.0912229914` to
`0.0002418502`, with 499 nonzero order-gradient steps after the initial zero
gate. Final alpha was `0.8955443`, **not** the target's `0.65`: fitting the
output with independently learned gates does not establish order recovery.

These fixtures test the learning connection, not pretrained quality, unique
parameter recovery or throughput. Independent gates are a structural
hypothesis, not evidence that fractional history improves language modeling.
Any pretrained comparison must retain pointwise and ordinary causal controls,
report parameter counts and additional work, and use the same data/update
schedule. Existing full-GL checkpoints and comparisons keep their meaning.

## WASM

The same classes expose explicit float32 arrays and owned handles:

```javascript
const kernel = new FractionalGlKernel(8, 0.7, 128, 1024);
const saved = kernel.forward(
  new Float32Array([1, 2, 3, 4]), new Uint32Array([1, 4, 1]), 1, 0.5);
const gradient = saved.vjp(new Float32Array([0, 0, 0, 1]));
const inputGradient = gradient.input;
const alphaGradient = gradient.alpha;
gradient.free(); saved.free(); kernel.free();
```

`saved.jvp(inputTangent, alphaTangent)` returns a `Float32Array`. The optimizer
and object lifetime are client-owned. JavaScript does not implement another GL
rule. `kernel.forward_history` returns the same handle type with the strictly
past map and its own VJP/JVP. The Node test executes the actual compiled wasm32
module; browser UI and WebGPU execution are not claimed by that test.

## Limits And Evidence

- Host float32 only. GPU Torch inputs incur CPU copies and synchronization;
  this is not resident GPU execution or a speed improvement.
- Nonempty rank 1..16 arrays, any valid axis, positive finite alpha and step.
  Output and gradients must remain finite; no silent clamping of unstable alpha.
- `max_values` bounds input element count; `max_products` bounds
  `input.len() * kernel_len` **per convolution**, not total work or bytes.
  Forward evaluates both the map and its alpha derivative and stores both.
  Binding conversion buffers are additional host allocations.
- The adapter takes `[batch,time,features]`, with no mixing between independent
  batch/feature lanes. Future tokens cannot change earlier outputs.
- Use complete unpadded prefixes and `use_cache=False` for both training and
  generation. There is no incremental history, KV-cache integration, attention
  mask support, packed-document reset or padding-aware state. Passing only a
  new token would reset its GL history and is not equivalent inference.
- Reverse AD and `torch.autograd.forward_ad` are first-order only. No higher
  derivatives, `torch.func`/vmap, mixed precision or automatic trainer patching.

Tests compare outputs and both gradients to an independent dense PyTorch
polynomial, including integer orders, non-unit step, noncontiguous inputs and
multiple axes. Rust also tests the joint JVP/VJP identity, frozen snapshots,
causality, invalid shapes and budgets. A random tiny HF model's actual language
loss updates both gate and alpha, preserves its frozen base, and resumes Adam
exactly. This establishes a learning path, **not pretrained quality**.

The real WASM synthetic fit learns target alpha `0.65` from `1.15`: 100 updates
reduce MSE from about `0.143687` to `6.68e-7`, with final alpha `0.651049`.
Run `node bindings/st-wasm/tests/fractional_learning.mjs <node-bindgen-module>`
to repeat it. These are correctness/learning checks, not a throughput benchmark.

Next quality comparisons should isolate no adapter, a pointwise gate,
fixed-order GL and learned-order GL under matched data/update schedules.
Report parameter counts explicitly: learning order adds one scalar. Their
extra history compute and host transfers must also be reported. Pure speed
comparisons remain restricted to the **same** mathematical computation.

The [paired language-model study](fractional_memory_study.md) implements these
controls with the public adapter and the existing restartable HF driver. It
isolates fixed and learned order without adding another geometric mechanism.
