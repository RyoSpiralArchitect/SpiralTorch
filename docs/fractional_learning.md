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
rule. The Node test executes the actual compiled wasm32 module; browser UI and
WebGPU execution are not claimed by that test.

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
