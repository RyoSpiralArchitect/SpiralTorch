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
    let alpha_only = saved.vjp_alpha(&[0.0, 0.0, 0.0, 1.0])?;
    let input_only = saved.vjp_input(&[0.0, 0.0, 0.0, 1.0])?;
    assert_eq!(alpha_only, gradients.alpha);
    assert_eq!(input_only, gradients.input);
    let tangent = saved.jvp(&[1.0; 4], 0.1)?;
    assert!(gradients.alpha.is_finite() && tangent.iter().all(|v| v.is_finite()));
    Ok(())
}
```

Both snapshots expose `vjp_input` and `vjp_alpha` in Rust, Python and WASM.
The alpha-only path validates and reduces the saved order differential without
computing an input-adjoint convolution or allocating its gradient tensor.
The input-only path omits the order reduction. Each requires a finite,
shape-matching upstream direction and a representable requested result;
overflow in an **unrequested** component does not reject a valid component.
Joint `vjp` still requires both, with unchanged component mathematics and
accumulation order. Forward snapshots still compute the order differential;
this is not a forward-only or resident-GPU optimization.

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

The AD bridge follows Torch's input-gradient requirements. For frozen hidden
states with trainable alpha, backward selects `vjp_alpha`; for trainable input
with frozen alpha, it selects `vjp_input`; when both are trainable it uses the
joint VJP. It does not drop older-lag order derivatives at integer alpha.
This changes unnecessary work, not the loss, optimizer or intended gradients.
Host transport remains explicit; a speed claim needs matched measurement.
The [selective-VJP validation](../benchmarks/results/2026-10-03-fractional-selective-vjp/README.md)
records native/WASM component parity and exact updates of saved pretrained
adapters, separately from timing or quality claims.
`tools/benchmark_fractional_learning.py` compares full forward plus the requested
order VJP through the joint Rust route, selective Rust route and a float32 Torch
causal-convolution reference. Use `--validate-only` before timing, a verified
release native build, and an idle host. It includes the native host transport;
it is not an end-to-end adapter/model or resident GPU benchmark.

### Bulk Float32 Transport

The optional AD bridge uses bulk buffers when its first-use Torch/NumPy interop
probe succeeds; otherwise it retains the sequence route. Only that capability
probe may fall back: errors during an actual operation are not silently retried.
Noncontiguous Torch inputs are materialized in C order, lazy negative views are
resolved, and device tensors still make explicit CPU host transfers. This does
not add another mathematical backend or change adapter/checkpoint schemas.

Native buffer methods also work without Torch or NumPy:

```python
from array import array
import spiraltorch as st

kernel = st.FractionalGlKernel(kernel_len=8, step=0.7)
saved = kernel.forward_history_buffer(array("f", [1, 2, 3, 4, 5, 6]), [2, 3], 1, 0.5)
output = memoryview(saved.output_buffer()).cast("f")
dx_bytes, d_alpha = saved.vjp_buffer(array("f", [1] * 6))
```

`forward_buffer`/`forward_history_buffer`, `vjp_buffer`,
`vjp_input_buffer`, `vjp_alpha_buffer` and `jvp_buffer` accept C-contiguous,
native-endian float32 exporters. Read-only and unaligned exporters are safe;
non-native byte order, wrong element types, noncontiguous/negative strides and
invalid sizes are rejected rather than reinterpreted. The existing sequence
methods and `output` list property remain available.

Inputs are copied to Rust-owned values before releasing the GIL. Output and
vector-derivative methods return fresh writable `bytearray` owners containing
native float32 bytes, not aliases of the saved snapshot and not a portable disk
format. The AD client retains a `memoryview` export around
[`torch.frombuffer`](https://docs.pytorch.org/docs/2.12/generated/torch.frombuffer.html),
so its CPU storage stays alive and its backing bytearray cannot resize. Mutating
an output never changes the snapshot used for derivatives. There is no end-to-end
zero-copy or resident-GPU claim.

Use `tools/benchmark_fractional_learning.py --compare-transport` for explicit
list versus buffer versus Torch comparisons of the same forward/order VJP.
Both Rust routes use the same native binary and kernel; the list baseline is
explicit, not an assumption about the current default transport.

The [complete transport comparison](../benchmarks/results/2026-10-04-fractional-buffer-transport/README.md)
records all 12 CPU processes and bitwise pretrained-update regression checks.
Its operator timings do not establish model throughput or learning quality.

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

## Fixed-Energy History

`FractionalL2HistoryAdapter` keeps the same independent gates and `2*F+1`
trainable parameters, but fixes the **strictly-past coefficient L2 norm**:

```text
c = (0, c_1(alpha), ..., c_(K-1)(alpha))
q = gain * c / ||c||_2
H_l2(x)[t] = sum(q[k] * x[t-k], k=1..min(t, K-1))
output = x + strength*tanh(local_gate)*x + strength*tanh(gate)*H_l2(x)
```

The norm includes the entire declared kernel, even at early prefix positions.
It is not recomputed per prefix or sample. This holds filter energy fixed,
**not output variance on correlated hidden states**, and is not gradient
normalization. `gain` is a positive, finite, constant float32, not a parameter.
The default is 1; matching a raw alpha=2 history filter at h=1 uses
`gain=sqrt(5)` with K>=3. Equality is mathematical up to float32 rounding,
not a promise of bitwise optimizer trajectories. For a K=3 versus K=32
comparison, keep gain, gates, initialization, data and update budget matched.

Rust owns the normalized taps, the normalization's alpha derivative, input
VJP and joint JVP. Both positive common factors `h^-alpha` and `alpha` cancel
analytically before normalization. This avoids an unnecessary step-scale
overflow and catastrophic cancellation near alpha=0. Internally, for k>=1,
`c_k=-alpha*p_k`, `p_1=1`, `p_k=(k-1-alpha)*p_(k-1)/k`. The factored polynomial
and its derivative must stay within the float32 range; recurrences and the
norm accumulate in float64 before checked float32 conversion. Unrepresentable
outputs/differentials raise errors rather than clipping alpha or gradients.

```python
adapter = st.FractionalL2HistoryAdapter(
    model.config.n_embd, initial_alpha=2.0, strength=0.1,
    kernel_len=32, gain=5.0**0.5,
).to(next(model.parameters()).device)
```

The extra-state schema `spiraltorch.fractional_l2_history_adapter.v1` includes
the canonical float32 gain and rejects raw-history/full-GL cross-loading.
Save Adam state too. Identity initialization and all full-prefix, float32,
padding, cache and first-order-only restrictions still apply. This is an
explicit alternative operator, **not a change to existing GL checkpoints**.

Lower-level entry points share the same immutable snapshot contract:
Rust `kernel.forward_history_l2(input, shape, axis, alpha, gain)`, Python
`kernel.forward_history_l2(...)` / `forward_history_l2_buffer(...)` and
`st.fractional_gl_history_l2_autograd(x, alpha, axis=1, kernel=kernel, gain=1.0)`,
WASM `kernel.forward_history_l2(float32Input, uint32Shape, axis, alpha, gain)`.
All remain CPU/host paths, not resident GPU kernels. K=1 or an axis of length
one has zero history and zero differentials. K=2 has one past tap `-gain`,
so its alpha gradient is exactly zero; learning order needs more past taps.

Tests compare to an independent Torch polynomial, cover finite differences,
integer/subnormal orders, buffer ownership, causal boundaries, and exact Adam
continuation in a tiny HF model with frozen base weights. Run the compiled
WASM counterpart:

```sh
node bindings/st-wasm/tests/fractional_history_l2.mjs <node-bindgen-module>
```

Its synthetic gate/order fit is an execution test,
not evidence of pretrained quality, unique order recovery or a speed win.

The [history-length x coefficient-energy protocol](fractional_history_factorial.md)
crosses K=3/K=32 with raw/L2-normalized learning from the same alpha=2 filter.
It tests whether the preceding effect survives a fixed coefficient norm,
without calling that constraint fixed hidden-state variance or a speed win.

These fixtures test the learning connection, not pretrained quality, unique
parameter recovery or throughput. Independent gates are a structural
hypothesis, not evidence that fractional history improves language modeling.
Any pretrained comparison must retain pointwise and ordinary causal controls,
report parameter counts and additional work, and use the same data/update
schedule. Existing full-GL checkpoints and comparisons keep their meaning.
The [four-arm independent-history protocol](fractional_history_study.md) fixes
those controls before observing pretrained outcomes.
Its completed three-seed experiment gives learned GL a small advantage over
ordinary EMA on both reused endpoints, but learned alpha approaches 1.
At step 1 that limit is the single-lag map `-x[t-1]`, so this is not evidence
that a long fractional tail is necessary; the next control should test that
ordinary short-memory explanation directly.

## Independently Learned Gain

The completed [history/energy factorial](fractional_history_factorial.md)
shows that fixing coefficient norm changes both the learned order and the
benefit of longer available history. `FractionalGainHistoryAdapter` makes
that amplitude an independent trainable parameter instead of silently
changing the fixed-gain operator:

```text
gain = exp(log_gain)
q(alpha, log_gain) = gain * c(alpha) / ||c(alpha)||_2
H(x) = causal strictly-past convolution with q
dH/dlog_gain = H(x)
```

Rust owns the exponential, normalized coefficients and input/alpha/log-gain
VJPs and joint JVP. The existing fixed-energy forward computes the map and
alpha differential; the new snapshot reuses its captured output for the
log-gain differential without a third full-size saved array. Shape and
amplitude coordinates are independent, **not guaranteed identifiable or
orthogonal under the data distribution**. Feature gates still modulate the
residual, so changing gain also changes their effective scale.

`log_gain` must be finite and `exp(log_gain)` must be positive and finite in
float32. Invalid/overflowing/underflowing gains fail rather than clamp,
including when history is empty. The positive step scale still cancels.
`FractionalGlKernel::gain_from_log_gain(log_gain)` exposes this exact checked
conversion without a history allocation in Rust; Python and WASM expose the
static `FractionalGlKernel.gain_from_log_gain(log_gain)` method. `FractionalGainHistoryAdapter.gain` reads
it for the current parameter. The [matched gain study](fractional_gain_study.md)
records it before/after each update and compares an ordinary learned short filter.
Only requested pullbacks are evaluated: `vjp_parameters` returns alpha and
log-gain gradients without allocating an input gradient; input-only,
alpha-only and log-gain-only methods are also available. Unrequested
pullback overflow does not invalidate a requested finite component.

```python
adapter = st.FractionalGainHistoryAdapter(
    model.config.n_embd, initial_alpha=2.0, initial_gain=5.0**0.5,
    strength=0.1, kernel_len=32,
).to(next(model.parameters()).device)

# Or use the differentiable low-level map on a complete float32 prefix:
y = st.fractional_gl_history_log_gain_autograd(
    x, alpha, log_gain, axis=1, kernel=st.FractionalGlKernel(kernel_len=32),
)
```

`alpha` and `log_gain` are scalar float32 Torch tensors; normal autograd
chains the adapter's log-alpha chart to the native alpha derivative.
All gates start at zero, so the adapter is identity initially and the two
scalar gradients are initially zero. Its `2*F+2` parameters include one
more scalar than the raw/fixed-energy adapters. The distinct extra-state
schema `spiraltorch.fractional_gain_history_adapter.v1` and saved `log_gain`
prevent silently interpreting old checkpoints with the new semantics.
Save and restore the optimizer as well. Float32 log/exp round trips need
not reproduce an arbitrary requested initial gain bit-for-bit.

Rust/Python/WASM use `kernel.forward_history_log_gain(input, shape, axis,
alpha, log_gain)`, returning `FractionalGlGainLearningBatch` with `output`
and the effective `gain`. Python also provides the native-f32 `_buffer`
transport. Joint VJP returns `(input, alpha, log_gain)` in Python and named
fields in Rust/WASM; `vjp_parameters` returns `(alpha, log_gain)` in Python
and a two-element `Float32Array` in WASM. `jvp(dx, da, dlog_gain)` shares
the same controls on all clients. WASM handles must be freed explicitly.

Finite-difference/adjoint checks, an independent ordinary Torch reference,
all seven nonempty reverse-AD combinations, forward AD, buffer ownership,
optional-Torch imports and frozen tiny-HF Adam continuation are tested.
`fractional_history_log_gain.mjs` fits a synthetic shape/amplitude target
through the compiled wasm32 module. No additional pretrained study or
speed benchmark is implied. Before a new quality comparison, match initial
maps, data/update budgets and scalar capacity with an ordinary short-filter
control, and account for the remaining gate/gain redundancy.

Raw and constant-gain operators/checkpoint schemas remain unchanged. All
host-float32, first-order-only, complete-prefix, no-padding/packed-document
and no-KV-cache limits below also apply to this new path.

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

The [matched operator timing](../benchmarks/results/2026-10-03-fractional-selective-vjp-timing/README.md)
records full/history forward plus order-only VJP on a release CPU build. The
selective path reduces observed work versus joint backward, but remains slower
than the ordinary Torch convolution reference. These ambient-load measurements
do not establish model throughput, GPU performance or a general speedup.

The [paired-forward follow-up](../benchmarks/results/2026-10-03-fractional-paired-forward/README.md)
now shares input reads between the output and cached alpha differential in
`FractionalGlKernel`. Contiguous trailing features are tiled in groups of 64,
with fixed-size f64 scratch accumulators; each output retains increasing-lag
accumulation order. The C-order input is borrowed rather than cloned. Public
shapes, budgets, full/history semantics and VJP/JVP methods are unchanged, as
are the general ND operators used as independent regression references. The
same Rust implementation reaches native Python and WASM, without a second
production formula in either client.

All 24 matched native-binary processes pass output/gradient hash equality.
Under this host's ambient load, Python-inclusive selective forward/backward
falls from roughly 21-22 ms to 16-17 ms. It still trails the Torch convolution
reference; list/Tensor transport is the next measured bottleneck, not a reason
to replace the mathematical operator. Six copied pretrained adapters also
preserve loss, gradients and Adam updates bit-for-bit across the two binaries.
These short regression probes are not additional quality runs or proof of
model-level throughput. The actual wasm32 tiled/scalar regression covers 128
axis, feature-width, order and spacing combinations in CI.

The [paired language-model study](fractional_memory_study.md) implements these
controls with the public adapter and the existing restartable HF driver. It
isolates fixed and learned order without adding another geometric mechanism.
