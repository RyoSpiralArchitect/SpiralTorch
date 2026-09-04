# Rust-owned autograd contract

SpiralTorch has one reverse-mode semantic core. `st-tensor::AutogradTensor`
owns graph construction, local derivative rules, gradient accumulation, and
backward-pass invariants. Rust applications call it directly; Python and WASM
hold handles to the same graph and transport its values and receipts.

The current contract is `spiraltorch.autograd.v1`, with semantic owner
`st-tensor`.

## Ownership boundaries

| Surface | Responsibility | Does not own |
|---|---|---|
| `st-tensor::Tensor` | Copy-on-write 2D storage, protected snapshots, and backend-dispatched tensor operations | Compute-graph or optimizer semantics |
| `st-tensor::AutogradTensor` | Immutable graph nodes, local VJP rules, atomic reverse-mode accumulation | Training-loop policy or higher-order solver choice |
| `st-core::autograd::hypergrad` | Unrolled and implicit higher-order differentiation over `AutogradTensor` | Primitive tensor derivative formulas |
| `AmegaHypergrad` / `AmegaRealgrad` | Z-space gradient accumulation and optimizer application | Compute-graph autograd |
| WGPU/CPU/device backends | Execute `Tensor` operations selected by the runtime | Change graph semantics or receipt fields |
| Python | Orchestrate experiments and expose Rust handles | Reconstruct gradients, normalization, or solver heuristics |
| WASM | Equal browser client over Rust handles | JavaScript fallback derivative formulas |

This separation is deliberate: execution backends can become faster without
changing derivatives, while clients can become richer without forking the
meaning of a backward pass.

## V1 invariants

- Graph edges are immutable and graph identity is assigned atomically.
- Leaves explicitly declare whether they require gradients.
- Leaves capture isolated snapshots in logical row-major order. Pre-existing
  mutable native/DLPack aliases cannot change graph values after construction.
- Saved gradients also use snapshots; mutating a seed or an exported gradient
  cannot change already committed gradient state.
- Implicit `backward()` is accepted only for a scalar `1 x 1` output.
- Non-scalar outputs require an explicit seed through `backward_with_grad()`.
- Seeds are interpreted in logical row-major order, including supported
  non-row-major Tensor layouts passed by direct Rust callers.
- A backward pass validates all local and accumulated gradients before any
  persistent gradient is committed.
- Repeated backward passes accumulate; `zero_grad()` and `zero_grad_graph()`
  clear gradients explicitly.
- `vector_jacobian_product()` is side-effect-free, ignores accumulated gradient
  state, and returns zero for a tracked input disconnected from the output.
- Concurrent backward calls cannot lose updates.
- Telemetry is emitted after the graph lock is released, so observers may read
  committed gradients without deadlocking.
- Every graph summary and binding receipt identifies the contract version and
  semantic owner.

V1 includes add, subtract, Hadamard product, matrix multiplication, scalar
scale, transpose, sum, mean, dot product, mean-squared error, broadcast row bias,
ReLU, tanh-approximate GELU, row softmax, row log-softmax, and integer-label
cross entropy with logits. These methods are additive;
snapshot isolation enforces the existing immutable-value invariant rather than
introducing a second contract version. Unsupported operations belong in this
Rust module first, with closed-form and finite-difference tests.

## Snapshot and interchange semantics

`Tensor::into_snapshot()` consumes uniquely owned native storage without copying.
Shared mutable storage and all foreign buffers are copied, including read-only
imports whose original owner may later regain write access. `snapshot()` borrows
the input and isolates it; an existing snapshot is cheap to clone.

`AutogradTensor::value()`, `grad()`, and `detach()` can share snapshot storage.
Rust mutation uses copy-on-write. Versioned DLPack exports mark it read-only;
legacy consumers receive a copy, or a copy-forbidden request fails. Explicit
copies remain writable. This protection applies to compliant DLPack consumers,
not unsafe writes that ignore read-only flags or storage lifetimes.

Ordinary mutable Tensor DLPack sharing is unchanged. These snapshots capture
values, not another framework's autograd graph. The caller must still prevent
concurrent writes to foreign memory while snapshot capture is reading it.

## Nonlinear training

| Operation | Forward and VJP contract |
|---|---|
| `add_row(bias)` | Bias has shape `(1, cols)`; its gradient sums over rows without implicit averaging. The CPU reduction accumulates in f64, then validates the final f32 result. |
| `relu()` | Derivative is zero for nonpositive inputs, including exactly zero. |
| `gelu()` | Uses the existing tanh approximation, not erf GELU; saturated derivatives stay finite. |
| `row_softmax()` | Uses the full coupled row VJP, not an elementwise approximation. A constant row cotangent gives zero. |

The kernels remain in Rust. GELU forward and the centered f64 softmax VJP run
on CPU; matrix products, softmax forward, and GELU backward can use existing
WGPU paths. This is not a claim that the whole graph stays GPU-resident. Empty
tensors preserve their shapes, and empty GELU backward is an explicit no-op
even with strict GPU fallback policy.
The upper-layer GELU CPU fallback delegates to `st_tensor::gelu_derivative`,
sharing the same saturation and finite-input rules instead of a second formula.

GELU parity tests use PyTorch's [tanh approximation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html).
The Rust VJP tests also use finite differences and cover shared branches,
nonuniform cotangents, bias reduction, invalid inputs, and failure atomicity.

Run the deterministic nonlinear fixture from either client:

```sh
cargo run -p st-tensor --example autograd_xor
python examples/autograd_xor.py
```

Both train a `2 -> 8 -> 2` GELU/softmax model for 600 steps on four XOR examples,
requiring all four predictions correct and a greater-than-tenfold MSE reduction.
This is evidence for working training mechanics, not an HF fine-tuning result
or a Z-space quality advantage.

The WASM methods are `addRow`, `relu`, `gelu`, and `rowSoftmax`. The executable
`bindings/st-wasm/tests/autograd_nonlinear.cjs` checks the real nodejs-target
WASM exports, including VJP and empty-shape behavior:

```sh
wasm-pack build bindings/st-wasm --target nodejs --out-dir /tmp/spiraltorch-wasm
node bindings/st-wasm/tests/autograd_nonlinear.cjs /tmp/spiraltorch-wasm/spiraltorch_wasm.js
```

## Classification from logits

`Tensor` and `AutogradTensor` both expose `row_log_softmax()` and
`cross_entropy_with_logits(labels, config)`. `CrossEntropyConfig` selects
`LossReduction::{None, Sum, Mean}`, `ignore_index` (default `-100`), and
`label_smoothing` (default `0`, valid finite range `[0, 1]`). Predictions have
shape `(samples, classes)` and labels contain one integer per row. Token logits
can use flattened sample/token rows; shifting language-model targets remains
the caller's responsibility.

- Mean divides by the count of non-ignored rows; an all-ignored or empty mean
  is an error rather than NaN or a silently successful training step.
- Unreduced output is `(samples, 1)` with zero at ignored rows; sum and mean
  return `(1, 1)`. All-ignored sum/unreduced output and gradients are zero.
- Labels outside the class range, invalid smoothing, non-finite logits
  (including ignored rows), and invalid backward seeds are rejected.
- Smoothing mixes the class target with the uniform distribution over classes.
  Class weights and soft-label distributions are not supported by this API.
- CPU f64 row partitions use a shifted `log1p` tail and only narrow the final
  reduction to f32. Large common offsets and tiny dominant-class residuals
  do not disappear through `log(softmax(...))` or subtraction of rounded ones.
  Log-softmax uses a compensated f64 cotangent sum so small seeds survive
  cancellation between large positive and negative upstream values.
- Direct Tensor backward methods take the original logits and a seed matching
  the loss output shape. Autograd owns labels/configuration and delegates these
  VJPs to the same kernels; clients do not implement another derivative.
- These new kernels are explicitly CPU, not WGPU kernels or a claim of GPU
  residency. `st-nn::CrossEntropyWithLogits` rejects strict WGPU execution.
  Allowed CPU fallbacks retain the requested WGPU route in adapter metadata.
- Kernel notifications from reverse mode are deferred until the graph lock is
  released, including when validation fails. Completed kernel events are not
  proof that a backward pass committed; only a successful backward receipt is.

The definition follows PyTorch's integer-target, unweighted
[cross entropy](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
and [log-softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.log_softmax.html).
The explicit all-ignored mean error and stricter finite checks are intentional
differences. Tests compare finite differences and an independent float64 PyTorch
oracle without depending on PyTorch for runtime computation.

```rust
use st_tensor::{AutogradTensor, CrossEntropyConfig, Tensor};

let logits = AutogradTensor::variable(Tensor::from_vec(2, 3, vec![2., 0., -1., 0., 2., -1.])?)?;
let loss = logits.cross_entropy_with_logits(&[0, 1], CrossEntropyConfig::default())?;
loss.backward()?;
# Ok::<(), st_tensor::TensorError>(())
```

Python spells the options as keyword arguments. For `ModuleTrainer`, use
`st.nn.CrossEntropyWithLogits` with an integral `(samples, 1)` target Tensor;
the adapter validates the transport and delegates to st-tensor. Its unreduced
backward uses an all-ones seed, matching the trainer's sum of unreduced losses.
Existing `CategoricalCrossEntropy`/`SoftmaxCrossEntropy` still take probabilities
and one-hot targets, and `CrossEntropy` remains the hyperbolic-loss alias.

```python
import spiraltorch as st

logits = st.AutogradTensor.variable(st.Tensor(2, 3, [2., 0., -1., 0., 2., -1.]))
loss = logits.cross_entropy_with_logits([0, 1], label_smoothing=0.05)
loss.backward()
print(loss.item(), logits.grad().tolist())
```

`python examples/autograd_classification.py` trains a `2 -> 12 -> 3` GELU
classifier for 300 steps on 72 synthetic points and checks 36 disjoint points.
The fixture asserts a minimum distance between training and validation points.
This is a working-pipeline fixture, not evidence of real-world generalization or
of a Z-space advantage over ordinary fine-tuning.
WASM exposes `rowLogSoftmax()` and
`crossEntropyWithLogits(Int32Array, reduction?, ignoreIndex?, labelSmoothing?)`.
Its executable `bindings/st-wasm/tests/classification.cjs` checks real WASM
exports, masking, tiny residuals, VJPs, and a 180-step classifier.

## Direct Rust graph use

```rust
use st_tensor::{AutogradTensor, Tensor};

let value = Tensor::from_vec(1, 3, vec![1.0, 2.0, -1.0])?;
let x = AutogradTensor::variable(value)?;
let loss = x.hadamard(&x)?.add(&x.scale(3.0)?)?.sum()?;
let receipt = loss.backward()?;

assert_eq!(x.grad().unwrap().data(), &[5.0, 7.0, 1.0]);
assert_eq!(receipt.leaf_gradient_count, 1);
# Ok::<(), st_tensor::TensorError>(())
```

## Client rule

Python's `spiraltorch.AutogradTensor` and WASM's `AutogradTensor` deliberately
mirror the Rust methods. A client may choose when to build a graph, which
backend to request, and where to store receipts. It must not silently supply a
non-scalar seed, clamp a gradient, or reimplement a derivative. Any semantic
change requires a Rust implementation, Rust tests, and a contract-version
decision before the client surfaces are updated.

When v1 was introduced, uncompiled approximate hypergradient and detached
device-CG files were removed rather than left as alternate semantic entry
points. A future device-resident Krylov solver belongs behind the canonical
`st-core::autograd::hypergrad` contract, with the same equation-residual
diagnostics and backend parity tests; it must not return as a parallel ops
module.
