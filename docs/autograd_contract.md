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
| `st-tensor::Tensor` | Immutable 2D values and backend-dispatched tensor operations | Compute-graph or optimizer semantics |
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
- Implicit `backward()` is accepted only for a scalar `1 x 1` output.
- Non-scalar outputs require an explicit seed through `backward_with_grad()`.
- A backward pass validates all local and accumulated gradients before any
  persistent gradient is committed.
- Repeated backward passes accumulate; `zero_grad()` and `zero_grad_graph()`
  clear gradients explicitly.
- Concurrent backward calls cannot lose updates.
- Telemetry is emitted after the graph lock is released, so observers may read
  committed gradients without deadlocking.
- Every graph summary and binding receipt identifies the contract version and
  semantic owner.

V1 includes add, subtract, Hadamard product, matrix multiplication, scalar
scale, transpose, sum, mean, dot product, and mean-squared error. Unsupported
operations should be added to this Rust module first, with closed-form and
finite-difference tests, before bindings expose them.

## Direct Rust use

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
