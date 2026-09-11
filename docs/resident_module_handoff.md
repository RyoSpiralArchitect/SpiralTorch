# Resident Learning Back to the Original Module

`InferencePlan::apply_parameters_to` closes an explicit round trip:

```text
Rust/Python Module -> resident NN graph -> GPU learning
       ^                                      |
       +--- checked weight handoff <--- owning plan snapshot
```

Intermediate predictions, cotangents and updates stay on the GPU. Read a weight
snapshot at the handoff boundary, not between each operation. The same Rust
checks serve Python and a plan exported by the WASM browser client. CPU-only
builds can import and apply the plan too; applying weights itself needs no GPU.

## Python

Keep the original baseline before compiling/fusing a graph. For example, after
the GPU loop in the [resident learner guide](resident_graph_learner.md):

```python
import spiraltorch as st

baseline = model.inference_plan([2, 2, 1])
learner = baseline.fuse_pointwise().compile_graph_learner_wgpu(
    gradient_policy="exact"
)
# Set the input, then forward/backward/sgd_weighted on GPU, as in that guide.
# Read the owning update receipts to check acceptance.
learned = learner.parameter_snapshot().read_plan()
count = baseline.apply_parameters_to(model, learned)
prediction = model.forward(st.Tensor(4, 1, [1.0] * 4))
```

The default `optimizer_state="reject"` rejects **any** attached hypergrad or
realgrad tape and any pending Euclidean gradient, including an all-zero one.
If starting a new training phase is intentional:

```python
baseline.apply_parameters_to(model, learned, optimizer_state="reset")
trainer = st.nn.ModuleTrainer(backend="cpu")
trainer.prepare(model)
```

`reset` discards parameter-local gradients and tapes. It does **not** reset an
old trainer's scheduling, telemetry, curvature, distributed or other state.
Create/configure a new trainer for the new phase. This operation is not an
optimizer checkpoint resume, and resident SGD does not become `ModuleTrainer`'s
hypergrad, band replay or geometry policy merely because weights are transferred.
The original baseline cannot be replayed after values have changed; snapshot a
new baseline before another resident phase.

## Rust and Browser

Rust calls the same core directly:

```rust,ignore
let count = baseline.apply_parameters_to(
    &mut model,
    &learned,
    st_nn::resident::ModuleOptimizerStatePolicy::Reject,
)?;
```

WASM learning exports the existing portable plan; it does not reimplement a
Python/Rust Module or the handoff policy. For a graph learner:

```javascript
const snapshot = learner.parameterSnapshot();
const learned = await snapshot.readPlan();
const payload = learned.toJson();
snapshot.free();
learned.free();
```

Pass `payload` through the application's transport and parse it with
`st.nn.InferencePlan.from_json(payload)` on the receiving side. The browser
mean-MSE trainer can instead export `stateSnapshot().readState().toPlan()`.
Free every owning browser handle after use, including intermediate snapshots.

## Acceptance Boundary

- Supported built-ins are `Linear`, `Gelu`, `Relu`, `Scaler` and `Sequential`.
  Weight, bias and gain slots all transfer; packed forward/transpose weight
  caches are invalidated. Existing supported host tensor layouts are preserved.
- Both the current Module and updated plan must match the baseline's input
  layout, graph program, parameter slots, roles and shapes. Only the existing
  Rust pointwise-fusion normalization is used to compare fused/unfused programs;
  arbitrary mathematical equivalence is not inferred.
- The current Module's logical parameter values must still match the baseline
  bit for bit. Signed-zero drift is a change. Imported values must satisfy the
  existing finite/bounded plan validation.
- Custom Rust Modules must opt into `resident_parameter_bindings` in lowering
  slot order. Bindings must identify the actual visited parameters, once each,
  with unique names. Duplicate names, tied slots and foreign parameter references
  are rejected rather than guessed. Inference lowering alone does not opt in.
- All checks and new tensor allocation precede mutation. Commit relies on the
  `Module` contract that parameter visitors remain stable and do not perform
  fallible work after successful callbacks. An implementation violating that
  contract is not a supported transactional participant.
- A matching plan is a value/program compatibility check, not object identity,
  provenance, lineage or authorization. Separately loaded matching Modules are
  allowed. The application owns transport and admission policy.

The public fixture uses a `[2, 2, 2]` plan with two Linear layers, GELU, ReLU and
two gains, trains on WebGPU, transfers all six parameters to the original host
model and starts a fresh `ModuleTrainer` phase. Both gradient policies and fused
and unfused plans are exercised. This is a bounded integration check, not a
throughput/model-quality claim or automatic migration of all `pure::Tensor`
operations to GPU storage.

This integration also exposed a column-major `pure::Tensor` transpose bug that
affected `Linear.backward` input gradients. Transpose now preserves logical
values on CPU and WGPU; the existing row-major CPU/kernel paths are retained.
The regression tests cover non-square/edge shapes and CPU copy-on-write. They
also check that exporting the result via DLPack does not expose the original
input to external writes, and that foreign-backed inputs produce independent
results. DLPack remains row-major-only; protected snapshots can share storage.
These tests do not certify
every tensor layout or optimizer's layout handling.
