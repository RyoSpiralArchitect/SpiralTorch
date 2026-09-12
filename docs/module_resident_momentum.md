# Resident Topos Momentum

The resident learner can now keep Topos' EMA gradient history on GPU. The
canonical transition and validation live in `st-kernel-contracts::momentum`;
ordinary `ToposOptimizerStateControl::gradient_step` and native/browser resident
updates consume that contract. Python and WASM transport settings and handles.

This is **not** PyTorch heavy-ball/Nesterov momentum or Adam:

`next_momentum = damping * previous_momentum + (1 - damping) * effective_gradient`

The initial history is zero, including on the first update. Damping must be
finite in `[0, 0.85]`, preserving Topos' range. Invalid settings preserve both
the previous configuration and the existing history. The default is disabled.

## Python Example

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Linear("classifier", 4, 3))
baseline = model.inference_plan([2, 3, 4])
learner = baseline.compile_graph_learner_wgpu(gradient_policy="exact")
learner.set_grad_clip_max_norm(1.0)
learner.set_momentum_damping(0.6)
device = learner.tensor_device()
x = device.upload([2, 3, 4], [(i % 7) / 8 for i in range(24)])
y = device.upload([2, 3], [0., 1., 2., 0., 1., 2.])
objective = st.nn.CrossEntropyWithLogits(label_smoothing=0.1)
receipts = []
for _ in range(32):
    learner.set_input_tensor(x)
    forward = learner.forward()
    loss = objective.evaluate_resident(forward.prediction_tensor(), y)
    gradient = learner.backward(forward, loss.prediction_gradient_tensor())
    learner.sgd(gradient, 0.1)
    receipts.append(learner.update_snapshot())
    # No tensor or optimizer-state readback in the update loop.
for receipt in receipts:
    receipt.read()
history = learner.momentum_tensors()  # Owning GPU snapshots, not host reads.
updated = learner.parameter_snapshot().read_plan()
baseline.apply_parameters_to(model, updated)  # Explicit weight-only handoff.
```

The same option applies to `sgd_weighted` and
[`sgd_accumulated`](module_resident_microbatch.md). The order is weighted raw
VJP sum, selected gain row normalization, optional global gradient clipping,
EMA, learning-rate multiplication, then whole-state commit. The global clip
limit bounds the new effective gradient under its documented epsilon rule;
it does not reclip old history or guarantee the EMA norm is below a newly
lowered limit. EMA has no bias correction or automatic learning-rate rescaling.

## Lifecycle And Atomicity

- `set_momentum_damping(d)` while enabled changes damping and preserves history.
  Damping zero still records the effective gradient; it is not disabled state.
- `reset_momentum()` zeros enabled history on GPU without reallocating buffers
  or changing weights. Resetting while disabled is an error.
- `clear_momentum()` disables the option. Subsequent plain-SGD steps do not use
  history. Re-enabling starts from zero, not stale history from before them.
- One invalid loss, VJP, composition, EMA or parameter candidate rejects the
  entire update: no weight or history slot is committed. Zero contribution
  weights cannot erase invalid sources. Recovery uses the last committed state.
- A zero-rate attempt validates its candidates but changes neither parameters
  nor history. This is the resident learner's probe rule, not PyTorch's behavior
  of advancing momentum at zero learning rate. Attempt counters and gradient
  invalidation still follow the existing resident update contract.
- `momentum_tensors()` returns owning GPU snapshots in parameter order. They
  survive later updates, resets, disabling and learner destruction. Snapshots
  after a rejected update still expose the valid last-committed history; only
  `update_snapshot().read()` proves that the attempted update was accepted.

Rust uses the same method names. WASM exposes `momentumDamping`,
`setMomentumDamping`, `clearMomentum`, `resetMomentum` and `momentumTensors`.
The disabled getter is Python `None` / Rust `None` / WASM `undefined`. Free WASM
handles when done. CPU-only bindings cannot compile a GPU learner.

## Boundaries

Two history-sized GPU buffers per parameter are allocated once on first enable
and reused. State snapshots allocate only when explicitly requested. With the
option disabled, the previous SGD/clip path is unchanged; no momentum dispatch
or host readback is added to it.

Gradient normalization and optional clip scaling now feed the EMA candidate
directly in one kernel per parameter. The intermediate effective-gradient
buffer write/read is removed from this path, not from ordinary SGD's observable
state. Norm reduction, global validation and whole-state commit remain separate.
This saves one dispatch per parameter per update; it does not imply a measured
end-to-end speedup. See [the matched benchmark protocol](resident_momentum_fusion.md).

The shared transition connects an existing Topos primitive, not the full
Topos RMS bias/clipping policy, ModuleTrainer's hypergrad tapes, or spectral
adaptation. Parameter snapshots/handoff are still weight-only; momentum
snapshots are observations, not a restorable optimizer checkpoint. Dropping a
learner loses its live optimizer state. Full optimizer resume remains separate.
f32 rounding, performance, generalization and CUDA require their own evidence.

## Verified Execution

The [source-bound native/browser run](../benchmarks/results/2026-09-12-module-resident-momentum/README.md)
passed 69 verification stages and 24 independent PyTorch CPU/MPS EMA replays
(38,472 tensor/scalar comparisons). This Python example also ran unchanged;
weight handoff reproduced the resident output exactly. The small learning
fixtures lowered loss, but all EMA runs finished worse than their clip-only
controls at 32 updates. These results establish correctness, not an optimizer
quality or performance advantage.

The later [preparation-fusion record](../benchmarks/results/2026-09-12-resident-momentum-fusion/README.md)
rechecks the same regression suite and this example, and adds two complete
native/browser/Torch timing matrices. Fewer GPU dispatches preserve numerical
results, but timing improvements are small and conditional; slower cases are
retained rather than hidden.
