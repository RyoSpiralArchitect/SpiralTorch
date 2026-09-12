# Resident Microbatch Accumulation

`ResidentGraphLearner` can now accumulate exact parameter gradients from
different input batches at one unchanged parameter state, then perform one
transactional update. Rust owns the state checks and checked GPU composition;
Python/WASM pass handles and explicit weights, not reconstructed derivatives.

This is distinct from `GraphGradientBatch`, which still combines objectives or
band cotangents from the same forward. Its existing same-forward contract and
256-term bound are unchanged. A `GraphGradientAccumulator` has a checked u64
contribution count and reuses fixed GPU parameter buffers rather than retaining
every source VJP. Explicit snapshots still allocate owning outputs.

## Python Example

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Linear("classifier", 4, 3))
baseline = model.inference_plan([2, 3, 4])
learner = baseline.compile_graph_learner_wgpu(gradient_policy="exact")
device = learner.tensor_device()
labels = [[0., 1., 2., 0., -100., -100.],
          [1., 2., 0., 1., 2., 0.],
          [2., 0., -100., -100., -100., -100.]]
batches = [
    (device.upload([2, 3, 4], [((i + j) % 7) / 8 for i in range(24)]),
     device.upload([2, 3], y), sum(v != -100 for v in y))
    for j, y in enumerate(labels)
]
objective = st.nn.CrossEntropyWithLogits(label_smoothing=0.1)
accumulator = learner.gradient_accumulator()
receipts = []
for _ in range(32):
    learner.zero_accumulator(accumulator)
    for x, y, valid in batches:
        learner.set_input_tensor(x)
        forward = learner.forward()
        loss = objective.evaluate_resident(forward.prediction_tensor(), y)
        gradient = learner.backward(forward, loss.prediction_gradient_tensor())
        learner.accumulate(accumulator, gradient, valid / 12)
    learner.sgd_accumulated(accumulator, 0.1)
    receipts.append(learner.update_snapshot())
    # No GPU value/gradient/parameter observation inside either loop.

for receipt in receipts:
    receipt.read()  # Single-use; check each attempted update.
updated = learner.parameter_snapshot().read_plan()
baseline.apply_parameters_to(model, updated)
```

This is a small API-composition example, not accuracy or throughput evidence.
Inputs retain the compiled N-D shape. Replacing values or same-shaped resident
views is supported; variable microbatch shapes do not trigger recompilation or
implicit padding. The weights above turn three per-batch mean losses with 4,
6 and 2 valid labels into a mean over all 12 labels. Weights are explicit: the
accumulator never guesses a sample count or divides by its contribution count.
For sum-reduced losses, use `1 / total_valid` for each contribution instead.
An all-ignored mean still fails even with zero weight; use sum reduction when
such microbatches must contribute a guarded zero to a nonempty window.

Rust exposes `gradient_accumulator`, `accumulate`, `zero_accumulator` and
`sgd_accumulated` on the same learner. WASM uses `gradientAccumulator`,
`accumulate`, `zeroAccumulator` and `sgdAccumulated`; create the accumulator
through its learner, not its constructor, and free handles when finished.
WASM contribution counts and `parameterGeneration` are BigInt.

## State And Failure Rules

- Contributions may come from different completed forwards/inputs, but must
  belong to this learner and the same parameter generation. The newest forward
  restriction for ordinary backward and same-forward SGD remains unchanged.
- `accumulate` is an ordered sum of `weight * exact_parameter_vjp`. It does not
  accumulate input gradients or introduce loss scaling, clipping or momentum.
  Negative finite weights are allowed; nonfinite weights fail before submission.
- Each attempted update, including zero-rate and GPU-rejected updates,
  invalidates old accumulators/gradients for further learning. Explicitly call
  `zero_accumulator` to reuse storage at the current generation; old gradients
  remain old. Resetting through a different learner is an error.
- Reset is logical, not a GPU read or reallocation. The next first contribution
  overwrites old sums and guards. An empty window cannot be observed or updated.
  Snapshots taken before reset remain independent and keep their original guard.
- Source failures and every checked multiplication/addition are sticky for the
  window. Zero weight or later cancellation cannot erase overflow or an invalid
  loss/VJP. One invalid parameter prevents the whole parameter transaction.
- `parameter_gradient_tensors()` (Rust `parameter_gradients`, WASM
  `parameterGradientTensors`) returns owning GPU snapshots in stable parameter
  order. It does not read them to the host. Capture only when needed; holding
  every diagnostic snapshot defeats the accumulator's bounded-storage benefit.

`exact` updates use the accumulated raw derivatives. `module_compatible` applies
the already-selected gain row average once, using the compiled microbatch row
count. It is the weighted sum of per-microbatch Module gradients, not a new
average over the entire virtual batch. The update receipt identifies its last
contribution's input/forward and the attempted update; it is not a list of all
contributing batches. Keep application batching metadata if that is needed.

Optional [global clipping](module_resident_gradient_clip.md) applies after this
policy and before the learning rate, once per update rather than per microbatch.
Use `learner.set_grad_clip_max_norm(1.0)` before the loop to enable it.
Optional [Topos EMA momentum](module_resident_momentum.md) follows clipping
and advances once per accepted nonzero-rate update, not per contribution.

## Boundaries

This supplies the missing cross-input accumulation primitive; it does not
silently replace `ModuleTrainer` hypergrad, roundtable bands, distributed
accumulators, spectral learning-rate adaptation or optimizer state with SGD.
Parameter handoff to an ordinary Module remains explicit and weight-only.
CPU-only clients cannot compile a GPU learner. Unsupported routes never
silently fall back. No CUDA or speed advantage is implied by this connection.

## Verified Execution

The [source-bound learning record](../benchmarks/results/2026-09-12-module-resident-microbatch/README.md)
includes six cases per native/browser route, each with 95 changing microbatches
and 32 updates, plus independent PyTorch CPU/MPS replay. Public Python/WASM
clients verify weighted means/sums, stale-state rejection and recovery; the
Python example above also ran unchanged. Small synthetic-data loss reductions
are reported separately from any future quality or performance experiment.
