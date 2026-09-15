# Pointwise Seeds Directly Into Graph VJPs

`ResidentGraphAutograd::backward_pointwise` and
`ResidentGraphLearner::backward_pointwise` evaluate an existing checked
`PointwisePlan` directly into the private cotangent buffer, then run the
graph's exact VJP in the **same GPU submission**.

This evaluates the supplied program as a cotangent. It does **not** differentiate
that program or infer a loss function. Ordinary `backward(forward, tensor)`
remains available, including for seeds from reductions or classification losses.

## Python

With a compiled learner `gpu`, a same-device negative target `negative`, and
a scalar normalization tensor `norm`:

```python
from spiraltorch.wgpu import WgpuPointwiseInputs

forward = gpu.forward()
error = forward.prediction_tensor().add(negative)
inputs = WgpuPointwiseInputs()
inputs.add(error)
inputs.add(norm)
quadratic = inputs.compile([("multiply", 1)])
quartic = inputs.compile([
    ("multiply", 0), ("multiply", 0), ("multiply", 1),
])
first = gpu.backward_pointwise(forward, quadratic, inputs)
second = gpu.backward_pointwise(forward, quartic, inputs)
```

For repeated steps, compile these plans once. Replace slot zero using
`inputs.set(0, new_error)`, then pass the current forward token. Plans retain
layouts and device identity, not the old input values.

Rust accepts `&PointwisePlan` and `&[&ResidentTensor]`. Browser clients call
`owner.backwardPointwise(forward, plan, inputs)` with `WgpuPointwiseInputs`.
Both clients delegate validation, seed evaluation and VJP to the same Rust code.

## Contract And Boundary

- The current owning forward token, exact prepared input layouts, output shape,
  and same device/queue are required. Host validation failures leave the tape
  and submitted-backward counter unchanged.
- N-D strides, offsets, broadcasting, original-input residual references and
  checked intermediates use the existing fused pointwise semantics.
- Invalid inputs, intermediate overflow, or invalid forwards guard the entire
  returned VJP. A later ReLU, zero cotangent or zero batch weight cannot hide them.
  Independent later VJPs can recover from a bad seed, not from a bad forward.
- Returned gradient tensors retain their existing owning-version lifetime.
  Weighted updates, clipping, Topos EMA and transaction rollback are unchanged.
- Relative to `plan.run(inputs, Fused)` followed by `backward`, this avoids one
  seed-value buffer, its seed-to-tape copy, and one submission per VJP. It does
  not remove the owning prediction capture, input-to-tape copy, intermediate
  gradient scratch, or acceptance receipt reads. It is not an automatic
  migration of `pure::Tensor` or `ModuleTrainer`.

## Matched Measurement

The native training runner's `--learner --direct-learner-seeds` option compares
**identical fused quadratic/quartic seed programs**: materialized in the baseline,
direct in the candidate. Both retain two independent VJPs and ordered weighted
updates. The browser runner uses `learner standard direct-learner-seeds`.
Ordinary, legacy seed-fusion and host-profile workloads remain separate.

Use the same source-bound executable/module for both lanes to isolate the route.
Three shapes, three seeds, three optimizer settings, eight updates per interval,
two warmups and eight retained blocks per cadence are the existing bounded
measurement matrix. Eager Torch remains an independent numerical/timing control,
not an equivalent finite-guard/rollback implementation. Fewer submissions/copies
alone do not establish a speedup; inspect the retained results per client.

The [two-round matched results](../benchmarks/results/2026-09-15-resident-pointwise-cotangent/README.md)
retain all conditions, including regressions. Saved-state validation passed;
group speed ratios were close to one, so this is not a universal speed win.
