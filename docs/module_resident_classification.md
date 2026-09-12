# Class-Last Resident Cross Entropy

`st.nn.CrossEntropyWithLogits.evaluate_resident(logits, labels)` now returns the
same owning `ResidentLoss` used by MSE. The existing Rust Loss delegates to the
WGPU kernel; Python and WASM transport configuration and handles, not another
softmax or derivative implementation.

## Python Learning

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Linear("classifier", 16, 7))
baseline = model.inference_plan([2, 8, 16])
learner = baseline.compile_graph_learner_wgpu(gradient_policy="exact")
device = learner.tensor_device()
x = device.upload([2, 8, 16], [0.25] * 256)
labels = device.upload([2, 8], [0., 1., 2., -100.] * 4)
learner.set_input_tensor(x)
objective = st.nn.CrossEntropyWithLogits(label_smoothing=0.05)

receipts = []
for _ in range(32):
    forward = learner.forward()
    loss = objective.evaluate_resident(forward.prediction_tensor(), labels)
    gradient = learner.backward(forward, loss.prediction_gradient_tensor())
    learner.sgd(gradient, 0.1)
    receipts.append(learner.update_snapshot())
    # No value, gradient or parameter observation here.

for receipt in receipts:
    receipt.read()  # Check every attempted update, not just the last.
updated = learner.parameter_snapshot().read_plan()
baseline.apply_parameters_to(model, updated)
```

This illustrates API composition, not a useful dataset or an accuracy result.
Parameter handoff is explicit and weight-only. The original Module is not
silently synchronized after each update. Plain SGD is not a replacement for
`ModuleTrainer` hypergrad, band replay, accumulation or optimizer-state policies.
Update receipts are single-use: the example consumes each once; request a fresh
snapshot for another observation.

In Rust, import `st_nn::Loss` and call
`CrossEntropyWithLogits::new(config)?.evaluate_resident(&logits, &labels)?`.
In WASM, use `new CrossEntropyWithLogits("mean", -100n, 0.05)` and
`evaluateResident(logits, labels)`. Its `ResidentLoss` has `lossTensor()` and
`predictionGradientTensor()`. Free WASM handles when finished. Out-of-range or
non-BigInt ignore IDs are rejected before they can wrap through the WASM ABI.

## Shared Contract

- Logits use the final axis for classes, for example `[batch, time, vocabulary]`.
  Labels match the sample axes, optionally ending in one: `[batch,time]` or
  `[batch,time,1]`. Neither shape flattening nor broadcasting is implicit.
- Labels are finite integral f32 values representable as i64, matching the
  ordinary Loss Tensor transport. Out-of-range IDs fail. An i64 ignore ID that
  no f32 can represent never matches a rounded neighbor. This is not an integer
  storage API for arbitrary large vocabularies.
- `mean` divides by non-ignored samples, not by classes or total padded tokens.
  `sum` sums them. `none` returns sample axes plus a singleton final axis, and
  its prediction gradient uses an all-ones loss seed, as ordinary Loss backward
  does. A custom weighted unreduced-loss VJP is not exposed by this method.
- Label smoothing mixes the class label with a uniform class distribution.
  Class weights and soft-label targets remain unsupported.
- An empty/all-ignored mean fails. Empty/all-ignored sum is zero; none produces
  zero ignored rows. Both outputs carry the same whole-loss guard, including
  input failures, invalid labels and reduction overflow. CPU-known shape errors
  fail immediately; GPU-known failures surface on observation or reject a
  learner update, even with a zero learning rate.

## Numerical Boundary

The CPU path retains its f64 row partitions. Resident execution is f32, not
bit-identical f64 emulation. It excludes one row maximum from the exponential
tail, retains `log1p`-scale small tails and computes the dominant-class gradient
without subtracting two rounded ones. Huge logit gaps and very small f64
smoothing coefficients use aligned significands and binary scales; mean
normalization precedes narrowing the loss. Ordinary-range exponentials retain
a direct subtraction fast path. GPU f32 subnormal behavior remains device-bound.

The implementation bounds exponent construction according to the
[WGSL builtin contract](https://www.w3.org/TR/WGSL/#ldexp-builtin).
This is not a throughput claim. The existing host Tensor forward/backward and
generic CPU autograd routes remain CPU implementations; strict host-WGPU
requests still reject instead of secretly transferring. Resident execution is
explicit and cannot bypass a committed tensor execution plan.

## Verified Example

The [source-bound learning record](../benchmarks/results/2026-09-12-module-resident-classification/README.md)
includes six 64-update classification cases on each native/browser route, 43
numerical probes per route, public Python/WASM update-reject-recover tests and
independent PyTorch CPU/MPS comparisons. The Python block above also ran as-is
against the frozen GPU library. Tiny-tail reference differences, extreme-value
reference choices and failed development runs are retained separately; these
small fixtures establish API/numerical behavior, not LLM quality or throughput.
