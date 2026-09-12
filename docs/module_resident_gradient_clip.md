# Resident Global Gradient Clipping

`ResidentGraphLearner` supports an optional global L2 norm limit. One limit
applies to all weight, bias and gain gradients together, not each tensor
individually. Rust owns the contract; Python and WASM only configure it.

## Usage

After compiling a learner from an ordinary Module's inference plan:

```python
learner.set_grad_clip_max_norm(1.0)
assert learner.grad_clip_max_norm == 1.0
# The setting applies to sgd, sgd_weighted and sgd_accumulated.
# The normal resident forward/loss/backward/accumulation loop is unchanged.
learner.sgd_accumulated(accumulator, 0.1)
receipt = learner.update_snapshot()
receipt.read()  # Explicit, single-use acceptance observation.
learner.clear_grad_clip()
assert learner.grad_clip_max_norm is None
```

See the [complete microbatch example](module_resident_microbatch.md) for model,
loss and accumulator construction. Set the limit before its update loop.
Rust uses the same method names and an `Option<f32>` getter. WASM exposes
`setGradClipMaxNorm`, `clearGradClip` and `gradClipMaxNorm` (`undefined` when
disabled). The setting is disabled by default. Nonpositive or nonfinite limits
fail without replacing the previous setting or invalidating parameter state.

## Calculation Order

1. Sum the explicitly weighted, exact parameter VJPs in contribution order.
2. Apply the selected gradient policy. `module_compatible` averages gain
   gradients over the compiled microbatch's rows; weight/bias gradients do not
   receive that extra average.
3. Compute a single norm over these effective gradients, then scale them all
   by `min(1, max_norm / norm)`.
4. Apply optional [Topos EMA momentum](module_resident_momentum.md) to the
   clipped gradient. The global limit does not reclip previous history.
5. Apply the learning rate, validate every candidate, and commit all parameters
   together or none.

The shared `st-kernel-contracts::gradient_clip` contract retains ModuleTrainer's
historical no-ops: norms at most `f32::EPSILON` are left alone, as are scales
within `f32::EPSILON` of one. Consequently, limits below that norm floor are
not a strict bound on arbitrarily tiny gradients. There is no added denominator
epsilon. This differs slightly from PyTorch's convenience clipping function.

Raw VJP handles and accumulator snapshots remain unmodified. Clipping happens
only at update, not per contribution, so cancellation across valid contributions
is preserved. A zero learning rate still validates the gradients. Clipping
cannot repair an already-invalid loss, VJP, or accumulated sum: their guards
still reject the entire update, even if the offending contribution has zero
weight. It can prevent overflow in the subsequent learning-rate multiplication
when the unclipped effective gradient is finite.

## Execution And Limits

The optional GPU workspace is prepared once on first enable and reused across
updates and disable/re-enable cycles. Native WGPU and browser WebGPU use the
same exponent-aware scaled norm reduction and ordered scale factors. No norm,
gradient or parameter is read back to make the decision. Disabled learners
retain the original parameter-update kernels and dispatch sequence.

The ordinary Rust trainer also uses the shared contract. Its norm no longer
rounds a finite f64 result to f32 infinity before choosing the scale, and very
small scale factors are split into normal-f32 factors instead of rounding to
zero prematurely. This is still f32 arithmetic: rounding and genuinely
subnormal gradient/output handling can differ by backend. It is not a promise
of bit-identical output or a new mixed-precision optimizer.

This does not replace ModuleTrainer's hypergrad, realgrad, spectral adapter or
band policies with SGD. It adds their common global-norm primitive to the
explicit resident learner. CPU-only bindings still cannot compile that learner.
Loss quality, throughput and CUDA behavior require separate measurements.

## Verified Execution

The [source-bound record](../benchmarks/results/2026-09-12-module-resident-gradient-clip/README.md)
includes native/browser learning, Python/WASM public clients, independent
PyTorch CPU/MPS replay and the original wide-norm failures. Restrictive clipping
slowed learning on these small fixtures compared with unclipped controls;
the result is update-control correctness, not an accuracy or speed advantage.
