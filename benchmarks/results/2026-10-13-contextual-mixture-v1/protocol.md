# Contextual Mixture Qualification, V1

Declared before GPU qualification. Base: `4e2e4290d29eca864a3c0d37fb9c375212f4aa2e`.
This is a connectivity/correctness qualification, not a corpus efficacy study.

Four fixed CPU-f32 Torch fixtures combine one/two residual blocks with zero/nonzero
query-projection initialization, on two four-byte input windows and 256-way next
byte outputs. Metrics are Poincare, Euclidean chord and categorical Fisher-Rao,
with the existing fixed scales. The one-block fixture isolates geometry with
zero Q/K; two-block fixtures use learned QKV, external biases and Topos.

- Unchanged absolute gate: `3e-6 + 5e-5 * abs(reference)`.
- Geometry parameter VJPs, router-to-coordinate contrast and router update
  deltas require reference L2 above `1e-8`, nonzero measured signal and relative
  L2 at most `0.002`. Zero-initial router-to-coordinate contrast is exempt because
  it is mathematically zero; its forward must recover the shared reference.
- Initial logits, mixture weights, all parameter/embedding/external-bias VJPs;
  sixteen SGD updates at `0.125`, every geometry VJP, parameter and mixture weight.
- Initial contextual projections use four input features and six output logits.
  Parameter tensor counts are 26/43 for one/two blocks. Generic linear projections
  include a trainable bias in addition to the existing shared intercept. This
  redundant intercept is explicit, not a matched-budget or matched-update control.
- Reject frozen router/projection updates in every step. Nonzero initial cases
  must detect the missing router-to-coordinate VJP against the detached reference.
- Prefix extension, suffix perturbation, document isolation and exact zero future
  input gradients; retained/foreign/stale tape checks; fresh-owner checkpoint
  resume at revision 7 through revision 16, bit-identical within the same runtime.
- An invalid late query projection must guard all parameter/embedding derivatives
  and reject the entire update at rates zero and `0.125`, without changing any
  parameter. A finite extreme-logit sibling must remain valid.

Native Metal and browser WebGPU run the same Rust harness serially against the
same exact fixture. CPU Torch defines the independent reference, never runtime
model semantics. Browser JavaScript only transports the fixture/report.
No promotion threshold may be relaxed after measurement. Failed attempts and
review repairs remain separately identifiable. Raw tensors, checkpoints,
fixtures and executables stay local; publish scalar checks, hashes and recipes.

This does not qualify large-corpus generalization, new metric kinds, long-context
scaling, a causal router conditioned on each block's hidden state, Python bindings,
cross-device bitwise resume, or a speed advantage over PyTorch.

## Failure Capture Amendment

The first native measurement passed. The first browser measurement failed the
unchanged update-delta gate at relative L2 `0.004374396586346818`. Only failure
reporting was then extended: retain complete raw trajectories and per-case
errors with `passed: false`, rather than discarding observations on the first
query-check error. No source math, fixture, reference or gate was changed for
the diagnostic browser rerun. A read-only recomputation compares each actual
update with the separately rounded FP32 SGD candidate from its own measured
gradient. This is diagnosis, not a replacement promotion criterion.

An independent review then found that empty mixture snapshot collections could
skip comparisons. Expected counts are now derived from the plan and enforced at
initialization, every update and resume. Both runtimes were remeasured after this
coverage repair; neither numerical gates nor model semantics changed.
