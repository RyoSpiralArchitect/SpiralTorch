# Shared Global Gradient Clipping For QKV

Status: native implementation, CPU binding compilation, one CPU extension build
and isolated Python execution verified. No pretrained-model study, historical
evaluation scoring, GPU execution, cache deletion, push or wheel publication.

The geometry optimizer already uses `st-kernel-contracts::gradient_clip`.
The QKV optimizer now uses that same operation instead of adding independent
normalization in Python or duplicating numerical thresholds in `st-nn`.

## Update Contract

Rust `NativeCausalLmQkvAdamW::new_with_gradient_clip` accepts optional
`GlobalNormClip`; `new` remains the unclipped constructor. Python adds the
keyword-only `max_gradient_norm=None` and a read-only getter. Python rejects
booleans and transports the scalar; the shared Rust contract admits positive,
finite f32 limits.

Every existing QKV update path reaches one common application point: finish
the full logical gradient mean, compute one global Q/K/V norm in f64, obtain
the shared ordered f32 factors, then send the effective gradient to AdamW.
This does not clip individual length groups, change answer weighting or apply
separate scales to Q, K and V. If the factors are empty there is no extra copy
of the gradient buffer. Raw gradient queries and old per-part norms do not change.

An enabled step report adds `gradient_clip`, with the configured limit, global
before/after norms and ordered factors. These describe the derivative before
AdamW, not the final weight displacement. Zero-rate calls still report gradient
processing but do not alter weights, moments or the optimizer clock.

The shared small-norm floor is f32 epsilon; near-one scale changes are also
skipped. These historical common-contract boundaries remain intact, including
when a configured cap is below the floor. This is not strict mathematical
projection or a claim of PyTorch API/numerical equivalence. Clipping cannot
repair a gradient that has already become nonfinite.

Native binary checkpoints optionally store `max_gradient_norm_bits`. Unclipped
checkpoints omit it and retain the previous byte serialization. Enabled
checkpoints restore the exact f32 policy; invalid limits are rejected even
after resealing a modified header. Serving identity excludes optimizer policy
and moments, so equal weights still have equal inference owners. Old readers
which reject unknown header fields will refuse a clipped checkpoint, rather
than silently discard its policy.

## Executed Checks

The initial offline native invocation passed 218 decoder tests in 19.87 seconds
including compilation (1.99 seconds for test bodies). Three added methods check:

- The actual smoothed ragged gradient mean produces the exact expected weights
  when passed through shared clip factors and AdamW candidates; clipping is
  active, not a vacuous high-threshold test. AdamW epsilon 0.1 makes gradient
  scaling observable in the update rather than mostly canceling on the first step.
- Exact checkpoint/restored policy, subsequent update/report equality, unchanged
  raw-gradient queries, and state preservation on zero rate or invalid input.
- Unspecified/explicit-None byte equality, omitted legacy fields, inactive-clip
  weight/moment payload equality, and invalid resealed limit rejection.

The CPU binding check passed in 16.32 seconds. One release extension build took
115.75 seconds. These reused the scoped Cargo target with incremental output
disabled, two build jobs and a 4 GiB free-space floor. The test/check caps were
180 seconds and the build cap 600 seconds. No guard fired and all four initial
invocations passed on their first execution.

The new extension was copied into a private 83-file source-package overlay,
not installed as a wheel. The previously saved generation extension remained
byte-identical. One fail-fast Python invocation passed all 70 tests, with zero
errors/skips and zero attempted or loaded Torch/NumPy/Transformers/Python
safetensors/Accelerate modules. It took 1.53 seconds in the bounded runner
(0.306 seconds of test bodies) against a 120-second limit.

The Python checks cover opt-in clipping, raw-query behavior, reports, strict
scalar rejection and exact continuation, plus the existing generation, snapshot
cache and file-handoff regressions. They use small synthetic fixtures, not a
fresh training-quality experiment. No convergence, calibration, speedup or
generalization improvement is established here.

Final review removed an unnecessary square-root/square round trip from the
test's expected clip-factor calculation. The reference now uses the gradient's
original f64 squared norm directly, without changing tolerances or production
code. A second native invocation on that refined test source again passed all
218 tests in 18.35 seconds. The production extension and Python code were not
rebuilt or changed; the initial passing records remain preserved.

Local evidence: `2026-11-10-qkv-gradient-clip-v1` in the SpiralTorch validation
store. Each invocation recorded an unchanged source inventory while it ran.
Only documentation changed after the final reference-test invocation.

| Evidence | SHA-256 |
| --- | --- |
| Tested source inventory | `7de9d7c2e76435895395298dc665d360d8c5e29a6e30ce4b7b099e277cfc3913` |
| Native 218-test log | `7b78dc8b89aaf6a6ebd94a81960ddb6a1efb65407c724e276e4a57b60399cad7` |
| CPU binding check log | `3422eb2637ba3f53c5a598d5fcef120aeeb892b09c4077e4b908b118c42dab10` |
| CPU extension build log | `4a8745b5937f460848469c70b4c0f8111d7701a194f58c55ef991d6b8df187ba` |
| CPU extension | `097373ef7d12c0c1d1bb9e50c12f81f993824bb71507b8e6e1f8595d5e14d28d` |
| Frozen package manifest | `e5a16e495f2835754a8588b9974f71d1755d9cfaa6fbaccffaa3cea221ecee2b` |
| Python 70-test log | `2c29a527dde66d1e4ea2679c7edd796464088c8804d6417af8b2a58003f1ddd7` |
| Final test-reference source inventory | `328b05e8ec2a191e6da582f84e25c05516254d56cd85d6e2d80baf2f3b7d7a47` |
| Final native 218-test log | `5c0759f12e381a594afd71e8d9783c92ed50d177b7a4b55bd7f56d5335466d2b` |
