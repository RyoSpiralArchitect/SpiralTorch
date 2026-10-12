# Resident Geometry Guard Batching

Status: **correctness-preserving candidate; performance unmeasured and not
adopted**. Source: `0f09558b9aa70b893ebfbfa0b4765634db24d24b`.
No formal PR, merge, CI or speed claim is made by this record.

Pair-bias (Poincare, Euclidean and Fisher-Rao), causal-wave, and headwise-mixture
forward/backward now encode their existing pre/post validity captures in the
operation's command encoder. Querywise mixtures reuse the headwise path.
Each of these six methods makes one explicit queue submission instead of
three. Shader arithmetic, reductions, captures and guard propagation remain.
This is not a whole-model submission count or a timing result.

## Executed Checks

- 27 backend tests passed with GPU execution enabled: validity 4, pair metrics
  11, causal wave 4, and shared/querywise mixture 8. Existing integration tests
  exercise strided views, retained tapes and late/nonfinite failures.
- 27 byte-decoder tests passed, including a serial rerun with GPU execution
  enabled. The added guard test records captures before and after a producer
  failure in one encoder and checks immutable earlier aliases and original data.
- Native and browser each reran the same four contextual full-LM fixtures,
  sixteen updates per case, with VJPs, checkpoint/resume and guard checks.
  Each complete raw report is byte-identical to its retained same-runtime
  baseline; [validation.json](validation.json) binds their hashes.

The original contextual-model browser qualification remains **failed**:
the zero-initialized two-block case exceeds the unchanged rounded router-delta
gate at update 14. Its reference delta is `6.812899059e-6`, relative error
`0.004374396586`, limit `0.002`. Batching preserves that failure, not fixes it.
Native still passes. See the [original result](../2026-10-13-contextual-mixture-v1/README.md).

## Review And Next Gate

Independent read-only local review found no actionable P1/P2 issues in the
five changed Rust files against `edf5e2008e810fca3e498d50bf80e11be714da36`.
That reviewer did not execute builds/tests or inspect helpers outside those
files. Parent-executed tests are not an independent execution replication.
Abandoning an unsubmitted encoder is not directly covered by the new test.

The [predeclared correctness protocol](pre-registered.md) remains fixed.
Before timing or adoption, separately freeze a repeated native/browser
performance protocol. Fewer submissions alone are not evidence of speedup.
The byte-corpus v8 study uses its original preserved binaries, not this candidate.

Reproduce the four-case check with the existing
[contextual geometry recipe](../../../docs/contextual_geometry_mixture.md)
and the same frozen fixture, then compare complete raw bytes with the baseline.
Run scoped Rust tests with `SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1` and
`--test-threads=1`. Raw tensors, checkpoints, binaries and logs stay local;
published hashes identify retained bytes, not execution attestation.
