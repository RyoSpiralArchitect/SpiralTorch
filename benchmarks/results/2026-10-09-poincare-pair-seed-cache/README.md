# Poincare pair-seed cache review repair

PR #2253's review identified repeated head reduction inside every coordinate
component of the metric pullback. This follow-up keeps the same mathematics
and fixed numerical criteria but computes the head-summed seed once per causal
pair, before the coordinate VJP dispatch.

The work changes from O(B*T^2*C*H) to O(B*T^2*(C+H)). The tradeoff is one extra
GPU dispatch and **16 bytes per batch/query/key pair** in a private tail of
each backward's packed gradient allocation. Returned gradient views do not
expose that tail, but retain its memory with the allocation. Extended values
are stored as integer words, preserving exponent and significand bits; they
are not prematurely narrowed to f32.

No new binding, upload or host readback is introduced. Forward coefficients
are not modified. Packed storage is checked before execution, including a
regression where the extra tail, rather than the forward cache, crosses the
binding-size limit. Separate backwards cannot alias each other's scratch.

## Verification

- Native: five primitive unit tests plus the shared integration test pass.
- Actual browser WebGPU: the same Rust shared integration probe passes.
- The original eight frozen independent PyTorch cases, seven extended-range
  cases, near-boundary control, causality and whole-family rejection checks
  pass without regenerated fixtures or relaxed tolerances.
- A new `[B,T,C]=[2,17,32]`, `H=8` control queues two distinct backwards before
  reading either, compares both VJPs with CPU, retains the earlier output and
  confirms a repeat is bitwise equal. Native and browser maximum CPU VJP error
  are both `6.705522537231445e-8` under the original numerical gate.
- Workspace formatting uses CI's pinned `nightly-2026-04-15` and passes.

[native.json](native.json), [browser.json](browser.json) and
[validation.json](validation.json) record the results, source hashes, retained
artifact hashes and commands. The [original evidence](../2026-10-09-poincare-bias/README.md)
is unchanged and describes the earlier kernel.

Reproduce with the commands in
[the metric API document](../../../docs/poincare_metric_attention.md).
This is a structural redundancy repair, not a timed speedup claim or a
qualification at LLM sequence lengths. The materialized O(B*T^2) storage and
remaining gain-gradient reduction still matter at larger shapes.
