# Headwise geometry in the QKV byte learner

The Rust resident learner now assigns Poincare, origin-scaled Euclidean or
categorical Fisher-Rao squared distance per block/head. It shares the existing
coordinate encoder and full-model optimizer. This is fixed routing, **not a
learned mixture or language-quality result**. See the
[API and reproduction guide](../../../docs/headwise_geometry_attention.md).

## Fixed Numerical Fixtures

The original [measurement plan](pre-registered.json) fixes two independent
CPU-f32, one-thread Torch fixtures before generation or GPU measurement:

- Seed 1761: one block, four heads `[Poincare,Poincare,Fisher,Poincare]`, Q/K
  zero, no external bias. Covers unequal consecutive groups and a repeated,
  nonadjacent metric; 23 tensors, 2,518 parameters.
- Seed 1863: two blocks, heads `[Fisher,Flat]` then `[Flat,Poincare]`, content
  QKV, external biases and Topos in the second block; 37 tensors, 2,678 parameters.
- Both: batch 2, four byte positions, width 4, hidden 6, geometry width 4,
  curvature -0.75, 16 SGD updates at 0.125, exact same-runtime resume after 7.
- One unchanged reference is shared by native Metal and actual browser WebGPU:
  `35e16d9885c9a3b32b23bdb66ecf4926550e367b9b8992c6645d15f381d10741`.

The original gates remain `3e-6 + 5e-5 * abs(reference)` for each scalar,
geometry-relative L2 at most `0.002`, nonzero measured geometry derivatives and
reference-family norm greater than `1e-8`. Losses, weights and both shared
encoder and external-bias pullbacks are checked, not just final outputs.

Two additional safeguards were added during qualification, without relaxing
those gates. Each head's initial gain VJP and 16-step gain-gradient history are
checked independently under the same relative rule. A separate finite-input
fault-injection control makes only the final Poincare group overflow and
requires guarded output/VJPs and an atomic, zero-rate SGD rejection.

## Results

Both cases pass on both runtimes, including every initial parameter VJP and
all 16 updates. All losses, parameters, geometry derivatives and embedding
derivatives in each fresh-owner step-7 resume equal that runtime's uninterrupted
trajectory bit-for-bit. v3 checkpoints bind every head route and saved tensor.

| Maximum discrepancy versus Torch | Native Metal | Browser WebGPU |
| --- | ---: | ---: |
| Initial logits | 8.95e-8 | 8.95e-8 |
| Training CE | 1.44e-6 | 4.77e-7 |
| Updated parameter | 4.48e-7 | 5.67e-7 |
| Initial parameter VJP | 3.58e-7 | 3.58e-7 |
| Geometry-family relative L2 | 8.19e-5 | 5.33e-5 |
| Per-head gain relative error | 1.08e-4 | 1.43e-4 |

The finite fault-injection baseline has maximum absolute final-head bias
`48.9681510925293`. Changing that head's raw gain to finite `f32::MAX` guards
the joined bias, model output and entire VJP family in both runtimes; no
parameter changes after rejected SGD. Causal prefix/document isolation,
off/detached controls, immutable retained results and stale-owner checks pass.

The retained uniform Fisher reference was also rerun with the new code. Native
and browser full result files are **byte-identical** to their respective
pre-change retained results, including checkpoint payloads. This is a bounded
regression check, not a guarantee for every possible uniform model.

## Failure Retained And Scope

The first native attempt stopped because the shared resume harness still
required checkpoint v2. Its failure log and binary are retained. The harness
now requires v3 and the exact head grid for heterogeneous cases, while
continuing to require v2 for the older flat/Fisher cases. Production checkpoint
validation was not loosened to bypass that failed check.

GPU-enabled Rust model/study tests pass (51); the stdlib verifier suite passes
(64), as does browser module syntax checking. A separate read-only reviewer
found no P1/P2 issues in the production and verifier/client changes, reran both
saved raw reports through the independent comparator and obtained byte-identical
comparison receipts. That review did not rerun the GPU or attest execution.

`native-comparison.json` and `browser-comparison.json` are independent scalar
verification results. Runtime scalar projections retain every training loss
and reported error, but deliberately omit raw weights, gradients and opaque
checkpoint JSON. Full originals remain local and are identified by hashes in
`validation.json`; the scalar files cannot substitute for complete comparator
inputs. Hashes identify retained artifacts, not execution attestation.

This establishes only the recorded small-model correctness and resume scope.
It does not measure corpus quality, long-run stability, speed, learned mixture
weights, separate coordinate encoders per head, or cross-runtime continuation.
Corpus studies v1-v5 remain unchanged; routing/mixture efficacy needs its own
strength-controlled corpus study.
