# Initial strength matching for causal geometric attention

Captured 2026-10-10 (JST), based on
`570f40a5782b2243a630e0a57874e2ef92b134c9`.
This is a separately scoped synthetic calibration/learning qualification,
not a new corpus study, quality improvement or speed result.

## What is now usable

Rust `CausalBiasMoments` measures per-head RMS after centering each causal
query row. `CausalBiasScaleMatch` fits the candidate's initial softplus gains
to a reference. The byte-decoder plan can replace only those gains without
changing its other parameters, layout, metric or optimizer ownership.

The model exposes its own resident geometry scores, excluding external biases
and Q/K content. The shared native/WASM harness explicitly snapshots them for
one-time initialization, fits the gains, then re-forwards the same training
windows. All subsequent forward/VJP/SGD uses the existing resident learner.
Calibration consumes no update and never runs again during checkpoint resume.
This is not a device-resident calibration reduction or a second Python learner.

The statistic includes all `k <= q` entries, including the diagonal and the
zero-energy first row. Batches are weighted by causal-pair count. Constant-row
offsets cannot affect the statistic, and zero-signal cases are rejected rather
than silently assigned a gain. Softplus inversion is stable in log space and
checks f32 representability before accepting the result.

## Measured qualification

The frozen recipe matches flat `4 * squared_chord` to Poincare bias strength.
Both metrics use the same learned bounded causal-wave coordinates. Two
preselected training batches yield 40 causal pairs per head (B=2, H=2, T=4).
The first model has one block with zero Q/K scores; the second has two blocks,
ordinary QKV, external biases and Topos. Each fitted model then runs all
initial parameter/embedding/external-bias VJPs and 16 SGD updates at rate .125,
including exact same-runtime restart at update 7.

Independent reference: PyTorch 2.12.1, CPU float32, one thread.
Native Apple M4/Metal and actual browser WASM/BrowserWebGpu both passed.

| Runtime | Maximum realized relative RMS error | Maximum parameter error over learning | Maximum geometry-gradient relative L2 |
| --- | ---: | ---: | ---: |
| Native Metal | 2.0660482e-7 | 5.3644180e-7 | 1.3319867e-4 |
| Browser WebGPU | 1.1916094e-7 | 5.0663948e-7 | 1.2946800e-4 |

The realized RMS gate was fixed at `1e-5` before running the devices. No
iterative retuning was allowed. The existing per-value gate remains
`3e-6 + 5e-5 * abs(reference)`, and each geometry gradient must have nonzero
reference norm above `1e-8` and relative L2 at most `0.002`.
See [native](native-comparison.json) and [browser](browser-comparison.json)
for every case and input hash.

The selected gains multiply initial flat strength by about 1.0763 and 1.1086
in these two fixtures. Matching this statistic does not make the pairwise
score pattern or attention probabilities identical. The relative raw-gain
sensitivity changes too; that diagnostic is recorded separately.

## Review and regression

Independent read-only review found one verifier gap: a one-ULP fitted-gain
splice could pass while retaining a different learner's report. The runtime
reconstruction was correct, but the evidence did not bind initialization
exactly. The corrected harness captures the actual learner's revision-zero
device checkpoint before its first forward. Rust and the independent verifier
require its complete state to match the fitted checkpoint exactly. Both
runtimes were rebuilt and rerun after the repair; the pre-review observations
remain local and are not the final qualification artifacts.

The focused re-review had no further findings. Ten new verifier tests include
the one-ULP splice; the existing twelve flat-verifier tests also pass. All
80 kernel-contract tests, eight byte-decoder checkpoint tests, strict
kernel-contract Clippy, pinned rustfmt and browser-module syntax checks pass.
Ordinary/out-of-range accessor controls, terminal-target changes and external
bias changes pass on both devices. Native wrong-fixture input is rejected at
the SHA gate. Both the shared Rust harness and browser page pin the frozen
fixture hash instead of silently accepting a replacement.

Regenerating the legacy flat reference is byte-identical to the retained
reference. Rebuilt native and browser legacy paths each reproduce their own
old raw report byte-for-byte. Corpus v3 and its identical-initial-weight guard
are untouched.

## Reproduce and limits

Use the [API and commands](../../../docs/causal_bias_calibration.md) and
[validation inventory](validation.json). Python is an independent oracle and
verifier only; native and WASM invoke the same Rust learner. Public files
contain scalar results, checks, hashes and reproduction instructions. Raw
fixtures, scores, weights, checkpoints, binaries and logs remain local.

Each engine calibrates its own computed scores independently. Its fitted f32
gains can differ by a few ULPs from another engine. This qualification therefore
does not assert identical cross-engine initialization or bitwise execution.
Checkpoint and resume equality is exact within each runtime. Report checks
and hashes are not cryptographic execution attestation.

Equal initial RMS does not imply equal distributions, equal SGD dynamics or
better language learning. This does not implement calibrated corpus arms,
head-wise mixtures, new geometry families or pretrained-model fine-tuning.
The next boundary is an explicitly versioned calibrated corpus comparison,
before attributing any quality difference to a distance rule or combination.
