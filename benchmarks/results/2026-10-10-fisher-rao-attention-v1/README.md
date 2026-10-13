# Categorical Fisher-Rao In QKV Byte Learning

Captured on 2026-10-10 (JST), based on
`1d3d35b0e46aa683d4a4276fa95d2f2166254677`.
This is synthetic numerical/gradient and same-runtime resume qualification,
not language-quality, long-context, general framework-speed or GPU-attestation
evidence.

## What Is Connected

The existing Rust categorical Fisher-Rao comparison now shares a CPU distance
contract with a resident WGPU metric. The byte learner selects
`categorical_fisher_rao_squared.v1` without adding parameter slots or another
optimizer owner:

```text
byte/position embeddings -> projection -> causal complex wave/bounded chart
 -> sqrt(softmax(coordinates)) -> Fisher-Rao squared pair distances
 -> per-head positive gain -> causal QKV attention scores
 -> complete VJP -> wave BPTT -> embeddings
```

This is an addition to ordinary QKV scores, not a replacement attention
algorithm. The categorical distribution is over latent coordinates, not
vocabulary tokens or attention probabilities. Negative curvature continues to
control the input chart, not the intrinsic Fisher geometry. Equal parameter
counts do not establish equal function classes or compute.

The second fixture also retains Topos and external attention biases. The
Poincare and flat paths are not relabelled or removed. Checkpoint v2 explicitly
preserves the new metric; old corpus-study v1-v4 remain their declared
experiments and do not silently gain a Fisher arm.

## Observed Correctness

The independent PyTorch 2.12.1 CPU float32, single-thread reference was frozen
before either full-model GPU run. No tolerances or reference values were
retuned. Both fixtures use B=2, T=4, width=4, two heads, 16 SGD updates at
0.125 and resume at update 7. One block has zero Q/K scores to isolate the
geometry (23 parameter tensors / 2,516 scalars); two blocks combine ordinary
QKV, Topos and external biases (37 tensors / 2,678 scalars).

| Runtime | Maximum all-step parameter error | Maximum geometry-gradient relative L2 |
| --- | ---: | ---: |
| Native Apple M4 / Metal | 5.8859587e-7 | 8.0769327e-5 |
| Actual browser / BrowserWebGpu | 6.8545341e-7 | 9.8943033e-5 |

Every value retains the gate `3e-6 + 5e-5 * abs(reference)`; each geometry
gradient also requires relative L2 <= `0.002` and reference norm > `1e-8`.
The same-runtime resumed losses, weights and retained gradients match bitwise.
This is not a claim of bitwise equality between native and browser.

The controls cover initial logits, all parameter VJPs, embedding/external-bias
VJPs, each update, causal prefix/document isolation, geometry-off and detached
pullbacks, guarded rejection and immutable checkpoint/tape ownership. All
geometry parameter families move by a nonzero amount by update 16. See
[scalar trajectories](scalar-trajectories.json), [native comparison](native-comparison.json)
and [browser comparison](browser-comparison.json). A declining synthetic loss
does not demonstrate better language modelling than another geometry.

## Review Repair

Initial full-model tests passed, but independent read-only review found a
valid finite-final-VJP case incorrectly rejected by intermediate float32
overflow. The native GPU reproduced this failure before repair:
`[.5,-.5,-.5,.5]` logits, one zero raw gain, and a causal seed of `3e38`.
The root cotangent exceeds float32 range while the final logit and gain
gradients are representable.

The fix retains private wide-arithmetic root cotangents through the final
simplex pullback, narrowing only final outputs. Shared guards still reject
nonrepresentable final outputs and invalid upstream inputs. The original
failed log, pre-review native observation and binary/WASM remain local; they
are not presented as evidence for the repaired source.

The corrected boundary ran on native and actual browser GPU. Both outputs
were independently checked using float64 analytic derivatives
(coordinate relative L2 `1.49e-7`, gain `4.30e-8`). The full model was then
rerun against the original immutable Torch fixture. Focused read-only
re-review found no further P1/P2 issues; its 16 stdlib tests and 12 additional
mutations are distinct from the parent's actual device runs.

Additional passing checks: 84 CPU kernel-contract tests, 13 core concept
diffusion tests, 11 native pair tests including actual GPU cases, 9 checkpoint
tests, 27 CPU study-policy tests, 3 Torch math tests, 4 Fisher verifier tests and
12 flat verifier tests. The optional-Torch CI route skips its 3 math tests
when Torch is absent; that skip is not math validation.

## Reproduction And Limits

Follow the [Rust API and commands](../../../docs/fisher_rao_attention.md#qualification-recipe).
Both runners use the same Rust learner and exact frozen reference bytes.
Native/browser GPU jobs run sequentially. The browser download preserves
opaque Rust JSON; `tools/verify_byte_flat_metric.py --fisher-rao` independently
checks every recorded numeric family and resumed checkpoint, plus the
large-cotangent boundary. Report consistency is not hardware attestation.

[Validation](validation.json) records source/raw/public hashes and scope.
Full tensors, weights, checkpoint JSON, logs, JS and WASM remain local.
Hashes bind retained bytes, not publicly available raw tensors. Compiler
warnings from existing vendored WGPU remain; no warning-free or strict-WGPU
Clippy claim is made.

This does not yet add Fisher-Rao to a corpus sweep, calibrate equal initial
bias strength, establish its advantage, or train a geometry mixture. The next
experiment should first qualify strength-matched single-metric trained/frozen
arms, then selected pairwise combinations without changing the old studies.
