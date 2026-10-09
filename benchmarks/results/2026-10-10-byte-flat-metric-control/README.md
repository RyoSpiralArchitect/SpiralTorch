# A parameter-matched flat metric in the complete byte learner

Captured on 2026-10-10 (JST), based on
`1a95eacb9ecab341ddc354f8f15c11dd1eb16e98`.
This is synthetic correctness and same-runtime resume evidence, not a language
quality result, a speed comparison, or a matched-compute corpus experiment.

## Control and ownership

The new choice replaces only the causal pair distance:

```text
Poincare: -softplus(raw_gain[h]) * d_c(x_q, x_k)^2
flat:     -softplus(raw_gain[h]) * 4 * ||x_q - x_k||^2
```

Projection, causal complex recurrence, bounded chart, initial parameter values,
byte/position embeddings, external biases, data, seed and optimizer ownership
are unchanged. The factor four matches the origin-local scale, not the full
Poincare distance. The flat arm still uses a nonlinear bounded encoder; this is
not a curvature-to-zero or a linear-encoder experiment.

The resident primitives share the pair workspace, finite guards, endpoint/head
reductions and extended intermediate arithmetic. Euclidean cache derivatives
are `(distance_scale * squared_chord, 2 * distance_scale, 0, 0)`.
The primitive accepts finite coordinates outside a ball; the model deliberately
retains the same bounded chart as the Poincare model.

The existing v1 model checkpoint remains the default. The flat arm exports
explicit `euclidean_chord_squared.v1` identity in model checkpoint v2; import
rejects unknown/null metrics and schema mismatches. All geometry VJPs reach the
same model-owned embeddings and parameters. Corpus-study request v1/v2 has not
been extended to a flat arm in this change.

## Measured correctness

Independent reference: PyTorch 2.12.1, CPU float32, one thread. Two fixtures
exercise one block with zero Q/K scores (23 tensors / 2,516 scalars) and two
blocks with ordinary QKV, external biases and Topos (37 tensors / 2,678 scalars).
Both have B=2, T=4, width=4, two heads and 16 SGD updates at rate 0.125.
Initial parameters and input/cotangent/bias values are exactly equal to their
existing Poincare counterparts.

Both native Apple M4/Metal and actual browser WASM/BrowserWebGpu passed:

- Initial logits, every parameter VJP, embedding VJP and external-bias VJPs.
- Geometry-off and detached-bias sensitivity controls, zero-Q/K isolation.
- Changed suffix, separate document and prefix-extension causality checks.
- Every CE, parameter value and geometry/embedding gradient across 16 updates.
- A new owner restored at update 7, reproducing all subsequent updates exactly.
  The retained raw resumed trace is independently checked against the
  uninterrupted trace, not just a Boolean success field.
- The existing immutable-capture, foreign-tape, rejected-update recovery and
  large decimal-string revision controls, now also under the flat metric.

| Runtime | Maximum parameter error across updates | Maximum geometry-gradient relative L2 |
| --- | ---: | ---: |
| Native Metal | 9.9092722e-7 | 3.1683416e-4 |
| Browser WebGPU | 1.2069941e-6 | 4.1411823e-4 |

The frozen gates remain `3e-6 + 5e-5 * abs(reference)` per value, and at most
`0.002` relative L2 for each geometry gradient with reference norm above
`1e-8`. See [native comparison](native-comparison.json) and
[browser comparison](browser-comparison.json) for all cases and input hashes.
Bitwise resume is within each runtime; it is not cross-device bitwise parity.

The new guard test initially tried to upload a NaN, which correctly failed at
host preflight before reaching the intended GPU VJP assertion. The corrected
test checks that rejection explicitly, then tests GPU-produced masked overflow,
late gradient overflow, whole-family rejection and retained tapes. The original
failed log is retained locally, alongside the successful seven-test GPU rerun.

Regression checks also passed: all 71 kernel-contract CPU tests, seven model
checkpoint tests, the four original full-model native/browser cases and the two
frozen-geometry native/browser cases. The latter also passed the independent
frozen-rate verifier against the unchanged reference. No new corpus comparison
or timing run was performed.

Kernel-contract strict Clippy and pinned rustfmt passed. WGPU strict Clippy
initially stopped because local Clippy 1.97 does not know an existing lint name;
the same strict command passed with installed Clippy 1.99. No lint allowance was
added. That tooling failure is retained separately from the successful run.

## Reproduce and limits

See the [API and commands](../../../docs/resident_byte_decoder.md#a-parameter-matched-flat-distance-control).
The local frozen reference is generated with
`tools/generate_resident_byte_geometry_torch_fixture.py --flat-metric`.
The native CLI and browser page invoke the same Rust full-model probe; Python
does not implement the production learner.

`tools/verify_byte_flat_metric.py` independently checks every retained numeric
family, identity, shape, revision and exact resumed trace. Its fabricated unit
fixtures test verifier rejection, never substitute for runtime evidence. An
independent read-only review found two false-positive gates: incomplete
checkpoint payloads and explicit runtime-control failures could pass. The
verifier now binds complete checkpoint topology and float32 parameter bits to
updates 7 and 16, rejects duplicate fields, and requires the reported
causality/tape/checkpoint controls. All 12 verifier tests pass; the stricter
verifier accepts the original, unmodified native/browser raw reports and emits
byte-identical scalar comparisons. These are report-consistency checks, not
runtime attestation. The focused independent re-review found no further issues;
its scope was static/stdlib review, not a separate device run. The earlier
accepted counterexamples and failed GPU test setup are not positive evidence.
Regenerating the default Poincare Torch fixture is byte-identical to the
previously committed fixture.

Public artifacts contain scalar comparisons, validation notes, hashes and
reproduction instructions only. Raw references, tensors, weights, checkpoint
payloads, binary/WASM/JS and logs remain local. Nothing here establishes which
geometry improves language learning, long-context behavior or generation.
The next experimental boundary is a separately versioned matched corpus study,
before widening to additional metrics or learned mixtures.

See [validation and hashes](validation.json),
[native frozen-rate regression](native-rates-regression.json) and
[browser frozen-rate regression](browser-rates-regression.json). Hashes bind
retained local bytes; they do not make unpublished tensors publicly inspectable.
