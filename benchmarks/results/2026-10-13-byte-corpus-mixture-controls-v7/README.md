# Learned And Fixed Geometry Mixtures: Byte-Corpus v7

The [resident mixture mechanism](../../../docs/learned_geometry_mixture.md) is
now exercised in actual document-held-out byte-language learning, not only a
synthetic derivative fixture. Native Metal and browser WebGPU each completed
51 cases with 128 SGD updates. Both pass the independent CPU-f32 Torch checks,
including every individual learned-router component.

**This establishes numerical connectivity, not a quality advantage.** The
learned-versus-fixed-router BPB differences are extremely small, with signs
that sometimes disagree across engines. Geometry choice changes the result;
the incremental benefit of learning the mixture is unresolved in this pilot.

## Frozen Design

The [protocol](pre-registered.md) was declared before preparation/measurement.
Keep all nine v5 arms for each of seeds 11, 23 and 37. Add P/E, P/F, E/F and
P/E/F, each with learned and fixed routers. P is Poincare squared, E is the
origin-scaled Euclidean chord squared, and F is categorical Fisher-Rao squared.
Both router arms train the coordinate encoder, causal wave, shared gains,
embeddings and ordinary QKV Transformer. Only the fixed router's rate is zero;
its full pullback is retained.

- Batch 2, 32 byte positions, width 16, hidden width 32, two heads, one block.
- Geometry width 4, curvature -0.75, Topos off, fixed 256-byte alphabet.
- 128 SGD updates at rate 0.05; 8,192 training target exposures per arm.
- 2,048 held-out target bytes at updates 0, 64 and 128. This is a tiny fixed
  repository-prose corpus, not pretrained-LLM fine-tuning or generation testing.
- Geometric baselines have 11,290 parameters. Two-/three-component mixtures
  have 11,294/11,296 parameters. Additional metrics cost extra computation;
  equal data/update budgets are not an equal-compute or speed comparison.
- Canonical input SHA-256:
  `7733e760748e14944744a91a6c55a16ad21982a53af0375a4a2ac95d00e9a010`.
- Unchanged prepared v5 source SHA-256:
  `95ef8c21c6c9c8221d7aced6adcda9d80ce4c0dccf6ec3819e40475c52c1db38`.

Routers begin at canonical +0 logits, hence uniform weights. Frozen component
units come from the v5 calibrated positive-gain ratios to P. Rust fits only
each mixture's initial shared gain on the original declared training windows.
Native/browser preparation RMS relative errors peak at `1.514e-7`/`1.599e-7`,
below the unchanged `1e-5` gate. No held-out fitting or resume refitting occurs.

The separately prepared browser request differs in 32 float32 gain scalars,
by at most `8.941e-8`. It is **not** substituted into browser learning. Every
learner consumes the same native-prepared opaque request bytes.

## Measured Differences

Primary contrast: learned-router minus fixed-router final held-out BPB. Negative
values favor the learned router. All twelve contrasts are retained:

| Seed | Mixture | Native | Browser | Torch |
| --- | --- | ---: | ---: | ---: |
| 11 | P/E | -5.374e-08 | +1.075e-08 | +2.150e-08 |
| 11 | P/F | -1.720e-07 | -1.182e-07 | -1.397e-07 |
| 11 | E/F | -2.472e-07 | -2.580e-07 | -2.472e-07 |
| 11 | P/E/F | -1.182e-07 | -9.674e-08 | -1.612e-07 |
| 23 | P/E | +2.150e-08 | +2.150e-08 | -2.150e-08 |
| 23 | P/F | -1.290e-07 | -8.599e-08 | -1.612e-07 |
| 23 | E/F | -1.075e-07 | -1.505e-07 | -1.720e-07 |
| 23 | P/E/F | -2.150e-08 | -5.374e-08 | -5.374e-08 |
| 37 | P/E | +2.150e-08 | -1.075e-08 | +0.000e+00 |
| 37 | P/F | -2.150e-08 | -1.075e-08 | +3.225e-08 |
| 37 | E/F | -3.225e-08 | +2.150e-08 | +0.000e+00 |
| 37 | P/E/F | -1.075e-08 | +2.150e-08 | +0.000e+00 |

Six native/Torch and five browser/Torch sign classifications disagree, including
three Torch-zero contrasts. The largest absolute primary contrast is only
`2.580e-7` BPB. Do not rank the learned routers as quality improvements from this
table. Three seeds are three samples, not twelve independent samples.

This is different from comparing metric subsets themselves. For example, native
P/F minus P is approximately `+5.675e-4`, `+8.545e-4`, and `-1.205e-4` BPB across
the three seeds: the geometric choice has a larger, seed-dependent effect.
All differences versus trained P, calibrated E and calibrated F are retained
in the comparison files, not filtered into a winning configuration.

Host-f64 diagnostics of retained final logits show a maximum absolute softmax
mass shift from uniform of approximately `9.549e-4` in each engine. These are
derived scalar diagnostics, not an additional GPU measurement or published
model weights. The router is per-block/head and shared over positions; it is
not an input-conditioned selector or a manifold Q/K/V projection.

## Verification And Continuation

Each runtime's 6,528 training losses, 4,896 held-out batch losses and 575,676
final parameter scalars pass `3e-6 + 5e-5 * abs(reference)`. Maximum CE error
is `1.431e-6`; maximum parameter error is `8.345e-7` in both runtimes.
All 54 individual learned-router components move; the smallest reference
update is about `5.917e-6`. Maximum per-component relative error is
`7.348e-5` native and `8.685e-5` browser, below `0.002`. Geometry-family update
errors also pass. All fixed routers retain their initial float32 bits.

Both runtimes preserve all 27 original v5 complete case reports exactly.
Separate stop-1 runs verify all 24 pairs per runtime. New router twins have
identical initial evaluations, first training losses and every non-router
parameter update, while the coordinate encoder and embeddings still learn.

Native pause-37/fresh-process resume and browser pause-37/fresh-page resume
pass for all 51 cases in each runtime. Completed no-op resumes also pass.
Within each runtime, uninterrupted/resumed/no-op report bytes match exactly,
and completed/no-op checkpoint bytes also match. This is not cross-runtime
bitwise identity.

Local tests pass: 41 Rust study tests, 5 native CLI tests, 59 Python corpus tests,
71 Python verifier tests and three browser module syntax checks. See the
[independent source review](independent-code-review.md) and
[native artifact audit](independent-native-audit.md) for their bounded scopes.
These checks are not a claim about formal CI or a general quality/speed gain.

The first browser attempt lost its owned page before a result could be
recovered. The cause is unknown. Its [unverified-attempt record](browser-attempt1-page-loss.json)
is retained; an unchanged-request replacement run produced the measured result.
No partial run is promoted and no numerical criterion is relaxed.

## Reproduction And Artifacts

Use the [v7 preparation and learning recipe](../../../docs/byte_corpus_mixture_controls.md)
with the exact v5 source above. Rust prepares the request; Python performs no
production fitting. Verify both preparation packets separately, then run
native, browser and Torch learning on **one** selected opaque request. Execute
separate stop-1 controls, stop-37/fresh-runtime resume and completed no-op resume.
Serialize GPU jobs and preserve unsuccessful attempts.

`native.json`, `browser.json` and `torch.json` contain scalar trajectories only,
not complete comparator inputs. Comparison/control/resume records bind the
checks; `router-diagnostics.json` contains only derived scalar changes.
`validation.json` and `SHA256SUMS` bind source, public and retained local
artifacts. Full requests, logits/weights, checkpoints, binaries and logs stay
local. Hashes identify artifacts, not a cryptographic execution attestation.
