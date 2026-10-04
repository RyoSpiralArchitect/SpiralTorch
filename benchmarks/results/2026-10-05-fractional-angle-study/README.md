# Chart-Aligned GL Learning

Completed offline CPU float32 study. Protocol: [angular comparison](../../../docs/fractional_angle_study.md).
Training revision: `758bc416d6e885e4c227464b385d474794f50a41`.
Verifier revision: `8385e4a1876781028013764f675530ccbb96b79f`.
Native runtime revision: `f8e8e6aa60f7c9a4211243e91f1f1a565d26e50b`.

**GL full (K=32) has lower endpoint cross-entropy than both short filters
on both evaluation sets in all three seeds.** The ordinary and GL short
filters now have almost identical endpoint losses. However, their final
parameter tensors fail the fixed preflight-tolerance comparison in every
seed. That failed criterion is retained, not relaxed or hidden by quality.

## Fixed Comparison

Frozen local pretrained GPT-2, first-MLP insertion, seeds 41/43/47, 512
updates per arm, Adam 0.001, strength 0.1, batch 2, context 128 and two CPU
threads. All arms have 1538 trainable parameters with identical initial
names, values and registration order: gate, local_gate, history_angle,
log_gain. Initial feature gates are zero and the initial strictly-past
filter is [-2,1] up to f32 rounding. No base-model parameters are trained.

Rust owns alpha=1+2*tan(angle), the normalized GL operator, gain and
differentials. The ordinary two-tap control computes its filter and
gradients independently in Torch; it observes the Rust chart only for the
common domain policy and telemetry. K=3 and K=32 both use the angular
coordinate, not the preceding study's positive log-order parameter.

All 4608 primary updates and 18 exact within-arm continuation updates
completed before endpoint scoring. No chart exit, clipping, projection,
duration selection or dropped run occurred. The separate admission used
six auxiliary training and six continuation-only updates, without endpoint
scoring; its records are included in `admission.json`.

The fixed data has 1204 training blocks, 16 diagnostic development blocks,
120 unused Pride-tail endpoint blocks and 32 Alice endpoint blocks. These
are reused exploratory books and may be in pretraining, not pristine
held-out confirmation. Development did not select any settings or outcomes.

## All Outcomes

Mean next-token cross-entropy across three minibatch-order seeds; lower is better.

| Arm | Pride | Alice |
| --- | ---: | ---: |
| Frozen base | 4.073713269 | 4.011807486 |
| Ordinary angular short | 4.023867468 | 3.982157086 |
| Rust angular GL short | 4.023867524 | 3.982156987 |
| Rust angular GL full | 3.992910082 | 3.964198959 |

All predeclared contrasts are retained; negative favors the first named arm.

| Contrast | Pride CE Difference | Alice CE Difference |
| --- | ---: | ---: |
| GL short - ordinary short | +5.629328e-8 | -9.934107e-8 |
| GL full - ordinary short | -0.030957386 | -0.017958127 |
| GL full - GL short | -0.030957442 | -0.017958028 |

GL full minus ordinary short by seed:

| Seed | Pride | Alice |
| --- | ---: | ---: |
| 41 | -0.032592207 | -0.018893406 |
| 43 | -0.029396059 | -0.016973265 |
| 47 | -0.030883890 | -0.018007711 |

`summary.json` retains all per-seed outcomes and descriptive sample SDs.
Three seeds on shared evaluation blocks do not establish significance.

Compared with the [completed log-order study](../2026-10-05-fractional-gain-study/README.md),
the short-filter loss gap largely disappears under the common angular
coordinate. The unchanged ordinary control reproduces every prior endpoint
block loss exactly (`control-reproduction.json`). This is a post-hoc
consistency check, not extra independent seeds or a rewrite of that study.

## Failed Equivalence Criterion

Each saved adapter and named Adam state matches its own recipe, cursor,
trajectory and endpoint receipts. Within-arm interrupted continuation is
exact. These successes do not imply cross-arm state equivalence.

At the unchanged preflight tolerance, rtol=3e-6 and atol=3e-7, final ordinary
versus GL short **parameter comparisons fail in all three seeds**. The
largest absolute parameter difference is 1.206994e-6. Named Adam tensors
pass this tolerance in all seeds, with maximum absolute difference
9.685755e-8, but neither parameter nor Adam maps are byte-equal. All tensor
hashes, per-name errors and failed statuses are retained in
`checkpoint-verification.json`; no tolerance was widened.

Endpoint loss agreement is therefore not evidence of bitwise learning or
full gradient-trajectory equivalence. The small final-state divergence is
observed, not proven to arise solely from floating-point rounding.

## Learned Coordinates

GL short ends at alpha 0.492598-0.518367 and gain 5.017911-5.202008.
GL full ends at alpha 0.087954-0.115421 and gain 5.174370-5.356466.
All nine runs have nonzero angle and gain gradients on every update after
the zero-gate first update. Initial loss/gate-gradient receipt criteria pass.

The full filter remains within the positive-order chart but approaches its
lower boundary; the smallest observed angular margin is 0.035796 radians.
This does not prove convergence or safety under arbitrarily longer training.
Gain and feature gates remain redundant amplitude controls.

Longer GL history also changes normalization and the first two taps. Thus
the lower loss does not uniquely isolate long-memory causation. A future
mechanistic test should separate tail contributions without renormalizing
the retained short taps, rather than silently turning K=32 into K=3.
That is a follow-up proposal, not an intervention performed in this study.

## Verification And Reproduction

### Portable Summary Correction

Review identified a host-libm dependency: `atan(0.5)` differs by one ULP
between supported environments. The current and archived summary tools
now use fixed binary64 chart bounds. Only the nine derived lower-bound
distances change; no losses, training records, saved tensors or tolerance
decisions change. The original summary, summarizer and verification are
retained as `original-*` files, with their source/artifact hashes intact.

`client-sha256.json` still describes the original frozen training client.
`verification-client-sha256.json` describes a separate read-only mirror
whose sole changed file is the portable summarizer. The saved-state
verifier was rerun against that mirror and the unchanged runtime/study,
producing the current `checkpoint-verification.json`. This is derived
reanalysis, not retraining, rescoring or a replacement of primary evidence.
Public reconstruction tests forbid platform `atan` for both the current
and archived tool; a lineage test bounds the correction to those nine fields.

Training, completed resume and saved-state verification exited zero. The
completed resume did not retrain or rescore; all 76 primary files, including
72 checkpoints, remained unchanged through resume and verification. Eight
client files, 71 runtime files, seven model assets and two corpora matched.
All 584 checked earlier study/client/runtime/preflight files were unchanged.

The recipe passed 707 Python regressions. The verifier expansion passed
751 regressions; its final finite-error hardening passed 86 targeted tests.
The initial publication-inclusive suite passed 755 tests without skips.
After the portability correction, 756 regressions and eight Torch-free
public-data checks passed without skips.
The diagnostics subtract f32 inputs in f64 so even large valid differences
remain finite JSON values; training and comparison tolerances are unchanged.
34 benchmark correctness tests passed without timing. Existing JIT
deprecation warnings remain. Publication checks validate numeric consistency,
not independent execution of private tensors.

Reconstruct the public summary without Torch, Transformers, native packages
or model weights, writing a new derived file:

```sh
python -B summarize_wave_gate_long_horizon.py \
  --plan plan.json.gz --results results.json.gz --journal journal.json \
  --output /tmp/fractional-angle-summary.json
cmp summary.json /tmp/fractional-angle-summary.json
shasum -a 256 -c SHA256SUMS
```

Private-state verification uses the frozen client/runtime and saved study:
follow the protocol's `verify_fractional_gain_study.py --coordinate angle`
command, with the separately inventoried verification-client mirror for
the corrected summary. Keep both verifier files together. Native and client build/launch
revisions differ intentionally; executable bytes are hash-bound.

No weights, corpus text, native packages or raw private logs are published.
No speed, browser/GPU residency, general LLM superiority, generated-text
quality, unique parameter recovery or pristine-corpus claim is made.
