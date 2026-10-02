# Anchored Elliptic Training: Completed Paired Study

**Fixed anchors improve the trained elliptic path, but ordinary tangent features
still win.** All twelve runs, continuation checks and endpoint evaluations
completed. The primary geometry contrast remains unfavorable in every seed on
both reused evaluation sets.

The [fixed protocol](../../../docs/elliptic_anchored_study.md) compares ordinary
tangent and Rust nonlinear elliptic features, each with a learned fixed-anchor
or causal-context correction. All four arms have 8451 parameters. Seeds 41/43/47
each receive 512 updates on local frozen float32 GPT-2, batch 2, context 128.
All projection/readout/gate tensors and minibatches are paired within each seed.
The completed total is 6144 primary updates plus 24 continuation-only checks.

Source was frozen at `af1c10c39a97ca862568dd3e3fb2259efa0e5ced` before launch.
`plan.json.gz` contains the original, unmodified hash-sealed plan. The study ID
is `229ccd7afba7893e7e18c7f354249957e977381afac80938c4c36cdcfd4b3ca8`.
Models, corpus text, packages, checkpoints and raw logs remain local.

## Outcomes

Mean cross-entropy over three seeds, lower is better. Pride uses 120 endpoint
blocks and Alice 32 transfer blocks. All active arms improve over the frozen base.

| Arm | Pride | Alice |
| --- | ---: | ---: |
| Frozen base | 4.073713 | 4.011807 |
| Anchored tangent | 3.563489 | 3.673359 |
| Anchored elliptic | 3.679653 | 3.738315 |
| Gated-context tangent | 3.569886 | 3.671798 |
| Gated-context elliptic | 3.708304 | 3.749487 |

Anchored minus gated-context elliptic is -0.028651 CE on Pride and -0.011172 on
Alice, improving in all three seeds on both sets. This is actual training from
identity, not the earlier fixed-weight intervention. Nevertheless, anchored
elliptic minus anchored tangent is still +0.116164 on Pride and +0.064956 on
Alice: geometry loses in every seed. A favorable mean interaction is not an
absolute geometry win.

Anchored tangent improves over its context version on Pride in all seeds
(-0.006397 mean), but slightly worsens Alice's mean (+0.001561). Alice seed 43
regresses by +0.012848 despite improvements in the other two seeds. Alice's
interaction is also unfavorable in seed 47 (+0.003133). These outcomes are kept
alongside every block loss, gradient record and per-seed contrast.

| Seed | Final effective anchored tangent gate | Final effective anchored elliptic gate |
| --- | ---: | ---: |
| 41 | -0.164029 | -0.471237 |
| 43 | -0.177503 | -0.432179 |
| 47 | -0.179219 | -0.425313 |

A negative anchor gate enlarges the deviation from the fixed anchor. It does not
establish geodesic learning, added expressivity, or an attention mechanism. Some
scaling can be absorbed by the readout, and underlying GPT-2 states are already
contextual. All twelve gates receive nonzero gradients on the 511 updates after
zero-readout startup; gate motion is not by itself the quality evidence.

## Verification

- The uninterrupted training/evaluation process exited 0. All 12 final checkpoint
  hashes, finite adapter/Adam contents, 512-step batch histories, paired initial
  weights, gate continuity and endpoint receipts were verified. The driver
  verified unchanged frozen-base weights and exact next adapter/Adam state in
  each run before unlocking evaluation.
- A completed `--resume` exited 0 without rerunning evaluation; plan, journal
  and result bytes remained identical. The original sealed plan and preflight
  are unchanged. No knobs were changed from intermediate development losses.
- All six gated-context controls exactly replay the earlier gated study's
  scores, training/development records, initial hashes, parameter counts and
  final gates. The frozen baseline also matches exactly. These are reproducibility
  checks, not six independent confirmations of a context result.
- Preflight had 165 tests, zero skips. The post-run suite also covers byte-exact
  reproduction of this completed summary and the earlier gated summary.
  Equivalent-formula VJP tests are numerical checks, not speed measurements.

`checkpoint-verification.json` contains saved-content checks and replay results;
`validation.json` adds process, resume, summary and source/log-hash receipts.
`results.json.gz`, `journal.json` and `summary.json` retain complete numeric outcomes.
`preflight.json` remains the historical launch-only record, not a completion
claim. `SHA256SUMS` covers all public artifacts. Weights, text, packages and raw
logs stay local.

## Reproduce And Interpret

Use the offline command in the protocol with unchanged frozen source/package/
data/recipe. Rebuild the derived summary without importing Torch or scoring again:

```sh
python -S tools/summarize_wave_gate_long_horizon.py \
  --plan benchmarks/results/2026-10-03-elliptic-anchored-study/plan.json.gz \
  --results benchmarks/results/2026-10-03-elliptic-anchored-study/results.json.gz \
  --journal benchmarks/results/2026-10-03-elliptic-anchored-study/journal.json \
  --output /path/to/new-summary.json
```

The output must match `summary.json` byte-for-byte. Three seeds share evaluation
blocks, and Pride/Alice already motivated this design; these are exploratory
outcomes, not a significance claim, untouched confirmation or general LLM
advantage. Equal parameters and updates do not mean equal compute. No speed claim
is made: mathematically equivalent PyTorch/native comparisons are separate.
