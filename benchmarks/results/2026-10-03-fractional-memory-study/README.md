# Fractional Memory Study

**Status: completed and locally verified.** All nine runs, their exact
adapter/Adam continuations, and delayed endpoint evaluations finished with
process exit code 0. This is not a fractional-geometry quality or speed win.
The [protocol](../../../docs/fractional_memory_study.md) compares pointwise,
fixed-order GL and learned-order GL against a shared no-adapter frozen baseline.

Three seeds, three active arms, 512 updates per run: 4608 primary updates and
18 additional continuation-only updates completed. The Rust-backed adapter
uses 32 causal GL coefficients, step 1 and initial alpha 0.5. CPU-f32 GPT-2
stays frozen; only the adapter's feature gates and, in the learned-order arm,
its one order scalar are trained. No second geometric mechanism is combined.

The controls have explicit total/trainable adapter parameter counts:
pointwise 768/768, fixed GL 769/768 and learned GL 769/769. Each seed uses
identical active-arm minibatches. Seeds vary shuffle order and therefore the
finite-budget subset of the 1204 training blocks, not random adapter weights.
Each sequence is a complete unpadded 128-token block with reset GL history.

The study finished all runs and exact adapter/Adam next-update checks before
scoring its 120 reserved-within-study Pride tail blocks and 32 Alice blocks.
These are previously inspected exploratory books, not untouched confirmation.
Development does not select alpha, kernel length, checkpoint or run duration.
No parameter-matched, compute-matched, speed or significance claim is made.

## Outcomes

Mean endpoint cross-entropy in nats per next token; lower is better. Active-arm
means cover all three minibatch seeds. The frozen baseline is shared.

| Arm | Pride (120 Blocks) | Alice (32 Blocks) |
| --- | ---: | ---: |
| Frozen baseline | 4.073713269 | 4.011807486 |
| Pointwise | 4.063162282 | 4.005609319 |
| Fixed GL | 4.065637651 | 4.007388651 |
| Learned GL | 4.065377066 | 4.007407176 |

| Paired Contrast | Pride Mean CE Difference | Alice Mean CE Difference | Improving Seeds (Pride / Alice) |
| --- | ---: | ---: | ---: |
| Fixed GL minus pointwise | +0.002475369 | +0.001779333 | 0/3 / 0/3 |
| Learned GL minus fixed GL | -0.000260585 | +0.000018525 | 3/3 / 0/3 |
| Learned GL minus pointwise | +0.002214784 | +0.001797857 | 0/3 / 0/3 |

Every active arm improves over the frozen baseline on both sets, but the
ordinary pointwise gate is better in every seed. Learning alpha slightly
improves fixed GL on Pride and slightly worsens it on Alice. It does not close
the pointwise gap or justify changing a default. `summary.json` preserves all
per-seed scores, differences and descriptive paired standard deviations.

| Seed | Learned Final Alpha | Nonzero Order-Gradient Steps |
| --- | ---: | ---: |
| 41 | 0.826886833 | 511 |
| 43 | 0.773333788 | 511 |
| 47 | 0.813042223 | 511 |

All fixed-order states remain exactly at alpha 0.5 with no order gradient or
Adam state. Learned order has zero gradient at the identity initialization and
actual nonzero gradients thereafter. These observations establish an active
learning mechanism, not a causal explanation of its remaining quality gap.
The GL map changes instantaneous gain and temporal correlations together;
isolating those effects is a future controlled question, not another result.

## Verification And Reproduction

`plan.json.gz` preserves the original plan bytes (gzip encoded); `preflight.json`
records 218 passed Python tests, the 67-file frozen package/native manifest hash,
the four frozen client/config files and source hashes. Source revision:
`21121092b6e5ecbe7ba4464046f1a02d7021d3a4`.
Tiny-HF tests establish actual gradients, fixed-order invariance, three-arm
interruption/restart equality and endpoint gating, not pretrained quality.

`checkpoint-verification.json` records direct inspection of all nine saved
adapter/Adam states, recipes, cursors and batch histories, including mode and
final-alpha agreement. The original training driver executed the frozen-base
and exact next-update checks. A completed `--resume` returned without rescoring;
plan, journal and result bytes remained unchanged. All 71 frozen client/package
files were checked against their launch hashes. `validation.json` separates
these checks from the later main-integration tests and binds local log hashes.

The post-run summarizer now visits arms in protocol order, not Python set/hash
order. This stabilizes serialized output without changing any number or frozen
experiment record. Rebuilding the summary under two distinct hash seeds is
byte-identical, and CI reproduces this committed summary alongside earlier
studies. Reproduce the numeric summary from the published inputs:

```sh
RESULT=benchmarks/results/2026-10-03-fractional-memory-study
python tools/summarize_wave_gate_long_horizon.py \
  --plan "$RESULT/plan.json.gz" --results "$RESULT/results.json.gz" \
  --journal "$RESULT/journal.json" --output "$NEW_SUMMARY_JSON"
cmp "$NEW_SUMMARY_JSON" "$RESULT/summary.json"
```

Use the [protocol](../../../docs/fractional_memory_study.md) for a fresh local
model run; the original training source is the revision recorded above, not a
later helper revision. Resume requires the same frozen client, native module,
configuration and assets. Public numeric records do not let readers inspect
private weights; saved-content verification and raw logs are retained locally.
Models, book text, checkpoints, packages and raw logs are not published. The
launch plan and historical `preflight.json` with `running` status are unchanged;
completion evidence is appended separately. No significance or speed claim.
