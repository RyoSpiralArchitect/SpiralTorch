# Frozen Context Ablation: Completed

All 30 conditions completed: six saved gated GPT-2 adapters, each evaluated with
five fixed context interventions on the original 120 Pride and 32 Alice blocks.
There were **zero training updates**. The original checkpoints, raw gates,
projections, readouts, base weights and parent study artifacts were not changed.

See the [protocol](../../../docs/elliptic_context_ablation.md) and the
[parent learning study](../2026-10-03-elliptic-gated-study/README.md).

## Outcomes

Mean cross-entropy over seeds 41/43/47; lower is better. Each row below changes
only the indicated feature term of the same saved model. These are not separately
trained models and must not be substituted into the parent learning comparison.

| Elliptic intervention | Pride | Alice |
| --- | ---: | ---: |
| Native learned attention correction | 3.708304 | 3.749487 |
| Local features only | 3.715178 | 3.761366 |
| Local gain only | 3.711077 | 3.755213 |
| Fixed chart-anchor correction | 3.705061 | 3.735555 |
| Inclusive prefix-mean correction | 3.708610 | 3.725398 |

Removing the entire elliptic correction hurts both sets in every seed. Keeping
only its local gain is mixed across seeds and worse on average. Replacing the
attention context with the fixed chart anchor improves every seed on both sets:
-0.003243 Pride and -0.013932 Alice mean CE relative to the native condition.
Prefix averaging is slightly worse on Pride (+0.000307; 2/3 seeds worse) but
improves every Alice seed (-0.024089 mean).

| Ordinary tangent intervention | Pride | Alice |
| --- | ---: | ---: |
| Native learned attention correction | 3.569886 | 3.671798 |
| Local features only | 3.573078 | 3.651302 |
| Local gain only | 3.579901 | 3.654062 |
| Fixed chart-anchor correction | 3.579549 | 3.654274 |
| Inclusive prefix-mean correction | 3.574155 | 3.659627 |

Thus the saved geometric model benefits from a correction, but these results do
not support attributing its learning gain specifically to content-dependent
attention. A fixed anchor is a useful next matched-training control, not an
already-proven better training method. The adapter may not need extra token
mixing; the underlying GPT-2 hidden states are still contextual. Even the best
geometric intervention here remains worse than the native ordinary control on
both sets. Reused endpoints, distribution shift and three seeds limit inference.

## Verification

- Six original native conditions reproduced every per-block loss exactly before
  any intervention ran. All 30 conditions retained their saved raw gates and
  complete adapter state. The restored model's base digest was unchanged.
- Evaluation exited 0. A completed resume exited 0 without rerunning conditions
  or changing plan, result or summary bytes. All parent study file hashes and six
  endpoint checkpoint hashes remained unchanged.
- Preflight: 130 tests passed, zero skips. The combined suite after the parent
  summary-review fix: 132 passed, zero skips. The new tests exercise actual tiny-HF
  replay, interruption/resume, invalid-state rejection, no-training enforcement,
  future/batch isolation and all control formulas at negative/zero/positive gates.
- `results.json.gz` decompresses to the exact recorded result, including all
  4560 block scores. `summary.json` retains every seed and paired native contrast.
  `plan.json`, `preflight.json` and `validation.json` bind the sources and receipts;
  `SHA256SUMS` covers the public record. Raw logs, weights and corpus text stay local.

## Reproduce

Use the [offline procedure](../../../docs/elliptic_context_ablation.md#run-offline)
with the completed parent study and its frozen package/helpers. Executed source:
`5e515c34bda5e073f3c68ce83d7bdd1d09267264`. The recipe and five modes were frozen
before evaluation; no condition was selected, retrained or dropped afterward.
This is a fixed-weight mechanism probe, not a throughput or model-quality win.
