# Learned Elliptic Context: Completed Paired Study

**The learned correction modestly improves mean elliptic loss, but does not
establish a geometry advantage over ordinary tangent controls.** All twelve
planned runs and endpoint evaluations completed. Every seed still favors the
ordinary control over geometry within the gated pair, on both evaluation sets.

See [the fixed design](../../../docs/elliptic_gated_study.md): local float32 GPT-2,
frozen base, first-MLP adapter, seeds 41/43/47, 512 updates each, batch 2 and context
128. Gated arms have 8451 trainable parameters each; pointwise arms have 8450,
without dummy parameters. All initial projections and batch schedules are paired.

## Outcomes

Mean cross-entropy over three seeds; lower is better. Pride uses 120 endpoint
blocks and Alice 32 transfer blocks. These sets have been inspected before.

| Arm | Pride | Alice |
| --- | ---: | ---: |
| Frozen base | 4.073713 | 4.011807 |
| Pointwise tangent | 3.566816 | 3.673728 |
| Pointwise elliptic | 3.717387 | 3.759478 |
| Gated tangent | 3.569886 | 3.671798 |
| Gated elliptic | 3.708304 | 3.749487 |

Gated elliptic minus pointwise elliptic is -0.009083 on Pride (2/3 seeds improve)
and -0.009992 on Alice (3/3 improve). Pride seed 43 slightly regresses by
+0.000181; do not hide it behind the mean. Gated tangent regresses on Pride in
all seeds (+0.003070 mean); its Alice mean improves by -0.001930 but only one
seed improves. Neither result supports a robust ordinary-context gain.

The primary geometry contrast, gated elliptic minus gated tangent, remains
+0.138418 on Pride and +0.077689 on Alice, unfavorable in every seed. A favorable
interaction is not an absolute geometry win. All per-seed scores, block losses,
training records and paired contrasts are retained in `results.json.gz` and
`summary.json`; no significance or generalization claim is made.

| Seed | Final effective tangent gate | Final effective elliptic gate |
| --- | ---: | ---: |
| 41 | +0.047123 | -0.519742 |
| 43 | +0.091536 | -0.450893 |
| 47 | +0.092418 | -0.435876 |

All six raw gates have nonzero gradients on the 511 updates after zero-readout
startup. Negative elliptic gates mean `y = (1+abs(g))*local - abs(g)*context`
algebraically, not interpolation on a manifold. This suggests a useful follow-up
separating local gain/centering from genuinely context-dependent benefit. It does
not prove why the correction helped. Gate movement alone is not quality evidence.

## Verification

- All 12 endpoint checkpoint hashes, parameter/Adam finiteness, 512-record batch
  histories, initial pairing and signed-gate continuity were verified. The driver
  verified frozen base weights and exact next parameters/Adam state for every run.
- There are 6144 committed primary updates and 24 continuation-only validation
  updates. The original process disappeared after four completed runs and a
  saved 64-step cursor in seed 43 tangent. Its exit status/cause and any unsaved
  updates are unknown. Recovery verified the frozen inputs/checkpoints and resumed
  that cursor without rerunning completed arms; `recovery.json` preserves this.
- The resumed training/evaluation process exited 0. A subsequent completed resume
  exited 0 without rerunning endpoints or changing result/journal bytes.
- All six pointwise arms exactly replay the prior causal study's scores, training
  records, development records, initialization hashes and parameter counts. The
  frozen baseline also matches exactly. These are reproducibility checks, not
  additional independent confirmations. Prior completed checkpoint receipts were
  unchanged by recovery.
- Preflight had 121 tests, zero skips, including actual tiny-HF interrupted resume
  and same-formula Torch/native VJP at B=2,T=128. The previous causal summary still
  reproduces byte-for-byte. This is not a throughput comparison.

`plan.json.gz` is the original sealed plan. `preflight.json` remains the historical
launch-only record, not a completion claim. `validation.json` contains completed
receipts, hashes and replay checks. `journal.json`, `results.json.gz` and
`summary.json` contain all final outcomes; `SHA256SUMS` covers the public records.
Models, corpora, binaries, raw logs and checkpoints remain local.

## Reproduce

Use the [offline launch procedure](../../../docs/elliptic_gated_study.md#run-offline)
with the public recipe and native/client revision recorded in the plan. Preserve
the original source/package identity for `--resume`. Rebuild only derived output:

```sh
python -S tools/summarize_wave_gate_long_horizon.py \
  --plan benchmarks/results/2026-10-03-elliptic-gated-study/plan.json.gz \
  --results benchmarks/results/2026-10-03-elliptic-gated-study/results.json.gz \
  --journal benchmarks/results/2026-10-03-elliptic-gated-study/journal.json \
  --output /path/to/new-summary.json
```

The output must match `summary.json` byte-for-byte. The summarizer never imports
Torch or evaluates a model, and rejects incomplete studies. Reused endpoints,
three seeds and the post-negative-result design make this exploratory evidence.
