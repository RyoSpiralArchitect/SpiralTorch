# Completed One-Lag And Fractional-Order Comparison

All twelve frozen-GPT-2 runs completed: four arms, seeds 41/43/47, 512 updates
each (6,144 primary updates), plus 24 continuation-only checks. The process
exited successfully before saved-state verification. Evaluation started only
after every planned run finished. This is a reused-book exploratory study,
not a pristine test, significance claim or general LLM quality result.

## Results

Mean next-token cross-entropy across three seeds, lower is better:

| Arm | Pride unused tail (120 blocks) | Alice transfer (32 blocks) |
| --- | ---: | ---: |
| Frozen base | 4.073713269 | 4.011807486 |
| Ordinary `lag1` | 4.057182088 | 4.002130352 |
| Rust GL, fixed alpha=1 | 4.057182088 | 4.002130352 |
| Rust GL, learned from alpha=1 | 4.054341469 | 4.000331007 |
| Rust GL, learned from alpha=0.5 | 4.056442695 | 4.001746317 |

Ordinary lag1 and fixed-one GL matched **exactly**, including all training
loss/gate receipts, diagnostic development scores, saved feature gates, named
Adam states and every endpoint block loss. They are two implementations of the
same map, not independent quality treatments.

Learning from one improved over fixed-one/lag1 in all three seeds on both sets:
mean CE differences were -0.002840618 and -0.001799345 respectively. It also
outperformed the half-order initialization in all paired seeds. Learned arms
have one extra trainable scalar: 1,537 versus 1,536, not equal capacity.

| Seed | Final alpha, initialized at 1 | Final alpha, initialized at 0.5 |
| --- | ---: | ---: |
| 41 | 2.097662687 | 1.034797907 |
| 43 | 1.949106097 | 0.988713503 |
| 47 | 2.002378225 | 0.995676756 |

Every learned-order run had 511 nonzero order-gradient updates; the first
update has zero order gradient because the history gate starts at zero.
The one-initialized orders finish at their observed maxima. Neither this
fixed update budget nor the endpoint proximity establishes convergence to two.
At exactly alpha=2 and step=1, strictly-past GL is the ordinary two-tap map
`-2*x[t-1] + x[t-2]`. Fixed-two controls and bounded-history ablations are
therefore more informative next steps than declaring a long-memory advantage.
These proposed controls have **not** been run in this record.

The repeated half-order arms exactly reproduce the previous independent-history
study's training records, adapter tensors, Adam states and endpoints in all
three seeds, despite the distinct recorded native binaries. This is a replay
check, **not three more independent seeds**. The earlier pointwise/EMA arms are
not pooled into this study's primary contrasts.

## Verification

- The frozen model, corpus hashes, batch schedules, K=32, step=1, CPU float32,
  two threads, learning rate 0.001, strength 0.1 and 2x128x768 hidden shape match
  the [protocol](../../../docs/fractional_lag_study.md).
- Saved checkpoints verify actual parameter counts, finite adapter/Adam
  contents, scalar trajectories, all 512 optimizer steps and exact continuation.
  The fixed scalar has no optimizer state; the pretrained base is unchanged.
- Completed resume exited successfully without rescoring endpoints. All 100
  sealed study files remained byte-identical. All five frozen client files and
  67 frozen package files remained unchanged; no selective-VJP patch was
  hot-loaded into this experiment.
- The read-only report is byte-identical under Python hash seeds 1 and 42.
  CI rebuilds the committed summary from the published compressed inputs.
- `preflight.json` deliberately retains its original launch-time
  `training_started_not_completed` status. `validation.json` records completion
  separately; historical records are not rewritten into success claims.

`checkpoint-verification.json` and `half-replay-verification.json` bind local
saved-state checks. `results.json.gz` contains all per-step numeric records,
development values and endpoint block losses, not only favorable aggregates.
The compressed plan and results decompress to the exact original bytes bound
by the journal and summary. `SHA256SUMS` binds this publication.

## Reproduce

Follow the protocol with the hash-matching local model and corpora. Freeze the
client and native package recorded in the manifests for the full run, then
verify the saved adapter/Adam contents independently before interpreting it.
The summary itself needs no model, corpus, Torch import or network request:

```sh
python tools/summarize_wave_gate_long_horizon.py \
  --plan benchmarks/results/2026-10-03-fractional-lag-study/plan.json.gz \
  --results benchmarks/results/2026-10-03-fractional-lag-study/results.json.gz \
  --journal benchmarks/results/2026-10-03-fractional-lag-study/journal.json \
  --output "$NEW_SUMMARY_JSON"
```

Weights, corpora, checkpoints, native packages and raw logs remain local.
There is no model-speed comparison here: ordinary lag1 and learned GL are
different maps. Seeds share the frozen model and evaluation blocks; blocks are
not independent experimental replicates. Both books are reused and may have
appeared in pretraining. No generation-quality, GPU or unique fractional-memory
claim follows from these cross-entropy results.
