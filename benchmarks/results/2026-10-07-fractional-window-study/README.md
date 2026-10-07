# Full-Normalization Matched Window Learning

Completed [two-arm learning protocol](../../../docs/fractional_window_study.md),
not merely a post-training intervention. Both arms normalize over K=32;
`history_window_short` retains [1,3), while `history_window_full` retains all
past taps. All filter math and order/gain derivatives are provided by Rust.

## Outcomes

All six runs completed 512 primary updates each (3072 total), plus 12 separate
exact next-update/Adam continuation checks. All planned endpoints were scored
only after every run finished. No seed, duration, checkpoint or setting was
selected. Each endpoint uses all 120 Pride blocks and 32 Alice blocks per arm
and seed; lower CE is better.

| Condition | Pride mean CE | Alice mean CE |
| --- | ---: | ---: |
| Frozen base | 4.07371327 | 4.01180749 |
| Retained short, full K normalization | 4.02455567 | 3.98255875 |
| Full history, same normalization | 3.99291008 | 3.96419896 |
| Full minus retained short | **-0.03164558** | **-0.01835979** |

Full history has lower mean CE for **all three paired seeds on both books**.
Unlike comparing separately normalized K=3 and K=32 filters, this experiment
holds the normalization rule, trainable capacity, initialization, optimizer,
batch order and update count fixed while changing available lag support.
Both arms have 1538 trainable parameters and start as exact identity.

Retained-short final orders are 0.5372, 0.5554 and 0.5413; full-history orders
are 0.0880, 0.1154 and 0.0991. Angle and amplitude remain trainable; all observed
chart states were valid. The full arm still approaches the lower chart
boundary. This result does not establish safety for longer training, justify
clipping or remove the terminal-domain guard.

## What This Establishes

In this fixed small-model setup, learning with long taps available improved
the measured endpoints relative to training the retained short taps alone.
This strengthens the earlier [saved-model dependence observation](../2026-10-07-fractional-window-diagnostic/README.md)
without confusing intervention on trained weights with matched training.

It does not establish unique fractional-memory causation, superiority over
every other long-history filter, general LLM quality or statistical significance.
These books and sample-order seeds are reused exploratory material and may
overlap pretraining. Equal update counts do not equalize arithmetic, so this
is not a Torch speed comparison. No GPU/browser performance is claimed.

The full arm exactly reproduces the prior angular study's base and all three
full trajectories, development scores, endpoint scores, initial parameter
hashes, and verified final parameter/named-Adam tensor receipts. Different
study recipes remain distinct. This is reproducibility evidence, **not three
new independent full-arm successes**. The old ordinary/GL-short final-parameter
tolerance failure remains failed and unchanged.

## Verification And Reproduction

- Training, completed no-op resume and actual saved-state verification exited 0.
- Completed resume did not rescore endpoints; all 52 sealed study files (48 checkpoints plus four records) stayed byte-identical.
- All six final adapters, exact recipes, finite parameters and named Adam states were verified against the sealed summary and journal.
- Before/after checks preserved the previous 76 study files, eight client files, 71 original-runtime files, 71 candidate-runtime files and two diagnostic records; the new nine-file client and two verifier files stayed unchanged.
- Training source: `cbf68cea3cf82678fc63171765394f6b3472820c`.
- Study ID: `0138640e620371d7dfa0cb760f6d59b96ac20b5ff21a5ca14fc6369e3962fcef`.
- Reused native build: `16379238c73f6890a4f754dccd4ad381436e047b`; native SHA-256 `18f4c1c2bb4b76fa0e8beeca59f7fbfa0dffbc85efd72b96c45c119a63befa58`.

`plan.json.gz` binds config, model, source/native hashes, token partitions and
all schedules. `results.json.gz` contains every primary scalar receipt and
endpoint block loss; `journal.json` binds checkpoint identities and completion.
`summary.json` is derived, and can be rebuilt without model libraries:

```sh
python -B benchmarks/results/2026-10-07-fractional-window-study/summarize_wave_gate_long_horizon.py \
  --plan benchmarks/results/2026-10-07-fractional-window-study/plan.json.gz \
  --results benchmarks/results/2026-10-07-fractional-window-study/results.json.gz \
  --journal benchmarks/results/2026-10-07-fractional-window-study/journal.json \
  --output /tmp/fractional-window-summary-new.json
cmp /tmp/fractional-window-summary-new.json benchmarks/results/2026-10-07-fractional-window-study/summary.json
```

Use a fresh output path. `checkpoint-verification.json` records newly checked
saved tensors and Adam, not a reexecution of training. `full-arm-reproduction.json`
compares numeric records and separately verified tensor hashes with the
unchanged previous publication. `validation.json` records exit codes,
regressions, preservation and private log hashes. `SHA256SUMS` covers every
public file. The archived summary and verifier source plus client/runtime
manifests pin reproduction independently of future repository changes.

For a new full execution, follow the study docs with the frozen recipe,
Python 3.12.6, Torch 2.12.1, Transformers 4.57.6, CPU float32/two threads,
paired seeds 41/43/47, Adam 0.001, strength 0.1, batch two, full unpadded
128-token contexts, first-MLP insertion and no KV cache or packed boundaries.
Use a NEW output directory; do not overwrite the sealed study or count a
deterministic replay as additional independent seeds.

Only numeric results, hashes, validation and reproduction code are published.
Weights, corpus text, native packages and raw private logs remain local.
