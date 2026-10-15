# Frozen-Weight Language Diagnosis

This follows the [negative greedy-generation result](../2026-10-15-language-greedy-serving/README.md).
It measures historical checkpoints, not new training or a serving promotion.
Raw corpus, checkpoints, weights and logit arrays remain private; the scripts,
complete result summaries and hashes are public.

## Recipe

Nine browser-origin checkpoints at update 128: seeds 11/23/37, each with ordinary
attention, Poincare/chord query routing trained, and the same routing with query
weights frozen. Width 16, one block, two heads, B2, T32, SGD 0.05. They saw 8192 target
bytes each, versus 143872 available complete-window target bytes in the training
documents. This is a tiny pilot, not a language model quality benchmark.

The unchanged historical held-out set has 2048 target bytes. The previously
selected autograd-prose, geometry-prose, rust-code cohorts add 512 each. All windows
and cases were fixed before this diagnostic. These are related repository
documents, not independent language domains or an untouched final test set.

Add-one unigram and conditional byte-bigram baselines use only transitions in
the 128 actual training batches. No validation labels, unused training windows
or future updates fit either baseline. Add-one bigram is a simple fixed control,
not a tuned or optimal byte model.

Native Python uses the private source-candidate wheel, Metal/Apple M4. Its
metadata still says 0.4.31; these source-only methods are not claimed to be in
the published 0.4.31 wheel. CPU float32 PyTorch 2.12.1 independently evaluates the
exact same checkpoint weights, with one thread and deterministic algorithms.
This is numerical comparison, not a speed benchmark.

## Results

Every model is worse than the train-only unigram on every evaluated cohort.
Historical held-out mean cross-entropy, in nats per byte:

| Seed | Ordinary | Query trained | Query frozen | Unigram |
| --- | ---: | ---: | ---: | ---: |
| 11 | 3.560238 | 3.560208 | 3.560208 | 3.400582 |
| 23 | 3.554492 | 3.552469 | 3.552469 | 3.400582 |
| 37 | 3.554529 | 3.553147 | 3.553147 | 3.400582 |

All nine models generate 50..64 spaces out of 64 new bytes, using the same two
preselected prompts and 32 greedy bytes per row. Native and Torch output bytes
agree for all nine. This does not establish useful generated language, and the
query-trained versus query-frozen differences do not support a query-learning
advantage at this budget.

For the first two batches of each cohort, reverse the first 31 input bytes while
preserving the final input byte and next target. The last-position distributions
do respond: cohort-mean total variation ranges roughly 0.0021..0.0071. This is
limited evidence of context sensitivity, not proof of useful context learning;
reversed prefixes are an intervention, not an in-distribution quality score.

Same-weight parity passes for 8,294,400 logits, including reversed-prefix last
positions. Maximum absolute error is 1.9073486328125e-6; maximum cohort mean-CE
discrepancy is 2.0445065640473103e-8. The predeclared per-logit gate remains
`abs(error) <= 3e-6 + 5e-5 * abs(reference)`.

## Verification And Reproduction

`native-result.json`, `torch-result.json` and `comparison.json` retain all cases
and cohorts, including raw-score hashes. The comparator recomputes baselines,
CE, entropy, accuracy and context sensitivity from source-bound inputs and private
raw scores; it rejects same-engine comparisons. Ten offline tests cover these
checks. A read-only independent review found three verifier issues; residual
association, engine identity and summary binding were fixed before both engines
were rerun. The prereview evidence remains private, not overwritten.

With the retained historical archive and serving gate:

```bash
python -I -S -B diagnose.py prepare --repo "$REPO" --archive "$ARCHIVE" \
  --gate "$GATE" --output "$PRIVATE/plan.json"
"$SOURCE_WHEEL_PYTHON" -I -B diagnose.py run --engine native \
  --plan "$PRIVATE/plan.json" --output "$PRIVATE/native"
"$TORCH_PYTHON" -I -B diagnose.py run --engine torch \
  --plan "$PRIVATE/plan.json" --output "$PRIVATE/torch"
python -I -S -B diagnose.py compare --plan "$PRIVATE/plan.json" \
  --native "$PRIVATE/native" --reference "$PRIVATE/torch" \
  --output "$PRIVATE/comparison.json"
python -I -S -B test_diagnose.py -v
```

The archive contains `input/request.json`, `browser-checkpoint-128.json` and
`browser.json` from the historical v8 study. This is not a corpus download or a
self-contained model distribution. Keep exact source bytes; the plan binds
their hashes and reference helpers. No model weights or prompt text are needed
in this public result bundle.

Next is a separately declared fresh-run 128/256/512-update pilot, preserving all
25 geometric control arms. Lower teacher-forced loss alone will not qualify its
free-running behavior or justify promoting a model.
