# GPT-2: ordinary affine versus a freely learned radius

This is a terminal-checkpoint transfer comparison, not a radius sweep. The
[prospective plan](plan.json) and [sample freeze](execution.json) came before
the three new fits or any held-out forward. The model is the real pretrained
GPT-2 small. Rust owns training, scoring, reductions and checkpoint parsing;
neither PyTorch nor NumPy is used. This compares two **native adapters**, not
native training against PyTorch, LoRA or full-model fine-tuning.

## What is matched

Each pair starts from the same frozen GPT-2 and four function-matched,
nonzero insertions at MLPs 8 through 11. AdamW sees the same permutation of
512 disjoint 64-token windows from *Pride and Prejudice* chapters 1-49,
256 updates and 32,256 next-token targets per fit. Both arms train 6,144
channel parameters; free also trains four log-radius scalars. No radius bound
is applied. Equal token and update budgets do not mean equal FLOPs.

- Seeds 101/103/107 reuse six original [public-wheel fits](../2026-10-31-native-book-ablation/README.md).
- Seeds 109/113/127 reuse three [development-wheel free fits](../2026-11-01-native-radius-drift-learning/README.md). Only their three missing affine controls are trained on that same wheel, observing every 32nd update.
- All twelve final owners, the frozen base and both initial controls are scored using **one** pinned development wheel. Analysis is paired within seed and reported separately by training cohort and book.

The held-out books are [Austen, *Sense and Sensibility*](https://www.gutenberg.org/ebooks/161),
[Shelley, *Frankenstein*](https://www.gutenberg.org/ebooks/84), and
[Faraday, *The Chemical History of a Candle*](https://www.gutenberg.org/ebooks/14474).
Each contributes six sections, twelve non-overlapping windows per section and
4,536 predicted tokens per state. These books are new to this study, **not**
known to have been excluded from GPT-2 pretraining. The old reserve is not opened.

## Reproduce

Use Python 3.12. `OLD` and `DRIFT` are the two private original study roots
with their fitted rosters, training windows, initialization and checkpoints.
The workers refuse missing or changed originals rather than silently refitting
them. The three raw UTF-8 files in `RAW` are Project Gutenberg's
`https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt`, with SHA-256 in the plan.
`MODEL` is the pinned local GPT-2 snapshot. `TOKENIZERS_SITE` contains
`tokenizers==0.22.2`; the isolated study interpreter `PY` instead contains
only the exact development wheel specified in the plan. Use a new private
`DEST`; generated windows, observations, weights and logs stay outside Git.

```sh
python3.12 -I -S -B test_transfer.py
python3.12 -I -S -B sample.py "$RAW" "$MODEL" "$TOKENIZERS_SITE" "$OLD/data/train.json" "$DEST/data"
python3.12 -I -S -B run_transfer.py run --old "$OLD" --drift "$DRIFT" --model "$MODEL" \
    --wheel "$WHEEL" --python "$PY" --out "$DEST/run" --data "$DEST/data" \
    --plan-pin 270d7ffe4b51cc5bf843a1cb20675f785160afea3c936e6ae4ffec4aee73756f \
    --data-pin 0631478d0d1c621c859af1bb7c2ee224430a1b66f65c2366c6428f0979f2013e
python3.12 -I -S -B run_transfer.py audit --old "$OLD" --drift "$DRIFT" --model "$MODEL" \
    --wheel "$WHEEL" --out "$DEST/run" --data "$DEST/data"
```

The audit is file-only; it checks all twelve fit streams, seed-paired input
IDs, checkpoint ownership, the 48 observed pairs in the second cohort, and
every scored denominator. Six seed pairs share the same books; they are not
six independent draws from a population of books. Generation and speed are
not measured.
