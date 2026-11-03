# GPT-2: ordinary affine versus a freely learned radius

This is a terminal-checkpoint transfer comparison, not a radius sweep. The
[prospective plan](plan.json) and [sample freeze](execution.json) came before
the three new fits or any held-out forward. The model is the real pretrained
GPT-2 small. Rust owns training, scoring, reductions and checkpoint parsing;
neither PyTorch nor NumPy is used. This compares two **native adapters**, not
native training against PyTorch, LoRA or full-model fine-tuning.

## Result

**Ordinary channel-affine had lower held-out NLL in all 18 book/seed pairs.**
Both adaptations improved on frozen GPT-2 in all 18. The preregistered
same-author criterion for free radius failed (0/6). The direction held in
all 108 sampled section/seed comparisons too; those are repeated evaluations
of the same sections, not 108 independent books.

| Book | Frozen NLL | Initial NLL (both) | Mean free minus affine, 101/103/107 | Mean free minus affine, 109/113/127 |
| --- | ---: | ---: | ---: | ---: |
| Sense and Sensibility | 4.220843 | 4.220789 | +0.003989 | +0.004044 |
| Frankenstein | 4.051046 | 4.050914 | +0.001635 | +0.001662 |
| The Chemical History of a Candle | 3.844245 | 3.844304 | +0.000477 | +0.000493 |

Positive favors affine. Starting NLL matched **exactly in every section**.
The finite transfer effect is larger in the same-author novel than in the
science lectures; do not infer a general genre effect from one book per type.
These results do not negate the earlier free-versus-fixed-radius comparison:
winning against fixed radius and beating an ordinary affine adaptation are
different claims. There is no basis here to promote free radius to a default.

[All paired values, section contrasts, model and step identities](results.json)
are retained, including changes from both controls. The independent file-only
audit reproduced the complete primary result with SHA-256
`9a2e7ec01b7dc1e730d05216e2d5ed7d038ae3033a33baa8d3723b916b282a40`.
To inspect the result **without any private inputs or installed ML library**,
run `python3.12 -I -S -B public_scores.py` here. It reconstructs the original
15 scalar score-file hashes and recomputes the reported contrasts from the
[270 section scores](section_scores.json).

## Backend Cross-check

On source [205f350a2](https://github.com/RyoSpiralArchitect/SpiralTorch/commit/205f350a2acb3ea1b94cf3f88d114da4bd236da1),
the Rust `NativeCausalLm.evaluate_variants` path shared eight frozen decoder
layers per active microbatch. For **15 states x 18 sections**, NLL, target
counts and data IDs matched the separately built scalar study wheel **exactly**.
This was an inference-only check on the same sealed states, not a second
training run or a wall-clock comparison.
See the [frozen check](shared_prefix_plan.json) and [original result](shared_prefix.json).

Independent review then tightened pre-admission of oversized lazy inputs,
duplicate microbatch shapes and direct Rust document counts. On the
[hardened source](https://github.com/RyoSpiralArchitect/SpiralTorch/commit/c10b315cdf20ef99b2be04ff7e6f6e9a2eea903b),
an additional [three-state mixed-length probe](masked_probe.json) reproduced
the entire scalar reports byte-for-byte, with 59 selected targets, 44 masked
targets and one fully skipped window. This latter check is **not** a second
270-score study on the hardened wheel. Local validation on that source:
1,065 Rust library tests, 103 installed-wheel Python tests, and a WASM
compile check (not browser GPT-2 execution). Neither development wheel is a
PyPI release.

The [integration validation](integration_validation.json) records the final
Python row-ownership fix separately. The installed wheel passed 105 tests,
including the two-thread replacement regression. The Rust numerical core did
not change after the mixed-length probe.

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

For the optional backend checks, use fresh isolated environments with the
respective exact development wheels. Set `RAYON_NUM_THREADS=4`,
`VECLIB_MAXIMUM_THREADS=1`, and `OMP_NUM_THREADS=1`. Invoke
[`verify_shared_prefix.py`](verify_shared_prefix.py) with the same inputs as
the audit, the comparison wheel in `--wheel`, the result/wheel SHA-256 pins
in [the receipt](shared_prefix.json), and a fresh `--receipt` file. For the
post-review probe use [`masked_probe.py`](masked_probe.py) with the hardened
wheel, `--old`, `--drift`, `--out`, `--model` and a new `--receipt`.
The public score audit requires none of these wheels.
