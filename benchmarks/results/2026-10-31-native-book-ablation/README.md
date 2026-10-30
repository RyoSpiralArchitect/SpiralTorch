# Native pretrained GPT-2: an initially matched book ablation

This is a prospective comparison on the published 0.4.33 macOS wheel. Rust
owns the frozen decoder, teacher-forced loss, geometry gradients, clipped AdamW
and evaluation. There is no Torch, NumPy or Transformers comparison runtime.
The ordinary control learns a channel-affine adapter, not base model weights.

The [protocol](plan.json) fixes three fresh data-order seeds and nine final-step
fits. Every arm sees the same 512 disjoint 64-token training windows in each
pair (256 updates and 32,256 shifted targets). Gates start at zero. Each radial
branch has a nonzero signed bias; ordinary affine copies the exact Rust radial
output as its bias. Thus all sites initially add the same vector on any input;
train-only site and decoder-logit probes check exact equality before updates.
This is **not** a constraint on residual magnitude after learning. Fixed and
learned radius start at 4; the learned arm has four extra scalar parameters.
Equal decoder/data budgets do not imply equal FLOPs or wall time.

Training uses chapters 1-49 of the original [Pride and Prejudice preparation](../2026-10-18-hf-book-controls/README.md).
The primary score is unsmoothed NLL on [Persuasion](https://www.gutenberg.org/ebooks/105),
an entirely different book from the FT corpus: eight disjoint windows per
chapter, 24 chapters and 12,096 targets. The secondary score uses eight windows
in each of the six previously used development chapters (3,024 targets).
The old reserve is never opened. The common trained-checkpoint roster is sealed
before model scoring of either split. These are fixed sampled excerpts, not
whole-book perplexity or a claim about what GPT-2 saw in pretraining.

The primary contrast is final learned-radius minus ordinary-affine NLL per seed.
The frozen base and frozen common insertion distinguish fitting from insertion
alone. Report all declared arms, seed contrasts, target counts and chapter
directions, including null or adverse outcomes. No best checkpoint, retry,
sweep, speed ranking, deployment promotion or free-generation quality claim.

## Reproduction

Use CPython 3.12, the original public macOS wheel (install offline with
`--no-deps --no-compile` in a fresh private venv), the exact GPT-2 snapshot and
the book files named in `plan.json`. No model, text, token arrays, learned
tensors or private keys are provided in the repository. Tokenization only uses
the separately installed `tokenizers==0.22.2` package. Use new private output
directories and independently retained data/plan/preflight pins.

```sh
python3 -I -S -B test_study.py -v
python3 -I -S -B prepare.py --model "$MODEL" --chapters "$CHAPTERS" \
  --external "$PG105" --tokenizers-site "$TOKENIZERS_SITE" --output "$PRIVATE/data"
python3 -I -S -B study.py preflight --python "$VENV/bin/python" --wheel "$WHEEL" \
  --model "$MODEL" --data "$PRIVATE/data" --plan-pin "$PLAN_PIN" \
  --data-pin "$DATA_PIN" --output "$PRIVATE/before"
python3 -I -S -B study.py run --python "$VENV/bin/python" --wheel "$WHEEL" \
  --model "$MODEL" --data "$PRIVATE/data" --plan-pin "$PLAN_PIN" \
  --data-pin "$DATA_PIN" --preflight "$PRIVATE/before/preflight" \
  --preflight-pin "$PREFLIGHT_PIN" --output "$PRIVATE/run"
```

The supervisor owns each worker process group, with a 40-minute per-worker and
four-hour shared wall-clock bound. A failed or interrupted worker is not a
completed fit. Its first logs/outputs remain in place; no alternate seed or arm
is substituted. Private checkpoints and per-update reports allow a later
data-only audit without another training run.
