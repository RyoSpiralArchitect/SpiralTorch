# Fixed, free, and bounded WaveGate learning across two kinds of prose

This is a fresh, matched native GPT-2 fine-tune, **not** another replay of a
previous fit. The [protocol](plan.json) was fixed before learning. The single
bound of 0.02 in log radius permits a multiplicative change of about 2% from
each site's initial radius of 4. It is not a per-step limit. No width search,
best checkpoint, test-set selection or replacement seed is allowed.

All three arms train the four WaveGate MLP placements on the same 512
Pride and Prejudice training windows as the original study, now with order
seeds 109, 113 and 127. Each arm gets 256 AdamW updates and 32,256 shifted
targets per seed. The free and bounded arms have identical initial geometry
and trainable parameter counts. Fixed has four fewer trainable scalars.

Evaluate the frozen base, common insertion, and all nine terminal checkpoints
on **two separate measures**: [Emma](https://www.gutenberg.org/ebooks/158)
for adaptation to another Austen novel, and
[On the Origin of Species](https://www.gutenberg.org/ebooks/1228) for retention
on scientific prose. Each score uses 14 chapters, eight disjoint windows per
chapter, and 7,056 targets. These books are new to this comparison, not known
to be absent from GPT-2 pretraining. Frames, contents and the Darwin index
are excluded. The previous reserve and eight authored prompts are not used.

Per seed, bounded must preserve at least half the free arm's Emma gain over
fixed (free must actually beat fixed), improve on free on Darwin, and exercise
the bound. The joint test requires all three seeds. Report each score against
the frozen base even if the criterion fails. There is no cross-domain average.

## Run and verify

Use a new private output directory, the explicitly pinned local source wheel,
model snapshot, raw book files, original initial artifacts and training-file
hash in the plan. No Torch/NumPy/Transformers training backend. The data-only
audit never advances a model:

```sh
python3 -I -S -B test_protocol.py -v
python3 -I -S -B protocol.py --books "$BOOKS" --model "$MODEL" --train "$TRAIN" \
  --tokenizers-site "$TOKENIZERS_SITE" --plan-pin "$PLAN_SHA" --output "$DATA"
python3 -I -S -B run_study.py --python "$DEVELOPER_PYTHON" --model "$MODEL" \
  --wheel "$WHEEL" --train "$TRAIN" --data "$DATA" --initials "$INITIALS" --output "$RUN" \
  --plan-pin "$PLAN_SHA" --data-pin "$MANIFEST_SHA" --wheel-pin "$WHEEL_SHA" --native-pin "$NATIVE_SHA"
python3 -I -S -B audit.py "$RUN"
```

One attempt, local CPU only, at most 40 minutes per worker and four hours total.
Save private observation/optimizer pairs at 32-update intervals. Stop on the
first failure and retain every partial file. Only a complete cohort can yield
a joint outcome. Holdout scores are not computed until all nine terminal pins
are sealed. No raw text, individual radii, weights or development wheels are
published.
