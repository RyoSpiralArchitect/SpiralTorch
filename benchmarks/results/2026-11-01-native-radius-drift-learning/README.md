# Fixed, free, and bounded WaveGate learning across two kinds of prose

**[Result](#observed-outcome):** free radius had the lowest NLL on both books
at every seed. The 0.02 bound was active but did not meet the joint criterion.

This is a fresh, matched native GPT-2 fine-tune, **not** another replay of a
previous fit. The [protocol](plan.json) was fixed before learning. The single
bound of 0.02 in log radius permits a multiplicative change of about 2% from
each site's initial radius of 4. It is not a per-step limit. No width search,
best checkpoint, test-set selection or replacement seed is allowed.

The [prior inference-only radius interchange](../2026-11-01-native-radius-interchange/README.md)
found that substituting learned radii on eight authored prompts slightly
increased NLL even though learned radius had improved the earlier book score.
That observation motivated a retention control; its prompts and scores are
not inputs to this study.

All three arms train the four WaveGate MLP placements on the same 512
Pride and Prejudice training windows as the original study, now with order
seeds 109, 113 and 127. Each arm gets 256 AdamW updates and 32,256 shifted
targets per seed. The free and bounded arms have identical initial geometry
and trainable parameter counts. Fixed has four fewer trainable scalars.
Its clipping norm also excludes gradients for those buffers. Thus fixed vs
free is not solely a change of inference radii; bounded vs free is the
matched-trainability comparison.

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
python3 -I -S -B trajectory.py "$RUN"  # separate, descriptive localization
```

One attempt, local CPU only, at most 40 minutes per worker and four hours total.
Save private observation/optimizer pairs at 32-update intervals. Stop on the
first failure and retain every partial file. Only a complete cohort can yield
a joint outcome. Holdout scores are not computed until all nine terminal pins
are sealed. No raw text, individual radii, weights or development wheels are
published.

## Observed outcome

The complete first attempt passed the integrity checks, **not** the learning
criterion: **0 of 3 seeds** passed the joint test. All nine fits completed
256 updates; the data-only audit verified 2,304 ordinary reports, 72 paired
intermediate checkpoints and 308 chapter scores. Both books had exactly
7,056 scored targets per model. A separate computation from the chapter files
agreed with the weighted NLLs and all three decisions. Both the original
private files and the [full scalar projection](results.json) are retained.

| Model | Emma NLL | Darwin NLL |
| --- | ---: | ---: |
| Frozen GPT-2 | 4.243032 | 3.993348 |
| Frozen common nonzero insertion | 4.242908 | 3.993321 |
| Fixed radius 4 | 4.233664 | 3.991559 |
| Freely learned radius from 4 | **4.231955** | **3.991254** |
| Bounded log radius, width 0.02 | 4.233563 | 3.991541 |

Trained rows average the three data-order seeds, with no selection. Lower
is better. Bounded was slightly better than fixed but worse than free on
both books at **every** seed. All trained arms beat the frozen base on these
samples: there was no measured Darwin loss above the frozen baseline to fix.

| Seed | Bounded minus free, Emma | Bounded minus free, Darwin | Free-over-fixed Emma gain retained | Individual radius candidates projected / 1,024 |
| --- | ---: | ---: | ---: | ---: |
| 109 | +0.001604 | +0.000290 | 5.89% | 920 |
| 113 | +0.001624 | +0.000288 | 5.84% | 933 |
| 127 | +0.001597 | +0.000285 | 5.88% | 931 |

The fixed bound changed 2,784 of 3,072 radius candidates (90.625%). The
largest sampled absolute log-radius displacement was 0.01999998 in the
bounded runs, versus 0.3793-0.3847 in free runs. A separate [trajectory
diagnostic](trajectory.json) found that the first different ordinary report
was at update 21, 20 and 20 respectively; at each boundary **only the
post-update geometry ID** differed. Up to that point the ordinary reports
matched byte-for-byte. This is a file-only localization, not an additional
endpoint or evidence about which individual radius moved first.

Keep the bound an opt-in constraint, not a quality default. The earlier small
authored prompt result did not predict the Darwin ranking. These two finite
samples do not establish general retention, the best possible width or an
advantage over ordinary affine FT (not an arm here). Do not tune a width
until it wins on these same two evaluations.

The [runtime checks](runtime_checks.json) separately cover old-reader
rejection, unbounded checkpoint compatibility, the optimized update loop,
and corpus-free inference from one real intermediate boundary. Those
mechanical passes do not turn the failed learning criterion into a pass.
