# Does the learned WaveGate radius itself change inference?

This is an inference-only intervention on the [nine frozen GPT-2 book fits](../2026-10-31-native-book-ablation/README.md),
not a retry of interrupted learning. The earlier scores motivate the question;
they are not the input or the outcome of this probe.

For each training-order seed, compare four cells. The first letter is the
recipient: keep **everything except the four log-radius values** from that
final checkpoint. The second is the source of the radii. F means fixed4
training, L means learned4 training. FF and LL are the original models; the
two off-diagonal cells are deliberately synthetic inference artifacts. Keep
the ordinary affine model per seed and the unadapted base as controls.
Both same-source reconstructions must have the original artifact IDs.

The [frozen design](plan.json) includes eight authored prompts with explicit
reference continuations. Primary scores teacher-force the *same new target
tokens*, masking the full prompt. Independently, free generation uses identical
greedy softmax/KV recipes and no supplied continuation. A divergence position
is a change in output, not evidence of a better output. A direct radius effect
is conditional on these finished weights; it is not a training mechanism or
a generalization result.

## Results

All 16 states completed all eight prompts on the public 0.4.33 CPU runtime:
91 selected targets per state and 2,048 generated tokens in total. The
prespecified hypothesis that learned radii lower NLL on both recipients at
every seed was **not** met. In fact, replacing the fixed radii with the learned
values raised NLL for every prompt at every seed on both recipients. These
are repeated measurements on the same eight hand-authored items, not 48
independent test samples.

| State | Target-weighted NLL (mean across data-order seeds) |
| --- | ---: |
| Unadapted base (one state) | 3.758424 |
| Ordinary affine | 3.762978 |
| FF: fixed-trained recipient, original fixed radii | 3.761237 |
| FL: fixed-trained recipient, learned radii | 3.761654 |
| LF: learned-trained recipient, original fixed radii | 3.761296 |
| LL: learned-trained recipient, learned radii | 3.761782 |

| Seed | FL minus FF | LL minus LF | LL minus FF |
| --- | ---: | ---: | ---: |
| 101 | +0.000408 | +0.000486 | +0.000534 |
| 103 | +0.000433 | +0.000499 | +0.000565 |
| 107 | +0.000411 | +0.000473 | +0.000536 |

The **direct** radius contrasts averaged +0.000418 and +0.000486 on the two
recipients. The recipient contrast at fixed radii averaged +0.000059;
the interaction averaged +0.000068. Do not infer a training-time mediation
percentage: these are interventions on final states, not on their histories.
In the earlier book comparison LL scored better than FF; on these authored
targets the ranking reversed and the unadapted base scored best. This is a
reason to test retention alongside adaptation, not a general benchmark win.

Of 48 paired **radius-only** free-generation comparisons, **one** changed:
FL versus FF, seed 107, prompt n1 at generated token 8. LL versus LF never
changed; LL versus FF diverged in that same case and LL versus affine did
not diverge under this 16-token greedy policy. These are not additional
independent changes or a judgment about prose quality.

See [execution.json](execution.json) for the pre-inference pins and
[results.json](results.json) for all scores, parent/daughter state IDs,
per-prompt commitments and divergence positions. To view all continuations
locally without another forward pass:

```sh
python3 -I -S -B private_samples.py --model "$MODEL" --tokenizers-site "$TOKENIZERS_SITE" \
  --run "$PRIVATE/run" --output "$PRIVATE/private-samples.txt"
```

## Reproduce

Use the exact original public 0.4.33 macOS wheel, pinned local GPT-2 snapshot,
the nine original private final checkpoints, and a separate offline
`tokenizers==0.22.2` installation. Use an empty private parent and a **new** run
directory. No original book text, train windows, reserve or prior replay are
needed. Pin the prepared input independently, then launch the one worker:

```sh
python3 -I -S -B test_interchange.py -v
"$PUBLIC_PYTHON" -I -B test_native_probe.py -v
python3 -I -S -B study.py prep --model "$MODEL" --tokenizers-site "$TOKENIZERS_SITE" \
  --plan-pin "$PLAN_SHA" --encoded "$PRIVATE/encoded.json"
python3 -I -S -B study.py run --original "$ORIGINAL" --wheel "$PUBLIC_WHEEL" \
  --python "$PUBLIC_PYTHON" --model "$MODEL" --output "$PRIVATE/run" \
  --plan-pin "$PLAN_SHA" --encoded "$PRIVATE/encoded.json" --encoded-pin "$ENCODED_SHA"
python3 -I -S -B study.py audit --output "$PRIVATE/run"
```

One CPU attempt, maximum 15 worker minutes; stop and keep the first failure
or timeout. Audit reads saved files only. No incomplete matrix is aggregated.
Generated IDs/step logs and all learned tensor values remain private.
