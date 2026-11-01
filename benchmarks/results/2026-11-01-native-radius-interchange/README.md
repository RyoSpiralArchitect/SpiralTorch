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
