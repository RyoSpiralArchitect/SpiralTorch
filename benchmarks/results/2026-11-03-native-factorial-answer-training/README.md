# Train the four Wave operators from a common origin

The [same-input local factorial](../2026-11-03-native-wave-factorial/RESULTS.md)
measured residuals and adjoints, not learning quality. Here each condition gets
its own actual joint fit. Nothing from the prior fits initializes these arms.

## Frozen comparison

Use the pinned native GPT-2 small base, deterministic attention output LoRA
(block 11, rank 2, alpha 4) and trainable zero gate/bias, strength 0.25.
The four arms are ordinary affine, saturation only (saturation 1, porosity
0.05), projection only (curvature -1, fixed log radius 0), and both (Wave v2
with the same fixed radius). All start as the identity residual addition with
the same first derivatives and **4,608 trainable scalars**. Use answer-masked
joint SGD, lr 0.01, one record per update. For each fixed order, repeat the
permutation four times with no early stop or shuffling between epochs:
120 x 4 x 4 arms x 3 orders = **5,760 arm-updates**. No hyperparameter sweep.

Exclude every triple from both prior train/eval splits (288 of 512). A
single deterministic bipartite matching supplies 120 fresh train records and
72 disjoint eval records, leaving 32 unused. Each field/value pair has 5 train
and 3 eval records; field position and template are balanced. Vocabulary and
prompt forms are shared by design. Three orders are not independent test sets.
The canonical specification, record texts and permutations are sealed by
`52c9c589de05f0c6a1f7d2b8b95493ac7a9df717730443c4b981f6ddf0f7b537`.

## Quality endpoints

Training includes the answer and one final LF; the prompt is masked. This is
a new stopping objective and a new split: do not compare its absolute scores
to earlier pilots. Score only at the common start (all four arms) and on the
12 final, hash-verified, reloaded checkpoints. Ordinary teacher-forced NLL and
top-1 exclude the LF and use only answer tokens. No epoch-end held-out probes.

Separately give the native KV greedy generator **only the independently encoded
prompt**, never the answer length or allowed values. Greedy, no logit control,
six tokens maximum regardless of answer length, stop IDs LF (198) and GPT-2
EOS (50256). Decode with the pinned tokenizer in a model-free process. Strict
line exact requires an emitted LF and an exact match up to its first occurrence,
including the intentional leading space. No trimming, normalization, or
canonical-token-ID equality. `" red"` without LF and `" red clay\n"` are not
exact for `" red"`. Termination and literal-prefix counts are descriptive only.

**Primary**: projection-only minus affine on identical answers. Declare a joint
advantage only if all three orders gain at least one strictly exact line **and**
answer-weighted mean content NLL improves beyond `log(1.001)` nats in the same
direction; the rule is symmetric for affine. A miss is not equivalence. Always
report the actual counts, gained/lost answers, NLL, all four arms, and the
factorial interaction `(both - saturation) - (projection - affine)`. No
selection of the best order or extrapolation to broad LLM quality.

## Reproduce

Use the local [fixed development wheel](../2026-11-03-native-wave-factorial/build.json).
The CPU-only preparation never opens model weights. The run consumes its
external manifest pin and the same prepared files. Two isolated workers fit
and adjudicate; no Tokenizers, Torch, NumPy or Transformers in the model
process. Preparation and execution have a cumulative 60-minute CPU and active
wall cap (a pause between phases does not consume training time). One attempt
per phase, no retry or overwrite; the first failure and partial journal survive.

```sh
"$VENV/bin/python" -I -B test_trial.py -v
"$VENV/bin/python" -I -B run.py prepare --model "$MODEL" --wheel "$WHEEL" --output "$NEW" --codec-site "$CODEC_SITE" --commit "$FROZEN_COMMIT"
"$VENV/bin/python" -I -B run.py run --model "$MODEL" --wheel "$WHEEL" --output "$NEW" --codec-site "$CODEC_SITE" --commit "$FROZEN_COMMIT" --manifest-pin "$PREPARED_PIN"
```
