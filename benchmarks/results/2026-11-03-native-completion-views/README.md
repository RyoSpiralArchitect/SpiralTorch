# One completion, three native views

See the [Python/Rust/WASM API](../../../docs/python/native-training-session.md#one-completion-three-uses)
and the [portable fixture](../../../crates/st-nn/tests/fixtures/causal_completion.json).
The independently encoded four-ID prompt must be the training prefix.
Only the five subsequent IDs are training targets; the last is the explicit LF,
excluded from the four content targets. No reference is passed to the generator.

On a synthetic two-layer width-four causal decoder, the previous CPU wheel
with manual IDs and the new wheel with `CausalCompletion` produced identical
two-update joint-SGD step reports, checkpoints, answer-level evaluations,
three greedy IDs, decision reports and generation cursors. Both model
processes forbade Torch, NumPy, Transformers and Tokenizers. Separately, the
optional local BPE test checked real boundary merges; the Node/WASM test
checked both raw ID transport and caller-supplied source offsets.

The [qualification record](validation.json) binds the original wheels, native
binaries, test logs and matched signature. From this directory, in independent
clean environments containing the indicated local wheels:

```sh
"$OLD_PY" -I -B check_installed.py manual "$OLD_WHEEL" "$OLD_SHA" --expect validation.json
"$NEW_PY" -I -B check_installed.py views "$NEW_WHEEL" "$NEW_SHA" --expect validation.json
```

These are synthetic compatibility checks, not a language-quality or speed
comparison. No pretrained weights, research corpus, GPU, or published wheel
was used or changed. The separate preregistered study still owns its frozen
wheel and input manifest.
