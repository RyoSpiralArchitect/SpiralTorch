# Input freeze (no pretrained decoder execution)

The trainable four-arm method and its materialization are separate:
source `ab63b0e3f64412b72814b8c32c89aebda9891964`;
[encoded input manifest](prepared-receipts/manifest.json) SHA-256
`72151805040cddbaa9d2149abc90ba5d424da8513eaa62fe2ca240363c7843c9`.
Preparation completed once, using the pinned tokenizer and installed native
wheel for mask alignment. It did not open the pretrained decoder weights.

| Input | Records | Input tokens | Scored targets | Longest input |
| --- | ---: | ---: | ---: | ---: |
| Train | 120 | 3,020 | 185 content + 120 LF | 29 |
| Held-out content NLL | 72 | 1,736 | 111 content; no LF | 28 |

Prompts for generation were encoded **without** the answer and verified equal
to the masked prefix. Every full source, prompt and answer round-tripped the
pinned tokenizer. No old score or checkpoint was used. The original exposure
set was excluded by triple identity, not string equality of the question.

To audit the metadata with no tokenizer or model:

```sh
python3 -I -S -B verify.py prepared-receipts --prepared-only
```

To check the separate original token arrays too, add `--originals "$NEW"
--model "$MODEL" --codec-site "$CODEC_SITE"`; this opens only tokenizer data,
never model weights. `test_trial.py` and `test_prepared.py` rehearse the four
native operators, actual multi-epoch synthetic SGD and checkpoint reload,
prompt-only greedy generation, independent line adjudication, epoch and mask
integrity, and rejection of unknown archive members.

This input receipt contains **no learning or quality outcome**. To run the
registered model trial, use the frozen source commit and the manifest pin
above. Only the original training run may create its exclusive fit attempt.
