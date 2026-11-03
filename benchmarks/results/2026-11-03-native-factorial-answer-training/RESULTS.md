# GPT-2 small: learning the four Wave operators

**All four arms learned the extraction task to a similar degree, but none
changed which answer the greedy decoder emitted within any training order.**
Strict exact lines rose from 2/72 to 15/72 or 16/72. Ordinary affine had
lower answer-weighted NLL than projection in all three orders. Neither met
the [registered joint advantage rule](README.md): NLL and strictly exact
answers must both improve in every order. A missed gate is **not equivalence**.

## Matched outcomes

NLL is the unweighted mean of the 72 per-answer mean content NLLs (nats;
lower is better). The LF is a training target but excluded from NLL. Within
each row the final three columns are the same for all four operators.

| Checkpoint / order | Affine NLL | Saturation NLL | Projection NLL | Both NLL | Teacher top-1 / 111 tokens | Greedy exact / 72 | Empty first line / 72 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Common start | 3.880304 | 3.880304 | 3.880304 | 3.880304 | 48 | 2 | 39 |
| 109, update 480 | 1.890072 | 1.895479 | 1.897441 | 1.897441 | 53 | 15 | 46 |
| 113, update 480 | 1.877675 | 1.883411 | 1.885468 | 1.885468 | 53 | 16 | 45 |
| 127, update 480 | 1.878368 | 1.883826 | 1.885776 | 1.885776 | 54 | 16 | 45 |

- Projection minus affine was **+0.007369, +0.007794, +0.007409** nats.
  Projection improved/tied/regressed on **0/0/72, 1/0/71, 3/0/69** answers.
  It gained **zero** exact answers and lost **zero** in each order.
- The NLL interaction `(both - saturation) - (projection - affine)` was
  -0.005408, -0.005736 and -0.005458. Projection and both had identical
  per-answer score cells and all 480 scalar training losses in each order.
  Their final LoRA IDs and numerical gate/bias arrays also matched on private
  checkpoint read-back; their geometry recipes are distinct. This does not
  establish that saturation is always inactive.
- All 72 generated ID sequences matched affine for every arm within each
  order. LF termination rose from 50/72 to 72/72, but 45-46 final answers
  began with LF. This is a failure to produce content, not a decoding cutoff.
  It motivates separating content learning from stopping behavior; it does
  not by itself identify why the model stopped early.

## Evidence boundary

One attempt on the fixed CPU wheel, no retries: 120 new training records,
four epochs, four operators and three orders = **5,760 arm-updates** (1,440
matched steps). Only the common start and the reloaded final checkpoints
were scored, on the *same* 72 held-out records in every order, not three
independent datasets. No Torch, NumPy or Transformers in the model worker;
no external API, GPU, adaptive hyperparameter selection or publication of weights.
Active wall time including preparation: 928.70 s; CPU: 897.77 s.

The independent original-file check reconciled all 16 aggregate scores,
recalculated the masked evaluation data ID, hashed all 12 checkpoints,
re-decoded all 1,152 prompt-only streams and reproduced the decision without
running the model. [Prepared manifest](prepared-receipts/manifest.json) and
[allowlisted receipts](receipts.zip) are public; input arrays, model weights,
checkpoints and raw execution logs are not included. Archive: 41 files,
561,097 bytes, SHA-256 `55bf04599e5a9d8a4c02b14d0b67991589b543b7dd2aa95b6df27025ac22b803`.
Result JSON SHA-256: `8f115d7460697880986f492bf20abc8089523614447495de08214b21066f5d6b`.

```sh
python3 -I -S -B verify.py receipts.zip
python3 -I -S -B summarize_outcome.py receipts.zip
```

The archive-only check validates the plan, receipts, decoded-line adjudication
and decision; independent tokenizer re-decoding and original byte identities
require the private inputs, as documented in [the input freeze](PREPARED.md).
