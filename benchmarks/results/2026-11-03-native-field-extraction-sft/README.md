# GPT-2 small: supervised field extraction on unseen records

Jointly training the last-block WaveGate with LoRA reduced answer NLL relative
to freezing that same WaveGate on **48/48** evaluation records. The average
effect was small: **-0.000787 nats per target with equal answer weight**.
Neither top-1 token hits nor all-correct answer counts increased.

| CPU f32 arm | Initial token-weighted NLL | Final token-weighted NLL | Final top-1 tokens | Final all-correct answers |
| --- | ---: | ---: | ---: | ---: |
| LoRA only | 3.631057 | 3.626278 | 31/74 | 5/48 |
| LoRA + frozen WaveGate | 3.631057 | 3.626278 | 31/74 | 5/48 |
| LoRA + learned WaveGate | 3.631057 | 3.625614 | 31/74 | 5/48 |

The preregistered comparison was final joint minus frozen WaveGate. The
token-weighted delta from f64 answer rows was **-0.000664**, and the individual
mean NLL differences ranged from -0.001794 to -0.000214. The prespecified pilot
criterion (lower equal-answer mean and no loss of all-correct answers) passed.
This is teacher-forced scoring on a single ordering, not generated exact match
or a general NLP quality claim. In particular, it does not isolate curvature
from the benefit of adding trainable gate, bias and radius parameters.

## What was fixed

Before opening pretrained weights, the [protocol](receipts/protocol.json) and
[input manifest](receipts/manifest.json) fixed 96 training records and 48
evaluation records (148 and 74 supervised tokens). Each instruction asks for
one color, shape or material from a displayed record. The partitions share
the 24 possible answers and two templates, but neither a record triple nor
a prompt is reused. Query fields and field positions are balanced within each
partition. Answers include their leading space; prompts, EOS and padding are
not loss terms. All documents fit one window (longest: 28 IDs).

Source at execution: `d8a1f2efd953c989bbc320374be48200fa4dd121`.
The local GPT-2 small base, weights, tokenizer and development wheel are
hash-pinned. One rank-2, alpha-4 LoRA and one zero-gate, zero-bias WaveGate
were placed on block 11. Joint also trains the log-radius (initially zero).
The zero frozen WaveGate is an identity control here, not a demonstration
that a nonzero fixed geometry is equivalent to bare LoRA.

All arms used the same 96 documents in the same order, one unpadded document
per SGD update, learning rate 0.01 and context cap 64. The shared frozen prefix
was eleven blocks. The same 48 records were scored only before training and
after reopening the final checkpoints. The 32/64 checkpoints were saved, **not**
evaluated or used to select a winner. There was one fit attempt, no retry, no
GPU, no model API and no free-running generation. The fit (including base
loading, scoring and saving) took 48.88 s wall time / 47.39 s process CPU time;
preparing the data took another 0.56 s process CPU time. This is not a benchmark.

## Inspect and reproduce

The [initial](receipts/scores-000.json) and [final](receipts/scores-096.json)
scores retain all three arms and every answer; the [step records](receipts/steps.jsonl)
retain every update, including the two control arms. [Result](receipts/result.json)
binds the endpoints, step sequence, attempt and checkpoint boundaries. Raw
training/evaluation IDs, original text and weights/checkpoints are retained
separately, not embedded in these result files.

```bash
D=benchmarks/results/2026-11-03-native-field-extraction-sft
python3 -I -S -B "$D/verify.py" "$D/receipts"
# To additionally verify every raw input, checkpoint and external boundary:
python3 -I -S -B "$D/verify.py" "$PRIVATE_RUN" --originals
```

For a **new** reproduction, use the same local model and a validated native
CPU wheel (this source API is not a claim about a published package). Install
the wheel without bytecode or dependencies; only data preparation needs a
separately installed `tokenizers==0.22.2`. Run `pilot.py prepare` then
`pilot.py fit` with the same `--output NEW_PRIVATE_DIR --model-dir MODEL_DIR
--wheel WHEEL --wheel-pin HEX --commit HEAD_SHA --deadline UNIX_TIME`.
Preparation also takes `--tokenizers-site PATH`; fit instead requires the
**exact manifest SHA-256 returned by preparation**. Both commands require
`python -I -B` and POSIX process/resource limits. The shared wall deadline is
at most one hour; fit also accounts for preparation's process CPU time. The
worker refuses to reuse an attempt. Do not retune on this evaluation split.
