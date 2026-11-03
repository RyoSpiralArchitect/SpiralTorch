# Matched native SGD on a synthetic decoder

This is a same-wheel CPU comparison of **three training arms together**, not
an increase in single-model training speed. The arms use bare LoRA, LoRA with
a frozen WaveGate, and trainable LoRA plus WaveGate, respectively. The wave
arms start with identical parameters. No pretrained weights or book data are used.

Each arm makes six SGD updates on the same 48 labeled targets. The generated
decoder has six blocks (width 64, four heads); all adapters are on block 5.
Independent steps execute five identical frozen blocks three times per batch.
Matched steps execute them once; each suffix and backward pass stays separate.
One untimed-for-analysis warmup per path precedes six measured pairs, with
alternating order. Each sample includes the Python calls and returned step
objects as well as native SGD and artifact validation/hashing. The protocol's
serialization/disk exclusion refers to explicit checkpoint/report serialization
and file I/O by the harness, not incidental runtime work.

| Six updates for all three arms | Median elapsed |
| --- | ---: |
| Independent | 39.46 ms |
| Matched | 23.59 ms |

The median of the six **paired** independent/matched ratios was **1.66x**
(range 1.51x to 1.90x). Every path and repeat had the same ordered report digest
and the same three final checkpoint IDs. In-sample masked scores were also
identical between execution paths in the warmup. This is one host, shape and
adapter placement, not a PyTorch, GPU, peak-memory or language-quality result.

The original [protocol](protocol.json), [all observations](observations.jsonl)
and [summary with hashes](result.json) are included. To reproduce from a
checkout with a matching locally built Python wheel:

```bash
python -I -B tools/bench_native_matched_sgd.py --output NEW_DIRECTORY
```
