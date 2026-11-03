# Exact native decoding on a synthetic, wide-vocabulary CPU model

Can a caller that only needs token IDs avoid ranking 50,257 logits, without
changing Z-Space control, the seeded sample stream, EOS, or portable resume?
The [Rust/Python APIs](../../../docs/python/native-generation.md) separate
this question from whether any model writes a better answer.

| Selection policy | Old report (ms) | New report (ms) | New single ID (ms) | New chunk (ms) | Old/new report | Old report/new chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary greedy, whole vocabulary | 56.84 | 24.54 | 4.10 | 4.03 | 2.32x | 14.29x |
| Seeded sample, top-32, p=0.9 | 45.48 | 4.13 | 4.22 | 4.12 | 10.96x | 11.20x |
| Z-Space greedy, top-32, p=0.9 | 45.46 | 4.14 | 4.23 | 4.12 | 10.91x | 11.17x |

Times are medians for two prompts, eight generated tokens per prompt.
Speed ratios are medians of the eight **within-block** ratios, not ratios
of the displayed medians. All 32 workers reproduced their warmup, and all
96 timed policy measurements agreed on IDs, stop reasons and checkpoint
hashes. Old/new detailed decisions agreed byte for byte (serialized JSON
digests). ID-only calls intentionally do not return probability/entropy.

Ordinary ID-only greedy scans the native f32 logits for their maximum.
Detailed greedy still constructs the distribution, while sampling and
Z-Space keep their true probability/control paths. For those two top-32
policies the expensive full-vocabulary sort was replaced by exact partial
selection. Returning IDs in a chunk did **not** independently speed them up
here: paired new-report/new-chunk medians were 0.965x and 0.970x.

## What was timed

- Fixed synthetic GPT-2-shaped decoder: two layers, width 16, two heads,
  32 positions; same model identity for all workers. Large vocabulary and a
  deliberately small transformer stress selection rather than transformer
  math. Not a pretrained GPT-2 throughput claim or a PyTorch comparison.
- KV prefill and appends, generator construction, and Python calls are in the
  timed section. Weight creation/loading, warmup, checkpoint hashing,
  report JSON serialization, correctness checks and disk writes are outside.
- Single local arm64 Mac. Five numerical-library thread limits set to one.
  Each worker measured all three policies, rotating the starting policy by
  block; four method orders counterbalanced over eight blocks. No foreign ML
  runtime or tokenizer. Local development wheels only, not new releases.

## Reproduce

Implementation and timing protocol: `f05bbac27`. Old wheel source:
`0720c527e`. Keep each wheel in its own clean environment without bytecode.
The installed package inventory and extension hash must match the pinned
wheel. [Attempt](receipts/attempt.json), [protocol and both build identities](receipts/protocol.json),
[raw timing cells](receipts/runs.jsonl), [paired results](receipts/result.json).

```sh
python3 -I -S -B benchmarks/results/2026-11-03-native-exact-decoding/verify.py benchmarks/results/2026-11-03-native-exact-decoding/receipts
"$NEW_PYTHON" -I -B tools/bench_native_decoding.py --output "$FRESH_DIR" \
  --old-python "$OLD_PYTHON" --old-wheel "$OLD_WHEEL" --old-pin "$OLD_SHA" \
  --new-python "$NEW_PYTHON" --new-wheel "$NEW_WHEEL" --new-pin "$NEW_SHA"
```

The verifier runs no model. This benchmark made no pretrained-model calls,
read no study documents, and made no weight updates. Browser/GPU throughput,
semantic quality and held-out accuracy are outside its scope.
See [validation](validation.json) for the Rust/Python suite counts and the
separate Node/WASM artifact and typed-declaration checks.
