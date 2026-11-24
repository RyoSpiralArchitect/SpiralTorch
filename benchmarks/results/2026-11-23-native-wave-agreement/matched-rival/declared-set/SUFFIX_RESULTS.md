# Exact candidate scores without running the frozen prefix twice

The finite-checked geometry step now reuses the same block-boundary tensor
that feeds the original geometry backward. The guarded candidate still runs
every block from the first attached geometry onward, gathers the same labeled
rows, and calculates fresh logits and both losses. No cache is shared between
steps. The Python API, result keys, strict acceptance rule and failure
semantics have not changed.

## Wall time

[Frozen protocol](SUFFIX_BENCH.md), commit `952a9ed9f35e`. Synthetic 12-block
decoder, width 64, four heads, 512 classes, two 16-token contexts per update,
WaveGate R=1. Apple M4 arm64, macOS 26.4.1, CPython 3.8.20, CPU release builds.
Both builds used identical model weights, input bytes, model config and
geometry initialization at each site.

| First geometry block (zero-based) | Old ms/step | New ms/step | Median paired speedup | All four paired ratios |
| --- | ---: | ---: | ---: | --- |
| 0 | 10.966 | 11.011 | 0.994x | 0.989, 0.989, 0.998, 1.015 |
| 11 | 4.540 | 2.735 | 1.662x | 1.629, 1.649, 1.676, 1.689 |

Times are medians across paired blocks of the three warm-chain measurements
per process. Speedups are medians of paired ratios, not a ratio of pooled
samples. At the last block, about 40% of elapsed update time was saved on
this workload. At block 0 there was no speedup. These are not production
GPT-2, GPU, cross-platform or PyTorch timing claims.

Across eight fresh processes the benchmark timed 192 calls after 64 warm-up
calls; these repeated the same four-step chain at two sites. All four updates
were accepted at each site. Complete receipts, returned geometry IDs and
final training-input logits agreed bit-for-bit, including across old/new
wheels. Canonical semantic SHA-256:
`9b538cd9317799e12bab9925fac90f552d0841436e94ddb169d287609cdbf556`.
The [paired timings](suffix_bench_summary.json) and [runner](suffix_bench.py)
allow the effect to be inspected independently of the training metric.

## Verification

The complete [four-case artificial control](RESULTS.md), including rejection
and infeasibility, reproduced its previous raw JSON byte-for-byte:
`cd05f97a13047a326721b759ee07d14644c230d1cbb971d886b1beb817df53eb`.
Rust regression tests independently compare explicit full re-decoding and
reused-prefix scoring with sparse labels, different head precision/storage,
frozen LoRA, key-lag, and multiple geometry placements. Owner or depth
mismatches fail before scoring. A separate 12-block test exercises both
acceptance and finite rejection at blocks 0 and 11: it counts 24 to 13
forward block calls at block 11, still one backward block replay and two
vocabulary projections. At block 0 the count stays at 24, with 12 replays
and two projections. Even with two negative first-order slopes, the large
step returns the original model in both placements.

303 Rust native LM tests and 70 installed Python facade tests passed. No
pretrained model calls, external evaluation, GPU run or release was needed.
