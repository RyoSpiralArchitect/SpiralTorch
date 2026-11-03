# Native SGD: reusing a frozen decoder prefix across epochs

**On this synthetic CPU workload, the median paired speed ratio for four
epochs was 1.274x (about 21.5% less elapsed time).** The cache was empty at
the start of every timed fit. Filling it in the first epoch did not speed up
that epoch; the benefit came from the repeated windows.

| Measurement | Result |
| --- | ---: |
| Median plain / cached, total run (six pairs) | 1.274x |
| Pair range, total run | 1.262-1.294x |
| Median plain / cached, initial fill epoch | 0.996x |
| Median plain / cached, epochs 2-4 | 1.398x |
| Median elapsed, plain / cached | 104.04 / 81.61 ms |

Six synthetic GPT-2-style blocks, width 64, four heads, vocabulary 97.
Paired cohorts: ordinary affine, saturation only, projection only and both,
each with its own joint LoRA/geometry SGD updates. Twelve synthetic windows,
length 25; batch 2, four epochs, last-block adapters, single CPU thread settings.
Both methods already share five frozen blocks **within** each matched update.
The new method additionally retained six batch boundaries, 18,432 f32 values
(72 KiB). It hit 18 times and avoided 90 prefix block calls beyond ordinary
within-step sharing. No suffix, target, gradient or optimizer update was reused.

One excluded warmup pair then six alternating paired runs, with identical step
report digests and four final checkpoint IDs in all 14 executions. Timings cover
the update loop only, not model construction, scoring, serialization, disk I/O,
or pretraining. Pairwise ratios need not equal the ratio of medians. This is not
a comparison to PyTorch, a GPU measurement, a quality result, or a timing for
the real GPT-2 field-extraction trial. That study remains pinned to its own wheel.

For scale only: the separately [prepared 120-record corpus](../2026-11-03-native-factorial-answer-training/PREPARED.md)
has 2,900 causal input rows after removing each last next-token target. At
width 768 and batch 1, keeping its 120 distinct inputs at one depth would
require 2,227,200 retained f32 values (8.496 MiB of payload plus metadata).
This is an input-size calculation, not a pretrained-model execution or an
amendment to the frozen study.

## Reproduce

The [benchmark](../../../tools/bench_native_training_prefix_cache.py) pins its
source, fixture, arm builder, installed wheel and native extension. It does not
read a pretrained checkpoint, a real dataset or held-out scores. Synthetic
fixtures are generated once; every timed fit starts from the same fresh models.

```sh
"$VENV/bin/python" -I -B tools/bench_native_training_prefix_cache.py --wheel "$CACHE_WHEEL" --output "$NEW_DIR"
python3 -I -B -S benchmarks/results/2026-11-03-native-training-prefix-cache/verify.py "$NEW_DIR"
```

[Protocol](receipts/protocol.json), [all timings](receipts/runs.jsonl) and
[summary and hashes](receipts/result.json). The [file-only verifier](verify.py)
recomputes every ratio and checks the shared trace/owner identities and budgets.
Native tests also count actual QKV operations on hits, vary the label mask,
prompt, batch layout and adapter depth, and inject a failing final arm to prove
that neither any cursor nor the cache is committed on failure.
