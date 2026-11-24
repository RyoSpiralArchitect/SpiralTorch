# Known suffixes after learning

The native Rust KV owner can now fork an existing prefix and process all
known suffix queries in one causal pass per layer. It keeps projected prefix
keys/values, extends each head once, and computes vocabulary logits only at
the last suffix position. Original contexts survive failed or successful
forks. Row-local geometry processes the new rows together; prefix-dependent
geometry replays its retained site inputs once for the extended prefix.
This is not a cache for training gradients, teacher-forced sequence scoring,
or parallel generation of unknown tokens.

For teacher-forced likelihoods of multiple terminated alternatives instead,
see [cached full-answer scoring and matched timings](CACHED_LINES.md).

## Numerical checks

The native tests cover every split of a two-head prompt, immutable branches,
invalid IDs, over-budget rectangular attention, numerical failure and later
continuation. A test counter sees one vocabulary projection for a chunk vs.
one per token for serial append. The installed Python context suite also
checks pointwise, elliptic causal and fractional-history recipes at attention,
MLP and block-output sites.

Node and headless Chrome executed the same WASM package. At 24 states of six
nonconstant artificial models (ordinary or key-lag, each bare or with a trained
WaveGate or fractional-history adapter), chunk logits were identical to the
same engine's full-prefix and serial logits. Maximum absolute difference vs.
the previous native Python build: 5.96046448e-8. Regenerating the entire oracle
with the new wheel gave identical bytes. The existing finite-step controls
also verify cached inference on the actual returned geometry, not the rejected
candidate. This is numerical compatibility, not evidence of language quality.

## CPU timings

Apple M4, macOS 26.4.1, CPython 3.8.20, native CPU release wheel. One synthetic
4-layer decoder, width 48, 4 heads, vocabulary 512, row-local WaveGate, fixed
24-token prefix. Both arms start with a fresh prefilled context outside the
timer, materialize the final vocabulary row, and compare it against the
full-prefix result. One warm-up per arm per length, eight alternating-order
pairs. Timings include the Python calls, not prefill or case construction.

| Known suffix | Serial median, ms | Chunk median, ms | Median paired serial/chunk |
| ---: | ---: | ---: | ---: |
| 1 | 0.050 | 0.046 | 1.075x |
| 8 | 0.385 | 0.179 | 2.137x |
| 32 | 1.653 | 0.727 | 2.291x |

All measured final rows matched the full-prefix reference exactly. For one
token the ratios straddled 1 (0.93-1.46); no consistent single-token gain.
The longer chunks save repeated cache copies, head projections and Python
calls. The measurement does not isolate those costs or compare PyTorch,
WASM speed, GPU or pretrained-model accuracy.
[Runner](cache_suffix_bench.py) and [all paired timings](cache_suffix_timings.json).
