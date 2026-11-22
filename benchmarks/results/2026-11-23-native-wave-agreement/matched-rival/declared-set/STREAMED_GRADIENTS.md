# Full-answer geometry gradients without retaining two full vocabulary matrices

For a large terminated set, ranked menu or accepted subset, the Rust decoder
now checkpoints its blocks once, normalizes the selected hidden rows once,
and reduces each vocabulary tile to the same full-support partitions used
by the dense reference. It retains the first tile, reprojects the others
after the objective is known, then writes the exact vocabulary cotangent
back over each tile using one row of scratch. Only hidden-width derivatives
are collected for decoder backward. No new decoder forward or optimizer step
is needed for this trade. The cold/warm head layout is chosen from the
original logical row count and held fixed during replay.

`max_tile_values` limits *one projected tile*, in f32 elements, not bytes,
RSS or total arithmetic. It must fit at least one vocabulary row and cannot
increase the original total-logit, activation or training-checkpoint limits.
The default targets 262,144 values (at least one row, within model limits).
When all rows fit, the existing single-pass dense implementation remains in
use. Otherwise, with N prefixes and T rows in the first tile, replay projects
N-T additional rows. Weights, checkpoints, sparse edge statistics and matrix
kernel scratch are not included in the logical work counters.

## What changed in a real CPU process

Previous dense wheel (source `9421cc6a`) vs automatic (5 rows) vs explicit
one-row tile. Artificial WaveGate, width 8, one layer, 2 heads, 50,257-token
vocabulary; the same eight-token prompt and four 32-token contents as the
[inference experiment](STREAMED_LINES.md). The model is not pretrained.
All 129 prefix rows and all vocabulary classes contribute. Twenty-seven fresh
processes, three cyclic orders per objective; one gradient call per process,
no weight updates. Per-arm medians on release CPython 3.8/macOS arm64:

| Objective | Peak RSS increase: dense / auto / one row (MiB) | Whole process peak: dense / auto / one row (MiB) | Time: dense / auto / one row (ms) |
| --- | ---: | ---: | ---: |
| Set NLL | 80.59 / 8.20 / 7.61 | 127.47 / 54.95 / 54.41 | 54.95 / 59.70 / 62.18 |
| Mixed gold/menu NLL | 80.67 / 8.23 / 7.69 | 127.45 / 55.03 / 54.42 | 55.24 / 60.34 / 63.42 |
| Accepted-subset NLL | 80.77 / 8.34 / 7.77 | 127.56 / 55.08 / 54.55 | 128.87 / 135.47 / 138.94 |

The RSS increase subtracts process high-water marks before/after the call.
It is not instantaneous RSS or a general training-memory guarantee. The larger
tile was faster than the one-row tile for all three medians; the old dense
route was fastest here. Each old/new loss report and every gate/bias derivative
agreed exactly in the Python records. [All process observations and raw file
hashes](stream_line_vjp_measurements.json) are available; timing is descriptive.
For the default, 51 forward head calls (124 reprojected rows) and 26 backward
head calls used 251,285 projected values plus 50,257 extra seed values. The
dense baseline needed 6,483,153 logical values in each matrix.

## Reproduce

With isolated, locally installed reference and candidate wheels, create an
unused synthetic fixture or reuse one whose pinned hash matches the driver:

```sh
"$OLD_PY" -IB stream_line_bench.py prepare "$NEW_FIXTURE"
python3 -ISB stream_line_vjp_bench.py compare "$OLD_PY" "$NEW_PY" "$NEW_FIXTURE" "$NEW_OUT"
python3 -ISB stream_line_vjp_bench.py verify "$NEW_OUT"
```

Separate small fixtures vary ordinary/key-lag attention and WaveGate/history
geometry. For all 12 loss/host cases the new one-row Python JSON was byte
identical to the dense wheel's JSON (84 nonzero parameters in total).
Node and headless Chrome compared both tile policies to these gradients and
tested snapshot-checked SGD; maximum absolute gradient difference from native
was below 1.11e-8. See [the WASM/Python example](../../../../../bindings/st-wasm/tests/native_causal_lm.md#train-on-complete-answers).
This is evidence about numerical agreement, work and memory, not model quality.
