# Guarded resident LayerNorm input VJP, 2026-09-24

Baseline: `ce21cccb7de207234f56c210f17a2a609fa6145b` (merged PR #2127).
Candidate code: `28ddea8d9149049a7f5e05e54b596510d8abc889`.

The input VJP can combine the row-constant `square_sum + epsilon_sum`
before its difference of products, avoiding one extended-range product and
addition per element. That rearrangement is **not** safe when epsilon is
distant from the row sum: an unguarded prototype returned zero where the
scale-direction derivative was `-0.0008760057`. The shipped candidate uses
the combined form only when epsilon is within 20 binary exponent steps of
the row square sum (or the sum is zero); otherwise it retains the original
split expression. This is a tested guard, not a proof for every possible
input. Ordinary CPU Tensor and non-LayerNorm paths are unchanged.

## Matched results

Apple M4 / Metal native release, 32 same-batch affine LayerNorm + MSE + all
VJPs + SGD steps. Input/target/parameters were preloaded, graph preparation
was outside the interval, and the terminal read was inside. Each executable
used 2 warmups, 5 timed repetitions, and alternating standalone/graph order.
The executables ran baseline, candidate, candidate, baseline. Prepared-graph
medians in milliseconds:

| Shape | Pair 1, old -> new | Pair 2, old -> new |
| --- | ---: | ---: |
| 2 x 3 | 23.179 -> 24.896 | 31.367 -> 24.723 |
| 32 x 256 | 43.576 -> 41.188 | 44.577 -> 41.276 |
| 128 x 1025 | 487.971 -> 456.131 | 498.316 -> 470.149 |

The small shape has substantial absolute-time drift and is not evidence of a
small-shape speedup. Within each executable, graph and standalone results
had `max_scaled_error = 0`; this is not a direct old/new comparison.

The new `resident_layer_norm_backward_masks` diagnostic repeats backward on
one prepared tape, with the same terminal readback for input-only,
affine-only, and both. At 128 x 1025, the old/new input-only medians were
`222.257 -> 190.206` and `221.884 -> 182.775` ms; affine-only was
`65.638 -> 64.234` and `64.004 -> 65.736` ms. This identifies the input
VJP as the source of the gain, but the intervals include host encode/submit
and are not GPU-only kernel timings.

Chrome 153 ran old/new WASM modules on the same WebGPU page, alternating route
order across 2 warmups and 5 repetitions. Browser wall-time medians in ms:

| Run | 32 x 256, old -> new | 128 x 1025, old -> new |
| --- | ---: | ---: |
| 1 | 39.9 -> 37.9 | 443.4 -> 407.8 |
| 2 | 39.8 -> 40.0 | 449.3 -> 412.4 |
| 3 | 39.4 -> 37.7 | 442.1 -> 408.6 |
| 4 | 39.9 -> 37.9 | 457.2 -> 413.0 |

All four browser runs had `max_scaled_error = 0` for prediction, input and
affine gradients, parameters, and loss. The independent 32 x 257 browser
correctness fixture also passed. The JS adapter probe reported Apple and
`isFallbackAdapter = false`; it is not independent proof of Rust runtime
device identity. Other adapters and browsers are unmeasured. Earlier
exploratory browser runs were noisier: one 128 x 1025 run regressed from
`608.6` to `625.9` ms. It and every other recorded run are retained in
[`results.json`](results.json), rather than being silently discarded.

## Correctness and rejected probes

- Actual Metal: 218 backend unit tests and 2 integration tests passed. Two new
  scale-direction tests cover the smallest nonzero f32 epsilon against an
  analytic derivative and sweep epsilon powers from `2^-30` through `1`
  across the fast/fallback boundary. Existing independent centered-f64,
  cancellation, shape-boundary, and requested-VJP tests passed.
- `st-nn --features wgpu` passed 774 unit tests and its integration suites.
  Workspace `cargo check`, scoped strict backend Clippy, nightly rustfmt, and
  WASM `webgpu` release build passed.
- A lane-local `Wide` cache was rejected: the 32 x 256 graph median worsened
  from about 49 to 57 ms, while 128 x 1025 improved only marginally.
- The unguarded combined expression was rejected even though the earlier
  21 LayerNorm tests passed. The new subnormal-epsilon test failed with
  `actual = 0`, `expected = -0.0008760057`; reverting to the split expression
  passed. The final guarded expression passes this and the exponent sweep.

`results.json` contains all 46 recorded native/browser comparison reports in
curated form, including every timed interval, local-original SHA-256, and
negative or exploratory runs. Raw original JSON remains local under
`~/Library/Logs/SpiralTorch/layernorm-backward-input-cache-v1/`.

Native old/new graph executables SHA-256:
`e699e330b6d39e125f9b2da0eedaa220dee73a11231265f8455542ebd712322e`
and `f69c2a9567432d1a0d1b7d9eafb5650a1ff5b7f45557bb5cf7d5b74934e8ca04`.
Native old/new mask executables SHA-256:
`bd30f29b061f3d774c47bfe09651c5874c30f5b26c8b739a3fd09f15543a4b40`
and `e1d39a9945d9ecd9ddb083b070bcd808f9dca433d177a15fd32d63ef8fc3b355`.
Old/new browser WASM SHA-256:
`343e596a2d171106b349a47ef88f85f0cb65f7127f2669334f34ceb144be0432`
and `42fa4b066b158fba4b8d70ba5fa4de13155fa482c92bcbba1da6edca434976b2`.
The unchanged graph benchmark source SHA-256 is
`725d0482ebbbc3d424f8a351a78d7ff8cf391c62223ccb5d695fe3a45006c99f`.
The new mask benchmark source SHA-256 is
`945812c2241a652136057afa717e2f8e65dc3556dff1a4bee60459cf60e4aa31`.

## Replay

Run on a non-CPU WGPU adapter:

```bash
cargo run --release -p st-backend-wgpu --example resident_graph_layer_norm_bench
cargo run --release -p st-backend-wgpu --example resident_layer_norm_backward_masks
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test -p st-backend-wgpu layer_norm
```

To rebuild the baseline mask executable, use the diagnostic example from the
candidate commit with the baseline library source; the example itself does
not change the backend. For the browser comparison, build each revision's
`spiraltorch-wasm` for `wasm32-unknown-unknown` with `--features webgpu`, bind
each with `wasm-bindgen 0.2.104 --target web`, then run:

```bash
NODE_PATH=/path/to/playwright/node_modules node tools/test_resident_browser.cjs \
  /path/to/candidate-module /path/to/Chrome /path/to/report.json \
  '' '' '' '' nn-graph-layer-norm-matched /path/to/baseline-module
```

This is a shape- and adapter-bounded throughput observation, not a PyTorch
comparison or a universal performance claim.
