# Prepared resident graph LayerNorm, 2026-09-24

Affine LayerNorm now lowers from a Rust `Module` into a prepared GPU-resident
inference/training graph. The portable graph plan uses v3 only when it contains
`layer_norm`; existing v2 graphs remain v2. The same v3 plan runs in the WASM
browser client. This is a connectivity and correctness milestone, not evidence
of a large training speedup.

## Matched native measurement

Apple M4 / Metal, release build, 32 affine LayerNorm + MSE + full VJP + SGD
steps per sample. Both routes start with preloaded input/target/parameters;
the standalone route uses the existing fused pointwise SGD update. Graph
preparation and initial upload are outside the timed window. The terminal
readback is inside it. Each run uses 2 warmups, 5 measured repetitions, and
alternates route order. Graph state readback includes additional diagnostic
fields, so terminal payloads are not byte-for-byte identical. Every compared
prediction, loss, input gradient, affine gradient, and final parameter was
finite and matched exactly in these runs (`max_scaled_error = 0` at tolerance
`5e-4 * (1 + abs(reference))`).

| Shape | Run | Standalone median, ms | Graph median, ms | Graph/standalone |
| --- | --- | ---: | ---: | ---: |
| 2 x 3 | first | 24.381 | 22.693 | 0.931 |
| 2 x 3 | final | 22.692 | 21.302 | 0.939 |
| 32 x 256 | first | 51.366 | 49.101 | 0.956 |
| 32 x 256 | final | 52.724 | 51.197 | 0.971 |
| 128 x 1025 | first | 662.750 | 659.425 | 0.995 |
| 128 x 1025 | final | 483.691 | 480.777 | 0.994 |

The 128 x 1025 absolute latency moved markedly between runs for *both*
routes. Treat its sub-1% median delta as parity, not a speedup. The smaller
cases show a modest host-end-to-end advantage only under this workload.
Neither run is a PyTorch comparison or a GPU-kernel-only measurement.

All measured intervals, losses, and exactness checks are in
[`results.json`](results.json). The unchanged source for both runs was
`crates/st-backend-wgpu/examples/resident_graph_layer_norm_bench.rs`, SHA-256
`725d0482ebbbc3d424f8a351a78d7ff8cf391c62223ccb5d695fe3a45006c99f`.
The complete local raw JSON files were retained under
`~/Library/Logs/SpiralTorch/resident-graph-layernorm-v1/`; their SHA-256 values
are in `results.json`.

## Browser and validation

Chrome 153 loaded the v3 plan and passed
forward values, `exact` and `module_compatible` affine gradients/updates, and
the zero-epsilon stage guard. Its JS adapter probe reported Apple and
`isFallbackAdapter = false`; this probe is not an independent device identity
for the Rust WGPU runtime. The browser page SHA-256 is
`5c4bfb98fad499c7b1061d67c6f00ad8ec496d23473a16baa1fa294fbe853866`;
the built WASM SHA-256 is
`582877d00e00aeabda65d4e05eeebda9a992dd34f8692ff4bb21d3b5de40419f`.
This confirms browser execution on this adapter, not parity across browsers.

Rust checks: `st-kernel-contracts` 32 tests, `st-backend-wgpu` 213 tests,
`st-nn --features wgpu` 774 unit tests plus integration suites; the focused
single/staged/stacked LayerNorm tests also passed on actual Metal. Native
Python binding `cargo check`, WASM `webgpu` build, and scoped strict Clippy for
backend/contracts passed. An isolated Python install passed all 10 graph
training tests on Metal, including v3 LayerNorm forward/training. Workspace
`cargo check` passed. Full `st-nn` strict Clippy remains blocked by 22
pre-existing warnings unrelated to this change.

Reproduce the native comparison with:

```bash
cargo run --release -p st-backend-wgpu --example resident_graph_layer_norm_bench
```

Reproduce the browser fixture after building `spiraltorch-wasm` for
`wasm32-unknown-unknown` with `--features webgpu`, then binding that artifact
with `wasm-bindgen 0.2.104 --target web`:

```bash
NODE_PATH=/path/to/playwright/node_modules node tools/test_resident_browser.cjs \
  /path/to/generated/module /path/to/Chrome /path/to/new-report.json \
  '' '' '' '' nn-graph-layer-norm
```
