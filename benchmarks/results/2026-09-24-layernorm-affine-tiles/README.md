# Resident LayerNorm affine tiling, 2026-09-24

The backward affine reduction now groups eight adjacent columns per workgroup
for 32..=128 rows and at least 256 columns. Other shapes keep the previous
schedule. The same Rust-owned dispatch is used by standalone resident tensors,
prepared training/autograd graphs, Python bindings, and browser WASM. This is a
shape-limited throughput result on Apple M4, not a PyTorch comparison or a
claim about all GPU adapters.

Baseline: `4c537872c0fce5dd2778a8bd54b1877c93271de5` (merged PR #2126).
Candidate code: `c10be57cb0fcc8e2f4513781eeef02880d9442b2`. The benchmark
source was unchanged in both builds (SHA-256
`725d0482ebbbc3d424f8a351a78d7ff8cf391c62223ccb5d695fe3a45006c99f`).

## Matched results

Native release benchmark: 32 same-batch affine LayerNorm + MSE + all VJPs +
SGD steps, preloaded input/target/parameters, preparation excluded, terminal
readback included. Each executable performs 2 warmups, 5 timed repetitions,
and alternates standalone-resident and prepared-graph route order. Executables
were run in baseline, candidate, candidate, baseline order. Values below are
prepared-graph medians in milliseconds; standalone medians and all intervals
are in [`results.json`](results.json).

| Shape | First baseline -> candidate | Second baseline -> candidate |
| --- | ---: | ---: |
| 2 x 3, unchanged route | 22.570 -> 23.228 | 29.705 -> 29.341 |
| 32 x 256 | 52.897 -> 42.423 | 61.147 -> 50.458 |
| 128 x 1025 | 493.315 -> 468.199 | 552.341 -> 529.929 |

The absolute latencies rose substantially in the second pair, including the
unchanged small-shape route. The relative improvement on the two selected
shapes persisted, but the data are too narrow for a universal speedup claim.
Within each native executable the standalone and graph routes had
`max_scaled_error = 0`; that check does **not** compare the old and new
executables directly.

Chrome 153 WebGPU ran the old and new WASM modules in the **same** page,
alternating route order with 2 warmups and 5 repetitions. Each trial releases
its graph/state after a terminal guarded read. Browser wall-time medians in ms:

| Shape | Browser run 1, old -> new | Browser run 2, old -> new |
| --- | ---: | ---: |
| 32 x 256 | 57.9 -> 42.4 | 56.6 -> 41.5 |
| 128 x 1025 | 477.6 -> 455.4 | 481.5 -> 451.7 |

Both browser runs compared prediction, input gradient, affine gradients,
parameters, and loss with `max_scaled_error = 0`. The JS adapter probe reported
Apple and `isFallbackAdapter = false`; it is not independent proof of the Rust
runtime's device identity. Browser timings include host encoding/submission
and readback, not GPU kernel time alone. Other browsers/adapters are unmeasured.

## Validation and provenance

The Rust backend's 216 unit tests and two integration tests passed on the
actual Metal adapter. Boundary and cancellation tests compare affine gradients
against an independent centered-f64 reference, including widths 256, 257,
and 1025. The 32 x 257 browser fixture passed forward, exact and
module-compatible training, affine update values, and the zero-epsilon guard.
Workspace `cargo check`, scoped strict backend Clippy, WASM `webgpu` release
build, Rust formatting, and browser fixture passed. All raw original reports
remain local under `~/Library/Logs/SpiralTorch/layernorm-affine-tiles-v1/`.

| Local original | SHA-256 |
| --- | --- |
| `baseline-final-1.json` | `b256d7ebe107212a5bf8a2fc8b33f7250680b3a032e38a09facc8e5e8939a219` |
| `candidate-final-1.json` | `0c45f5d4d90dec32659d6b2f69af27ca6a4d6b79643f78018b6ddee6877402e7` |
| `candidate-final-2.json` | `5feb26212f8c80d4ff284a35ea132a6627b977709272f544be86e8dc33bb5b3b` |
| `baseline-final-2.json` | `ea6f62b135299ceaa5e78a7a2138df161a07ea66a56ffd6f1a71b620e97c01d7` |
| `browser-final.json` | `a946fff32b285c08b63b5d71657f6bc9d4c061d03f1b0d99142e68aaa2c44b9b` |
| `browser-final-matched-1.json` | `022bf798efe42b0a5fbcbb27b7ef485ff127f573efbe5a9814a18eaf1473cbc2` |
| `browser-final-matched-2.json` | `baeba087ae7836de6fd461740d0d8ab81304e705ef424933d7846e67280f9b5d` |

The native baseline/candidate executables were SHA-256
`ee8e5ada70cfb08c91c9ea2097908bdb333fe81994a97d369f2fc6d170e59e08`
and `44dfd229e60670ec4d7959498aafd4228808307f17e1db032366dbfb778a9ae2`.
The browser baseline/candidate WASM payloads were
`1cba653d69d0e1f95e1bd3fc3b08a2104ef565d945355adfd307d817c7e5b3a1`
and `343e596a2d171106b349a47ef88f85f0cb65f7127f2669334f34ceb144be0432`.
The final matched browser page SHA-256 is
`85b6e7d530a4204e4dd03fff50698766c7e3b90b627d1862c28250fed4c44f38`.

Rebuild the two native revisions and run the unchanged benchmark with:

```bash
cargo run --release -p st-backend-wgpu --example resident_graph_layer_norm_bench
```

Build each revision's `spiraltorch-wasm` for `wasm32-unknown-unknown` with
`--features webgpu`, bind each artifact with `wasm-bindgen 0.2.104 --target web`,
then compare them in Chrome using:

```bash
NODE_PATH=/path/to/playwright/node_modules node tools/test_resident_browser.cjs \
  /path/to/candidate-module /path/to/Chrome /path/to/new-report.json \
  '' '' '' '' nn-graph-layer-norm-matched /path/to/baseline-module
```
