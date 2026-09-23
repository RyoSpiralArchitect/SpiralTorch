# Resident LayerNorm grouped readback (exploratory)

Source commit: `6ac4837c42ddf124b9c84ab6b35ce1b90567a5db`.
Apple M4 / Metal native, with a separate Chrome 153 browser-WebGPU check.
The browser's non-fallback Apple Metal probe is separate from, not an
attestation of, the Rust runtime's browser adapter. No default backend route
changes here.

The fixed six shapes and inputs match the earlier LayerNorm benchmark. Each
route includes four CPU-owned outputs (forward, dx, dgamma, dbeta), uses three
warmups and 18 timed intervals, and alternates route order. The first scope
reads already computed resident tensors; the second includes uploads, forward,
all VJPs, and readback. Both routes are checked against the CPU-centered
reference with the unchanged scaled `2e-5 * (1 + abs(reference))` tolerance.
All 12 shape/scope/route comparisons passed; the maximum scaled error was
`0.001133` (128x1025). Times below are medians in milliseconds from the
source-pinned run on a shared machine, not speed guarantees.

| Shape | Terminal serial | Terminal grouped | Full serial | Full grouped |
| --- | ---: | ---: | ---: | ---: |
| 2x3 | 0.982 | 0.185 | 3.786 | 3.254 |
| 8x257 | 0.685 | 0.187 | 3.713 | 3.018 |
| 32x256 | 0.673 | 0.185 | 3.455 | 2.964 |
| 64x768 | 0.727 | 0.244 | 7.493 | 7.032 |
| 128x1025 | 0.833 | 0.375 | 16.132 | 15.661 |
| 256x256 | 0.742 | 0.254 | 7.590 | 6.976 |

This isolates serial snapshots from one `snapshot_many` submission and
map/spill policy. It does not establish a PyTorch advantage. The earlier
PyTorch CPU/MPS comparison remains a separate, negative overall-performance
control; this table must not be spliced into that historical route's timings.
The pre-commit exploratory run also favored grouped readback at every shape,
but its source was not pinned, so its medians are not the table above.

## Verification and originals

- Native Rust backend: 206/206 library tests passed with real GPU tests enabled;
  the existing ordered readback-batch integration test passed. Strict Clippy
  passed for the backend and the new benchmark example. Formatting passed.
- Chrome browser: `spiraltorch.resident_layer_norm.browser.v3` passed the
  400-step learning check and four added batched-snapshot checks (offset view,
  transposed view, empty view, inherited guard). First/last loss:
  `1.9879963397979736` / `7.817134717313934e-10`.
- The full local originals are in
  `/Users/ryospiralarchitect/Library/Logs/SpiralTorch/resident-readback-20260923/`.
  The source-pinned benchmark JSON is `readback-bench-final.json` (SHA-256
  `03ee59e347a1c8f5d44253c4a83be7266d44c5caffec4347cd90cc29e1b20df3`);
  the browser report is `browser-report-final.json` (SHA-256
  `c66c3ee4b3a02bf8cde5ad473f4b26ae9e6877426f0ce0a7d161f6bf60c834c3`).
  The emitted browser WASM is SHA-256
  `85110b56510f873dd310fb0b8e248070180cd41215946532d79be9c1e63026bf`.
  The pre-commit exploratory JSON is retained as `readback-bench.json`
  (SHA-256 `3666028a857597afc4626b84c7716f3f4402103c116f07798c3b6e23264c0a7a`).

## Replay

Use a fresh output directory; do not overwrite the originals above. On a
compatible GPU host, run:

```sh
OUT=$(mktemp -d)
CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
  cargo +1.98.0 test --locked -p st-backend-wgpu --lib -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
  cargo +1.98.0 test --locked -p st-backend-wgpu --test readback_batch
SPIRALTORCH_STRICT_GPU=1 CARGO_BUILD_JOBS=4 \
  cargo +1.98.0 run --locked --release -p st-tensor \
  --no-default-features --features cpu,wgpu_dense \
  --example layer_norm_readback_bench > "$OUT/readback-bench.json"
cargo +1.98.0 build --locked --release -p st-backend-wgpu \
  --target wasm32-unknown-unknown --example layer_norm_resident_browser
wasm-bindgen --target web --out-name spiraltorch_wasm --out-dir "$OUT/module" \
  target/wasm32-unknown-unknown/release/examples/layer_norm_resident_browser.wasm
node tools/test_resident_browser.cjs "$OUT/module" "$CHROME" \
  "$OUT/browser-report.json" '' '' '' '' layer-norm-resident
```

Here `OUT` is a new directory, `CHROME` points to a Chrome executable,
`wasm-bindgen` is version 0.2.104, and the browser harness needs `playwright`
available to Node. The benchmark JSON validator checks six shapes, two scopes,
18 intervals per route and scaled error at most one; the browser fixture
rejects an incomplete report.

```sh
jq -e '.schema == "spiraltorch.layer_norm.readback_exploratory.v1" and
  .warmup == 3 and .iterations == 18 and (.cases|length) == 6 and
  all(.cases[]; (.scopes|length) == 2 and all(.scopes[];
    (.intervals|length) == 36 and
    ([.intervals[]|select(.route == 0)]|length) == 18 and
    ([.intervals[]|select(.route == 1)]|length) == 18 and
    all(.max_scaled_error[]; . >= 0 and . <= 1)))' "$OUT/readback-bench.json"
shasum -a 256 "$OUT/readback-bench.json" "$OUT/browser-report.json"
```
