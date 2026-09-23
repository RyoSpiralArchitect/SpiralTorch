# Statistics-only resident LayerNorm VJP (exploratory)

Implementation and benchmark source: `70a0bff181c0e6c92f067b69cd61f9f86f5f04a4`.
Native measurements used an Apple M4 / Metal integrated GPU. The separate
Chrome 153 WebGPU probe reported Apple Metal and no fallback adapter; that
probe is not an attestation of the Rust browser runtime's device. The new
native `Tensor::layer_norm_affine_backward_resident` is **opt-in**. Neither
`Auto` nor the existing hybrid WGPU training route changed.

The Rust-owned statistics-only tape computes centered values and row moments
without evaluating the forward affine value. Thus an overflowing, unused
forward output does not poison finite requested VJPs. Input, gamma and beta
gradients remain resident until one grouped terminal readback. All eight
gradient masks, empty batches, logical column layout, zero-epsilon constant
rows, upstream guards, and requested overflow were checked against the CPU
contract. No intermediate readback is used; nonempty gradient masks use one
terminal staging map, and the empty mask uses none.

## Native comparison

Six fixed shapes and inputs; CPU centered-f64, existing hybrid WGPU, and the
new statistics-only resident WGPU route each return all three CPU-owned VJPs.
WGPU timings include uploads, compute, and terminal readback. Each of three
independent runs alternated route order, used three warmups and 18 timed
intervals per route and shape. Every output passed the unchanged scaled
`2e-5 * (1 + abs(CPU reference))` tolerance. The largest scaled error was
`0.027889` for hybrid WGPU and `0.001133` for resident WGPU, both below 1.
Cells show each run's median milliseconds in run order, not a pooled median.

| Shape | CPU r1 / r2 / r3 | Hybrid WGPU r1 / r2 / r3 | Resident WGPU r1 / r2 / r3 |
| --- | ---: | ---: | ---: |
| 2x3 | 0.000605 / 0.000750 / 0.000542 | 4.090 / 4.397 / 4.495 | 2.048 / 2.973 / 3.183 |
| 8x257 | 0.014062 / 0.013667 / 0.012479 | 3.536 / 4.005 / 4.404 | 1.477 / 1.978 / 4.439 |
| 32x256 | 0.039375 / 0.040480 / 0.039583 | 3.675 / 4.497 / 4.297 | 1.495 / 1.720 / 3.293 |
| 64x768 | 0.228688 / 0.229375 / 0.244458 | 5.295 / 5.609 / 5.653 | 6.088 / 6.305 / 7.377 |
| 128x1025 | 0.602855 / 0.601250 / 0.601417 | 7.819 / 8.001 / 7.941 | 14.468 / 15.120 / 15.192 |
| 256x256 | 0.299354 / 0.298792 / 0.299125 | 5.937 / 5.976 / 6.000 | 6.176 / 6.359 / 6.432 |

CPU was fastest at every shape. Resident beat hybrid in all three runs at
2x3 and 32x256, in two of three at 8x257, and in none at the three larger
shapes. An earlier unpinned pre-cache exploratory run lost to hybrid on all
six shapes; an unpinned cached run won on five of six. Those runs are retained
below as negative and variability evidence, but neither is substituted for
the source-pinned table. The new route establishes a GPU-resident semantic
building block, **not** a general speedup or a PyTorch advantage. Host-to-host
timings here do not measure a multi-operation resident training graph.

## Verification and originals

- Native real-GPU backend library tests: 207/207 passed. `st-tensor` library
  tests: 506 passed, 1 ignored. LayerNorm autograd and numerics integration
  tests: 12/12 and 4/4 passed.
  Strict Clippy passed for backend all targets and the affected Tensor test
  and benchmark example. Nightly Rust formatting and `git diff --check`
  passed. Vendored WGPU emitted its existing upstream warnings.
- Chrome browser: `spiraltorch.resident_layer_norm.browser.v4` passed seven
  cases with eight masks each, 400 learning steps, four existing guard and
  four grouped-readback checks, and four added statistics-only VJP checks.
  The first/last loss was `1.9879963397979736` / `7.817134717313934e-10`.
- Full local originals remain in
  `/Users/ryospiralarchitect/Library/Logs/SpiralTorch/resident-vjp-20260924/`.
  Source-pinned benchmark JSON SHA-256: r1
  `bbce88fdd6c5b25bef30c2f9d7ebef6cf31ba7d0b7c9821067a587408710e626`,
  r2 `916cd3d1c18605d56dc92ee0fbedf66aefddd0cb21397c3ac25931a3d504c56e`,
  r3 `ee52563ab0aeb9af1df282b1fe19f3e16b63df7496a2ab5fe8492438e7f801e0`.
  Browser report SHA-256:
  `37d9fa27dac5d4cc2fa7acc9beb1141c2022cc6fa6eb16b54dd664f684376b5a`;
  emitted browser WASM SHA-256:
  `b44444234060cd3bfce82fb6493c3ae5d09aa8b716fd4853512c87d1385b446e`.
  Unpinned exploratory pre-cache and cached JSON SHA-256:
  `a988517322f2a6611ebef757ee0c8ecaeb8685fb1a4fd42cbb0de492a9bdefde`
  and `189133bb66ec45772e0cec936ea2233721e3121ae96f5d87fc5a1bd0f6be2b73`.

## Replay

Use a fresh output directory instead of overwriting the originals. Run on a
compatible GPU host with Rust 1.98.0, `wasm32-unknown-unknown`, matching
`wasm-bindgen` 0.2.104, Node with `playwright`, and Chrome:

```sh
OUT=$(mktemp -d)
CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
  cargo +1.98.0 test --locked -p st-backend-wgpu --lib -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
  cargo +1.98.0 test --locked -p st-tensor --no-default-features \
  --features cpu,wgpu_dense --lib -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
  cargo +1.98.0 test --locked -p st-tensor --no-default-features \
  --features cpu,wgpu_dense --test layer_norm_autograd \
  --test layer_norm_numerics -- --test-threads=1
for run in 1 2 3; do
  SPIRALTORCH_STRICT_GPU=1 CARGO_BUILD_JOBS=4 cargo +1.98.0 run \
    --locked --release -p st-tensor --no-default-features \
    --features cpu,wgpu_dense --example layer_norm_vjp_bench \
    > "$OUT/vjp-bench-$run.json"
done
cargo +1.98.0 build --locked --release -p st-backend-wgpu \
  --target wasm32-unknown-unknown --example layer_norm_resident_browser
wasm-bindgen --target web --out-name spiraltorch_wasm \
  --out-dir "$OUT/module" \
  target/wasm32-unknown-unknown/release/examples/layer_norm_resident_browser.wasm
node tools/test_resident_browser.cjs "$OUT/module" "$CHROME" \
  "$OUT/browser-report.json" '' '' '' '' layer-norm-resident
```

The benchmark rejects a CPU adapter, checks all requested gradients against
the CPU oracle, and emits every interval. Independently validate completeness
and numerical bounds before interpreting medians:

```sh
jq -e '.schema == "spiraltorch.layer_norm.vjp_exploratory.v1" and
  .warmup == 3 and .iterations == 18 and (.cases|length) == 6 and
  all(.cases[]; . as $case | (.intervals|length) == 54 and
    all([0,1,2][]; . as $r |
      ([$case.intervals[] | select(.route == $r)]|length) == 18) and
    all(.max_scaled_error[]; . >= 0 and . <= 1))' "$OUT"/vjp-bench-*.json
shasum -a 256 "$OUT"/vjp-bench-*.json "$OUT/browser-report.json"
```
