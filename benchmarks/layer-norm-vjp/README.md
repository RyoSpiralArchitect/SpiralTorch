# Statistics-only resident LayerNorm VJP (exploratory)

Implementation and benchmark source: `6ddb1102cc7ed73c61e8355ab78666685a34c308`.
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
rows, inherited input/gamma guards, upstream guards, and requested overflow
were checked against the CPU contract. No intermediate readback is used;
nonempty gradient masks use one terminal staging map, and the empty mask uses
none.

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
| 2x3 | 0.000625 / 0.000709 / 0.000563 | 3.707 / 4.085 / 4.045 | 1.432 / 2.478 / 2.901 |
| 8x257 | 0.012271 / 0.012500 / 0.011834 | 3.557 / 3.615 / 3.654 | 1.817 / 1.385 / 2.068 |
| 32x256 | 0.039604 / 0.040458 / 0.040000 | 3.698 / 3.828 / 3.697 | 1.498 / 1.514 / 1.487 |
| 64x768 | 0.228334 / 0.242229 / 0.230667 | 5.193 / 5.860 / 5.222 | 5.936 / 6.008 / 6.006 |
| 128x1025 | 0.611917 / 0.637771 / 0.612167 | 7.851 / 8.350 / 7.795 | 14.379 / 15.003 / 14.561 |
| 256x256 | 0.300792 / 0.299313 / 0.299646 | 5.995 / 5.755 / 6.030 | 6.418 / 6.130 / 6.720 |

CPU was fastest at every shape. Resident beat hybrid in all three runs at
the three smaller shapes and in none at the three larger shapes. An earlier
unpinned pre-cache exploratory run lost to hybrid on all six shapes; an
unpinned cached run won on five of six. The first source-pinned run at
`70a0bff181c0e6c92f067b69cd61f9f86f5f04a4` preceded the inherited-guard
fix and is retained separately, not substituted for the final table. These
records expose both performance variability and the corrected semantic gap.
The new route establishes a GPU-resident semantic
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
  four grouped-readback checks, and six added statistics-only VJP checks,
  including inherited input and gamma guards.
  The first/last loss was `1.9879963397979736` / `7.817134717313934e-10`.
- Full local originals remain in
  `/Users/ryospiralarchitect/Library/Logs/SpiralTorch/resident-vjp-20260924/`.
  Final source-pinned benchmark JSON SHA-256: r1
  `6b521ea5b32ead7a7ccd67043588065df3aaa7e5b03f64afc4cda1d7532e199d`,
  r2 `337436d9977777dcff4406c85ee8c380bb93f9da37b8e3e4ed818c6a75580234`,
  r3 `95f8acffcd9f3ca61c9c6df698534ee455bef683d5021510f5cb5501bb65ab3e`.
  Browser report SHA-256:
  `9c0c140ee1e70a9e22d8e125a576eb742407d0cfc67b9602d8fd6d90d1c307a0`;
  emitted browser WASM SHA-256:
  `aaab15873f96102652704bd7c0058435173c605d9b7c8feb7a620637d03eb779`.
  Superseded source-pinned benchmark r1/r2/r3 JSON SHA-256:
  `bbce88fdd6c5b25bef30c2f9d7ebef6cf31ba7d0b7c9821067a587408710e626`,
  `916cd3d1c18605d56dc92ee0fbedf66aefddd0cb21397c3ac25931a3d504c56e`,
  `ee52563ab0aeb9af1df282b1fe19f3e16b63df7496a2ab5fe8492438e7f801e0`.
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
