# Independently Learned Fractional History Gain

Implementation validation for source revision
`12d7ed80c6fae5720fd3a2fe919464a19f353c4b`, not a new pretrained-model study.
The [operator contract](../../../docs/fractional_learning.md#independently-learned-gain)
separates normalized history shape from positive learned amplitude:

```text
q(alpha, log_gain) = exp(log_gain) * c(alpha) / ||c(alpha)||_2
```

Rust owns the map, input/order/log-gain VJPs and joint JVP. Python and WASM
expose the same owned snapshot. Existing raw and fixed-energy history
operators and checkpoint schemas are unchanged. The new adapter has
`2*F+2` parameters; feature gates and gain still provide redundant amplitude
controls. This is not a claim of identifiability or orthogonality.

## Verification

- 135 Rust tests passed, including five new learned-gain tests.
- 586 Python regression tests passed, including 29 learned-gain tests.
- 34 benchmark correctness tests passed; no hardware timing comparison ran.
- Seven compiled WASM fixtures passed, including the new joint learning loop.
- TypeScript checking, native/WASM `st-frac` strict Clippy, default native
  release build, explicit CPU/text binding check and formatting checks passed.
- Binding Clippy passed with the existing `too_many_arguments` and
  `type_complexity` categories allowed; it is not a fully strict binding lint.
- All 71 candidate runtime files matched source/build artifacts. The 352
  recorded files of the preceding two studies and their clients/runtimes
  remained unchanged. They were not rerun or reinterpreted as new trials.

Python checks include independent ordinary Torch reference math, all seven
requested-gradient combinations over both transports, forward AD, ownership,
domain/overflow guards, and a frozen randomly initialized tiny HF model.
The latter makes 12 updates and verifies exact next-update/Adam continuation;
it is not pretrained-model quality evidence. There were 18 Torch JIT-related
warnings in the regression run, with no skipped or failed tests.

## Compiled WASM Learning

The synthetic fixture fits only order and gain through Rust scalar VJPs;
JavaScript owns SGD and the log-alpha chart, not the history operator.

| Measurement | Observed |
| --- | ---: |
| Updates | 600 |
| Initial MSE | 0.5310602356173215 |
| Final MSE | 0 |
| Final alpha (optimizer coordinate) | 0.6500000040213405 |
| Final effective gain | 1.5 |
| Nonzero order-gradient updates | 338 |
| Nonzero gain-gradient updates | 338 |

Zero MSE means float32 output agreement on this synthetic fixture. It does
not establish unique parameter recovery, generalization, browser/GPU
execution, pretrained quality or a speed improvement. The actual WASM alpha
input is rounded to float32 from the reported optimizer coordinate.

## Reproduce

From the repository root, with Rust 1.98.0, a Python environment containing
the built current SpiralTorch extension, Torch and Transformers, and Node:

```sh
cargo +1.98.0 test --locked -p st-frac
python -P -B -m pytest --import-mode=importlib -q \
  bindings/st-py/tests/test_fractional_history_log_gain.py
cargo +1.98.0 build --locked -p spiraltorch-wasm \
  --target wasm32-unknown-unknown --release
# Use wasm-bindgen-cli 0.2.104, matching Cargo.lock.
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm \
  --target nodejs --out-dir /tmp/spiraltorch-learned-gain-wasm
node bindings/st-wasm/tests/fractional_history_log_gain.mjs \
  /tmp/spiraltorch-learned-gain-wasm/spiraltorch_wasm.js
```

The full regression file list, build/check commands, versions and hashes are
in [validation.json](validation.json). [wasm-learning.json](wasm-learning.json)
is the emitted numeric fixture result. `SHA256SUMS` covers these three files.
Native runtimes, model assets, checkpoints and raw local logs are not published.

Before a pretrained learned-gain experiment, add scalar gain trajectories to
the shared study driver and freeze a capacity-matched ordinary short-filter
control, common initial maps, data, update budget and seeds. No such long
training run was launched for this implementation validation.
