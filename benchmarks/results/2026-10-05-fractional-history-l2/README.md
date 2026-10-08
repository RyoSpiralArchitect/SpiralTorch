# Fixed-Energy Fractional History: Execution Validation

This validates a new Rust-owned GL history operator, not a language-model
quality result or speed benchmark. The full declared strictly-past coefficient
vector is scaled to a constant L2 norm. Rust differentiates that normalization;
Python and WASM only transport inputs, snapshots and derivatives.

The common positive alpha and sample-spacing factors cancel analytically.
This avoids unstable subtraction near alpha=0 without clipping orders or
renormalizing gradients. Existing raw/full GL semantics and checkpoint schemas
are unchanged. See [the contract](../../../docs/fractional_learning.md#fixed-energy-history).

## Observed Checks

- Rust: 130 tests passed; strict st-frac clippy passed on native and wasm32.
- Python: 504 geometry/study regression tests and 34 matched-math benchmark
  tests passed, with no skips. These benchmark tests check correctness and
  published receipts; they do not run new hardware timing measurements.
- Compiled WASM: six Node fixtures passed, including the new normalized
  history loop, raw/full GL, selective VJP, tiled forwarding and numeric
  boundaries. TypeScript declarations passed the strict no-emit check.
- A tiny, randomly initialized HF causal LM trains local/history gates and
  fractional order with a frozen base. Saving and restoring the adapter and
  Adam reproduces the next update exactly for both raw and normalized history.
- Default native release and explicit CPU/text-only builds passed. The final
  native artifact is byte-identical to the artifact initially tested.

The compiled-WASM synthetic task ran 500 SGD updates: MSE went from
0.2195702135724794 to 0.0000676152559847477, with 499 nonzero order-gradient
steps. Final order was 0.7231869829086626, not the generating order 0.65.
This shows the learning connection, **not unique order recovery**, pretrained
quality, browser/WebGPU execution, speed, or superiority over raw history.

Strict binding clippy with `--no-deps -D warnings` still reports nine existing
argument-count/type-complexity findings in unchanged binding files. A scoped
run allowing those two existing lint categories passed. An earlier run that
also linted dependencies stopped on existing st-nn findings. No repository-wide
lint rule was relaxed; the two new native methods have narrow argument-count
allows to keep Rust/Python/WASM controls aligned.

## Reproduce

Use Rust 1.98.0, matching wasm-bindgen 0.2.104, and an isolated release Python
package built from this source. The recorded Python runtime is 3.12.6 with
Torch 2.12.1 and Transformers 4.57.6, CPU float32. No model download is needed.

```sh
cargo +1.98.0 test --locked -p st-frac
cargo +1.98.0 clippy --locked -p st-frac --all-targets -- -D warnings
PYTHONPATH="$PACKAGE_ROOT" python -P -B -m pytest --import-mode=importlib -q \
  bindings/st-py/tests/test_fractional_history_l2.py \
  bindings/st-py/tests/test_fractional_history.py
PYTHONPATH="$PACKAGE_ROOT" python -P -B -m pytest --import-mode=importlib \
  --confcutdir=tools --rootdir=tools -q tools/test_benchmark_fractional_learning.py
cargo +1.98.0 build --locked -p spiraltorch-wasm --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm \
  --target nodejs --out-dir "$WASM_NODE_OUTPUT"
node bindings/st-wasm/tests/fractional_history_l2.mjs "$WASM_NODE_OUTPUT/spiraltorch_wasm.js"
tsc --noEmit --strict --skipLibCheck --lib es2020,dom bindings/st-wasm/tests/fractional_learning_types.ts
```

The full Python regression file list is in `validation.json`. The explicit
tools test root prevents unrelated ancestor packages from being collected.
Source hashes describe the changed tree on the recorded parent commit, not
an assertion that the parent alone built the new binary. Native/WASM packages,
raw build/test logs and the full runtime file manifest stay local. Published
files contain numerical results, verification metadata and hashes only;
`SHA256SUMS` binds those bytes. No corpus, model weights or private paths are
included. Completed primary studies and their frozen runtimes were not changed.
