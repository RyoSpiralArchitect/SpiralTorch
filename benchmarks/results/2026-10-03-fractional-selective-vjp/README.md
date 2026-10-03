# Selective Fractional VJP Validation

This is correctness and learning-path evidence, **not a throughput benchmark**.
Rust now exposes `vjp_input` and `vjp_alpha` alongside the unchanged joint VJP.
The order-only path skips the input-adjoint convolution and its gradient
allocation. Python selects the requested components from Torch's gradient
requirements; the same operations are available to Rust and WASM callers.
Forward snapshots, including their cached order differential, are unchanged.

## Evidence

- 120 Rust tests passed, including bit-exact selective/joint component checks
  across full/history maps, axes, orders, step sizes and kernel lengths. The
  15 learning integration tests also passed in release mode.
- 327 isolated-package Python tests passed in both development and release
  packages, zero skipped, plus 13 non-timing benchmark-reference tests. Includes selective
  dispatch observation, independent Torch mathematical references, empty-history
  guards, optional-PyTorch imports, exact tiny-HF learning and resume checks.
- 54 actual wasm32 selective/joint comparisons passed. Full-GL 100-update and
  independent-history 500-update synthetic clients now use alpha-only VJP.
- Six saved pretrained GPT-2 adapters (three full-GL, three history; seeds
  41/43/47) each ran two auxiliary updates through each route. Across these
  24 auxiliary updates, losses, every parameter gradient, final parameters and
  Adam states matched exactly. The frozen base and original study files did
  not change; no held-out losses were computed.
- Rustfmt, Clippy with the WGPU feature and TypeScript declarations passed.
  Existing vendored-WGPU warnings and Torch deprecation warnings remain.

`pretrained-parity.json` records all auxiliary update scalars and original
checkpoint identities. The reference is the **joint VJP in the same new native
binary**, not a concurrently loaded old extension. Tests also verify that an
overflow in an unrequested component cannot reject a finite requested gradient,
while invalid upstream shapes/nonfinite values still fail. At alpha=1, older
coefficient derivatives remain active even when those coefficients are zero.

## Boundaries And Reproduction

The pretrained parity and WASM checks used development builds; a separately
built release native package also passed the 327 Python checks and matched
the Torch benchmark reference at the actual 2x128x768 shape for both maps.
The `benchmark-validation-release-*.json` files contain **no timings**. No timing is used
to claim a speedup over PyTorch, the old bridge or an end-to-end model. The
running twelve-run single-lag study kept its original frozen package and
client files; this change was not hot-loaded into it.

Rebuild the Python extension and run `test_fractional_selective_vjp.py` with
`--import-mode=importlib` against that installed package. The test reproduces
the old joint backward on the same snapshot and compares actual dispatch and
tiny-HF losses, gradients, parameters and Adam state. Native and WASM checks:

```sh
cargo test --locked -p st-frac
cargo clippy --locked -p st-frac --all-targets --features wgpu -- -D warnings
cargo build --locked -p spiraltorch-wasm --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/debug/spiraltorch_wasm.wasm \
  --target nodejs --out-dir "$NEW_WASM_DIRECTORY"
node bindings/st-wasm/tests/fractional_selective_vjp.mjs "$NEW_WASM_DIRECTORY/spiraltorch_wasm.js"
node bindings/st-wasm/tests/fractional_learning.mjs "$NEW_WASM_DIRECTORY/spiraltorch_wasm.js"
node bindings/st-wasm/tests/fractional_history.mjs "$NEW_WASM_DIRECTORY/spiraltorch_wasm.js"
tsc --noEmit --strict --skipLibCheck --target ES2020 --module commonjs \
  bindings/st-wasm/tests/fractional_learning_types.ts
python -m pytest --import-mode=importlib --confcutdir=tools --rootdir=tools -q \
  tools/test_benchmark_fractional_learning.py
python tools/benchmark_fractional_learning.py --native-profile release \
  --validate-only --output "$NEW_VALIDATION_JSON"
```

The benchmark includes forward, order-only backward and native host transport.
After the active training/build work finishes, run on an idle host without
`--validate-only` and with a fresh output path. Record all three routes, the
release binary hash and balanced execution order; do not extrapolate these
operator timings to an entire adapter/model or different geometric maps.

For the pretrained check, use the completed fractional-memory/history study
checkpoints, restore their exact adapter/Adam states, and use each seed's planned
next training minibatch twice per route. Compare all losses and gradients after
each update and the final adapter/Adam tensors. This is an auxiliary check on
copies, not additional primary study steps or endpoint rescoring.

`validation.json` binds source/native/WASM hashes and local evidence logs;
`SHA256SUMS` binds this published record. Weights, corpora, checkpoints, native
packages and raw logs remain local. No quality, significance, resident-GPU or
unique fractional-memory claim follows from this validation.
