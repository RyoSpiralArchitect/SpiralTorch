# Anchored Elliptic Learning: Cross-Client Validation

Source: `f5c43e1911073dc3c754ae3bd953141b5baae5dc`. The
[operator contract](../../../docs/elliptic_anchored_learning.md) implements
`(1-tanh(raw_mix))*phi(x) + tanh(raw_mix)*phi(1,0,0)` in Rust with input and
sum-reduced shared-gate VJPs. Python/HF and browser WASM call that same core.

This validates the implementation and actual learning wiring. It is **not** a
pretrained language-quality or speed result. The earlier frozen-checkpoint probe
is unchanged and is not being relabeled as an anchored-training experiment.

## Checks

- Rust: 12 tests passed, zero skipped (four new anchored tests plus eight existing
  elliptic/causal tests); clippy with `-D warnings` and workspace formatting passed.
- Python: 149 related tests passed, zero skipped. Equivalent-formula Torch VJPs
  agree at negative/zero/positive gates, including `[2,128,3]`, a single row and
  empty leading axes. Exact zero initialization, row independence, shared-gradient
  sum, bounds, immutable snapshots and saved-tensor version checks are covered.
- Actual tiny random HF GPT-2 loss updates the readout, orientation and gate;
  frozen base weights remain unchanged. Adapter plus full Adam continuation is
  exact. Cached token-by-token logits match full-context logits after training
  (`atol=2e-6`, `rtol=2e-5`). This is not universal HF compatibility evidence.
- MPS inputs match CPU forward/VJP exactly through explicit Rust CPU transport;
  this does not establish resident accelerator execution.
- WASM build, strict TypeScript surface and actual browser fixture passed.
  Largest finite-difference error: orientation `0.000204331`, gate `0.000100704`.
  A 150-update synthetic gate fit reduced MSE from `0.0222102961` to
  `0.00000307836`, learning raw gate `-0.59115672` toward target `-0.6`.

## Reproduce

```sh
cargo +1.98.0 test -p st-core --no-default-features \
  --test elliptic_anchored --test elliptic_gated_causal --test elliptic_learning
cargo +1.98.0 clippy -p st-core --no-default-features --test elliptic_anchored -- -D warnings
cargo +nightly-2026-04-15 fmt --all -- --check
cargo +1.98.0 build -p spiraltorch-py
cargo +1.98.0 build -p spiraltorch-wasm --target wasm32-unknown-unknown
tsc --noEmit --strict --lib es2020,dom \
  bindings/st-wasm/types/spiraltorch-wasm.d.ts \
  bindings/st-wasm/tests/elliptic_anchored_types.ts
```

Run `bindings/st-py/tests/test_elliptic_anchored.py` and the related geometry suite
against the freshly built native package with Torch/Transformers available.
Generate web output using wasm-bindgen 0.2.104, serve it at `/module/`, and open
`bindings/st-wasm/tests/elliptic_anchored_learning.html`. Read its completed
`#result` rather than treating a loaded page as a passing test.

`validation.json` binds sources, runtime, binary hashes and local log hashes.
`browser.json` contains the actual derivative errors and entire synthetic loss
trajectory. `SHA256SUMS` covers these public records. Binaries and raw logs stay
local; four obsolete incremental caches were removed to recover build space,
without touching frozen packages, corpus text, weights or earlier experiments.
