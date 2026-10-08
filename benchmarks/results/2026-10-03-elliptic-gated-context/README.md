# Gated Elliptic Context: Learning-Path Validation

Scope: a new Rust-owned signed contextual correction, not a pretrained language
quality study or a speed benchmark. The earlier causal factorial's negative
outcome remains unchanged. See [the contract and examples](../../../docs/elliptic_gated_context.md).

- Rust: 8 tests passed, including gate/orientation finite differences, exact
  pointwise zero initialization, causal boundaries and summed shared gradients.
- Python: 106 related tests passed, zero skips. Includes equivalent Torch
  forward/VJP at B=2,T=128, missing optional Torch imports, MPS host transport,
  exact Adam resume, and actual tiny-HF loss reaching all three parameter paths
  without modifying the frozen base.
- Browser WASM: finite-difference maximum absolute error 0.000221113 for
  orientations and 0.000045602 for the raw gate over negative/zero/positive cases.
  A 150-update synthetic gate-only fit reduced MSE from 0.006494068 to
  0.000233491, starting at raw gate 0 and ending at 0.466313511 (target 0.6).
  This establishes a trainable browser path, not LLM quality or convergence.
- Strict TypeScript fixture, Rust formatting, and st-core test Clippy passed.
- Native and WASM builds passed. The first native link ran out of disk space;
  removing three obsolete incremental Python-build directories allowed retry.
  A later Rust-test rebuild also exhausted space; two obsolete core incremental
  directories and this run's disposable raw WASM build copies were then removed.
  Frozen experiment packages, checkpoints, corpora and user-deleted artifacts
  were not changed. Existing dependency warnings are not new operator failures.

`validation.json` records source/artifact/log hashes. `browser.json` contains
the numeric browser checks and complete synthetic loss sequence. Native binaries,
test logs and generated WASM remain local. `SHA256SUMS` covers the public records.

## Reproduce

```sh
cargo +1.98.0 test -p st-core --test elliptic_gated_causal --test elliptic_learning --offline
cargo +1.98.0 clippy -p st-core --tests --offline -- -D warnings
cargo +1.98.0 build -p spiraltorch-py --offline
cargo +1.98.0 build -p spiraltorch-wasm --target wasm32-unknown-unknown --offline
wasm-bindgen target/wasm32-unknown-unknown/debug/spiraltorch_wasm.wasm --target web --out-dir /path/to/module
tsc --noEmit --strict --target es2020 bindings/st-wasm/types/spiraltorch-wasm.d.ts bindings/st-wasm/tests/elliptic_gated_causal_types.ts bindings/st-wasm/tests/elliptic_causal_types.ts
```

Install the fresh native binding and matching Python client into an isolated
environment with Torch and Transformers 4.57.6. Run the geometry tests listed in
`validation.json` with `python -I -m pytest --import-mode=importlib -q ...` from
the repository root; tests do not download a model. The original host used
Python 3.12.6 and Torch 2.12.1 on Apple Silicon. Serve the browser fixture and
fresh `/module/` output locally, then inspect its visible passed JSON.

A next LLM study must use a gated tangent control with the same additional
scalar and paired budgets. No advantage can be inferred from this record alone.
