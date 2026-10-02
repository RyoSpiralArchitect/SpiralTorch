# Causal Elliptic Bridge Validation

This is correctness and small-learning evidence, not a pretrained-model quality
study or a speed benchmark. See [the contract and usage](../../../docs/elliptic_causal_learning.md).

Rust owns causal attention and the chart VJP. The same snapshot is callable from
Rust, Python and browser WASM. All tied query/key/value derivatives are included.
The existing pointwise learning experiments and their frozen runtimes were not
modified by this implementation.

## Verified

- All 47 kernel-contract tests pass, including finite differences for every
  attention input/bias, multiple batches/heads, causal offsets and empty shapes.
- All five elliptic core tests pass, including finite differences through the
  composed feature map and causal attention, pair budgets and batch isolation.
- All 75 Python geometric-learning tests pass with no skips. Seven new cases
  compare against equivalent PyTorch attention, check causal prefix/future/batch
  boundaries, exact resumed parameters/Adam state, incompatible checkpoint
  rejection, actual tiny-HF gradients with a frozen base, and MPS CPU transport.
- The full Python native extension and wasm32 module build successfully.
- Browser WASM reports `passed`. The maximum finite-difference VJP error is
  0.000256497. A 100-update coordinate-learning check reduces MSE from 0.0353041
  to 0.000104894. That is a small feature-matching task, not language-model FT.
- Kernel-contract Clippy, targeted Python Ruff and source whitespace checks pass.

`browser-report.json` contains the visible browser result and all learning losses.
`validation.json` records runtime/source hashes and the verification scope.
`SHA256SUMS` covers public receipts. Raw local Python logs and generated binaries
stay local. No corpus text, model weights or private machine paths are published.

## Reproduce

```bash
cargo test -p st-kernel-contracts
cargo test -p st-core --test elliptic_learning
```

Build/install the current native Python binding, require its new
`EllipticCausalLearningBatch` export, and run
`bindings/st-py/tests/test_elliptic_causal_learning.py` with pytest importlib mode.
The existing offline geometry CI suite includes the new test file and checks the
native/class exports before collection, so a stale extension cannot silently skip
the new path. Local verification is distinct from remote CI completion.

For the browser, build `spiraltorch-wasm` for `wasm32-unknown-unknown`, generate
wasm-bindgen web output and serve the HTML/module routes described in the guide.
The browser test is self-contained and needs no model download or network API.
It must display `status: passed`, not merely finish compilation.

The numerical tests do not establish a speedup, resident execution, a geodesic
metric, padding/cache support or a language-quality advantage. Real comparative
pretrained learning on this new causal mechanism is the next experiment.
