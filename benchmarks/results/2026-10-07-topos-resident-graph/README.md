# Shared Topos Gates In Resident NN Graphs

Correctness measurements of source
`d4c61e03a664f0dc15fc9c02eebca55374db8008`, immediate parent
`7f43b48f4b511ed34d61a2acf22f0f6e6ddfaae8`, compared with kernel baseline
`74248e46e2f26826dcaa389220a60d2e132bd83b`.

The shared gate now participates in resident NN inference, arbitrary-seed VJP,
transactional graph SGD and parameter handoff to `Sequential`. Python and WASM
construct the same Rust-owned v5 graph. Both gradient policies sum the gate
gradient without adding another row mean.

## Measurements

On Apple M4 / Metal, native Rust tests completed 300 synthetic updates across
two porosities, two gradient policies and three ranks. Python independently
compared 100 updates with CPU Torch 2.12.1, carrying separate weights throughout
each trajectory. Maximum absolute errors were `5.9604645e-8` for output/loss,
`2.9802322e-8` for input gradient, `1.1920929e-7` for gate gradient and
`1.4901161e-8` for the updated gate (`rtol=5e-4`, `atol=3e-5`).

Chrome 154.0.8037.98 executed another 100 WebGPU graph updates and 4,119 checks
against the scalar Rust/WASM reference. Its adapter probe confirmed Apple and
`is_fallback_adapter=false`; this reference shares Rust semantics and is not
an independent mathematical implementation. These four trajectories cover
shape `[2, 2, 3]`, porosity `0`/`0.3`, and `exact`/`module_compatible` policies.
They include **explicit per-step verification readbacks**, not transfer-free
training. Four existing graph/autograd/learner/module browser fixtures also
passed without page errors or console messages.

At the corrected source, 781 default NN tests, 53 real-WGPU resident NN tests,
45 Python tests plus 32 subtests, Node-hosted WASM contract checks and touched
Rust formatting passed. Contracts (49), backend graph training (32) and graph
forward (26) also passed at the earlier integration source `d769b25e`; these
are recorded separately rather than presented as corrected-source reruns.

Read-only independent review found two P2 issues: rejected parameter handoff
cleared a Topos capture, and graph lowering skipped optimizer/topos alignment.
Both failed regression tests before the fix, then passed. Follow-up static
review found no actionable remaining or new issue; it did not rerun tests.

## Files And Limits

- `browser.json.gz`: all 100 synthetic inputs, targets, predictions, input/gate
  gradients, losses and updated gates, plus browser/runtime asset hashes.
- `python-check.json`: independent Torch comparison summary, tested source and
  reference hashes, and native runtime identity. It is not a Python trajectory.
- `verification.json`: source/binary/log hashes, validation scope, retained
  initial failures and known limits. Complete logs and binaries remain local.
- `SHA256SUMS`: integrity of the four public files, not execution attestation.

Strict local NN all-target Clippy failed with 39 diagnostics in 22 files
unchanged between the kernel baseline and corrected source. This is a remaining
lint gap, not a successful lint run. Vendor WGPU warnings, `block` future
incompatibility and stable rustfmt's nightly-option warnings also remain.
Initial test-harness failures and the genuine pre-fix regressions are retained,
not rewritten. No pretrained weights or corpus text are included.

The graph does not materialize the complete core semantic audit. VJP recomputes
the finite unroll, owning outputs can require copies, and allocation remains.
Elementwise Topos modules are still host-only. This is one deterministic
synthetic family, not multi-seed, CUDA, speedup or language-model quality evidence.

## Reproduction

Use a fresh output directory and a WGPU-enabled build of the measured source.
Python needs an isolated, unpatched Torch environment and that source's native
extension, not a previously installed wheel. `-I` alone does not disable site
startup hooks; verify the native extension hash and actual adapter separately.

```sh
cargo test --locked --release -p st-nn --lib
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-nn --features wgpu --lib resident:: -- --nocapture
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -P bindings/st-py/tests/test_nn_resident_topos.py
cargo build --locked --release -p spiraltorch-wasm --target wasm32-unknown-unknown --features webgpu
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm --target web --out-dir /tmp/topos-graph-new-web
node tools/test_resident_browser.cjs /tmp/topos-graph-new-web "$CHROME" /tmp/topos-graph-new.json "" "" "" "" topos-resident-graph
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm --target nodejs --out-dir /tmp/topos-graph-new-node
node bindings/st-wasm/tests/topos_graph_contract.cjs /tmp/topos-graph-new-node/spiraltorch_wasm.js
python -I -m pytest --import-mode=importlib --confcutdir=tools --rootdir=tools -q tools/test_topos_resident_graph_results.py
```

Use wasm-bindgen 0.2.104 and a Playwright-equipped Node runtime. Set `CHROME` to
an installed Chrome executable. The harness creates an isolated browser and
serves only its allowlisted local assets; it does not use an existing profile.
The publication tests need no GPU, Torch or native extension: they check saved
numbers and SGD/MSE consistency, not independent execution provenance.
