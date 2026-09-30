# Shared resident SGD candidate contract

`st-kernel-contracts::sgd::SgdStep` validates finite, nonnegative rates and
provides a CPU oracle for `parameter - rate * gradient`. Its generated WGSL
helper is used by dense training, graph training, global-norm clipping, and
Topos EMA updates. Loss reduction, gain policy, clipping, and EMA still run
before the shared candidate rule. The existing whole-update decision and
commit ordering remain unchanged.

The rule checks the gradient, multiplication, and subtraction separately.
This rejects an overflowing multiplication even when a fused expression
could produce a finite final value. A zero rate still validates derivatives;
the existing commit kernels preserve parameters and EMA history at zero rate.
The diagnostic masks remain 4096 for a plain gradient, 8192 for the change,
and 16384 for the candidate. EMA supplies its existing 32768 history mask.

## Evidence

- Native Metal: 237 backend and 35 kernel-contract tests passed with GPU tests
  enabled. The new direct shader comparison checks 12 finite/non-finite and
  zero-rate/overflow cases against the Rust oracle, including both plain and
  EMA diagnostic masks. Existing graph tests cover clipping, momentum,
  microbatch accumulation, rejection, and recovery.
- The mandatory `st-core` library suite passed all 1,001 tests.
- The [learner browser report](2026-09-30-shared-sgd-learner-browser.json)
  covers both gradient policies, 32 custom-loss updates per policy, weight-only
  checkpoint replay, stale/foreign gradients, invalid zero-weight inputs, and
  an isolated SGD-candidate overflow followed by a successful retry.
- The [microbatch browser report](2026-09-30-shared-sgd-microbatch-browser.json)
  covers 12 conditions: two gradient policies, mean/sum loss, and plain,
  clipped, or clipped-plus-EMA updates. Each runs 32 updates from 96 microbatches,
  then rejects update 33 and accepts a zero-rate recovery at update 34. The
  fixture checks retained gradients, weights, and EMA history against its
  scalar reference after the learner is reused or dropped.
- Native and wasm32 strict Clippy passed for both affected crates. The browser
  reports came from a fresh release WASM build in Chrome 154.0.8037.58, with
  no page or console errors.

The learner fixture uses an absolute error bound of `2e-5`; the microbatch
fixture uses `2e-5 + 2e-4 * abs(reference)`. Both report the Rust runtime as
`BrowserWebGpu` with device type `Other`; this does not identify the physical
adapter. These are bounded behavior-preservation checks, not performance
measurements or end-to-end ConvNeXt training results. Model-owned resident
parameter updates and versioned gradient handoff are still the next work.

## Replay

From the repository root, with Rust 1.98.0, `wasm-bindgen 0.2.104`, Playwright,
and `CHROME_EXECUTABLE` set to the installed Chrome executable:

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-kernel-contracts -p st-backend-wgpu --lib
cargo +1.98.0 build --locked -p spiraltorch-wasm --features webgpu --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm --target web --out-dir /tmp/spiraltorch-shared-sgd-web --out-name spiraltorch_wasm
node tools/test_resident_browser.cjs /tmp/spiraltorch-shared-sgd-web "$CHROME_EXECUTABLE" /tmp/spiraltorch-shared-sgd-learner.json '' '' '' '' nn-learner-clients
node tools/test_resident_browser.cjs /tmp/spiraltorch-shared-sgd-web "$CHROME_EXECUTABLE" /tmp/spiraltorch-shared-sgd-microbatch.json '' '' '' '' nn-microbatch-clients
```

Choose new report paths on repeated runs; the harness does not overwrite
existing evidence. The reports include hashes of every served module asset.
