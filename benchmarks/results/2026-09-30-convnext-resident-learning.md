# Model-owned resident ConvNeXt learning

## Scope

The existing SpiralTorch ConvNeXt-style backbone now compiles into a fixed-shape
GPU-owned model. Forward and VJP read its parameter owner, and all-or-none plain
SGD updates the values consumed by the next forward. The original host model is
an independent snapshot, not an implicit cache authority over device learning.
See [the API and its limits](../../docs/resident_convnext_training.md).

Convolution geometry is reused from `st-nn` layers. Linear, LayerNorm and GELU
share existing graph-autograd; explicit resident rebinding copies weights
GPU-to-GPU and preserves validity guards. Its guard-capture pipeline is reused
instead of recompiled on every step. No host weight or gradient mapping is
required between steps. Candidate/output allocations and GPU-to-GPU copies
remain; this is not a throughput benchmark.

## Shared Native And Browser Fixture

Both runtimes execute `examples/support/convnext_learning_checks.rs` in
`st-vision`. The fixture uses two synthetic `[2, 8, 8]` images, two stages with
widths `[3, 4]` and depths `[1, 1]`, patch size `[2, 2]`, curvature -1 and
epsilon 0.001. This tests the complete stem/block/downsample/final-norm topology,
not a pretrained ConvNeXt-Tiny or real-image task.

Eight forward/MSE/VJP/SGD steps run with **zero readbacks inside the learning
loop**. Every retained prediction, loss, input derivative, all 22 parameter
derivatives and updated weights are compared with the host Rust CPU model after
submission. A ninth forward checks the updated weights. All five parameter
groups (stem, block, downsample, block, final norm) change.

Seventeen named checks also cover strided/offset inputs, host-source isolation,
pending host-gradient rejection, shape/rate validation, foreign model and stale
tokens, whole-model rejection, invalid zero-rate updates, valid retry, and
retained parameters/derivatives/receipts after reuse and model destruction.
Rejected updates preserve all 22 tensors bit-for-bit. The retry is checked
against the next CPU update, not just a successful submission.

At learning rate **0.0001**, native GPU and Chrome WebGPU both produce initial
MSE **0.5028321743** and post-eight-step MSE **0.2255446166**. The full loss
sequence and hashed assets are in the [browser report](2026-09-30-convnext-resident-learning-browser.json).
The comparison threshold is `abs(actual - expected) / (1 + abs(expected)) <= 2e-4`.
This tolerates f32 backend arithmetic differences; it is not a bitwise-equivalence
claim. Rejection/snapshot-retention checks separately require identical bits.

The initial fixture attempt at learning rate **0.003** matched CPU arithmetic
but failed the loss-decrease check: losses were
`[0.5028322, 1.0135162, 0.9709197, 0.92030203, 0.8462911, 0.70729446, 0.31087738, 0.9851692]`,
with final MSE `0.7747575`. The final fixture rate was lowered; no tolerance was
relaxed. Even the successful run is not monotonically decreasing. This evidence
does not establish long-run optimization stability, generalization, a Z-space
advantage, or superiority to PyTorch.

The real browser is Chrome 154.0.8037.58. The Rust runtime reports BrowserWebGpu
with an empty device name; a separate browser adapter probe reports non-fallback
Apple/Metal. That separate probe is not physical-device attestation of the Rust
runtime. The browser fixture invokes Rust directly, not a new JavaScript model API.

## Regression Checks

- GPU-enabled library tests: backend 242, NN 788, vision 73 and core 1,027 passed;
  one existing vision and one existing core test ignored.
- CPU-only vision: 61 passed.
- Native and wasm32 strict backend Clippy passed. Non-strict vision all-target
  Clippy passed with existing warnings and no diagnostics in the new files.
- The broader strict NN/vision command remains blocked by existing NN lints
  (24 library diagnostics, 38 including tests); these were not silenced here.
- Scoped nightly rustfmt, JavaScript syntax, and whitespace checks passed.
- CI now includes the full-model native GPU test and the WASM example check.

## Replay

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-vision --features wgpu --lib resident_convnext_learning -- --nocapture
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-backend-wgpu -p st-nn -p st-vision -p st-core --features st-nn/wgpu,st-vision/wgpu --lib
cargo +1.98.0 test --locked -p st-vision --no-default-features --features nn --lib
cargo +1.98.0 build --locked -p st-vision --features wgpu --example convnext_resident_learning_browser --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/examples/convnext_resident_learning_browser.wasm --target web --out-dir /tmp/convnext-learning-web --out-name spiraltorch_wasm
node tools/test_resident_browser.cjs /tmp/convnext-learning-web "$CHROME_EXECUTABLE" /tmp/convnext-learning-new-report.json '' '' '' '' convnext-learning-resident
```

Use wasm-bindgen 0.2.104 and Playwright; every report output path must be new.
Generated WASM binaries are not committed. Checkpoint/resume, host handoff,
classification-head/data-pipeline integration, Python/JS bindings, and matched
real-image accuracy/throughput remain separate gates.
