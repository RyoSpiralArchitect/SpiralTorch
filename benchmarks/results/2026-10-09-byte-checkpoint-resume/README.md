# Complete byte-model checkpoint and resume

Source: `cafe944b9cce71a511b481bad9f9fd6803b0a6a7`, based on
`0700a15d07d02b60037761b645640e04f3728fb0`. Results were captured on
2026-10-09. This is a correctness result, not a language-quality or speed claim.

## What changed

The Rust byte decoder now captures and imports its complete model: byte and
position tables, all residual graphs, fused QKV/output projections, Topos gates,
head, and optional causal projection/wave/Poincare geometry. The same portable
graph contracts validate native and WASM imports. Restore starts a fresh owner
at the captured **attempted** SGD revision; old tapes/gradients are not reusable.
The new live checkpoint template retains no additional host weight arrays.

The decimal-string revision survives values above JavaScript's exact-integer
range. There are no momentum slots in this stateless SGD path. Data cursors,
corpus identity, RNG, learning-rate schedules, external biases, acceptance
history and runtime/kernel selection remain application state. This is not a
complete application replay or streaming-state checkpoint.

## Observed results

Both [native Metal](native.json) and [actual BrowserWebGpu](browser.json) passed
the same shared Rust verification. Four configurations cover 18, 31, 23 and 37
parameter tensors: plain one-block, two-block Topos, learned causal geometry,
and learned causal geometry plus two-block Topos.

For every configuration the resume probe:

- Applies two updates on different byte batches/rates, captures revision 2,
  advances the original model to revision 3, then reads the old capture.
- Imports the complete JSON, creates a fresh owner and checks exact restored
  values, logits, CE and every parameter gradient.
- Rejects a foreign tape even when submission index and revision match, and
  rejects gradients bound to the original owner.
- Applies the same subsequent batches/rates and requires exactly identical
  canonical model JSON after updates, plus a nonzero-update sensitivity control.
- Rejects an invalid gradient, preserves all weights at attempted revision 5,
  then restores and reproduces the next accepted update at revision 6.
- Restores revision `9007199254740993` and observes exactly
  `9007199254740994` after an update, with a string-valued browser receipt.

The resume probes use the self-contained forward, not caller-owned external
biases. Their JSON snapshots were 31,375-35,270 bytes. Exact agreement is required
**within each runtime**; no cross-device bitwise equality is claimed. The existing
independent frozen PyTorch checks also pass: all four models' full VJPs and all
16 CE/SGD updates, using the unchanged elementwise and geometry-relative gates.

Native unit tests: 14 passed with GPU tests enabled. Existing portable graph
unit tests: 7 passed. CPU-only library check and pinned-nightly workspace format
check passed. Library Clippy completed with 23 warnings in unchanged files and
none in changed files; it was not a clean strict/all-targets Clippy run.

An independent read-only local review found one P2: generic graph lowering could
multiply an invalid huge input rank across many stages before the byte-model
rank check. Every nested graph is now checked at rank three before lowering.
Regression tests cover all six graph-location categories and a sub-1MiB input
with 200,000 dimensions and 4,096 identity stages, requiring the exact early
preflight error. Re-review found no remaining concrete findings. Final native
and actual-browser regressions were rerun after the fix; each final report is
byte-identical to its own pre-fix report. Earlier local logs were retained.

## Reproduce

See [the API and ownership contract](../../../docs/resident_byte_decoder.md).
Native commands used the debug profile and default features plus `wgpu`:

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-nn \
  --features wgpu --lib resident::byte_decoder -- --test-threads=1
cargo test --locked -p st-nn --features wgpu --lib resident::portable
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-nn \
  --features wgpu --test resident_byte_decoder -- --nocapture
cargo check --locked -p st-nn --no-default-features --lib
cargo +nightly-2026-04-15 fmt --all -- --check
cargo clippy --locked -p st-nn --features wgpu --lib
```

The browser used Rust 1.97.0, the release profile and wasm-bindgen 0.2.129:

```sh
env -u CARGO_ENCODED_RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS \
  -u CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS \
  -u LIBRARY_PATH -u PKG_CONFIG_PATH RUSTFLAGS= \
  cargo build --locked --release -p st-nn --no-default-features \
  --features wgpu --target wasm32-unknown-unknown \
  --example resident_byte_decoder_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_byte_decoder_browser.wasm \
  --target web --out-dir target/resident-byte-decoder-web
python3 -I -S -m http.server 8771 --bind 127.0.0.1
```

Open `http://127.0.0.1:8771/crates/st-nn/tests/byte_decoder_browser.html` in a
WebGPU browser. The page checks the v3 report and every checkpoint assertion,
then provides a result download. It executes the Rust learner, not a JS model.

[validation.json](validation.json) records source/fixture/report hashes and
hashes of the locally retained binary, WASM, JS and command logs. Public reports
contain verification metrics, not newly captured model weights or raw corpora.

Remaining limits: native-to-browser checkpoint handoff is not yet tested;
signed-zero/subnormal roundtrip is tested at Rust JSON level, not through a
JavaScript parse/stringify cycle. Keep the Rust JSON string unchanged for
bit-preserving transport. There is no new Python facade or release bump here.
