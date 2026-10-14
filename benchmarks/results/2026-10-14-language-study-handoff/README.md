# Study Checkpoint To Executable Language Model

This is a source-only handoff qualification after 0.4.31, not a new language
quality experiment or a PyPI release. No additional training updates were run
on the historical artifacts. Raw weights, corpus and local build outputs stay
in the private validation archive; public results and identities are in
[`qualification.json`](qualification.json).

## Contract And Results

`ByteCorpusStudy::case_checkpoint` validates every case against the exact study
request before extracting one existing `ByteDecoderCheckpoint`. Python releases
the GIL around that Rust operation; the standalone WASM example calls the same
validator. Extraction does not acquire a GPU or make a promotion decision.

- Rust portable study suite: 47 passed, including malformed unselected cases,
  foreign requests, partial cursors, frozen values and learned query/router bits.
- Installed WGPU candidate: 10 passed, one CPU-artifact-only skip. The v8 test
  restores a learned contextual router in a fresh isolated Python process and
  requires exact logits, loss and model state.
- Installed CPU-only candidate: 8 passed, three opt-in GPU skips. Model
  extraction succeeds without WGPU; explicit GPU restore still rejects.
- The new wasm32 binary was built with stable Rust and wasm-bindgen 0.2.129.
  The in-app browser imported the existing 75-case browser checkpoint at update
  128 with no new updates, extracted `mixture-11-0-train-query-train`, and rejected
  a whitespace-changed request. The downloaded opaque model JSON SHA-256 equals
  the Python export: `c69dc04834a97b2584a91cff26a61eab5c7af0744f7ea19ff84ee6eb9f7dce8e`.

Historical v8 qualification selected the first ordinary arm and the first
learned/frozen contextual-router pair, from each of the native and browser
128-update archives: six model handoffs, not all 75 models' inference. All six
fresh-process runs exactly matched the parent process's 16,384 logits, loss,
revision and model JSON on Apple M4/Metal. Complete study envelopes were
validated, and their source bytes remained unchanged.

Historical browser-to-native loss is deliberately recorded separately: one
frozen-query arm differs from the saved browser value by about 3e-7 (one f32
ULP after decoding the saved f32). Do not infer universal cross-device bitwise
equality. Rust's shortest f32 JSON spelling is compared by f32 bits in tests,
not directly to Python's expanded f64 spelling; no tolerance was widened.

Additional strict `cargo +1.99.0 clippy -p st-nn --lib --no-default-features -- -D warnings`
did not pass: it reports 23 diagnostics in other files byte-identical to the
release base `1b525b7e`. None targets the handoff changes. This is not a clean
whole-crate strict-lint claim; those unrelated findings were not suppressed.

## Reproduction

Build and install a source candidate with the [Python build guide](../../../docs/python/build-and-validation.md).
Use separate CPU-only and WGPU virtual environments; do not shadow the installed
extension with `PYTHONPATH`. Then run:

```sh
cargo test --locked -p st-nn --lib resident::byte_study --no-default-features
python -I -B bindings/st-py/tests/test_nn_byte_study.py -v
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B bindings/st-py/tests/test_nn_byte_study.py -v
```

The synthetic tests contain the preparation, learning, extraction and fresh
inference recipe. For the retained real-corpus run, use the original v8 request
and native/browser checkpoint hashes in `qualification.json`, select its three
explicit case names, and apply the same fresh-process probe. Retain the exact
request and full study checkpoint; the model alone is not a provenance record,
frozen-rate resume policy, HF adapter or optimizer-state snapshot.

Build the standalone WASM example using the [corpus browser recipe](../../../docs/byte_corpus_learning.md#browser-execution).
Open `byte_learning_resume_browser.html?resume=1&stop=128&case=mixture-11-0-train-query-train`
with the pinned completed v8 checkpoint. The page exposes the selected-model
download only after the all-case and foreign-request checks succeed.
