# Language-First Serving Gates

Source-only after 0.4.31. This qualification concerns **measured model selection
and exact-owner rollback**, not new language-quality evidence or a speed result.
The private candidate wheels have 0.4.31 metadata but are not the immutable PyPI
release and must not be published as that version.

## Runtime Contract

Rust fixes evaluation data/windows and explicit loss/geometry/router limits at
slot bootstrap. Each candidate assessment replays both models on that same
runtime and gate. Only a passing process-local typed permit, bound to the exact
slot, generation and incumbent, can activate. Reports cannot be imported as
permits. Rollback preserves the previous owner rather than reconstructing it;
retained forwards keep their original artifact identity.

Python and the standalone WASM client call the same Rust implementation.
The packaged `st-wasm` module and HF PEFT executor are unchanged. This is not a
network service, durable registry, scheduler, KV cache or streaming decoder.
See [the API and metric definitions](../../../docs/python/byte-serving-gates.md).

## Existing Models, No New Training

The private v8 native/browser studies each contain 75 cases at update 128.
Before measurement, `verify_serving.py` selects the first ordinary case as
incumbent and the first train/frozen contextual-router pair as candidates in
each archive. **Four paired comparisons** use all 32 existing evaluation
batches. These are not independent cohorts or new held-out data.

The script fixes deliberately permissive mechanics thresholds before any
measurements: mean CE <=10, mean/worst-batch CE regression <=2, merged geometry
RMS <=100, router mass error <=1e-5 and normalized entropy >=0. These are not
recommended operating thresholds or evidence-based quality promotion rules.

All four passed those explicit gates, rejected stale permits and restored the
exact pre-switch logits. All original archive bytes remained unchanged, with
**zero additional training updates**. Native replay mean CE deltas were about
-3.02e-5. Actual contextual-router entropy stayed near 1; its being measurable
does not demonstrate that meaningful routing specialization was learned.
The full assessments, identities, per-window losses, geometry RMS, router rows,
hashes and transitions are in [qualification.json](qualification.json).

An actual browser then evaluated the browser-trained ordinary/query-train pair
using `BrowserWebGpu` (device type `Other`). It exercised accepted activation,
cross-slot/stale/reused/JSON-permit rejection, stale rollback and exact output
restoration. A separate deliberately impossible relative gate rejected the
candidate without changing the owner. The first inference window had 16,384
logits; the before/rollback f32 little-endian SHA256s were identical:
`08bd7e1b5d6b166420cb2f8d233a807f11668faa1403d3c6e976f23bfced1e27`.
See [browser-receipt.json](browser-receipt.json). This is same-browser equality,
not a promise of cross-device bitwise equality.

## Validation And Review

[validation.json](validation.json) binds source files, binaries, raw-log hashes
and test counts. Independent read-only review found that entropy initially used
unnormalized masses: `[0.5, 0.25]` could appear uniform. The fix divides by row
mass, rejects zero mass and keeps the separate mass-error check. Regression
tests protect mean-only and worst-window-only failures independently.

Installed-wheel GPU tests also compare actual shared/contextual router readbacks
on two blocks and two heads, then inject a collapse in only block 1/head 0 and
verify entropy-only rejection. Those two-block fixtures duplicate existing
single-block parameters for inference coverage; they explicitly fail study
companion validation and are not labelled as trained two-block models. The
original failed test assumption (multi-block study calibration) remains in the
private attempt log. No study policy was broadened just to make that test pass.

The follow-up independent review found no further actionable issues. It was
static/read-only, not an independent rerun of our numerical tests. Concurrent
async scheduling and cross-process durable deployment remain outside this slice.

## Reproduce

Install a fresh source candidate as described in the [build guide](../../../docs/python/build-and-validation.md).
Use separate CPU/WGPU environments; do not shadow the installed package with
the checkout's Python files. Serialise GPU tests on one device.

```sh
cargo test --locked -p st-nn --lib resident::byte_study --no-default-features
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-nn --lib resident::byte_study::serving --no-default-features --features wgpu -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B bindings/st-py/tests/test_nn_byte_serving.py -v
python -I -B tests/test_release_learning_smoke.py -v
python -I -B benchmarks/results/2026-10-15-language-serving-gates/verify_serving.py PRIVATE_V8_ARCHIVE NEW_PRIVATE_OUTPUT
```

The final command reads the original `input/request.json`,
`native-checkpoint-128.json` and `browser-checkpoint-128.json` with the hashes in
`qualification.json`. It writes the fixed gate and selection before measurement,
then private per-case receipts and a `serving-input.json` for browser replay.
It neither trains nor overwrites source models. Source files are necessary for
exact historical replay; public synthetic tests need none of those private data.

Build `resident_byte_learning_browser` using the existing
[WASM recipe](../../../docs/byte_corpus_learning.md#browser-execution) and matching
wasm-bindgen 0.2.129. Serve the generated JS/WASM plus `serving-input.json` at
`/target/resident-byte-learning-web/` and the qualification page at
`/crates/st-nn/tests/byte_serving_browser.html` on **loopback only**. Prefer an
explicit route allowlist, not serving the whole private archive. Wait for
`data-status="completed"` and download the receipt. Shut down the server after
qualification. No weights or corpus belong in public benchmark output.

Hashes verify byte identity, not authenticity or historical training. Useful
operating thresholds, independent evaluation cohorts, human selection criteria
and HF integration are follow-up work, not implied by these passing mechanics.
