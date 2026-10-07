# Captured Topos Learning In The Browser

The existing browser page now trains through `kernel.capture` and `batch.vjp`
instead of recomputing the recurrence in `kernel.backward`. A shared JavaScript
contract runs in the rendered page and in Node CI; mathematics and derivatives
remain in Rust. Batches and pullbacks are explicitly freed. Stateless execution
remains an independent comparison path inside this test, not the training path.

## Observed Results

Actual Codex in-app browser, desktop 1280x720, scalar `rust_f32_wasm`:

- Forward, input-VJP and gate-VJP maximum differences from the current Python
  native fixture are all **0**. Finite-difference maximum error is
  `0.000012156062425106029` within the declared test tolerance.
- The 100-update captured gate trajectory matches stateless learning exactly.
  Synthetic MSE moves from `0.4857680486936758` to `0.003528845506002011`.
  This is a four-element gate-fitting wiring test, not language-model quality.
- JSON round-tripped gate state produces the exact same next update. This is
  saved-gate continuation with fixed configuration, not a full optimizer/model
  checkpoint or resume guarantee.
- Seventy rejection checks pass, along with empty capture, repeated VJPs,
  source/returned-output mutation and a tape surviving kernel disposal.
- Browser, Node ESM and Node CommonJS results match in every result field.
  CommonJS is exercised by the real WASM CI job. Node execution is explicitly
  labeled separately from the rendered browser observation.

No WebGPU, speed, mobile layout, broad browser compatibility or pretrained FT
claim is made. Only `native_fixture()` from the existing Python example was
called; no HF model was instantiated, trained or scored. No previous study,
runtime, weight or result bundle was rewritten or removed.

## UI And Negative Checks

The expected page/title rendered real JSON results, without a framework error
overlay. Final browser console error/warning capture was empty. Reloading after
restoring the fixture returned the page to `passed` with the same results.
Screenshots and full local observations are retained privately and hashed here.

Two initial issues remain recorded rather than discarded:

1. The first staging attempt copied generated JS/WASM but omitted its generated
   `snippets` directory. The page displayed an import error; staging the complete
   module fixed it without a runtime change.
2. Removing the fixture initially left a cached successful result; an explicit
   error-selector wait timed out. The page now fetches fixtures with
   `cache: "no-store"`. A missing fixture then renders HTTP 404 as an error,
   and restoring it permits a clean successful reload. Dynamic imports and
   fetch failures are caught, not left as a permanent `running` page.

The Node runner was also given an intentionally incorrect native output. It
wrote an error receipt and exited **1**, rather than accepting the bad fixture.

## Reproduction And Identity

Client source: `9ea9a7c8386094cc19798c17a704897706d759a8`.
Runtime source: `8e1071e1e1caf1bfcb4be6dd794e601ff3eadde0`.
The tracked fixture records the native library hash; results bind source files,
all served assets including generated snippets, both WASM module formats and
local evidence. Hash consistency is not independent execution provenance.

After generating a complete `--target web` module using wasm-bindgen 0.2.104:

```bash
mkdir -p "$WEBROOT/module"
cp -R "$MODULE_DIR/." "$WEBROOT/module/"
cp bindings/st-wasm/tests/topos_resonator_learning.{html,mjs} "$WEBROOT/"
cp bindings/st-wasm/tests/topos_resonator_learning_fixture.json "$WEBROOT/learning.json"
python -I -S -B -m http.server 8000 --bind 127.0.0.1 --directory "$WEBROOT"
```

Open `http://127.0.0.1:8000/topos_resonator_learning.html`. Use a new staging
directory, not an existing frozen runtime. The CLI accepts either a complete
web module or a generated Node module:

```bash
node tools/probe_topos_browser_learning.mjs "$MODULE_DIR" \
  bindings/st-wasm/tests/topos_resonator_learning_fixture.json "$NEW_REPORT"
python -I -B -m pytest --import-mode=importlib --confcutdir=tools --rootdir=tools \
  -q tools/test_topos_browser_learning_results.py
```

The publication tests check closed hashes and numerical receipt consistency.
They do not substitute for executing the browser or WASM tests.
