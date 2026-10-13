# Causal Wave Batch-Gradient Retention

This increment fixes premature rounding of shared decay/phase gradients in the
pure Rust contract and resident WGPU wave used by geometric byte-decoder QKV
learning. It does **not** resolve the existing projective trajectory failure.

## Result

Before the fix, both CPU and WGPU produced `0.25` rather than the analytic f32
`0.2475000023841858` for the one-step decay contributions `[2^24, 1, -2^24]`.
Both also rejected opposite, individually overflowing parameter partials even
when the final parameter gradient was zero and every public VJP was finite.
Both new regression tests failed before implementation (exit 101); these logs
are retained and hashed. This is a reproduced defect, not a hypothetical gain.

The pure contract now uses compensated f64 batch sums; WGPU retains its existing
extended `Wide` representation through the batch reduction. Final f32 outputs
are still checked. Same-sign final overflow and upstream invalidity still
reject the whole gradient family. The forward recurrence and public API are
unchanged. See the [implementation contract](../../../docs/causal_wave_batch_gradients.md).

| Check | Native | Browser WASM/WebGPU |
| --- | --- | --- |
| Analytic decay/phase, exponents 24/48/96, all six permutations | 36/36 bit-exact | 36/36 bit-exact |
| Opposing partial overflow, all public VJPs finite | 2/2 | 2/2 |
| Same-sign final overflow rejects all four VJPs | 2/2 | 2/2 |
| Frozen Torch primitive cases / radial chart controls | 10/10 + 9/9 | 10/10 + 9/9 |
| Wave parameter updates and streaming/guard controls | 16 updates, passed | 16 updates, passed |
| Original six-case QKV qualification | **4/6, still failed** | **4/6, still failed** |
| Same-runtime paired query controls | 2/2 | 2/2 |
| QKV full-mode exact checkpoint continuation | 6/6 | 6/6 |

The original two-block projective phase gate at step 12 remains above `0.002`:

| Runtime | Before | After |
| --- | ---: | ---: |
| Native | 0.0033325099062151368 | 0.0033326298509409537 |
| Browser | 0.003370400643695823 | 0.00336997867247487 |

The slight native increase is retained, not rounded away or relabeled a pass.
The separate legacy mixed-runtime query gate also still fails. No previously
passing case or gate failed. All six initial forwards and every recorded
post-update parameter tensor remain unchanged in these specific runs. Thus this
fix recovers signals in the analytic cancellation cases but does not establish
an improvement in the six small models, let alone language quality.

## Cost And Evidence

The change adds `12*B*C` bytes of logical temporary GPU payload: extended
scratch grows by `16*B*C`, and old f32 partial storage shrinks by `4*B*C`.
Storage-limit preflight covers the larger allocation. Binding and dispatch
counts are unchanged; no shader-f64 capability or CPU fallback is added.
No throughput measurement or speed improvement is claimed.

- [Frozen protocol](protocol.json) predates the baseline regression runs.
- [All scalar results](results.json) retain every analytic case, wave learning
  step, QKV loss/parameter-delta step, old/current failure and paired control.
- [Validation](validation.json) records source, raw-result, binary and log
  hashes, terminal exit codes, and test scope.
- [Independent review](review.md) states the read-only review scope and limits.
- [Checksums](SHA256SUMS) cover the public evidence files.

The first native integration build failed because a nested support module used
an implicit path; it was corrected to an explicit path and the rerun passed.
That failed build log is preserved. It is separate from the two genuine
pre-fix numerical regressions.

Rust tests: pure wave 9, WGPU wave 6, native primitive integration 1, and resident
byte-decoder 32. The publication validators passed 18 Python tests (11 new,
7 existing), and the actual browser batch checker passed 6 Node tests, including
both retained runtime receipts.
All GPU tests ran with the runtime opt-in, not skipped. Native adapter was
Apple M4 / Metal. Raw model tensors, checkpoints, executables and screenshots
remain local; public output is scalar measurements, hashes and this recipe.

The 36 cases are not a guarantee for arbitrary exponent spans, ordering,
sequence lengths or ill-conditioned inputs. `Wide` is not exact arithmetic.
No large-combination qualification, corpus/LLM FT benefit, CUDA result, release,
PR or merge is claimed here. Earlier rejected evidence is not superseded.

## Reproduce

Use a fresh private `RAW` directory. `PROJECTIVE_BASELINE` holds the original
projective-v1 final `reference.json`; `PAIRED_BASELINE` holds the previous paired
run's `native-legacy.json`, `browser.json` and `diagnostic/stopped-reference.json`.
The [previous recipe](../2026-10-13-query-pullback-paired-v1/README.md#reproduce)
describes recreating those inputs. The QKV fixture SHA-256 must remain
`07b87441cd15ff7d2478cf87dd460c21d58b1118ee78509992ae231014795358`.
Use Rust 1.97.0 and wasm-bindgen 0.2.129 with the repository lockfile. The frozen
wave Torch fixture is checked into the repository, SHA-256
`6d999fa45310ada0514624a003712c699f5d81418b947e8722013cd038ece721`.

```sh
export CARGO_INCREMENTAL=0
export SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1
cargo test --locked -p st-kernel-contracts causal_wave:: --lib -- --test-threads=1
cargo test --locked -p st-backend-wgpu resident_tensor::causal_wave:: --lib -- --test-threads=1
cargo test --locked -p st-nn --no-default-features --features wgpu \
  resident::byte_decoder:: --lib -- --test-threads=1
cargo test --locked -p st-backend-wgpu --test causal_zspace_wave \
  -- --nocapture --test-threads=1 > "$RAW/native-wave.log" 2>&1
node -e 'const fs=require("fs");const line=fs.readFileSync(process.argv[1],"utf8").split("\n").find(s=>s.startsWith("{")&&s.includes("spiraltorch.causal_zspace_wave.validation.v1"));if(!line)throw Error("missing report");fs.writeFileSync(process.argv[2],JSON.stringify(JSON.parse(line),null,2)+"\n",{flag:"wx"});' \
  "$RAW/native-wave.log" "$RAW/native-wave.json"
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric
"$CARGO_TARGET_DIR/release/examples/resident_byte_flat_metric" \
  "$PROJECTIVE_BASELINE/reference.json" --projective > "$RAW/native-projective.json"
"$CARGO_TARGET_DIR/release/examples/resident_byte_flat_metric" \
  "$PROJECTIVE_BASELINE/reference.json" --query-pullback > "$RAW/native-paired.json"
```

The `--projective` command intentionally exits 1. Keep that failure and its
report; do not let a shell wrapper relabel it. Set `CARGO_TARGET_DIR` explicitly
or substitute the normal `target` directory above. Run builds/GPU tests serially.

```sh
cargo build --locked --release -p st-backend-wgpu --target wasm32-unknown-unknown \
  --example causal_zspace_wave_browser
wasm-bindgen "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/examples/causal_zspace_wave_browser.wasm" \
  --target web --out-dir "$RAW/wave-web"
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --target wasm32-unknown-unknown --example resident_byte_decoder_browser
wasm-bindgen "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/examples/resident_byte_decoder_browser.wasm" \
  --target web --out-dir "$RAW/lm-web"
```

On loopback only, expose the wave JS/WASM as `target/causal-zspace-wave-web/`,
and the decoder JS/WASM plus frozen `reference.json` as
`target/resident-byte-query-pullback-web/`. Serve the corresponding HTML pages
`crates/st-backend-wgpu/tests/causal_zspace_wave_browser.html` and
`crates/st-nn/tests/byte_query_pullback_browser.html`. Run once and retain their
downloads as `$RAW/browser-wave.json` and `$RAW/browser-lm.json`. Do not expose
the entire private artifact directory. The decoder page intentionally displays
the original qualification failure separately from the paired pass.

```sh
python3 -I -S -B -c 'import sys,unittest;sys.path.insert(0,"tools");r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(["test_validate_causal_wave_batch_results","test_validate_resident_query_pullback"]));raise SystemExit(not r.wasSuccessful())'
ST_WAVE_NATIVE_REPORT="$RAW/native-wave.json" ST_WAVE_BROWSER_REPORT="$RAW/browser-wave.json" \
  node --test tools/test_causal_wave_batch_browser.cjs
python3 -I -S -B -c 'import sys,runpy;sys.path.insert(0,"tools");sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/validate_causal_wave_batch_results.py "$PAIRED_BASELINE" \
  "$PROJECTIVE_BASELINE" "$RAW" "$RAW/scalar-results.json"
```

The publisher uses explicit scalar fields, independently checks the 36 analytic
results and paired gradients, and refuses new QKV gate failures, forward drift
or incomplete trajectories. Existing QKV gates are still evaluated in the Rust
harness; this publication step does not replace or relax them. Its output path
must be fresh so previous evidence cannot be overwritten.
