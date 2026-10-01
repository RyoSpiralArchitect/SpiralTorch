# Resident CE Rounding And Exact Feedback Restoration

This follows the [retained feedback replay failure](../2026-10-01-vision-feedback/README.md),
without rewriting that result or relaxing its comparison. Implementation source:
`c24743dd922988ffddd63380dfae921ed9702afc`.

## Two Actual Fixes

The two first-divergence model probes produced identical logits and gradients
on native Metal and browser WebGPU, but different losses. An intermediate-value
probe found matching exponential and standalone logarithm values; the
`log_one_plus` correction differed. Its `(1 + tail) - 1` rounding boundary could
be erased, invalidating the intended correction. The kernel now reuses the
existing integer-defined f32 addition for the sum and subtraction. It retains
the multiplicative correction and the original tiny-tail early return.

This was also an accuracy defect, not only a replay difference. On 7,513 binary
logit gaps, the public native loss API's maximum per-row relative error against
an f64 `log1p(exp(-gap))` reference fell from approximately `0.999981` to
`3.72e-6`. One small loss was about `1.19e-7` instead of `5.96e-8`. Absolute
errors are small; these figures are not evidence of a training-quality benefit.
None/Sum/Mean gradients are unchanged in this domain. The final Python and WASM
APIs agree on every loss and gradient bit for all three reductions.

An additive-correction candidate is retained as a rejected alternative: it
improved accuracy but still differed across runtimes on 1,039 probe inputs.
The protected multiplicative candidate matched all 7,513. No loss quantization,
feedback threshold change or relaxed comparison was used. This does not promise
bitwise equivalence across every GPU, compiler, shape or input.

Standalone Rust then exposed a separate bug: JSON deserialization changed one
feedback f64 by one ULP at step 2, and a resident trainer could reject its own
checkpoint's hash. `st-core` now enables `serde_json/float_roundtrip` itself,
rather than inheriting that correctness property from a WASM/client build.
The integrity checks remain strict; existing saved histories are not rewritten.
The loss correction changes numerical behavior, so reproducing an old build's
future trajectory still requires that build.

## End-To-End Checks

Both fixed-rate and warmup/cosine cases use 100 attempts, 90 accepted and ten
rejected updates, and a 37/63 restart. Final builds exercise all eight browser
phases again. Native and browser uninterrupted records, resumed records, loss
history, gate state and final checkpoints agree exactly. Python-to-browser and
browser-to-Python continuation both pass, including all 24 tensors / 456 values.
The original failed receipts remain available in the earlier result.

Validation completed with Rust 1.98.0:

- `st-core`: 1,002 passed, including the standalone exact-JSON regression.
- `st-backend-wgpu`: 246 passed with GPU runtime tests enabled.
- `st-vision`: 112 passed / one ignored with WGPU; 78 passed CPU-only.
- Public Python feedback tests: five passed, none skipped, including reverse
  continuation and preservation of old feedback checkpoint histories.
- Fresh native wheel and WASM release builds, generated/shipped TypeScript
  contracts, scoped strict Clippy and nightly formatting passed.

These are separate test configurations, not a count of unique tests. Both
numerical regressions were reproduced before their fixes; those failures and
the failed intermediate standalone-vision run are retained locally.

## Reproduce And Scope

```bash
cargo +1.98.0 test -p st-core --lib json_checkpoint_preserves_every_feedback_float
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test -p st-backend-wgpu \
  --lib resident_tensor::classification::tests -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test -p st-vision \
  --features wgpu --lib resident_trainer -- --test-threads=1
```

Use the [feedback client guide](../../../docs/resident_vision_feedback.md) to
build matching Python/WASM artifacts and run the eight browser phases. Then:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
SPIRALTORCH_VISION_FEEDBACK_BROWSER_DIR="$BROWSER_RECEIPTS" \
SPIRALTORCH_VISION_FEEDBACK_LEGACY_FIXTURE="$PRE_FIX_NATIVE_FIXTURE" \
python -I bindings/st-py/tests/test_vision_trainer_feedback.py -v
```

`summary.json` records all measured conditions, current build hashes and local
receipt hashes, including diagnosis scripts and rejected candidates. Raw runs,
diagnostic source and binaries remain local; only results and validation records
are published. `shasum -a 256 -c SHA256SUMS` verifies this note and summary.

The native adapter identifies Apple M4 / Metal. The browser reports
BrowserWebGpu / Other with no physical model name; none is inferred. This
closes the measured observation/restart boundary, not full producer-state
migration, real-image policy advantage, throughput or memory gates.
