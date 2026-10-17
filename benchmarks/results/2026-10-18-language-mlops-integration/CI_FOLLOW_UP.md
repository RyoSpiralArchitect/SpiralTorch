# Language MLOps CI Follow-Up

Base source: `37fd32273a74347ccbfcfa141e5296e0b6bfaf74`. The
[validation record](results/ci-follow-up.json) pins the changed source files and
retained local logs/results. This is a build/client-boundary repair, not a new
language-quality, long-run numerical-equivalence or performance claim.

## Repair

[CI run 39065331035](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/39065331035)
failed the WASM all-target backend check because the new `language_local_vjp`
native diagnostic called synchronous readback, which is unavailable on WASM.
Both it and `resident_byte_gradient_probe` now explicitly compile their native
implementation only for non-WASM targets. Their WASM entrypoints fail with an
explanatory message rather than pretending that a diagnostic ran. Packaged
browser training/inference exports, math and acceptance thresholds are unchanged.

CI now generates a small public synthetic study/checkpoint/serving-gate fixture,
with no corpus download, private model or Python package dependency. It tests
the actual `nn` and `webgpu` WASM packages via generated Node glue. The shared
runner accepts both web and Node glue. Cursor and foreign-checkpoint assertions
require the intended validation errors, so an unrelated device-unavailable
failure can no longer pass those checks.

## Local Verification

Rust 1.99.0, wasm-bindgen 0.2.129 and Node 24.14.0:

| Check | Result |
| --- | --- |
| Backend WASM `--all-targets` Clippy with `-D warnings` | Pass |
| Native backend diagnostic Clippy with `-D warnings` | Pass |
| NN diagnostic `cargo check`, native and WASM | Pass |
| Generated Node glue, `nn` / `webgpu` transport checks | 20 / 30 pass |
| Generated web glue under Node, same transport checks | 20 / 30 pass |
| Injected device-error negative controls | All 3 correctly rejected |
| Native local VJP diagnostic | All 4 fixture cases execute |
| Native saved-model gradient diagnostic | 3 models x 4 batches; checkpoints unchanged |

The VJP replay preserves the three earlier denominator witnesses and the later
wide-rounding midpoint repair (`1 + 2^-23`); it is not byte-identical to the
older pre-rounding-repair report. The saved-model probe is read-only, not another
1,024-update training run. Node transport tests do not execute GPU training.

Additional strict `st-nn` Clippy checks **fail with 24 existing library lints**
in both native and WASM configurations. The reported files are unchanged from
the AdamW parent `b21b3aacd3be160072302c81cc778b2ccb3a97ff`. They were not
suppressed or represented as passes. Vendor and future-incompatibility warnings
also remain. These extra checks differ from the failing backend CI command.

Independent read-only reviews found no introduced actionable issue in the
native target guards, fixture, workflow or generated-glue runner. Review also
prompted the stricter rejection-reason checks; re-review confirmed their source
patterns and rejected generic device-error substitutions. Reviews did not run
GPU work; the runtime checks above were performed separately.

Two stale local debug fingerprints reused older parent-worktree artifacts that
lacked `f32_sum`. Only those small generated fingerprint directories were
retired to private validation storage before successful rebuilds; no source,
weights, evidence or broad build caches were deleted. Use worktree-specific
build caches for different source snapshots.

## Reproduce

From the repository root, use fresh output directories and the matching
wasm-bindgen CLI. Run the `nn` block first, then repeat it with `features=webgpu`.

```sh
cargo +1.99.0 clippy --locked -p st-backend-wgpu \
  --target wasm32-unknown-unknown --all-targets -- -D warnings
cargo +1.99.0 clippy --locked -p st-backend-wgpu \
  --example language_local_vjp -- -D warnings
cargo +1.99.0 check --locked -p st-nn --no-default-features --features wgpu \
  --example resident_byte_gradient_probe
cargo +1.99.0 check --locked -p st-nn --no-default-features --features wgpu \
  --target wasm32-unknown-unknown --example resident_byte_gradient_probe
python3 -I -S -B bindings/st-wasm/tests/byte_lm_fixture.py > "$PRIVATE/transport.json"
features=nn
cargo +1.99.0 build --locked -p spiraltorch-wasm --release \
  --target wasm32-unknown-unknown --features "$features"
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm \
  --target nodejs --out-dir "$PRIVATE/node-$features"
node bindings/st-wasm/tests/byte_lm_node.mjs \
  "$PRIVATE/node-$features/spiraltorch_wasm.js" "$PRIVATE/transport.json" "$features"
```

For negative controls, temporarily replace `advanceWebGpu` (all calls, then only
the beyond-total call) and `resumeWebGpu` in the generated class prototype with
a function throwing an unrelated device error. Each `cpuChecks` invocation must
fail with `wrong rejection for` the corresponding assertion; restore the method
after each check. No actual GPU request is made by these controls.

Native diagnostics reuse the private fixtures/checkpoints from the
[VJP study](../2026-10-18-language-vjp-isolation/README.md) and the
[wide-rounding follow-up](../2026-10-18-wide-rounding/README.md). Their raw model
data and full outputs remain private. Remote exact-head CI and merge status
remain separate from this local record. No package or model is promoted here.
