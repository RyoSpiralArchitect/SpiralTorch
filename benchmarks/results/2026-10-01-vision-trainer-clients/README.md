# Unified Vision Trainer: Native And Browser Restart

The Rust model/input/schedule owner is now reachable through the public Python
wheel and actual WASM/WebGPU clients. This is bounded synthetic correctness
evidence, not throughput, real-image accuracy, or a Z-space-policy advantage.

## Measured Conditions

All cases use 20 synthetic 3x4x4 images, batch 2, a two-stage ConvNeXt classifier
with 24 parameter tensors / 456 values, seeded shuffle and horizontal flips,
normalization, and one deliberately invalid class ID. Both constant SGD and
warmup/cosine complete 100 attempts: 90 accepted and ten rejected. Rejections
consume input but do not advance the accepted-update scheduler.

| Route, repeated for both schedules | Comparison | Result |
| --- | --- | --- |
| Native Python process restart | 100 vs 37 + process exit + fresh process + 63 | Exact input/rate/acceptance records and complete final checkpoint |
| Browser restart | 100 vs 37 + tab closure + fresh document/WASM instance + 63 | Exact input/rate/acceptance records and complete final checkpoint |
| Python checkpoint to browser | Native prefix 37, browser continuation 63 vs native continuation | Exact input/rate/clock state; all 456 weights have zero observed difference |
| Browser checkpoint to Python | Browser prefix 37, native continuation 63 vs browser continuation | Exact input/rate/clock state; all 456 weights have zero observed difference |

The browser executed in Codex's in-app Chromium through `BrowserWebGpu`.
The adapter API reported `device_type: Other` with an empty name; do not infer
a publicly identified browser adapter or cross-device equivalence. Native
execution used the Apple M4 Metal adapter. Cross-runtime weight acceptance was
specified as `abs(actual-expected)/(1+abs(expected)) <= 2e-4`, not a universal
bitwise guarantee. The observed zero difference is specific to these cases.

The native processes exit normally and the browser test recreates its document;
neither establishes sudden-power-loss durability or an OS-browser crash recovery
guarantee. The browser dataset is in-memory; this is not a streaming JavaScript
DataLoader implementation.

## A Failure That Changed The Backend

The first scheduled cross-runtime continuation failed at attempt 69 (accepted
schedule step 62). Platform `f32::cos` produced learning-rate bits `978780404`
on native versus `978780405` on wasm32. Other input/acceptance fields matched.
`cosine-pre-fix.json` preserves this failed observation and the original artifact
hashes. The failed run is not relabeled as passing.

`st-nn` now uses the same pinned Rust `libm::cosf` on both targets. The formula
and acceptance clock are unchanged; no Python/JS arithmetic or relaxed rate-bit
tolerance was introduced. A golden Rust regression checks this precise case.
Fresh wheel/WASM builds then repeated all eight browser phases and both native
restart trajectories successfully. Old checkpoints still parse, but historical
native rate rounding requires the historical build for bitwise reproduction.

## Evidence And Replay

`summary.json` contains every condition, checkpoint hashes, original receipt
hashes, Python fixture/wheel and actually served WASM/JS asset hashes. Its
`source_capture_commit` captures the measured source content; the build was
performed before committing that content. Full synthetic fixtures, checkpoints,
browser receipts, failed-run receipts, build outputs and logs are retained locally,
not duplicated into the public repository.

Follow the [public client guide](../../../docs/resident_vision_trainer_clients.md)
to generate a fresh Python fixture, serve the real WASM module, close/reopen the
browser prefix document, and exercise both handoff directions. Reduce retained
receipts without ML dependencies:

```bash
node tools/summarize_vision_trainer_clients.cjs PYTHON_FIXTURE BROWSER_DIR \
  REVERSE_REPORT BUILT_WHEEL NEW_SUMMARY_JSON
```

Verification on the measured source: native `st-nn` 734 tests; WGPU `st-vision`
104 passed with one pre-existing manual diagnostic ignored; CPU-only
`st-vision` 54; fresh-wheel vision tests 20; generated and shipped TypeScript
contracts; scoped `st-vision` strict Clippy; nightly formatting; and the Python
quick-start example. GPU and browser runs are explicit local checks, not implied
by the CPU-only Python CI surface test.
