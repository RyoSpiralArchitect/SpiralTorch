# Vision geometry in browser WebGPU: bounded result

This slice starts from `origin/main` at
`e9a01251af9267cfaf05969efac01d75f7dde2bc` and connects the existing
`st-vision` transform planner to an asynchronous browser WebGPU readback.
Consecutive resize, center crop, and seeded horizontal flip stages upload once,
stay on GPU between stages, then return a host image once. Browser callers use
`VisionTransformPipeline.createGpu(seed)`; `createCpu(seed)` runs the same Rust
pipeline without a GPU dispatcher.

The real Chrome 153 browser fixture passed 12 consecutive seeded frames with
3x12x14 input and 3x6x6 output, including a probability-0 and probability-1
flip. The maximum absolute CPU/WebGPU difference was
`5.960464477539063e-8`; invalid crop dimensions were rejected. The actual
Rust runtime reports `BrowserWebGpu` and non-CPU device type. An independent
browser probe reported Apple `metal-3` with a non-fallback adapter; that probe
is not an attestation that the Rust runtime chose the exact same adapter.
Native `st-vision` geometry-sequence tests passed 4/4, backend transform tests
passed 3/3, and the complete WGPU-enabled/no-NN vision suite passed 57/57.
A fresh 0.4.27 wheel installed into an isolated venv passed
the Python 12-frame CPU/WGPU parity test from outside the source tree. WASM
target check and release build passed.

The paired browser wall-time sample uses the same 3x128x128 input, resize to
80x96, seeded flip, and center crop to 64x64. Five warmups precede 20
alternating-order pairs. Median WASM CPU was about 0.10 ms versus browser
WebGPU about 0.60 ms, **including upload and readback**. This is a single-host,
single-shape latency result, not a WebGPU speedup or training-throughput claim.
Raw per-pair timings, outputs, runtime metadata, and hashes are in
[`browser-report.json`](browser-report.json). The initial browser attempt is
preserved as [`browser-initial-failure.json`](browser-initial-failure.json): it
failed solely because the fixture demanded a nonempty adapter name, which
Chrome does not promise. The acceptance check now uses backend and device type.

Replay the Rust and browser checks from this source:

```sh
cargo +1.98.0 test --locked -p st-vision --no-default-features --features wgpu geometry_sequence
cargo +1.98.0 test --locked -p st-backend-wgpu transform::tests
cargo +1.98.0 build --locked --release -p spiraltorch-wasm --features webgpu --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir /tmp/spiraltorch-vision-wasm \
  target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm
NODE_PATH=/path/to/playwright/node_modules node tools/test_resident_browser.cjs \
  /tmp/spiraltorch-vision-wasm /path/to/Chrome /tmp/vision-report.json \
  '' '' '' '' vision-transforms
```

For Python, build a new wheel from `bindings/st-py/Cargo.toml`, install it in
an isolated venv, then run `test_vision_wgpu_pipeline.py` from outside the
source tree so the repository's `spiraltorch` shim cannot shadow the wheel.
`SHA256SUMS` covers the published result and source inputs. The generated WASM
module and wheel are kept locally under
`~/Library/Logs/SpiralTorch/vision-wasm-async-v1/` rather than committed.
The bound WASM module SHA-256 is
`f95b1b302dd2cd83d2d9173da4c27475dc98cf55e8064eb3f5406b5b9c521125`;
the fresh Python wheel SHA-256 is
`0a6b87cbb8866d9067af06983c9ed62e4ad2bc7d0e67b21914535a8f57d8cda5`.

Scope limit: this public browser API intentionally supports geometry-only
pipelines. Normalize, ColorJitter, multi-image batches, and direct GPU-resident
handoff into a model graph remain follow-up work.
