# Vision-to-NN GPU-resident handoff: bounded result

This slice starts from `origin/main` at `68d87b4b99c57b5323bea6ce098c74cf45c3315f`.
It adds a Rust-owned handoff from `st-vision` geometry transforms to the
existing guarded `ResidentTensor`. The final transform buffer is adopted
without a host readback or GPU value copy; a GPU finite-value scan produces
the tensor validity flag before downstream operations. A device mismatch,
unsupported transform, invalid crop, or non-finite host input fails explicitly.
The seeded flip state advances only after a successful submission. CPU-visible
`apply` and browser async `apply` remain unchanged.

The real Chrome 153.0.8010.54 fixture passed 12-frame CPU/WebGPU parity and
an independent geometry-to-`Sequential(Scaler)` resident forward. The model
output matched the CPU reference exactly in the 1x8x8 to 1x6x6 handoff case.
The terminal NN output was the first readback; the input reshape shared GPU
storage. Invalid-crop retry and non-finite input rejection passed. Native
Rust tests exercised the same handoff and GPU guard propagation. A newly
built 0.4.27 wheel, installed into an isolated venv and tested from outside
the repository, passed the three Python vision WGPU tests including an actual
resident NN forward.

The paired browser comparison used a 3x128x128 CHW image, resize to 80x96,
seeded flip, crop to 64x64, then `Sequential(Scaler x1.1)`. Both arms used
WebGPU and the same Rust NN model. The baseline read the transformed image
back to the host, reuploaded it, then read the NN output; the resident arm
read only the NN output. Five warmups preceded 20 alternating-order pairs.
The browser wall-time medians, including initial upload and final readback,
were **1.0 ms round-trip** and **0.7 ms resident** in the final build. An
earlier build before the strict external-buffer-size check measured 0.9 ms
and 0.6 ms respectively; its complete report is preserved as
[`browser-report-initial.json`](browser-report-initial.json). Both runs' terminal
values matched exactly. This is a single-host, single-shape result, not a general
WebGPU speedup, training-throughput result, or PyTorch comparison. The
browser's non-fallback Apple adapter probe is separate from the Rust runtime
metadata (`BrowserWebGpu`, device type `Other`); it does not attest to an
identical adapter identity. Raw per-pair values and hashes are in
[`browser-report.json`](browser-report.json).

Replay from this source:

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-backend-wgpu checked_import
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked -p st-backend-wgpu resident_geometry_handoff
cargo +1.98.0 test --locked -p st-vision --no-default-features --features wgpu geometry_resident_output
wasm-pack build bindings/st-wasm --target web --out-dir /tmp/spiraltorch-vision-resident --release -- --features webgpu
NODE_PATH=/path/to/playwright/node_modules node tools/test_resident_browser.cjs \
  /tmp/spiraltorch-vision-resident /path/to/Chrome /tmp/vision-resident-report.json \
  '' '' '' '' vision-transforms
```

For Python, build a new WGPU-enabled wheel from
`bindings/st-py/Cargo.toml`, install it into an isolated venv, then run
`test_vision_wgpu_pipeline.py` from outside the repository so the source
package cannot shadow the wheel. The wheel and generated WASM package stay
local under `~/Library/Logs/SpiralTorch/vision-resident-handoff-v1/` and are
not committed. The final generated WASM module SHA-256 is
`80fe08c8caa38bedcff775a2676eb3ae9e851eea2c1dccdc58844b305792a0b5`.
The final local wheel SHA-256 is
`af5a7d296a5f7b7bf52f8b15efe3c23c42e59fa42cc4f1260f8f0c631a87aac7`.
`SHA256SUMS` binds the published report and source inputs.

Scope limit: this interface currently accepts one CHW image, geometry-only
transforms, and a downstream resident inference graph. Batched images,
Normalize/ColorJitter, end-to-end training, and a shape crossover sweep remain
open work.
