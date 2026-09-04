# Resident WebGPU Matmul

This source-checkout API keeps float32 inputs, outputs, and the pipeline on the
GPU. Rust owns the implementation in `st-backend-wgpu::resident_matmul`; Python
and WASM expose the same workspace, and host Tensor matmul uses the same WGSL.
It is not yet part of the public 0.4.27 wheels.

## Python

Build a WGPU-enabled wheel, then:

```python
from spiraltorch import Tensor, WgpuMatmul

first = WgpuMatmul(2, 3, 2)
first.upload(Tensor(2, 3, [1, 2, 3, 4, 5, 6]),
             Tensor(3, 2, [1, 0, 0, 1, 1, 1]))
first.dispatch()

second = WgpuMatmul(2, 2, 1)
second.upload_rhs(Tensor(2, 1, [1, 1]))
second.set_lhs_from(first)  # GPU-to-GPU copy, no host readback
second.dispatch()
print(second.readback().tolist())  # [[9.0], [21.0]]
print(first.adapter_info())
```

`spiraltorch.wgpu.WgpuMatmul` is the same class. CPU-only wheels expose a
constructor that raises `NotImplementedError`, not a silent CPU substitute.
Python inputs are Tensors; logical row-major conversion occurs before upload.
Native waits release the GIL and have a 30-second host-poll timeout.

## Browser / WASM

Build with the explicit `webgpu` feature and a matching wasm-bindgen CLI:

```bash
cargo build --locked --release -p spiraltorch-wasm \
  --target wasm32-unknown-unknown --features webgpu
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm \
  --target web --out-dir wasm-webgpu
```

Serve the generated module from localhost or HTTPS in a WebGPU-capable browser:

```js
import init, { WgpuMatmul } from "./wasm-webgpu/spiraltorch_wasm.js";
await init();
const model = await WgpuMatmul.create(2, 3, 2);
model.upload(new Float32Array([1, 2, 3, 4, 5, 6]),
             new Float32Array([1, 0, 0, 1, 1, 1]));
model.dispatch();
const pending = model.readback(); // snapshots now, before the await
model.uploadRhs(new Float32Array(6));
model.dispatch();                // does not overwrite pending's snapshot
console.log(await pending);      // Float32Array [4, 5, 10, 11]
await model.synchronize();
model.free();
```

Creation and readback are asynchronous. No blocking host-poll function runs in
the browser. Concurrent creation shares the browser-thread runtime, allowing
`setLhsFrom` to connect workspaces without a device mismatch. Input arrays must
be actual Float32Arrays, including valid cross-realm arrays. Dimensions and
repeat counts are validated before narrowing JS numbers to u32.
wgpu 0.20 does not implement browser queue-completion callbacks. Browser
`synchronize()` therefore enqueues and maps a four-byte completion fence;
this small transfer is included in browser timings, not hidden as kernel time.
The workspace pins a [minimal upstream limit-mapping backport](../vendor/wgpu-0.20.1/SPIRALTORCH_PATCH.md)
for current browsers. It removes the obsolete `maxInterStageShaderComponents`
device requirement without upgrading native WGPU or modifying browser globals.

## Execution Boundaries

- This is a fixed-shape 2D execution workspace, not an autograd graph. It does
  not implicitly register gradients, choose a route, or change training policy.
- Precision is always float32, independent of the host Tensor int8 opt-in.
  The current portable tile is 8x8x16; this is not an autotuned speed claim.
- `upload` validates both lengths before changing either operand. Input changes
  invalidate the output. Reading or copying requires a dispatch of current inputs.
- `dispatch` enqueues work. `output_is_current` / `outputIsCurrent` denotes
  logical freshness, **not completed GPU execution**. Use synchronize/readback.
- `dispatch(repetitions)` accepts 1..1024 and repeats the same multiplication
  into the same output; it does not represent multiple optimizer/training steps.
- A readback owns its staging buffer. It remains independent of later dispatches
  and keeps the device alive even after its workspace is dropped/freed.
- GPU chaining requires compatible matrix shapes and the exact same device and
  queue. No implicit cross-device migration is performed.

## Verification And Benchmarks

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
  cargo test --locked --release -p st-backend-wgpu resident_matmul
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
  python bindings/st-py/tests/test_wgpu_resident.py
python tools/bench_resident_vs_torch.py \
  --torch-device cuda --expected-adapter "NVIDIA GeForce RTX 5090" \
  --output resident-vs-torch.json
node tools/test_resident_browser.cjs /absolute/wasm-webgpu \
  /absolute/chrome-executable /absolute/new-browser-result.json
```

The browser runner needs Playwright and launches an isolated headless test
profile. It does not inspect existing browser sessions. Its
`--enable-unsafe-webgpu` flag is recorded; no CPU computation fallback is used.
The page runs numerical, ownership, and invalid-input gates before timing.

The PyTorch comparison uses identical float32 input bytes and float64 reference
gates. Both sides preallocate output and run the same number of Python-loop
dispatch calls, followed by their own GPU synchronization. Allocation,
upload, and readback are excluded. Results retain adapter/module hashes and
raw samples. Timings still include API, queue submission, and host wait costs;
they are **not kernel timestamp measurements**. Browser timings are a separate
boundary and must not be divided by the Furnace PyTorch timings.
CUDA runs require exactly one visible CUDA device and matching expected adapter
names; this is not portable cross-API UUID attestation. Browser adapter metadata
comes from a separate browser probe: wgpu 0.20 itself reports only the
`BrowserWebGpu` backend, not an exact adapter identity. Do not present that probe
as the Rust device's identity.
