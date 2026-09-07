# Existing NN Modules, Resident WebGPU Execution

`st_nn::resident::InferencePlan` lowers existing `Linear`, `Gelu`, and nested
`Sequential` modules into shared WGPU dense kernels. This is an explicit Rust
inference path with Python/WASM clients, not a second model definition API or a change to ordinary
`Module::forward` or backward.

```text
Sequential(Linear -> Gelu -> Linear)
                  |
         frozen InferencePlan
                  |
one input upload -> GPU [matmul+bias+GELU -> matmul+bias]
                  |
         one final output snapshot
```

Weights and activation buffers stay on the device. A producer's output buffer is
the next stage's input buffer: there is no intermediate host readback or
device-to-device copy. One command submission contains all stages, with a
compute pass per stage. Requesting a snapshot adds a copy submission containing
the final output and per-stage validation flags. These are structural transfer
boundaries, not hardware-counter measurements.

## Rust Use

Enable `st-nn/wgpu` and use the existing model:

```rust
use st_backend_wgpu::runtime;
use st_nn::{layers::Gelu, resident::InferencePlan, Linear, Sequential};
use st_tensor::NdLayout;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut model = Sequential::new();
model.push(Linear::new("up", 4, 7)?);
model.push(Gelu::new());
model.push(Linear::new("down", 7, 3)?);

let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[2, 5, 4])?)?;
let (runtime, _) = runtime::ensure_default_runtime_blocking("my.inference")?;
let mut gpu = plan.compile_wgpu(runtime)?;
gpu.upload(&[0.25; 2 * 5 * 4])?;
gpu.dispatch()?;
let snapshot = gpu.snapshot()?;
assert_eq!(snapshot.layout().shape(), &[2, 5, 3]);
let output = snapshot.read()?;
assert_eq!(output.len(), 2 * 5 * 3);
Ok(())
}
```

On WASM, request the runtime asynchronously with
`WgpuRuntime::request_headless(...).await` and consume the same snapshot with
`read_async().await`. The execution/lowering semantics remain Rust-owned.
`compile_wgpu_with_options` accepts the shared matmul tile, scalar/register-2x2
kernel, and sequential/tiled/compensated accumulation options; the default is
8x8x16, scalar, sequential. This is not an autotuned fastest-kernel promise.

## Python To Browser

Use a wheel built from this source with the `nn` and `wgpu` features (the
default build includes both). Define the model with existing modules:

```python
import spiraltorch as st

model = st.nn.Sequential()
model.add(st.nn.Linear(4, 7, name="up"))
model.add(st.nn.Gelu())
model.add(st.nn.Linear(7, 3, name="down"))

plan = model.inference_plan([2, 5, 4])
gpu = plan.compile_wgpu()
gpu.upload_values([0.25] * 40)
gpu.dispatch()
snapshot = gpu.snapshot()
assert snapshot.shape == (2, 5, 3)
values = snapshot.read_values()  # the only output readback; validates every stage
payload = plan.to_json()        # pass these exact bytes to the browser
```

For a 2D plan, `gpu(st.Tensor(...))` is a host-input/host-output convenience
call with the same resident intermediate buffers. N-D plans deliberately use
flat values plus snapshot shape, not a silently flattened 2D return value.
`upload(Tensor)` accepts the exact leading-axes matrix in logical row-major
order. Snapshot reads consume the snapshot once.

Build the public WASM package with `bash scripts/build_wasm_web.sh --features
webgpu`, then use its generated web module in a WebGPU-enabled secure context:

```javascript
import init, { InferencePlan } from "./pkg/spiraltorch_wasm.js";
await init();
const payload = await (await fetch("./plan.json")).text(); // Python's plan.to_json()
const plan = InferencePlan.fromJson(payload);
const pending = plan.compileWebGpu();
plan.free(); // compilation already owns a Rust snapshot
const gpu = await pending;
gpu.upload(new Float32Array(40).fill(0.25));
gpu.dispatch();
const snapshot = gpu.snapshot();
const shape = snapshot.shape; // Uint32Array([2, 5, 3])
gpu.free();                  // the output snapshot owns its buffers
const read = snapshot.readValues();
snapshot.free();             // the pending read also owns its buffers
const values = await read;
```

Rust `InferencePlan::{to_json, from_json}`, Python `nn.InferencePlan.from_json`,
and WASM `InferencePlan.fromJson` share the same Rust parser and lowering.
The wire schema is `spiraltorch.nn.inference_plan.v1`: contiguous input shape
and fixed f32 Linear parameters with optional GELU. It is not a checkpoint,
executable code, or a device image. The default import budget is 64 MiB of
UTF-8 JSON; callers may explicitly raise `max_bytes`. It bounds transport
size, not total parsing or GPU allocation memory. The portable format requires
u32-addressable input, parameter, and stage buffers on every platform;
device-specific limits are checked separately during compilation. Unknown
fields, non-finite parameters, wrong shapes, and unknown schemas are rejected.

A CPU-only Python build can create, export, and import plans, but
`compile_wgpu()` raises `NotImplementedError`. WASM built with `--features nn`
similarly handles plans while `compileWebGpu()` explicitly fails; `webgpu`
includes `nn`. Neither client reconstructs NN operations or inserts a CPU
fallback when resident compilation is unavailable.

## Boundaries

- Plans own immutable parameter snapshots. Rebuild the plan and recompile after model updates;
  changing the source model does not update an existing resident plan.
- `dispatch()` means enqueued, not validated success. Reading a snapshot checks
  every stage's finite-value flags, including GELU square/cubic overflow that a
  final saturated output could otherwise hide. An earlier failure is retained
  even if a later stage masks it. Errors identify the stage and flag mask.
- Snapshots own their outputs and flags across re-upload, redispatch, and
  workspace destruction. Invalid host uploads leave the previous state intact.
- Unknown modules and unsupported fusion sequences fail explicitly. No CPU
  fallback is inserted into a compiled chain.
- `NdLayout` provides checked element-stride `permute`, `narrow`, `reshape`, and
  storage indexing. This executor accepts nonempty contiguous inputs at offset
  zero and applies Linear to the last axis. Other views are rejected, not copied
  or silently reinterpreted. Scalars/empty arrays have layout semantics but are
  not supported by this inference executor.
- Existing `pure::Tensor` is still host-backed and two-dimensional. This does
  not add general N-D broadcasting or device-resident autograd/optimizer updates.
  The public clients expose this explicit inference subset, not an automatic
  replacement for all NN execution.

## Verification And Measurement

`bindings/st-py/examples/resident_nn_roundtrip.py` executes the existing Python
model and its resident plan against an independent small f64 oracle, then
exports three 1D/2D/3D fixtures. The browser test imports those exact plan bytes:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I bindings/st-py/tests/test_nn_resident.py -v
python -I bindings/st-py/examples/resident_nn_roundtrip.py --output /tmp/nn-fixture.json
OUT_DIR=/tmp/nn-client-webgpu EXAMPLES_DIR=/tmp/no-example-sync \
  bash scripts/build_wasm_web.sh --features webgpu
cp /tmp/nn-fixture.json /tmp/nn-client-webgpu/nn-fixture.json
node tools/test_resident_browser.cjs /tmp/nn-client-webgpu /path/to/chromium \
  /tmp/new-nn-client-report.json '' '' '' '' nn-clients
```

The runner requires Playwright in the Node environment and launches an isolated
headless browser, not the user's profile. For a plan-only WASM build, use
`--features nn` and the `nn-clients-cpu` fixture. CPU-only Python tests omit
the GPU opt-in environment variable.

```bash
cargo test -p st-kernel-contracts
cargo test -p st-nn --lib --no-default-features
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test -p st-nn --features wgpu \
  --test resident_inference -- --test-threads=1
cargo check -p st-nn --no-default-features --features wgpu \
  --target wasm32-unknown-unknown
```

For browser correctness, build the dedicated cdylib example and process it with
a `wasm-bindgen` CLI matching Cargo.lock. The fixture runs the real Rust model
and compares against WASM CPU output, including 32 source operations, owned
snapshots, and overflow guards. It does not time browser execution.

```bash
cargo build -p st-nn --example resident_browser --release --features wgpu \
  --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir /tmp/nn-browser --out-name spiraltorch_wasm \
  target/wasm32-unknown-unknown/release/examples/resident_browser.wasm
node tools/test_resident_browser.cjs /tmp/nn-browser /path/to/chromium \
  /tmp/new-nn-browser-report.json '' '' '' '' nn
```

The native comparison requires a clean committed source tree and a binary whose
embedded build identity matches it. `resident_mlp` also accepts JSONL requests
such as `{"shape":[2,8,64],"depth":16,"seed":17}` for direct diagnostics.

```bash
cargo build --release -p st-nn --features wgpu --example resident_mlp
python tools/bench_resident_nn_vs_torch.py \
  --executable target/release/examples/resident_mlp --device cuda \
  --output /tmp/new-nn-comparison.json
```

`--device mps` admits a single Apple GPU on macOS, with CPU fallback disabled.
It records contention as **unknown**, not uncontended. CUDA rejects observed
foreign compute processes. Both use nine fixtures over three seeds, two warmups
and twelve retained host-input-to-host-output samples, with setup excluded and
output parity checked outside timing. Native eager/resident controls alternate;
PyTorch is measured afterward. Rust retains intermediate finite guards while
the eager PyTorch baseline does not add them for these finite fixtures. These
results are diagnostic inference comparisons, not fastest-PyTorch or training
throughput claims.

The [first retained study](../benchmarks/results/2026-09-07-resident-nn/README.md)
contains three complete MPS comparisons and browser correctness evidence,
including slower cases and the still-unmeasured CUDA boundary.
