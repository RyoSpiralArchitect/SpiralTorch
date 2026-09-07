# Existing NN Modules, Resident WebGPU Execution

`st_nn::resident::InferencePlan` lowers existing `Linear`, `Gelu`, and nested
`Sequential` modules into shared WGPU dense kernels. This is an explicit Rust
inference path, not a second model definition API and not a change to ordinary
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

## Boundaries

- Plans own immutable parameter snapshots. Recompile after model updates;
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
  not add general N-D broadcasting, device-resident autograd/optimizer updates,
  or a Python/JS public NN compilation API. Rust browser execution is tested by
  the example below; general client bindings are a subsequent step.

## Verification And Measurement

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
