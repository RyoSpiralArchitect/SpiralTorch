# GELU Backward: Live Outputs, Not Discarded Work

The ordinary Tensor/Module/autograd path previously called the full fused GELU
backward helper and discarded its residual and bias gradients after computing
and reading all three outputs. It now uses a backend-owned derivative-only
pipeline, with one output and no residual/bias buffers or reduction pass.
The three-output helper remains available and batches its terminal readback.
CPU routing, finite-value rejection, layout conversion and fallback policy
stay at the existing Tensor boundary.

## Shared Implementation

The backend owns the canonical saturated tanh-GELU derivative, embedded
plain/fused shaders, 32/16-byte fused uniforms, bindings and validated geometry.
Tensor pipelines still compile lazily. Resident VJP/training use the same WGSL
derivative source without changing their deferred guard machinery.

The old filesystem loader failed to compile its override-sized workgroup array
on WGPU 0.20. Specializations now become host-selected constants, on both the
filesystem and embedded paths. Native and browser callers can encode the
embedded pipelines without filesystem access, implicit submissions or readback:

```rust,ignore
let geometry = st_backend_wgpu::gelu_back::Geometry::default();
let pipelines = st_backend_wgpu::gelu_back::Pipelines::from_embedded(device, geometry)?;
let plan = st_backend_wgpu::gelu_back::Plan::new(rows, cols, stride, geometry, true, &device.limits())?;
// Allocate nonaliasing buffers, upload plan uniforms and create the bindings.
pipelines.encode_into(&mut encoder, &fused_binding, &reduce_binding, &plan)?;
```

Plans reject empty/overflowing shapes, short strides, storage/index bounds and
unsupported dispatch geometry. Low-level callers still own buffer contents,
matching uniforms, nonaliasing and device identity. Plain data is contiguous;
Tensor first converts non-row-major input/seed to logical row order.
Residual accumulation preserves the existing contract: residual_seed + gZ.
Bias accumulation retains the existing per-tile and cross-tile order.
The full helper also avoids a redundant host residual clone and host zero-fill
of partials that the GPU always writes before reading. Batched full-output
snapshots can have a higher peak staging footprint than separate reads;
per-buffer limits are not an aggregate memory budget.

## Comparison Protocol

Six shapes, one/three-output contracts, bursts 1/4, three warm-up blocks and
nine measured blocks. Three routes rotate through every position exactly
three times per condition. Three serial rounds rotate native/browser/Torch
runtime order. No concurrent GPU benchmark or tuning during these rounds.

- One output: legacy Tensor fused shader + three separate reads; the identical
  fused shader + selected gZ read; canonical derivative-only shader + gZ read.
- Three outputs: legacy Tensor fused shader + separate reads; canonical shared
  derivative fused shader + separate reads; the same canonical shader + batch.
- Torch: actual ATen tanh-GELU backward, preallocated packed output views, CPU
  and MPS, four intra-op/one inter-op threads, no fallback or torch.compile.
  One owning CPU copy observes all live outputs. This is not a deliberately
  fragmented Torch readback control or a comparison to its default exact-erf GELU.

The prepared interval includes encoding/bindings, per-operation submissions,
per-operation GPU residual reset where needed, and terminal owning CPU
completion. Allocation, uploads of fixed operands, compilation, checks, list
conversion and JSON export are outside. A burst repeats the same operation
before observing its last result. Every residual update starts from the same
seed; it does not accumulate across burst iterations.

The two WGPU readback controls use the same Rust owning lease/decoder, separately
or in a batch. They are not a byte-identical measurement of the old native
read_buffer helper. High-level Tensor routing is covered by strict GPU tests;
these prepared timings are not an end-to-end Tensor or model multiplier.

Frozen accuracy: abs=2e-6 + rel=1e-5*abs(reference), with only the bias sum's
absolute allowance multiplied by row count. Independent scalar f64 and Torch
CPU f64 references check all outputs. Twenty-four domain fixtures additionally
cover tails, padded rows, finite extremes and residual accumulation. The
historical backend formula is observed after only repairing its syntax; its
output strings/nonfinite count are diagnostic, not a passing correctness gate.
The original loader failure is preserved separately.

## Replay

```sh
cargo test --locked --release -p st-backend-wgpu --lib --test wgsl_syntax
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-backend-wgpu --example gelu_backward_bench -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 cargo test --locked --release -p st-tensor --features wgpu --test wgpu_gelu_liveness
cargo run --locked --release -p st-backend-wgpu --example gelu_backward_bench
cargo build --locked --release -p st-backend-wgpu --target wasm32-unknown-unknown --example gelu_backward_bench_browser
wasm-bindgen --target web --out-name spiraltorch_wasm --out-dir NEW_WASM_DIR target/wasm32-unknown-unknown/release/examples/gelu_backward_bench_browser.wasm
node tools/test_resident_browser.cjs NEW_WASM_DIR CHROME_EXECUTABLE NEW_REPORT.json '' '' '' '' gelu-backward-bench
python3 -I -B benchmarks/gelu-backward/torch_bench.py
python3 -I -B benchmarks/gelu-backward/test_protocol_gelu.py
python3 -I -B benchmarks/gelu-backward/test_evidence.py
```

Use a fresh report path per run. Browser cases stream to a bounded JSONL sink
outside timing before the owned driver reconstructs the report. Record source,
toolchain/device identities and all failed attempts. Public archives contain
intervals, all conditions, validation and raw hashes; arrays, binaries, generated
WASM/JS and logs remain local. Archive verification rehashes/recomputes recorded
evidence and is not numerical GPU reexecution.
