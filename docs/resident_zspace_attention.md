# Resident Z-space attention

The language-specific resident primitive and frozen projection chain share Rust
semantics between native WGPU and browser WebGPU. They do not replace a complete
decoder or the high-level attention module's uncertainty output.

## Contract

`st_kernel_contracts::attention::AttentionSpec` owns shape, scale, bias and
causal-offset validation. `attention_reference` is a CPU numerical oracle, not
a fallback route. The WGPU entry point is
`ResidentTensor::scaled_dot_attention`:

```text
softmax(scale * QK^T + z_bias + pair_bias, structural_mask) V
Q / output : [batch, heads, queries, head_dim]
K / V      : [batch, heads, keys, head_dim]
z_bias     : [batch, heads, keys]
pair_bias  : [batch, heads, queries, keys]
```

Biases are additive, after scaling the dot product. Use `broadcast_to` explicitly
for shared biases. A zero bias is an identity control. A nonzero Z-space bias
must also be supplied to the PyTorch reference; comparing it only with plain
attention would compare different functions.

`AttentionMask::Causal { query_offset }` allows keys through
`query_offset + query_index`, inclusive. Prefill starts at zero. A one-token
query against five cached keys uses offset four, not a top-left one-row mask.
The query range must fit in the key sequence. Biases cannot expose future keys.
This API consumes existing keys/values; it is not a KV-cache manager.

Q/K/V, biases and output stay on the owning device and queue. Permuted, sliced
and broadcast views are packed on GPU if needed. Online normalized softmax
does not allocate the quadratic score/probability matrix. A caller-supplied
pairwise bias is still quadratic. Only explicit snapshots read values back.

For at least 128 keys and head dimension at most 32, the portable 64-thread kernel
computes four key dot products in parallel with 16 lanes each, amortizing
workgroup barriers. Short sequences and unmeasured wider heads retain one key
per tile with 64 lanes. Each pipeline specialization is cached lazily. This
bounded policy follows full-chain measurements, not a universal speed claim.
Online normalization and value
accumulation still visit keys in order; dot-product reduction order changes, so
agreement is tolerance-based, not bitwise. Tail tiles and causal visibility
must not load masked keys. This uses core WGSL, not subgroups or native-only
instructions, and is exercised by the browser probe as well as native tests.

All inputs must be finite. Non-finite score/weighted-output arithmetic sets an
owned failure guard, inherited by downstream operations and checked on readback.
Even a masked-out invalid input retains its upstream failure. Different floating
point reduction orders are not promised to be bitwise identical.

Empty batch/head/query axes are accepted; keys and head dimension must be
nonzero. The initial GPU kernel supports `head_dim <= 256`, float32, no dropout,
no implicit GQA expansion, and no backward tape. Unsupported shapes or devices
return an error rather than silently executing on CPU. Python bindings and a
complete decoder graph remain follow-up work. The browser examples exercise the
same Rust implementation, not a separate JavaScript attention implementation.

## Existing NN Parameters And Z-RBF Geometry

`st_nn::resident::AttentionInferencePlan` freezes existing Q/K/V/output `Linear`
parameters with `from_linears`, or four `(weight, bias)` pairs with
`from_parameters`. Weights use `[in, out]`, biases `[1, out]`. Q/K/V have the same
width, divisible by the number of heads; the output width may differ from input.
This is self-attention with input `[B,T,I]`, either unmasked or causal offset zero,
not cached decode. Empty sequences are rejected at this higher-level boundary.

QKV weights are packed together once during plan construction, not every forward.
The plan owns frozen parameter versions; updating a source module requires a new
plan and compilation. `compile_wgpu` rejects unsupported limits before building
the graphs. The resident chain is:

```text
input [B,T,I]
  -> fused QKV Linear [B,T,3*H*D]
  -> select/permute head views [B,H,T,D]
  -> attention with optional key-wise/pairwise bias
  -> GPU head merge [B,T,H*D]
  -> output Linear [B,T,O]
```

The split is a storage-sharing N-D view. Noncontiguous heads are subsequently
packed on GPU for the attention kernel. Head merge and projection outputs also
remain on GPU. This uses multiple submissions and GPU copies, not a single
fused dispatch. There are no intermediate activation readbacks or CPU fallbacks;
only the caller's final snapshot materializes the output. Inputs, biases and
both graphs must share the owning device and queue. Outputs retain their own
storage and deferred failure guards across graph reuse.

The existing `ZRBFAttention` supplies two explicit adapters:

- `kernel_bias(frame, queries, keys)` returns `[H*Q,K]` host geometry metadata
  using the same checked product kernel, metric and per-head ARD policy as its
  ordinary forward. Rectangular geometry is supported. Upload as `[1,H,Q,K]`
  and explicitly broadcast when multiple batch entries share the frame.
- `mean_inference_plan(input, mask)` freezes that module's actual projection
  parameters. Pass its `kernel_bias` as the resident **pair-bias** to preserve the
  original mean function. It does not manufacture variance or entropy, nor does
  it silently carry a frame: callers own the geometry/bias lifecycle.

Geometry construction remains CPU-side metadata work and pairwise bias takes
quadratic memory. Reuse a bias only while its frame, indices and geometry
parameters remain unchanged. Runtime strength changes can multiply the resident
bias without reading activations back. This bridge is frozen inference, not
parameter training, an autograd tape, or a new competing Z-space policy.

## Verification

See the [dated numerical evidence](../benchmarks/results/2026-10-02-resident-zspace-attention/README.md).
The same 20 frozen PyTorch SDPA cases are checked by the Rust oracle, native
resident kernel and browser example. Native tests additionally cover head widths
through 256, GPU view packing, ownership, invalid inputs and inherited guards.

```bash
cargo test --locked -p st-kernel-contracts
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked \
  -p st-backend-wgpu attention -- --test-threads=1 --nocapture
```

Without the runtime-test environment variable, GPU-specific unit tests return
early. A green default test run alone is not GPU-execution evidence.

Browser build (wasm-bindgen CLI must match Cargo.lock, currently 0.2.104):

```bash
env -u CARGO_ENCODED_RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS \
  -u CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS \
  -u LIBRARY_PATH -u PKG_CONFIG_PATH RUSTFLAGS= \
  cargo build --locked -p st-backend-wgpu --example resident_attention_browser \
  --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/debug/examples/resident_attention_browser.wasm \
  --target web --out-dir target/resident-attention-web
python3 -m http.server 8782 --bind 127.0.0.1
```

Open `http://127.0.0.1:8782/crates/st-backend-wgpu/tests/attention_browser.html`.
The probe checks WebGPU validation errors as well as numerical output. Build
success and Naga validation alone did not catch the first browser issue: the
decimal spelling of negative f32 MAX exceeded the browser compiler's accepted
range. Both the resident and legacy fused shaders now use its exact bit pattern.

Performance is measured separately on the complete QKV/attention/output-projection
chain, distinguishing resident-only timing from transfers and execution routes.
The numerical slice alone establishes neither a speedup nor a learning-quality gain.

## Full-Chain Verification

The [full-chain evidence](../benchmarks/results/2026-10-02-resident-attention-chain/README.md)
checks two shapes with plain, zero-geometry and nonzero Z-RBF controls, both
unmasked and causal: 12 cases each on native Metal and browser WebGPU. The
independent PyTorch oracle includes all four projections and computes the same
geometric bias; Rust geometry is separately checked against it. These are small
analytic fixtures, not trained-model or throughput results.

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked -p st-nn --features wgpu \
  attention -- --test-threads=1 --nocapture
cargo run --locked -p st-nn --features wgpu --example resident_attention_chain
env SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0 \
  python3 -I tools/generate_attention_chain_torch_fixture.py --output /tmp/attention-chain-oracle.json
cmp /tmp/attention-chain-oracle.json crates/st-nn/tests/fixtures/attention_chain_torch.json
```

Exact fixture regeneration uses PyTorch 2.12.1 on CPU float32; disable optional
global patches before Python starts. The fixture is committed so Rust/browser
checks do not need a PyTorch installation or model/data downloads.

```bash
env -u CARGO_ENCODED_RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS \
  -u CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS \
  -u LIBRARY_PATH -u PKG_CONFIG_PATH RUSTFLAGS= \
  cargo build --locked -p st-nn --features wgpu \
  --example resident_attention_chain_browser --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/debug/examples/resident_attention_chain_browser.wasm \
  --target web --out-dir target/attention-chain-web
python3 -m http.server 8782 --bind 127.0.0.1
```

Open `http://127.0.0.1:8782/crates/st-nn/tests/attention_chain_browser.html` and
require `passed: true`, 12 output checks and two geometry checks. Native runtime
tests also compare the full chain with the original `ZRBFAttention` mean and
exercise frozen parameters, noncontiguous inputs, retained outputs and inherited
non-finite guards. The GPU CI lane runs the full-chain fixture; the recorded
browser probe was run locally, not by that CI lane.

## Bounded Performance Comparison

`resident_attention_chain_bench` consumes the independent fixture with three
larger shapes. `bench_attention_chain_vs_torch.py` rotates native executables and
eager PyTorch CPU/MPS runs, requires complete numerical/sample coverage, and
records every sample rather than just favorable medians. Build native executables
in release mode, and keep baseline/candidate binaries separate for paired runs.

```bash
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -I tools/generate_attention_chain_torch_fixture.py --suite benchmark --output "$FIXTURE"
cargo build --locked --release -p st-nn --features wgpu --example resident_attention_chain_bench
python3 -I tools/bench_attention_chain_vs_torch.py --fixture "$FIXTURE" \
  --native "st_candidate=target/release/examples/resident_attention_chain_bench" \
  --devices cpu mps --rounds 3 --samples 7 --warmup 50 --burst 4 --output "$NEW_RESULT"
```

Add `--native "st_baseline=$BASELINE_BINARY"` and use four rounds to rotate four
engines through every order position. Resident timings include CPU encoding,
allocation and GPU completion of a burst, but no output readback. Host-to-host
timings include fresh input/bias uploads and an owning output readback; weights
remain resident. Compilation, fixed geometry/mask preparation and numerical
checks are outside both timers. All burst outputs are checked after timing.
PyTorch uses its default eager SDPA dispatch, not a forced CPU math backend for
GPU timing; its own outputs must pass the frozen CPU math reference. MPS fallback
and optional global patches must be disabled before Python starts.

These are full-chain inference observations, not kernel timestamps. ST's
non-finite guards remain enabled; Torch is not given equivalent guard kernels.
CPU, Metal/WGPU and MPS are different routes even on one machine, and short
wall-clock samples are noisy. No CUDA or training advantage follows from them.
The default 50 warmup blocks follow a short-sequence sensitivity check; the
earlier three-warmup measurements remain published rather than overwritten.
Even this is not a proof of clock stability or statistically bounded regression.
See the [comparison and rejected broad rollout](../benchmarks/results/2026-10-02-attention-key-tiling/README.md).

To check both kernel specializations in the browser, build the WASM probe above,
place the generated benchmark fixture at
`target/attention-chain-web/benchmark-fixture.json`, and open the same local page
with `?suite=benchmark`. Require 18 output checks and three geometry checks. This
larger browser run is numerical validation, not a browser timing benchmark.
