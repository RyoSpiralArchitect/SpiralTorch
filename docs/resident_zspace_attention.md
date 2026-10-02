# Resident Z-space attention

The first language-specific resident primitive shares Rust semantics between
native WGPU and browser WebGPU. It does not replace the high-level attention
module or make the entire decoder resident yet.

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

All inputs must be finite. Non-finite score/weighted-output arithmetic sets an
owned failure guard, inherited by downstream operations and checked on readback.
Even a masked-out invalid input retains its upstream failure. Different floating
point reduction orders are not promised to be bitwise identical.

Empty batch/head/query axes are accepted; keys and head dimension must be
nonzero. The initial GPU kernel supports `head_dim <= 256`, float32, no dropout,
no implicit GQA expansion, and no backward tape. Unsupported shapes or devices
return an error rather than silently executing on CPU. Python bindings, native
NN graph lowering and automatic geometric-bias construction remain follow-up
work; the browser example is a thin Rust-kernel probe, not that public facade.

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

The next performance gate is a matched complete QKV/attention/output-projection
chain, separating resident-only timing from transfers and comparing like devices.
This numerical slice establishes neither a speedup nor a learning-quality gain.
