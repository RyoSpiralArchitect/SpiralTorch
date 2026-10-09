# Resident attention learning primitive

`ResidentTensor::scaled_dot_attention_vjp` connects the shared Rust attention
contract to first-order GPU gradients. It complements the
[resident forward and frozen NN projection chain](resident_zspace_attention.md).
This is not yet a trainable decoder, an implicit autograd tape, or a quality gain.
These client APIs are source additions after 0.4.29, not part of that PyPI wheel.

## Contract and ownership

Q and upstream are `[B,H,Q,D]`; K/V are `[B,H,K,D]`. Optional Z-bias is `[B,H,K]`
and pair-bias is `[B,H,Q,K]`. The same `AttentionSpec`, causal offset, finite-input
rules and `D <= 256` limit apply to forward and VJP. Scale is explicit and finite,
including zero or negative values. No dropout, hidden averaging, implicit GQA,
KV-cache ownership or scalar scale gradient is introduced.

Returned `ResidentAttentionGradients` owns `query`, `key`, `value`, `z_bias` and
`pair_bias`. Bias gradients are absent exactly when their inputs are absent.
Shapes are the **logical input shapes**, even for broadcast inputs. Reduce shared
bias/head/batch dimensions explicitly before updating the underlying source.
Causally masked pairs have exactly zero derivative; bias cannot unmask them.

Inputs may be immutable strided/broadcast views. Output gradients are contiguous
views at possibly nonzero offsets into one allocation, with one operation-wide
validity guard. Keeping one gradient retains that allocation, including scratch.
Use `contiguous()` when a separate offset-zero allocation is necessary. Subsequent
forwards/VJPs and dropping other handles do not overwrite retained gradients.

Three compute passes recompute normalization, query/pair gradients and key/value/
Z gradients. One submission, no host mapping, no CPU fallback, no floating-point
atomics, and **nine scratch words per query**. No quadratic probability tape is
retained; a requested pair gradient still necessarily occupies `B*H*Q*K` values.
Packed allocation size, both dispatch grids, all views, device identity and the
eight-storage-buffer requirement are preflighted before pipeline creation.

Scores retain checked f32 operations **and the forward's selected dot reduction
order**, including its eight-lane key tile. Normalization and derivatives reuse the
extended-exponent significand arithmetic used by resident LayerNorm; small
probabilities are range-reduced before multiplying large derivatives. This is
not IEEE f64, not bitwise equality with the f64 Rust oracle, and not a throughput
optimization claim. Every inherited input failure, including a failed value
hidden by a crop or empty query, invalidates every output. Unrepresentable
**requested** gradients also invalidate the whole result. An absent bias gradient
does not make an otherwise representable Q/K/V result fail.

## Python and WASM

Python only converts arguments and releases the GIL around the same Rust calls:

```python
from spiraltorch import WgpuTensorDevice

device = WgpuTensorDevice.create()
q = device.upload([1, 1, 1, 2], [0.2, -0.1])
k = device.upload([1, 1, 2, 2], [0.3, 0.1, -0.2, 0.4])
v = device.upload([1, 1, 2, 2], [1.0, 0.0, 0.0, 1.0])
z = device.upload([1, 1, 2], [0.1, -0.1])
upstream = device.upload([1, 1, 1, 2], [0.5, -0.25])
output = q.scaled_dot_attention(k, v, 0.5, causal_offset=1, z_bias=z)
gradients = q.scaled_dot_attention_vjp(
    k, v, upstream, 0.5, causal_offset=1, z_bias=z,
)
next_z = z.add(gradients.z_bias.mul(device.upload([], [-0.01])))
print(next_z.snapshot().read_values())  # The explicit host boundary.
```

`causal_offset=None` is unmasked; `0` means causal prefill, not unmasked. These
methods are available on `WgpuTensor`, not the host `Tensor`. A CPU-only wheel
rejects `WgpuTensorDevice.create()` rather than pretending to execute on GPU.

The WASM API uses the same shapes, arithmetic, guards and ownership. A small
`WgpuAttentionBiases` container borrows optional inputs safely because wasm-bindgen
does not support `Option<&exported type>`. Setters clone only Rust handles, not
device storage; calls do not consume the bias container or input tensors.

```javascript
const biases = new WgpuAttentionBiases(); // Empty means neither bias.
biases.setZBias(z);
const output = q.scaledDotAttention(k, v, 0.5, 1, biases);
const gradients = q.scaledDotAttentionVjp(k, v, upstream, 0.5, 1, biases);
const dz = gradients.zBias; // Independently retained handle.
gradients.free();
biases.free();
// dz remains valid; snapshot/readValues is the explicit async host boundary.
```

Use `undefined`/`null` in place of offset for unmasked attention. The container has
`setZBias`, `setPairBias`, `clearZBias`, `clearPairBias`. Caller-owned JS handles,
including snapshots, should be freed when no longer needed.

## Verification

The independent frozen CPU-f32 PyTorch fixture contains 18 conditions, each
checked in canonical and reversed/padded layouts on native Metal and browser
WebGPU. It covers all bias modes, causal offsets, batches/heads, 65/256-wide heads,
131-key tails, zero/negative scale and empty queries. It also records 16 MSE/SGD
updates of Q/K/V and both biases. The Rust probe executes every update on the GPU
and reads outputs/parameters only after the update sequence for validation.
Both probes also cover 12 inherited-failure cases, two wide-range cases and three
adversarial cancellation cases across the forward specialization boundary. See
the [dated kernel and public-client evidence](../benchmarks/results/2026-10-09-resident-attention-vjp/README.md).
The full WASM package passes all 18 public-client fixture conditions in the
browser, and the default-feature Python wheel passes the three public API tests
on macOS WGPU, including the same 18 conditions and parent-handle lifetime checks.

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release \
  -p st-backend-wgpu --lib resident_tensor::attention -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release \
  -p st-backend-wgpu --test resident_attention_vjp -- --nocapture
cargo check --locked --manifest-path bindings/st-py/Cargo.toml
cargo check --locked -p spiraltorch-wasm --features webgpu \
  --target wasm32-unknown-unknown
```

GPU-specific native tests return early without the environment variable. A green
default test run is not execution evidence. The browser probe is standalone:

```bash
cargo build --locked --release -p st-backend-wgpu \
  --example resident_attention_vjp_browser --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_attention_vjp_browser.wasm \
  --target web --out-dir target/resident-attention-vjp-web
python3 -I -S -m http.server 8783 --bind 127.0.0.1
```

Match the wasm-bindgen CLI to Cargo.lock (`0.2.129`). Open
`http://127.0.0.1:8783/crates/st-backend-wgpu/tests/attention_vjp_browser.html` and
require `passed: true`, 36 checks and all 16 update checks. The public binding
regressions are `bindings/st-py/tests/test_wgpu_attention.py` and
`bindings/st-wasm/tests/resident_attention_clients.html`; the latter expects a
fresh full `webgpu` package in its adjacent `module/` directory.

CI also executes the Python regression with the real-GPU opt-in on macOS, using a
fresh default-feature wheel and an isolated Python import. The WASM job uploads
the already-built browser package as `resident-attention-webgpu` for one day.
For a checkout with limited disk space, download that artifact from the CI run
whose head matches the intended PR revision into a fresh directory under
`target/`. Do not substitute a package from an earlier run. For example, a package
under `target/attention-client-ci/` can be exercised at:

```text
http://127.0.0.1:8783/bindings/st-wasm/tests/resident_attention_clients.html?module=/target/attention-client-ci/spiraltorch_wasm.js
```

Only same-origin module URLs are accepted. Require `passed: true` and all 18
fixture cases; the page exposes a downloadable JSON result. Preserve the CI run
and head IDs, downloaded module/WASM hashes, fixture hash and result together.
This is public-client execution evidence, not a replacement for the separate
kernel SGD, strided-layout and wide-range tests. Uploading an artifact or passing
its declaration check alone is not browser execution evidence.

The generator `tools/generate_resident_attention_vjp_torch_fixture.py` refuses
to overwrite an existing fixture. Regenerate to a new path with PyTorch 2.12.1
and optional global patches disabled, then compare bytes. This bounded synthetic
learning proves gradient/update connectivity, not an advantage over ordinary FT.
QKV/output projection training, complete LLM graph integration and matched
throughput measurement remain separate follow-ups.
