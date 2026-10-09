# Resident embeddings: the byte-input learning boundary

Attention consumes vectors, not necessarily BPE tokens. A byte LM can therefore
use a fixed 256-symbol alphabet, trainable input vectors and causal attention
without a learned tokenizer dictionary. This still has discrete input/output
symbols; it is not a claim of representation without any segmentation.

This slice supplies the missing resident input primitive, not a complete LM:

```text
host integer IDs -> once-prepared resident indices
                          |
resident [V,C] table -> embedding [B,T,C] -> caller's model/loss
                          ^                       |
                          +-- exact table VJP ----+
```

## Rust contract

`TensorDevice::upload_embedding_indices(shape, rows, &[usize])` validates exact
integer IDs, prepares stable CSR grouping on CPU in O(rows + tokens), and uploads
immutable u32 transport once. IDs are never rounded, clamped or cast through f32.
The plan is reusable on its owning device; GPU-produced integer IDs are not yet
supported. This input preparation is distinct from activation readback.

`table.embedding(&indices)` returns a `ResidentEmbeddingForward`. Its
`prediction()` is a resident tensor with the original sample axes followed by the
embedding width. `backward(&cotangent)` produces a fresh `[V,C]` logical-table
gradient. Repeated IDs accumulate in original input order with checked f32
addition, without float atomics, frequency scaling or a hidden batch average.
Gather work is O(tokens * C); pullback work is O((rows + tokens) * C), not a scan
of every token for every vocabulary row. These are complexity bounds, not measured
speedups. Noncontiguous value/cotangent views are packed on GPU.

Zero-sized axes and scalar ID shapes are supported. Failed table guards survive
unselected rows and empty lookups; failed forward/cotangent guards propagate into
the complete table VJP. Any nonfinite intermediate sum rejects the whole VJP,
even if later cancellation would yield a finite value. This resident f32 policy
is intentionally stricter than the existing host `Tensor` f64 scatter oracle.
Both routes share the same exact-ID/stable-grouping Rust contract.

The tape owns its index plan and output, so source handles can be dropped and
multiple pullbacks remain valid. It does not own model parameters or perform an
optimizer step. Caller-owned `ResidentParameters` provides revision binding and
one all-or-none update across embeddings, blocks and heads. Broadcast table VJPs
refer to the logical table; a caller that owns a smaller aliased parameter must
apply its separate broadcast adjoint.

## Minimal learning connection

Given an existing `TensorDevice`, this non-contextual lookup classifier exercises
embedding -> cross entropy -> table VJP -> version-bound SGD. It is not a byte LM.

```rust
use st_backend_wgpu::resident_training::parameters::ResidentParameters;
use st_kernel_contracts::classification::{ClassReduction, CrossEntropySpec};

let ids = device.upload_embedding_indices(&[2, 3], 5, &[4, 1, 4, 0, 1, 2])?;
let table = device.upload(&[5, 3], &[0.; 15])?;
let targets = device.upload(&[2, 3], &[2., 0., 2., 1., 0., 1.])?;
let mut parameters = ResidentParameters::new(vec![table])?;
let version = parameters.snapshot();
let forward = version.values()[0].embedding(&ids)?;
let loss = forward.prediction().cross_entropy_with_logits(
    &targets, CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?,
)?;
let gradient = forward.backward(loss.prediction_gradient())?;
let bound = version.bind_gradients(vec![gradient])?;
let update = parameters.sgd(&bound, 0.125)?;
// Explicit native observation; on WASM use read_async().await instead.
let accepted_revision = update.snapshot()?.read()?;
```

There is no Python/JavaScript model facade added in this slice. The browser probe
calls the same Rust code through a small wasm-bindgen test entry point.

## Decoder and geometry follow-through

The [resident byte decoder](resident_byte_decoder.md) now owns token/position
tables, every residual attention block and the final 256-way head in **one**
`ResidentParameters`, not separate optimizers. It reuses an owner-free residual
executor rather than duplicating block math, and joins the entire derivative
family after embedding accumulation so a late input-gradient failure cannot
leave an earlier head gradient apparently valid.

Full-sequence next-byte training requires structural causal masking with offset
zero, explicitly shifted targets, document isolation and a tested position/reset
policy. Prefix invariance must be tested before interpreting learning results.

For a trainable Z-Space front end, the missing research layer is a **causal**
byte/prefix-to-latent encoder and its exact parameter pullback. Attention already
returns `z_bias` and `pair_bias` derivatives; returning them is not the same as
training the encoder that generated them. The existing whole-text DFT in
`LanguageWaveEncoder` must not supply one-shot autoregressive training features:
suffix-dependent features would leak future bytes. Learned variable-size latent
units and a streaming decoder are later, separately evaluated mechanisms.

## Verification

The independent CPU-f32 Torch fixture is frozen at
`crates/st-backend-wgpu/tests/fixtures/resident_embedding_torch.json`.
The shared native/browser harness covers 12 forward/VJP cases (six cases, each
contiguous and strided), broadcast/signed-zero/order/subnormal edges, 10 invalid
shape/ID cases, seven guard cases, two atomic update rejections, two foreign-device
rejections and all 16 CE/SGD steps before any activation readback. Four negative
controls ensure unrelated errors cannot count as a numerical update rejection.

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release \
  -p st-backend-wgpu --test resident_embedding -- --nocapture
cargo build --locked --release -p st-backend-wgpu \
  --target wasm32-unknown-unknown --example resident_embedding_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_embedding_browser.wasm \
  --target web --out-dir target/resident-embedding-web
python3 -I -S -m http.server 8771 --bind 127.0.0.1
```

Open `http://127.0.0.1:8771/crates/st-backend-wgpu/tests/embedding_browser.html`.
WASM builds must use target-appropriate Rust flags, not inherited host linker
flags. Actual native/browser observations and hashes live in
[the verification record](../benchmarks/results/2026-10-09-resident-embedding/README.md).
Correctness is not a throughput or language-quality result.
