# A resident causal byte decoder

Attention consumes vectors; it does not require a BPE vocabulary. The byte
decoder composes existing resident primitives into a complete next-byte training
path with a fixed 256-symbol alphabet. This is tokenizer-dictionary-free, not a
claim that discrete symbols or sequence boundaries disappear.

```text
within-document bytes [B,T+1]
       | explicit one-byte shift
       +--> input IDs [B,T] --> learned byte table + learned position table
       |                                  |
       |                         causal residual blocks
       |                     (LayerNorm / Attention / MLP / optional Topos)
       |                                  |
       +--> targets [B,T] <--- CE <--- tokenwise head [B,T,256]
                                           |
                            exact VJP through every parameter
                                           |
                         ONE owner/revision and all-or-none SGD
```

## Rust ownership and data contract

`ByteDecoderPlan::from_plans(token_table, position_table, blocks, head)` freezes
the host parameters. Each block must preserve `[B,T,C]` and use
`AttentionMask::Causal { query_offset: 0 }`; the head must return `[B,T,256]`.
The graph stages currently admitted by these plans are tokenwise outside
attention. There is at least one block and enough learned positions for T.

`ByteLmBatch::from_documents(documents, selections, steps)` takes `(document,
byte_offset)` selections and rejects any T+1 window that crosses a document
boundary. `from_windows` is the lower-level form for already-selected equal-size
windows; the caller owns their provenance. Neither method decodes UTF-8,
normalizes text, concatenates documents, pads, or adds EOS. Every value from 0 to
255 is an ordinary byte. Positions reset to zero per window and documents occupy
separate attention batch rows. This is fixed-window training, not packed variable
lengths or a streaming/KV-cache decoder.

The model owns byte and position tables, optional causal geometry, all residual blocks, and the head in one
`ResidentParameters`. `parameter_layout()` exposes the exact untied order:
byte table, position table, optional geometry range, each block's parameter range,
then the head range. The geometry range further names projection parameters,
raw decay, raw phase and one raw-gain vector per block. Without geometry the
original ordering is unchanged.
Same-shaped blocks have separate execution workspaces. The owner-free residual
executor is shared with the existing single-block trainer, not a second copy of
the attention/skip-path mathematics.

Given a composed `plan` and a `WgpuRuntime`:

```rust
use st_kernel_contracts::classification::{ClassReduction, CrossEntropySpec};
use st_nn::resident::ByteLmBatch;

let documents: &[&[u8]] = &[b"abaca", b"xyxyz"];
let host = ByteLmBatch::from_documents(documents, &[(0, 0), (1, 0)], 4)?;
let mut model = plan.compile_training_wgpu(runtime)?;
let batch = model.prepare_batch(&host)?;
let forward = model.forward(&batch)?;
let loss = forward.next_byte_loss(
    CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?,
)?;
let gradients = model.backward(&forward, loss.prediction_gradient())?;
let update = model.sgd(&gradients, 0.125)?;
// Explicit observation; on WASM use read_async().await instead.
let accepted_revision = update.snapshot()?.read()?;
```

Batch preparation uploads exact integer IDs and targets. Forward, CE, VJP and
SGD do not read activations or gradients back to the host. Receipts and tensors
are observed explicitly. The shifted targets belong to the forward's captured
batch; a later batch cannot silently replace them.

Only the latest forward of the same owner/revision can be differentiated. A
successful forward invalidates earlier tapes. Host-side input/bias preflight
errors preserve the current tape; every submitted update attempt invalidates
it, including a GPU-rejected update. Parameter snapshots and returned gradients
remain immutable. A whole-family guard runs **after both embedding pullbacks**:
a late duplicate-byte or position accumulation overflow invalidates every head,
block, table, embedding-output and external-bias derivative together. A rejected
update preserves all parameter values, then the next forward rebinds them at the
new attempted revision.

## The geometry boundary

Default `forward` has no external score bias. Topos can already be part of a
tokenwise residual feed-forward graph, so its gate participates in the same
byte-loss VJP and parameter update rather than only changing a report.

`forward_with_external_biases` is a deliberately advanced seam. It accepts one
`ByteDecoderBias` per block and returns exact `z_bias`/`pair_bias` gradients, but
**causality is conditional on the producer of those biases**. A causal attention
mask cannot repair a feature computed using future bytes. The model does not
own or update external bias producers.

`ByteDecoderPlan::with_causal_geometry` now connects a tokenwise projection,
[causal wave](causal_zspace_wave.md), interior chart and
[Poincare squared-distance bias](poincare_metric_attention.md) under that same
full-model owner:

```text
summed byte/position embeddings --> residual blocks --> head --> next-byte CE
             |                         ^
       tokenwise projection            |
             |                   per-block/head gains
       causal wave + chart ----------> Poincare pair biases
```

```rust
use st_nn::resident::ByteDecoderGeometryPlan;

// projection: an InferencePlan from [B,T,C] to [B,T,2P].
// decay/phase have P values; gains contains one H_i-vector per block.
let geometry = ByteDecoderGeometryPlan::new(
    &projection, &raw_decay, &raw_phase, &raw_gains, -0.75,
)?;
let plan = plan.with_causal_geometry(geometry)?;
// compile_training_wgpu / forward / next_byte_loss / backward / sgd as above.
```

Both roles of each coordinate in every consuming block's metric contribute to
one summed chart cotangent before wave BPTT. Its drive VJP traverses the
projection and adds to the ordinary residual-path embedding cotangent. Only
then do the byte/position scatters and whole-model guard run. No geometry-only
optimizer, update receipt, host readback or second semantic implementation is
introduced. External pair bias is additive; only caller-supplied biases appear
in the returned external-bias gradient list.

State resets to zero at each selected document window, with zero terminal-state
cotangent. This gives a full-window causal encoder, not streaming decoder/KV
ownership or gradients across windows. Curvature is fixed. Gains are shared
across positions within each block/head, not an input-dependent router.
Softplus gains can shrink through training, but raw gain zero is **not** off;
omit `with_causal_geometry` for the ordinary model/control.

The full-text DFT and length-dependent envelope in `LanguageWaveEncoder` are not
safe one-shot autoregressive features. Learned variable-size latent units are a
later, separate experiment, not something the 256-byte model already implements.

## Reproducible verification

`tools/generate_resident_byte_decoder_torch_fixture.py` independently computes
CPU-f32 PyTorch results without importing SpiralTorch. The checked-in fixture is
frozen; it is not regenerated by tests. The fixed elementwise criterion is
`abs(actual - expected) <= 3e-6 + 5e-5 * abs(expected)`.

The shared native/browser Rust probe retains two original models: one plain block with 18
parameter tensors, and two blocks with Topos plus position-only geometric bias
and 31 parameter tensors. It checks logits, a strided cotangent VJP for **every**
parameter, summed-embedding-output gradients, four external bias derivatives in
the geometric model, and all parameter values after each of 16 CE/SGD updates.
All updates are queued before their losses, receipts or parameters are read.

`tools/generate_resident_byte_geometry_torch_fixture.py` adds two independent
CPU-f32 fixtures with learned projection/wave/metric parameters: 23 tensors for
one block and 37 for two blocks with Topos and external biases. The same
elementwise criterion covers every full-model VJP and all 16 CE/SGD updates.
Each geometry gradient also must be nonzero and meet relative L2 error <=0.002;
an absolute-tolerance pass cannot disguise a disconnected small derivative.
Zero-Q/K scores isolate the metric's role in the one-block fixture. Geometry-off
and detached-coordinate controls distinguish prediction influence from the
embedding pullback. These are fixed-weight correctness contrasts, not matched
language-quality experiments.

Separate controls change suffix bytes, compare prefix-alone versus extended
execution, check document isolation and zero future-byte gradients, and verify
positive input sensitivity. Ownership tests cover foreign same-counter tapes,
superseded tapes, invalid-final-bias preflight, good/bad/good pullbacks, retained
gradient immutability, atomic rejection, stale gradients and recovery. Native
unit tests additionally inject a candidate overflow in each of the 12 plain and
18 geometry-enabled parameter slots of smaller two-block models and isolate late byte-only and position-only
scatter overflow.

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-nn \
  --no-default-features --features wgpu --test resident_byte_decoder -- --nocapture
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-nn \
  --no-default-features --features wgpu --lib resident::byte_decoder -- --test-threads=1
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --target wasm32-unknown-unknown --example resident_byte_decoder_browser
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_byte_decoder_browser.wasm \
  --target web --out-dir target/resident-byte-decoder-web
python3 -m http.server 8771 --bind 127.0.0.1
```

Use a wasm-bindgen CLI version matching Cargo.lock, then open
`http://127.0.0.1:8771/crates/st-nn/tests/byte_decoder_browser.html` in a WebGPU
browser. The page rejects incomplete result counts and exports its result JSON.
This is the Rust model executing in WASM, not a JavaScript reimplementation.

These are tiny synthetic correctness tests, not a corpus evaluation, a speed
comparison, or evidence that geometric features improve language quality. The
models are different configurations/seeds, not a matched quality ablation.
No Python/JavaScript full-model facade, checkpoint/import API, tied embeddings,
streaming generation or release-version change is included in this slice.

The [2026-10-09 native/browser results](../benchmarks/results/2026-10-09-resident-byte-decoder/README.md)
record the tested source and fixture hashes, full per-parameter errors and
per-update trajectories. Neither the criteria nor the fixture was loosened
during implementation or review.

The [causal geometry owner results](../benchmarks/results/2026-10-09-byte-causal-geometry-owner/README.md)
extend that original record with the integrated encoder, detached/off controls
and per-update geometry-gradient comparisons. Historical primitive/model records
remain unchanged and do not retroactively claim this integration.

The [pair-cache integration regression](../benchmarks/results/2026-10-09-byte-causal-geometry-owner-pair-cache/README.md)
reruns those full-model controls after the metric's head-reduction optimization;
native and actual browser reports preserve every prior validation result.

For data-driven training outside the synthetic fixtures, see the
[paired corpus learning runner](byte_corpus_learning.md). It uses this same
Rust model on native and browser WGPU, with document-held-out evaluation and
an independent PyTorch comparison; it is not a new model implementation.
