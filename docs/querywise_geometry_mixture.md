# Query-Local Geometry Mixtures

`ResidentTensor::querywise_mixture` mixes components `[B,H,Q,S,K]` with
logits `[B,H,Q,K]`, yielding `[B,H,Q,S]`. Softmax spans components; the
logit pullback sums over keys only, retaining batch, head and query identity.
`K` can be 2 through 8. This is a general resident primitive, not a second
Python implementation of runtime semantics.

```rust
let tape = components.querywise_mixture(&query_logits)?;
let bias = tape.output();
let masses = tape.weights(); // [batch, heads, queries, components]
let gradient = tape.backward(&attention_bias_gradient)?;
// gradient.components(): [B,H,Q,S,K]
// gradient.logits(): [B,H,Q,K]
```

The Rust contract groups contiguous `[B,H,Q]` into independent rows and reuses
the existing wide-reduction mixture kernels. Strides are materialized on the
owning GPU; no host activation readback, routing or softmax is inserted.
Shape, context and device limits are checked before forward materialization.
All component failures propagate even when their current softmax mass is zero.
Parameter-owner SGD can bind these derivatives and reject a whole update
atomically, including a zero-rate update.

## Why Query-Local

The [v7 corpus study](byte_corpus_mixture_controls.md) learns one mixture per
block/head, shared over all positions. This smaller contract makes that sharing
explicit instead of assuming that different queries must prefer the same
geometry. A causal contextual projection can produce the new logits; its own
pullback must receive `gradient.logits()` and join the model's parameter owner.

The primitive itself does **not** implement a trainable contextual LM router.
The optional [byte-decoder connection](contextual_geometry_mixture.md) now adds
that separate learning path; shared routing remains the default. Nor does this
primitive mask future keys or make arbitrary supplied logits causal. The logit producer
and attention mask must enforce those boundaries. No manifold Q/K/V projection,
quality improvement or throughput advantage is claimed.

## Reproduce The Bounded Probe

Keep raw fixtures and model weights local. `REQUEST` and `NATIVE` identify the
unchanged complete v7 request and native report; `RAW` is a new directory.

```sh
python3 -I -B tools/probe_byte_corpus_mixture.py \
  "$PWD" "$REQUEST" "$NATIVE" "$RAW/sensitivity.json" "$RAW/corpus-fixture.json"
python3 -I -B tools/generate_querywise_mixture_fixture.py \
  "$RAW/corpus-fixture.json" "$RAW/qualification.json"
cargo build --locked --release -p st-backend-wgpu --example querywise_mixture
"$CARGO_TARGET_DIR/release/examples/querywise_mixture" \
  "$RAW/qualification.json" > "$RAW/native.json"
```

The first tool is a read-only CPU-f32 diagnostic over **all** 128 declared
training batches for twelve learned-mixture arms at initial and retained native
update-128 weights. It makes no optimizer update and uses no held-out windows.
It records row-centered component cosine similarity, geometric attention shift,
and how query/batch gradient sums cancel. Normal shared-parameter gradients
should sum; cancellation is not a bug or evidence that local routing will help.

The second tool retains 24 first-training-batch cotangent fixtures plus three
synthetic cases with nonuniform query logits. Native and browser use the same
Rust harness: forward, both VJPs, a one-step primitive SGD check, retained tapes
and failure guards. This SGD step updates component/logit tensors, not the
full LM or a contextual router network.

For browser qualification, build `querywise_mixture_browser` for
`wasm32-unknown-unknown`, run the matching pinned `wasm-bindgen` with `--target web`,
and serve its JS/WASM plus `fixture.json` under `target/querywise-mixture-web/`
on loopback. Open `crates/st-backend-wgpu/tests/querywise_mixture_browser.html`.
JavaScript only transports the unchanged fixture and downloads the scalar
report. Serialize GPU runs. Retain the full raw inputs and source hashes;
reported maxima alone are not independently replayable tensor observations.
