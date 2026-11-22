# A finite-checked geometry update in WASM

Build the `st-wasm` web package with the `nn` feature, for example
`wasm-pack build bindings/st-wasm --target web --dev --features nn` from the repository root.
Serve the resulting `bindings/st-wasm` directory over loopback HTTP and open
[`tests/native_causal_lm.html`](native_causal_lm.html). For Node, run
`node bindings/st-wasm/tests/native_causal_lm_node.mjs bindings/st-wasm/pkg/spiraltorch_wasm.js`.
Both execute the same four artificial controls and compare against native Python results.

## Cached continuation

For a known sequence (for example, a document suffix or one answer candidate),
reuse a prefix rather than running the full decoder for every token:

```js
const prefix = model.prefill(Uint32Array.of(1, 2), 1024);
try {
  const branch = prefix.withSuffix(Uint32Array.of(3, 4, 5));
  try { console.log(branch.nextLogits(), branch.tokenCount); } finally { branch.free(); }
} finally { prefix.free(); }
```

The same operation is `prefix.with_suffix(&ids)?` in Rust and
`ctx.with_suffix(ids).logits().values()` in Python. In WASM, pass an ordinary,
non-shared `Uint32Array`; the explicit cache budget is at most 1,048,576
persistent f32 values per context, excluding transient storage and other forks.
The source remains usable after either success or failure. The child owns its
model and may outlive the JS handles for the source and model.

Only the last suffix position gets vocabulary logits. Rust applies the absolute
position offset and causal mask to all new queries; existing decoder keys and
values are copied into the fork, not reprojected. A causal geometry site may
still replay its retained input history. The attention budget charges the
rectangular query-by-key shape, including masked slots. A large chunk can
exceed it even if separate single-token calls fit. This API neither guesses
future tokens nor scores intermediate ones: extend by one when each next
token depends on the prediction. A successful chunk does not certify the
vocabulary logits or loss of any skipped intermediate row.

The browser and Node controls also include six nonconstant models (ordinary
and key-lag attention; bare, trained WaveGate and trained fractional history).
Their fixture was generated with the original full-prefix and one-token Python
interfaces using `native_causal_lm_fixture.py --cache OUTPUT`.

## Finite-checked learning

```js
// M is the initialized web package. configJson and weights (Uint8Array)
// identify a local decoder. Pin the geometry JSON to both its digest and this base.
const base = M.NativeCausalLm.fromGpt2(configJson, weights);
const art = M.GeometryAdapterArtifact.fromJson(geometryJson, artifactId, base.baseModelIdentity);
const origin = base.withGeometry(art);
const batch = M.CausalLmDeclaredBatch.fromJson(JSON.stringify({
  ids: [2, 0], shape: [1, 2], contrasts: [[1, 0, [1]]],
}));
try {
  const step = origin.stepDeclaredSetGeometryAtDistance(batch, 1 / 65536);
  const receipt = JSON.parse(step.reportJson());
  const chosen = step.intoModel(); // consumes step; do not free it again
  try {
    console.log(receipt.accepted, chosen.nextLogits('[2]'));
    const saved = chosen.geometryArtifact();
    try { console.log(saved.artifactId, saved.toJson()); } finally { saved.free(); }
  } finally { chosen.free(); }
} finally { batch.free(); origin.free(); art.free(); base.free(); }
```

Every declaration is `[unshifted_flat_position, gold_id, sorted_rival_ids]`.
Optional unshifted `labels` use `-100` for ignored targets. Without a declaration
an active token contributes ordinary CE and zero coverage loss. Rust scores the
proposed weights on the **same training batch**: both mean `-log P(gold | set)`
and mean `-log P(set)` must decrease. A finite tie or regression returns the
origin; an infeasible direction throws without modifying it. `report.proposed`
can therefore describe weights **not** selected by `intoModel()`.

This is a small-model CPU client, not a WebGPU benchmark or a general HF loader.
Ingress accepts an ordinary, non-shared safetensors byte view up to 64 MiB;
total base parameters are limited to 8,388,608 elements. Each activation, attention,
logit and training-checkpoint budget is 1,048,576 elements. No tokenizer, model
download, Python, Torch or NumPy is required at runtime. IDs and geometry use
the same contracts as native Rust and Python.

To check a WASM-to-Python handoff, pass an unused output path as the Node
runner's final argument. With a native development wheel supporting the
declared-set APIs, run `native_causal_lm_fixture.py --verify-wasm FIXTURE_JSON
SELECTED_JSON` using Python. The same script with one unused output path
regenerates the tiny fixture from deterministic weights; it never opens a pretrained model.
