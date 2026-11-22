# A finite-checked geometry update in WASM

Build the `st-wasm` web package with the `nn` feature, for example
`wasm-pack build bindings/st-wasm --target web --dev --features nn` from the repository root.
Serve the resulting `bindings/st-wasm` directory over loopback HTTP and open
[`tests/native_causal_lm.html`](native_causal_lm.html). For Node, run
`node bindings/st-wasm/tests/native_causal_lm_node.mjs bindings/st-wasm/pkg/spiraltorch_wasm.js`.
Both execute the same four artificial controls and compare against native Python results.

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
