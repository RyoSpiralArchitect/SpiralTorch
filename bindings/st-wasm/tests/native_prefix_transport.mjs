// Artificial decoder only. Check WASM SGD, pinned reload, and a compact cache against native Python.
export function checkPrefixTransport(M, fixture, reference) {
  const need = (ok, why) => { if (!ok) throw Error(why); };
  const turns = ['identity', 'quarter_turn', 'inverse_quarter_turn'];
  need(Object.keys(reference).length === 3, 'unexpected prefix transport reference');
  const prompt = [3, 1, 9, 5, 2, 10];
  const f = fixture.cases.find(c => c.name === 'ordinary/wave');
  const base = M.NativeCausalLm.fromGpt2(f.config, Uint8Array.from(atob(fixture.weights), c => c.charCodeAt(0)));
  let max = 0, cases = 0;
  function near(a, b) {
    need(a.length === b.length && a.length > 0, 'prefix transport dimensions');
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(a[i] - b[i]);
      need(d < 3e-6, 'prefix transport mismatch: ' + d);
      max = Math.max(max, d);
    }
  }
  try {
    for (const turn of turns) {
      const owned = [], keep = x => (owned.push(x), x);
      try {
        const r = reference[turn];
        need(JSON.parse(r.origin).placements[0].recipe.rotation === turn, 'wrong prefix transport mode');
        const art = keep(M.GeometryAdapterArtifact.fromJson(r.origin, r.origin_id, base.baseModelIdentity));
        const model = keep(base.withGeometry(art));
        const g = keep(model.geometryTokenMenuGradients(Uint32Array.from(prompt), Uint32Array.of(7, 8), 0, 1));
        const p = JSON.parse(g.parameterGradientsJson())['transformer.h.1.mlp'];
        near(p.gate, r.grads.gate); near(p.bias, r.grads.bias); near([JSON.parse(g.reportJson()).loss], [r.loss]);
        const trained = keep(model.stepLineGeometrySgd(g, 1 / 32));
        const post = keep(trained.geometryArtifact());
        const pinned = keep(M.GeometryAdapterArtifact.fromJson(post.toJson(), post.artifactId, base.baseModelIdentity));
        const restored = keep(base.withGeometry(pinned));
        near(restored.nextLogits(JSON.stringify(prompt)), r.trained);
        const full = restored.nextLogits(JSON.stringify([...prompt, 7]));
        near(full, r.suffix);
        const root = keep(restored.prefill(Uint32Array.from(prompt), 90));
        const child = keep(root.withSuffix(Uint32Array.of(7)));
        need(root.cacheValues === 78 && child.cacheValues === 90, 'wrong compact memory accounting');
        near(child.nextLogits(), full); near(root.nextLogits(), r.trained);
        let refused = false;
        try { keep(child.withSuffix(Uint32Array.of(7))); } catch { refused = true; }
        need(refused, 'accepted an over-budget suffix');
        cases++;
      } finally { for (const h of owned.reverse()) h.free(); }
    }
  } finally { base.free(); }
  return {trained_cases: cases, max_abs_error: max};
}
