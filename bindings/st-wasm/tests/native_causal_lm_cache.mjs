// Compare against a separate native build as well as the current decoder's full-prefix path.
export function checkCachedSuffix(M, fixture) {
  const need = (ok, why) => { if (!ok) throw Error(why); };
  const diff = (x, y) => {
    need(x.length && x.length === y.length, 'vocabulary shape changed');
    const d = Math.max(...x.map((v, i) => Math.abs(v - y[i])));
    need(d <= 3e-6, 'cached logit mismatch: ' + d);
    return d;
  };
  const words = ids => Uint32Array.from(ids);
  const {prompt, chunk, tails} = fixture;
  const paths = [prompt, [...prompt, ...chunk], ...tails.map(t => [...prompt, ...chunk, ...t])];
  let maxNative = 0, maxFull = 0, maxSerial = 0, rows = 0, refusals = 0;
  function refuses(fn) {
    try { fn(); } catch { refusals++; return; }
    throw Error('unexpectedly admitted cached request');
  }
  for (const f of fixture.cases) {
    const base = M.NativeCausalLm.fromGpt2(f.config, Uint8Array.from(atob(fixture.weights), c => c.charCodeAt(0)));
    const art = f.geometry && M.GeometryAdapterArtifact.fromJson(f.geometry, f.geometry_id, f.base_id);
    const model = art ? base.withGeometry(art) : base;
    let root;
    let full;
    const perToken = f.name.endsWith('/history_gain') ? 15 : 12; // f32 K + V + optional site history
    try {
      full = paths.map(p => model.nextLogits(JSON.stringify(p)));
      const view = words([999, ...prompt, 999]).subarray(1, prompt.length+1);
      root = model.prefill(view, 7*perToken);
      view.fill(999);
      for (const budget of [0, 1, 2.5, true, '128', 1048577]) refuses(() => model.prefill(words(prompt), budget));
    } finally {
      if (art) { model.free(); art.free(); }
      base.free();
    }
    const owned = [root], keep = h => (owned.push(h), h);
    try {
      const child = keep(root.withSuffix(words(chunk)));
      const endpoints = [root, child, ...tails.map(t => keep(child.withSuffix(words(t))))];
      for (const [i, ctx] of endpoints.entries()) {
        const p = paths[i], logits = Array.from(ctx.nextLogits());
        need(ctx.tokenCount === p.length && ctx.cacheValues === perToken*p.length, 'cache dimensions');
        need((ctx.geometryArtifactId ?? null) === f.geometry_id, 'lost geometry identity');
        maxNative = Math.max(maxNative, diff(logits, f.refs[i]));
        maxFull = Math.max(maxFull, diff(logits, full[i]));
        let serial = root;
        for (const t of p.slice(prompt.length)) serial = keep(serial.withSuffix(Uint32Array.of(t)));
        maxSerial = Math.max(maxSerial, diff(logits, serial.nextLogits()));
        rows++;
      }
      need(endpoints[2].nextLogits().some((v,i) => Math.abs(v-f.refs[3][i]) > .01), 'branches collapsed');
      child.nextLogits().fill(99);
      for (const bad of [[], [3], new Int32Array([3]), new Uint32Array(), words([3,999]),
                         new Uint32Array(3), new Uint32Array(65537)]) {
        refuses(() => child.withSuffix(bad));
      }
      if (typeof SharedArrayBuffer !== 'undefined') {
        refuses(() => child.withSuffix(new Uint32Array(new SharedArrayBuffer(4))));
      }
      diff(child.nextLogits(), f.refs[1]);
      diff(keep(child.withSuffix(words(tails[0]))).nextLogits(), f.refs[2]);
      need(root.tokenCount === prompt.length, 'source cache changed');
    } finally { for (const h of owned.reverse()) h.free(); }
  }
  need(fixture.cases.length === 6 && rows === 24, 'incomplete cache controls');
  return {states: rows, max_abs_native: maxNative, max_abs_full: maxFull, max_abs_serial: maxSerial, refusals};
}
