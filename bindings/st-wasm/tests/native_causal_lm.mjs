// Shared by Node and the browser. Only the checked-in artificial safetensors are loaded.
export function checkNativeCausalLm(M, fixture, onSelected) {
  const need = (ok, why) => { if (!ok) throw Error(why); };
  const equal = (a, b, why) => need(JSON.stringify(a) === JSON.stringify(b), why);
  const near = (a, b, tol, what) => need(Math.abs(a - b) <= tol, what + ': ' + a + ' vs ' + b);
  let refusals = 0;
  function refuses(f, pattern = null) {
    try { f(); } catch (e) {
      if (pattern) need(pattern.test(String(e)), 'unexpected error: ' + e);
      refusals++;
      return;
    }
    throw Error('accepted invalid input: ' + pattern);
  }
  const wire = {shape: [1, 2], ids: [2, 0], contrasts: [[1, 0, [1]]]};
  const batch = obj => M.CausalLmDeclaredBatch.fromJson(JSON.stringify(obj));
  const input = batch(wire), cases = [];
  const score = (m, b = input) => JSON.parse(m.declaredSetNllJson(b));
  const keys = ['within_nll', 'coverage_nll', 'full_nll'];
  const toBytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  function checkSnapshot(m, expected) {
    const logits = m.nextLogits('[2]');
    need(logits instanceof Float32Array && logits.length === expected.logits.length, 'logit row');
    const p = Array.from(logits, Math.exp);
    const pair = p[0] + p[1], total = p.reduce((a, b) => a + b, 0);
    const direct = [Math.log(pair/p[0]), Math.log(total/pair), Math.log(total/p[0])];
    const s = score(m);
    for (let i = 0; i < p.length; i++) near(logits[i], expected.logits[i], 2e-6, 'native logits');
    for (const [i, key] of keys.entries()) {
      near(s[key], direct[i], 1e-11, 'independent probability ' + key);
      near(s[key], expected.scores[key], 2e-6, 'native score ' + key);
    }
    need(s.target_tokens === 1 && s.contrasted_tokens === 1, 'row counts');
    return s;
  }
  try {
    for (const f of fixture.cases) {
      const bytes = toBytes(f.weights);
      const owned = [], keep = x => (owned.push(x), x);
      try {
        const base = keep(M.NativeCausalLm.fromGpt2(f.config, bytes));
        bytes.fill(0);
        need(base.baseModelIdentity === f.base_id && base.geometryArtifact() === undefined, 'frozen base');
        const adapter = keep(M.GeometryAdapterArtifact.fromJson(f.geometry, f.origin_id, f.base_id));
        const origin = keep(base.withGeometry(adapter));
        const start = score(origin);
        for (const exp of f.steps) {
          const advance = () => origin.stepDeclaredSetGeometryAtDistance(input, exp.distance);
          if (exp.name === 'infeasible') {
            refuses(advance, /feasible interval/);
            checkSnapshot(origin, exp);
            cases.push({name: exp.name, selected: 'origin'});
            continue;
          }
          const step = advance(), report = JSON.parse(step.reportJson());
          const result = keep(step.intoModel());
          const selected = checkSnapshot(result, exp);
          const accept = exp.name !== 'finite_reject', g = report.proposal_guard;
          equal(report.before, start, 'same pre-update scores');
          need(report.accepted === accept && (g.mix > 0) === (exp.name === 'repair'), 'wrong decision');
          need(g.applied_loss_dots.every(x => x < 0), 'non-descending proposal');
          for (const key of keys) near(report.proposed[key], exp.proposed[key], 2e-6, 'native proposal ' + key);
          equal(selected, accept ? report.proposed : report.before, 'rescored selected model');
          need(accept
            ? report.proposed.within_nll < start.within_nll && report.proposed.coverage_nll < start.coverage_nll
            : report.proposed.within_nll < start.within_nll && report.proposed.coverage_nll > start.coverage_nll,
            'expected finite improvements or coverage regression');
          const saved = keep(result.geometryArtifact());
          need(saved.artifactId === report.result_geometry_id && g.pre_geometry_id === f.origin_id &&
            g.post_geometry_id !== f.origin_id &&
            report.result_geometry_id === (accept ? g.post_geometry_id : f.origin_id), 'selected artifact identity');
          const pinned = keep(M.GeometryAdapterArtifact.fromJson(saved.toJson(), saved.artifactId, f.base_id));
          const restored = keep(base.withGeometry(pinned));
          equal(score(restored), selected, 'reloaded scores');
          equal([...restored.nextLogits('[2]')], [...result.nextLogits('[2]')], 'reloaded logits');
          onSelected?.({name: exp.name, base_id: f.base_id, geometry_id: saved.artifactId,
            geometry: saved.toJson(), scores: selected, logits: [...result.nextLogits('[2]')]});
          cases.push({name: exp.name, selected: accept ? 'proposal' : 'origin',
            proposed_set_probability: Math.exp(-report.proposed.coverage_nll),
            returned_set_probability: Math.exp(-selected.coverage_nll)});
        }
        equal(score(origin), start, 'origin mutated');
        need(keep(origin.geometryArtifact()).toJson() === f.geometry, 'origin bytes changed');
        // Multiple windows, ignored targets, one undeclared active token. Denominator is three.
        const mixed = keep(batch({shape: [2, 3], ids: [2,0,1, 2,0,2], labels: [-100,0,-100, -100,0,2],
          contrasts: [[1,0,[1]], [5,2,[0]]]}));
        const s = score(base, mixed), v = JSON.parse(f.config).vocab_size;
        need(s.target_tokens === 3 && s.contrasted_tokens === 2, 'shift or ignored row');
        near(s.within_nll, (2*Math.log(2)+Math.log(v))/3, 1e-11, 'mixed within');
        near(s.coverage_nll, 2*Math.log(v/2)/3, 1e-11, 'mixed coverage');
        for (const contrasts of [[[0,2,[1]]], [[1,1,[0]]], [[1,0,[1,1]]]]) {
          refuses(() => score(origin, keep(batch({...wire, contrasts}))));
        }
        refuses(() => score(origin, keep(batch({...wire, labels: [-100,-100]}))));
        refuses(() => origin.stepDeclaredSetGeometryAtDistance(input, '1'));
        refuses(() => origin.stepDeclaredSetGeometryAtDistance(input, NaN));
        refuses(() => base.stepDeclaredSetGeometryAtDistance(input, 1/65536));
        refuses(() => origin.nextLogits('[4294967296]'));
        refuses(() => origin.nextLogits('[]'));
      } finally { for (const h of owned.reverse()) h.free(); }
    }
    for (const bad of [{...wire, shape: [4294967295, 2]}, {...wire, ids: [-1,0]},
                       {...wire, ids: [2.5,0]}, {...wire, labels: [0]}, {...wire, extra: 1}]) {
      refuses(() => batch(bad));
    }
    const f = fixture.cases[0], raw = toBytes(f.weights);
    refuses(() => M.NativeCausalLm.fromGpt2(f.config, new Uint16Array(12)), /Uint8Array/);
    refuses(() => M.NativeCausalLm.fromGpt2(f.config, raw.subarray(0, 7)), /byte budget/);
    refuses(() => M.NativeCausalLm.fromGpt2(f.config, raw.subarray(0, 16)));
    if (typeof SharedArrayBuffer !== 'undefined') {
      refuses(() => M.NativeCausalLm.fromGpt2(f.config, new Uint8Array(new SharedArrayBuffer(16))));
    }
    const other = fixture.cases[1];
    const a = M.GeometryAdapterArtifact.fromJson(f.geometry, f.origin_id, f.base_id);
    const b = M.NativeCausalLm.fromGpt2(other.config, toBytes(other.weights));
    try { refuses(() => b.withGeometry(a), /different native base/); } finally { a.free(); b.free(); }
    equal(cases.map(x => x.name), ['repair', 'infeasible', 'accept', 'finite_reject'], 'four fixed controls');
    return {cases, refusals, pretrained_calls: 0};
  } finally { input.free(); }
}
