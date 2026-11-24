// The reference contains complete reports and all gradients from the prior dense CPU wheel.
export function checkLineGradients(M, fixture, references) {
  const need = (ok, msg) => { if (!ok) throw Error(msg); };
  let checks = 0, components = 0, refusals = 0, maxGrad = 0, maxReport = 0;
  function refuse(fn) {
    try { fn(); } catch { refusals++; return; }
    throw Error('invalid line training request admitted');
  }
  function close(a,b) {
    if (typeof b === 'number') {
      const e = Math.abs(a-b);
      need(typeof a === 'number' && e <= 3e-6, 'line VJP numeric mismatch: '+e);
      return e;
    }
    need(a && b && typeof a === 'object' && Array.isArray(a) === Array.isArray(b), 'line VJP shape');
    const keys = Object.keys(b).sort();
    need(JSON.stringify(Object.keys(a).sort()) === JSON.stringify(keys), 'line VJP keys');
    return Math.max(0,...keys.map(k=>close(a[k],b[k])));
  }
  const prompt = Uint32Array.from(fixture.prompt), flat = Uint32Array.of(1,1,2,3,4), lens = Uint32Array.of(1,2,2);
  const specs = {set:{kind:'set'},menu:{kind:'menu',gold_index:1,mix:0.5},subset:{kind:'subset',accepted_indices:[0,2]}};
  for (const f of fixture.cases.filter(x => x.geometry)) {
    const base = M.NativeCausalLm.fromGpt2(f.config,Uint8Array.from(atob(fixture.weights),c=>c.charCodeAt(0)));
    const art = M.GeometryAdapterArtifact.fromJson(f.geometry,f.geometry_id,f.base_id);
    const model = base.withGeometry(art);
    const run = (spec, cap) => model.geometryLineGradients(prompt,flat,lens,0,JSON.stringify(spec),cap);
    try {
      const start = model.nextLogits(JSON.stringify(fixture.prompt));
      for (const [kind,spec] of Object.entries(specs)) {
        const ref = references.find(x=>x.host === f.name && x.kind === kind);
        need(ref && ref.geometry_id === f.geometry_id, 'unknown gradient reference');
        for (const cap of [undefined,11]) {
          const g = run(spec,cap);
          try {
            const w = JSON.parse(g.trainingWorkJson()), params = JSON.parse(g.parameterGradientsJson());
            need(w.prefix_rows === 5 && w.vocabulary === 11 && w.forward_head_calls === (cap ? 9 : 1) &&
                 w.backward_head_calls === (cap ? 5 : 1) && w.replayed_head_rows === (cap ? 4 : 0) &&
                 w.peak_logit_values === (cap ? 11 : 55) && w.peak_extra_seed_values === (cap ? 11 : 55), 'training work');
            maxGrad = Math.max(maxGrad,close(params,ref.parameters));
            maxReport = Math.max(maxReport,close(JSON.parse(g.reportJson()),ref.report));
            components += Object.values(params).flatMap(Object.values).flat().length;
            if (cap) {
              const next = model.stepLineGeometrySgd(g,0.01);
              try {
                const saved = next.geometryArtifact();
                try { need(saved && saved.artifactId !== f.geometry_id,'geometry not updated'); } finally { saved?.free(); }
                refuse(()=>next.stepLineGeometrySgd(g,0.01));
                refuse(()=>model.stepLineGeometrySgd(g,true));
              } finally { next.free(); }
            }
            checks++;
          } finally { g.free(); }
        }
      }
      for (const cap of [10,1048577,true,null,1.5]) refuse(()=>run(specs.set,cap));
      for (const bad of [{kind:'bad'},{kind:'set',junk:1},{kind:'menu',gold_index:0,mix:2},
                         {kind:'subset',accepted_indices:[]}]) refuse(()=>run(bad,11));
      refuse(()=>model.geometryLineGradients(prompt,flat,Uint32Array.of(9,9),0,'{"kind":"set"}'));
      refuse(()=>base.geometryLineGradients(prompt,flat,lens,0,'{"kind":"set"}'));
      const end = model.nextLogits(JSON.stringify(fixture.prompt));
      need(start.length === end.length && start.every((x,i)=>x===end[i]),'origin mutated');
    } finally { model.free(); art.free(); base.free(); }
  }
  need(checks === 24 && references.length === 12, 'incomplete line gradient coverage');
  return {checks,components,refusals,max_abs_grad:maxGrad,max_abs_report:maxReport};
}
