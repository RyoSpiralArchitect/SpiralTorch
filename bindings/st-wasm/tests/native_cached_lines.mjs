// Two independent oracles: a prior native padded scorer, and one full WASM
// prefix per teacher-forced target (including the stop after each candidate).
export function checkCachedLines(M, models, oracle) {
  const check = (ok, msg) => { if (!ok) throw Error(msg); };
  const words = ids => Uint32Array.from(ids);
  const {contents, stop, cases} = oracle;
  const flat = words(contents.flat()), lens = words(contents.map(x => x.length));
  let tokens = 0, refusals = 0, maxPadded = 0, maxFull = 0, maxTiled = 0;
  const error = (a, b) => {
    const d = Math.abs(a-b);
    check(d <= 3e-6, 'line score changed: ' + d);
    return d;
  };
  function refuse(fn) {
    try { fn(); } catch { refusals++; return; }
    throw Error('admitted invalid line scoring input');
  }
  for (const f of models.cases) {
    const base = M.NativeCausalLm.fromGpt2(f.config, Uint8Array.from(atob(models.weights), c => c.charCodeAt(0)));
    const art = f.geometry && M.GeometryAdapterArtifact.fromJson(f.geometry, f.geometry_id, f.base_id);
    const m = art ? base.withGeometry(art) : base;
    const root = m.prefill(words(models.prompt), 1024);
    try {
      const start = Array.from(root.nextLogits());
      const padded = cases[f.name], report = JSON.parse(root.scoreLinesJson(flat, lens, stop)), s = report.scores;
      const tiled = JSON.parse(root.scoreLinesJson(flat, lens, stop, 11));
      const perToken = f.name.endsWith('/history_gain') ? 15 : 12;
      check(report.cached_prompt_tokens === 2 && report.decoded_content_tokens === 7 &&
            report.decoder_chunks === 6 && report.peak_fork_cache_values === 12*perToken, 'sharing counters');
      for (const [r, batches, peak] of [[report,6,22],[tiled,7,11]]) {
        check(r.readout_batches === batches && r.peak_projected_logit_values === peak &&
              r.scanned_logit_values === 88 && r.retained_edge_logits === 12 && r.decoder_chunks === 6,
              'streamed vocabulary counts');
      }
      for (const k of ['stop_id','target_counts','prefix_nodes','decoder_lanes','shared_content_tokens','winner_index']) {
        check(JSON.stringify(s[k]) === JSON.stringify(padded[k]), 'changed ' + k);
        check(JSON.stringify(tiled.scores[k]) === JSON.stringify(padded[k]), 'changed tiled ' + k);
      }
      for (const k of ['line_nlls','conditional_nlls','token_nlls','menu_set_nll']) {
        const a = [s[k]].flat(2), b = [padded[k]].flat(2);
        check(a.length === b.length, 'padded score dimensions');
        const t = [tiled.scores[k]].flat(2);
        check(t.length === b.length, 'tiled score dimensions');
        for (let i=0; i<a.length; i++) {
          maxPadded = Math.max(maxPadded, error(a[i], b[i]));
          maxTiled = Math.max(maxTiled, error(t[i], b[i]));
        }
      }
      for (const [i, line] of contents.entries()) {
        const prefix = [...models.prompt], targets = [...line, stop];
        check(s.token_nlls[i].length === targets.length, 'missing termination cost');
        let total = 0;
        for (const [j, id] of targets.entries()) {
          const v = Array.from(m.nextLogits(JSON.stringify(prefix))), hi = Math.max(...v);
          const cost = Math.log(v.reduce((a,x) => a + Math.exp(x-hi), 0)) + hi - v[id];
          maxFull = Math.max(maxFull, error(cost, s.token_nlls[i][j]));
          total += cost; prefix.push(id); tokens++;
        }
        maxFull = Math.max(maxFull, error(total, s.line_nlls[i]));
      }
      for (const args of [[flat, words([1]), stop], [flat, words([0, flat.length]), stop],
          [words([2,1]), words([1,1]), stop], [words([1,1]), words([1,1]), stop],
          [words([0]), words([1]), stop], [words([99]), words([1]), stop],
          [flat, lens, true], [flat, lens, -1], [flat, lens, 0.5], [flat, lens, 11],
          [new Int32Array(flat), lens, stop], [flat, [1], stop], [flat, new Uint32Array(33), stop]]) {
        refuse(() => root.scoreLinesJson(...args));
      }
      const small = m.prefill(words(models.prompt), 12*perToken-1);
      try { refuse(() => small.scoreLinesJson(flat, lens, stop)); } finally { small.free(); }
      for (const cap of [0,10,1048577,-1,11.2,true,null,'11',NaN]) {
        refuse(() => root.scoreLinesJson(flat,lens,stop,cap));
      }
      const end = root.nextLogits();
      check(root.tokenCount === models.prompt.length && start.every((v,i) => v === end[i]),
            'source changed after scoring/refusals');
    } finally {
      root.free();
      if (art) { m.free(); art.free(); }
      base.free();
    }
  }
  check(tokens === 96, 'incomplete score comparisons');
  return {models: models.cases.length, tokens, tiled_tokens: tokens, refusals,
          max_abs_old_padded: maxPadded, max_abs_tiled_padded: maxTiled, max_abs_full: maxFull};
}
