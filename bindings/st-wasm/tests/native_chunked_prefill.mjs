const schema = 'spiraltorch.native_causal_lm.chunked_prefill_plan.v1';
const routes = ['prefillChunked', 'prefillChunkedLastOutput'];

function need(ok, why) {
  if (!ok) throw Error(why);
}

function using(handle, run) {
  try { return run(handle); } finally { handle.free(); }
}

function logitBits(x) {
  need(x instanceof Float32Array && x.length > 0 && x.every(Number.isFinite), 'invalid logits');
  return new Uint32Array(x.buffer, x.byteOffset, x.length);
}

function bitsEqual(a, b, label) {
  need(a.length === b.length && a.every((word, i) => word === b[i]), label + ': changed logit bits');
}

// A real long prefix with tiny generated weights. No larger archive or cache budget is needed.
function longFixture() {
  const width = 8, vocab = 11, positions = 1153, layers = 2, inner = 12;
  const cfg = {model_type: 'gpt2', vocab_size: vocab, n_positions: positions,
    n_embd: width, n_head: 8, n_layer: layers, n_inner: inner};
  const entries = [];
  const head = {};
  let bytes = 0;
  function add(name, shape) {
    const count = shape.reduce((a, b) => a * b, 1);
    head[name] = {dtype: 'F32', shape, data_offsets: [bytes, bytes + 4 * count]};
    entries.push({name, count, start: bytes});
    bytes += 4 * count;
  }
  add('wte.weight', [vocab, width]);
  add('wpe.weight', [positions, width]);
  for (let i = 0; i < layers; ++i) {
    for (const norm of ['ln_1', 'ln_2']) {
      add('h.' + i + '.' + norm + '.weight', [width]);
      add('h.' + i + '.' + norm + '.bias', [width]);
    }
    for (const [op, input, output] of [
      ['attn.c_attn', width, 3 * width], ['attn.c_proj', width, width],
      ['mlp.c_fc', width, inner], ['mlp.c_proj', inner, width],
    ]) {
      add('h.' + i + '.' + op + '.weight', [input, output]);
      add('h.' + i + '.' + op + '.bias', [output]);
    }
  }
  add('ln_f.weight', [width]);
  add('ln_f.bias', [width]);
  const text = JSON.stringify(head);
  const header = new TextEncoder().encode(text.padEnd(Math.ceil(text.length / 8) * 8));
  const weights = new Uint8Array(8 + header.length + bytes);
  const view = new DataView(weights.buffer);
  view.setBigUint64(0, BigInt(header.length), true);
  weights.set(header, 8);
  for (const [seed, {name, count, start}] of entries.entries()) {
    for (let i = 0; i < count; ++i) {
      const value = /ln_.*\.weight$/.test(name) ? 1 : ((i * 17 + seed * 11) % 101 - 50) / 1024;
      view.setFloat32(8 + header.length + start + 4 * i, value, true);
    }
  }
  return {config: JSON.stringify(cfg), weights};
}

export function checkChunkedPrefill(M, fixture) {
  let roots = 0, forks = 0, refusals = 0, maxAbsReference = 0;
  const hasSharedBuffers = typeof SharedArrayBuffer !== 'undefined';
  const close = (xs, ys, label) => {
    need(xs.length > 0 && xs.length === ys.length, label + ': changed vocabulary');
    const d = Math.max(...xs.map((x, i) => Math.abs(x - ys[i])));
    need(d <= 3e-6, label + ': reference error ' + d);
    maxAbsReference = Math.max(maxAbsReference, d);
  };
  const rejects = (run, reason) => {
    let value;
    try { value = run(); }
    catch (error) {
      need(reason.test(String(error)), 'wrong rejection: ' + String(error));
      ++refusals;
      return;
    }
    value?.free?.();
    throw Error('unexpectedly accepted: ' + reason);
  };
  function compare(model, {prefix, tails, refs, cap, unit, cache, geometry, label}) {
    const n = prefix.length;
    const payload = Uint32Array.of(999, ...prefix, 999);
    const ids = payload.subarray(1, n + 1);
    function compareOwned(a, b, steps, ref) {
      need(a.tokenCount === steps && b.tokenCount === steps, label + ': token count');
      need(a.cacheValues === steps * unit && b.cacheValues === a.cacheValues, label + ': cache words');
      need((a.geometryArtifactId ?? null) === geometry && (b.geometryArtifactId ?? null) === geometry,
        label + ': geometry ownership');
      bitsEqual(logitBits(a.nextLogits()), logitBits(b.nextLogits()), label);
      if (ref) close(a.nextLogits(), ref, label);
    }
    using(model.prefillChunked(ids, cache, cap), a =>
      using(model.prefillChunkedLastOutput(ids, cache, cap), b => {
        payload.fill(999);
        compareOwned(a, b, n, refs[0]);
        ++roots;
        const original = logitBits(b.nextLogits()).slice();
        for (const [i, tail] of tails.entries()) {
          const suffix = Uint32Array.from(tail);
          using(a.withSuffix(suffix), x => using(b.withSuffix(suffix), y => {
            compareOwned(x, y, n + tail.length, refs[i + 1]);
            ++forks;
          }));
        }
        b.nextLogits().fill(NaN);
        bitsEqual(logitBits(b.nextLogits()), original, label + ': source after forks');
        compareOwned(a, b, n, refs[0]);
      }));
  }

  const prefix = [...fixture.prompt, ...fixture.chunk];
  const {tails} = fixture;
  const weights = Uint8Array.from(atob(fixture.weights), c => c.charCodeAt(0));
  for (const f of fixture.cases) {
    using(M.NativeCausalLm.fromGpt2(f.config, weights), base => {
      const unit = f.name.endsWith('/history_gain') ? 15 : 12;
      const check = model => {
        for (const cap of [undefined, 1, 2, 3, 64]) {
          const steps = Math.min(prefix.length, cap ?? 64);
          const plan = JSON.parse(model.planChunkedPrefillJson(prefix.length, 7 * unit, cap));
          need(plan.schema === schema && plan.chunk_steps === steps &&
            plan.chunks === Math.ceil(prefix.length / steps), f.name + ': plan changed');
          compare(model, {prefix, tails, refs: f.refs.slice(1), cap, unit, cache: 7 * unit,
            geometry: f.geometry_id, label: f.name + ' chunk ' + (cap ?? 'default')});
        }
      };
      if (f.geometry) {
        using(M.GeometryAdapterArtifact.fromJson(f.geometry, f.geometry_id, f.base_id), artifact =>
          using(base.withGeometry(artifact), check));
      } else {
        check(base);
      }
      if (f.name === 'lag/history_gain') {
        const record = JSON.parse(f.geometry);
        record.placements[0].path = 'transformer.h.1.mlp';
        using(M.GeometryAdapterArtifact.createJson(JSON.stringify(record)), artifact =>
          using(base.withGeometry(artifact), model => {
            const refs = [prefix, ...tails.map(t => [...prefix, ...t])]
              .map(p => model.nextLogits(JSON.stringify(p)));
            compare(model, {prefix, tails, refs, cap: 2, unit, cache: 7 * unit,
              geometry: artifact.artifactId, label: 'last-block causal geometry'});
          }));
      }
      if (f.name === 'ordinary/plain') {
        const ids = Uint32Array.from(prefix);
        for (const route of routes) {
          for (const invalid of [prefix, new Int32Array(ids)]) {
            rejects(() => base[route](invalid, 1024), /Uint32Array/);
          }
          for (const invalid of [new Uint32Array(), new Uint32Array(65537)]) {
            rejects(() => base[route](invalid, 1024), /token count/);
          }
          if (hasSharedBuffers) {
            const shared = new Uint32Array(new SharedArrayBuffer(4));
            rejects(() => base[route](shared, 1024), /ArrayBuffer/);
          }
          rejects(() => base[route](ids, prefix.length * unit - 1), /cache/);
          rejects(() => base[route](Uint32Array.of(2, 11), 1024), /token ID/);
          for (const bad of [0, 1.5, false, null, '1024', 1048577]) {
            rejects(() => base[route](ids, bad), /cache value limit/);
          }
          for (const bad of [0, 1.5, false, null, '2', 65]) {
            rejects(() => base[route](ids, 1024, bad), /chunk step limit/);
          }
        }
        for (const bad of [0, 65537, '5', 1.5]) {
          rejects(() => base.planChunkedPrefillJson(bad, 1024), /token count/);
        }
        rejects(() => base.planChunkedPrefillJson(prefix.length, 1024, 65), /chunk step limit/);
      }
    });
  }

  const long = longFixture();
  const longPrefix = Array.from({length: 259}, (_, i) => (3 + i * 7) % 11);
  const longTails = [[3], [8, 4]];
  const longCache = 32 * (longPrefix.length + 2);
  let longPlan, geometryBudget;
  using(M.NativeCausalLm.fromGpt2(long.config, long.weights), model => {
    const ids = Uint32Array.from(longPrefix);
    rejects(() => model.prefill(ids, longCache), /attention pairs/);
    longPlan = JSON.parse(model.planChunkedPrefillJson(ids.length, longCache));
    need(longPlan.schema === schema && longPlan.chunk_steps === 64 && longPlan.chunks === 5 &&
      longPlan.attention_pairs === 667792 && longPlan.geometry_prefix_work === 0,
      'long prompt did not use the expected plan');
    const refs = using(model.prefill(ids.subarray(0, 64), longCache), root =>
      using(root.withSuffix(ids.subarray(64)), oracle =>
        [oracle.nextLogits(), ...longTails.map(t =>
          using(oracle.withSuffix(Uint32Array.from(t)), c => c.nextLogits()))]));
    for (const cap of [undefined, 17]) {
      compare(model, {prefix: longPrefix, tails: longTails, refs, cap, unit: 32,
        cache: longCache, geometry: null, label: 'long prompt chunk ' + (cap ?? 'default')});
    }
    for (const route of routes) {
      rejects(() => model[route](new Uint32Array(1152), 32 * 1152), /total attention pairs/);
    }
    rejects(() => model.planChunkedPrefillJson(1152, 32 * 1152), /total attention pairs/);

    const artifact = M.GeometryAdapterArtifact.createJson(JSON.stringify({
      schema: 'spiraltorch.geometry_adapter_artifact.v1',
      base_model_identity: model.baseModelIdentity,
      policy: 'full_prefix_f32_no_cache.v1',
      training: false,
      placements: [{
        path: 'transformer.h.0.attn', training: false,
        recipe: {schema: 'spiraltorch.elliptic_causal_residual_adapter.v1',
          features: 8, strength: 0.5,
          warp: {curvature_radius: 1.25, sheet_count: 2, spin_harmonics: 2}},
        tensors: [['orientation.weight', [2, 8]], ['orientation.bias', [2]], ['readout.weight', [8, 9]]]
          .map(([name, shape]) => ({name, shape, role: 'parameter', requires_grad: false,
            values: Array.from({length: shape.reduce((a, b) => a * b, 1)}, (_, i) => (i % 7 - 3) / 64)})),
      }],
    }));
    using(artifact, a => using(model.withGeometry(a), decorated => {
      const n = 512, unit = 40;
      const plain = JSON.parse(model.planChunkedPrefillJson(n, unit * n, 1));
      need(plain.attention_pairs === 2101248 && plain.geometry_prefix_work === 0,
        'ordinary one-token chunks should pass the attention cap');
      const coarse = JSON.parse(decorated.planChunkedPrefillJson(n, unit * n));
      need(coarse.chunks === 8 && coarse.geometry_prefix_work === 835584,
        'elliptic prefix work did not accumulate over chunks');
      rejects(() => decorated.planChunkedPrefillJson(n, unit * n, 1), /geometry prefix work/);
      for (const route of routes) {
        rejects(() => decorated[route](new Uint32Array(n), unit * n, 1), /geometry prefix work/);
      }
      geometryBudget = {prompt_tokens: n, ordinary_attention_pairs: plain.attention_pairs,
        accepted_chunk_steps: coarse.chunk_steps, geometry_prefix_work: coarse.geometry_prefix_work,
        refused_chunk_steps: 1};
      const small = longPrefix.slice(0, 9);
      const refs = [small, ...longTails.map(t => [...small, ...t])]
        .map(p => decorated.nextLogits(JSON.stringify(p)));
      compare(decorated, {prefix: small, tails: longTails, refs, cap: 3, unit, cache: unit * 11,
        geometry: a.artifactId, label: 'elliptic causal replay'});
    }));
  });
  need(fixture.cases.length === 6 && roots === 34 && forks === 68 &&
    refusals === (hasSharedBuffers ? 50 : 48), 'incomplete chunked prefill checks');
  return {roots, forks, refusals, max_abs_reference: maxAbsReference, exact: true,
    shared_buffers_checked: hasSharedBuffers, long_plan: longPlan, eager_refused: true,
    geometry_budget: geometryBudget};
}
