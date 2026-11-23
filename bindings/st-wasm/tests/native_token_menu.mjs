// Exercise the real WASM handle against independent next-logit costs. Generated weights only.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const url = pathToFileURL(resolve(process.argv[2]));
const M = await import(url);
await M.default({module_or_path: readFileSync(new URL(url.href.replace(/\.js$/, '_bg.wasm')))});
const data = JSON.parse(readFileSync(new URL('./native_causal_lm_cache_fixture.json', import.meta.url)));
const f = data.cases.find(x => x.name === 'ordinary/wave');
const owner = JSON.parse(readFileSync(new URL('./native_line_owners_bridge.json', import.meta.url)));
const base = M.NativeCausalLm.fromGpt2(f.config, Buffer.from(data.weights, 'base64'));
const art = M.GeometryAdapterArtifact.fromJson(f.geometry, f.geometry_id, f.base_id);
const geo = base.withGeometry(art), active = geo.withLoraJson(owner.lora, owner.lora_id);
const frozen = active.freezeLora(), model = frozen.withSingleGeometryAt('transformer.h.1.mlp');
const ids = Uint32Array.of(4, 1, 2), choices = Uint32Array.of(1, 3, 6);
const logits = Array.from(model.nextLogits('[4,1,2]'));
const ce = menu => Math.log(menu.reduce((s,x) => s + Math.exp(x-logits[3]), 0));
try {
  for (const mix of [0, .5, 1]) {
    const g = model.geometryTokenMenuGradients(ids, choices, 1, mix);
    try {
      const r = JSON.parse(g.reportJson());
      assert.equal(r.target_tokens, 1); assert.equal(r.rival_ids, 2);
      assert.ok(Math.abs(r.loss - ((1-mix)*ce(logits)+mix*ce(Array.from(choices, i=>logits[i])))) < 3e-6);
      assert.equal(JSON.parse(g.trainingWorkJson()), null);
      const step = model.stepLineGradientAtDistance(g, 1/256);
      const trace = JSON.parse(step.reportJson()), moved = step.intoModel();
      try {
        assert.equal(trace.mode, 'geometry'); assert.ok(trace.gradient_dot_step < 0);
        assert.equal(moved.frozenLoraJson(), owner.lora);
      } finally { moved.free(); }
    } finally { g.free(); }
  }
  for (const [options,gold,mix] of [[choices,0,true], [choices,.5,.5], [choices,3,.5],
                                   [Uint32Array.of(1,1),0,.5], [choices,0,NaN]]) {
    assert.throws(() => model.geometryTokenMenuGradients(ids, options, gold, mix));
  }
  console.log('3 first-token objectives, 3 fixed-distance steps, 5 rejections');
} finally { model.free(); frozen.free(); active.free(); geo.free(); art.free(); base.free(); }
