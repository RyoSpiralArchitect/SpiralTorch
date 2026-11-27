import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const [pkg, nativePath, dest] = process.argv.slice(2);
assert(pkg && nativePath && dest, 'usage: PKG.js NATIVE.json NEW_WASM.json');
const fixture = file => JSON.parse(readFileSync(new URL(file, import.meta.url)));
const pin = text => 'sha256:' + createHash('sha256').update(text).digest('hex');
const word = x => {
  assert(Number.isFinite(x));
  return new Uint32Array(Float32Array.of(x).buffer)[0];
};
const native = JSON.parse(readFileSync(nativePath));
const plan = JSON.parse(native.plan);
assert.equal(pin(native.plan), 'sha256:d8deba486de69e07582505001e2a67e2356d2b88039f9278b3a335d40eb5f263');
const anchors = plan.update.anchor_examples.map(i => plan.examples[i].prompt_ids);
const f = fixture('native_causal_lm_cache_fixture.json');
const owners = fixture('native_line_owners_bridge.json');
const model = f.cases.find(c => c.name === owners.host);
const glue = pathToFileURL(resolve(pkg));
const M = await import(glue);
await M.default({module_or_path: readFileSync(new URL(glue.href.replace(/\.js$/, '_bg.wasm')))});
const owned = [];
const own = value => (owned.push(value), value);
try {
  const base = own(M.NativeCausalLm.fromGpt2(model.config, Buffer.from(f.weights, 'base64')));
  const artifact = own(M.GeometryAdapterArtifact.fromJson(model.geometry, model.geometry_id, model.base_id));
  const geometry = own(base.withGeometry(artifact));
  const origin = own(geometry.withLoraJson(owners.lora, owners.lora_id));
  const policies = [origin.hostGeluTanh, origin.hostAttentionExp];
  assert.deepEqual(policies, ['portable_libm_tanh_f32.v1', 'portable_libm_exp_f64.v1']);
  assert.equal(pin(native.before.payload), native.before.id);
  const s = own(M.NativeCausalLmLineTrainingSession.fromJson(
    origin, native.plan, native.before.payload, native.before.id));
  function snapshot() {
    const ck = s.checkpoint();
    const m = s.modelSnapshot();
    try {
      assert.equal(pin(ck.payload), ck.id);
      const rows = anchors.map(p => Array.from(m.nextLogits(JSON.stringify(p)), word));
      return {payload: ck.payload, id: ck.id, cursor: s.completedUpdates, rows};
    } finally {
      m.free();
      ck.free();
    }
  }
  const before = snapshot();
  assert.equal(before.cursor, 4);
  assert.deepEqual(before, native.before);
  const e = plan.examples[before.cursor % plan.examples.length];
  const atCut = own(s.modelSnapshot());
  const g = own(atCut.lineAdaptationGradients(
    Uint32Array.from(e.prompt_ids), Uint32Array.from(e.candidates.flat()),
    Uint32Array.from(e.candidates.map(c => c.length)), e.stop_id, JSON.stringify(e.objective),
    'joint', plan.max_tile_values));
  const blocks = Object.fromEntries(Object.entries(JSON.parse(g.parameterGradientsJson()))
    .map(([site, tensors]) => [site, Object.fromEntries(Object.entries(tensors)
      .map(([name, values]) => [name, values.map(word)]))]));
  assert.deepEqual(blocks, native.blocks);
  assert.deepEqual(snapshot(), before);
  const attempt = JSON.parse(s.attemptStepJson());
  assert.equal(attempt.status, 'applied');
  const step = attempt.step;
  assert.deepEqual([step.update, step.mode], [5, 'joint']);
  assert.notEqual(step.pre_lora_id, step.post_lora_id);
  assert.notEqual(step.pre_geometry_id, step.post_geometry_id);
  const after = snapshot();
  assert.equal(after.cursor, 5);
  const receipt = {plan: native.plan, policies, before, after, blocks, loss: word(step.loss)};
  assert.deepEqual(receipt, native);
  writeFileSync(dest, JSON.stringify(receipt) + '\n', {flag: 'wx'});
} finally {
  for (const value of owned.reverse()) value.free();
}
