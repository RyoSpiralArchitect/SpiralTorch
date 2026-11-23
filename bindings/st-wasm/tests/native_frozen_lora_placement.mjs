// Cold-reconstruct the frozen decoder in real WASM, resume native SGD, and return the next cut.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const [pkg, bridge, returned] = process.argv.slice(2);
const url = pathToFileURL(resolve(pkg)), M = await import(url);
await M.default({module_or_path: readFileSync(new URL(url.href.replace(/\.js$/, '_bg.wasm')))});
const read = path => JSON.parse(readFileSync(path));
const data = read(new URL('./native_causal_lm_cache_fixture.json', import.meta.url));
const owner = read(new URL('./native_line_owners_bridge.json', import.meta.url));
const bytes = readFileSync(bridge), ref = JSON.parse(bytes);
assert.equal(ref.schema, 'spiraltorch.synthetic_frozen_lora_placement.v1');
const f = data.cases.find(c => c.name === 'ordinary/wave');
const base = M.NativeCausalLm.fromGpt2(f.config, Buffer.from(data.weights, 'base64'));
const art = M.GeometryAdapterArtifact.fromJson(f.geometry, f.geometry_id, f.base_id);
const geo = base.withGeometry(art), active = geo.withLoraJson(owner.lora, owner.lora_id);
const frozen = active.freezeLora(), Session = M.NativeCausalLmLineTrainingSession;
const ck = s => { const c = s.checkpoint(); try { return [c.payload,c.id]; } finally { c.free(); } };
const pin = bytes => createHash('sha256').update(bytes).digest('hex');
let worst = 0, n = 0;
function near(a,b) {
  if (typeof b === 'number') {
    assert.ok(Number.isFinite(a) && Math.abs(a-b) <= 3e-6, a+' vs '+b);
    worst = Math.max(worst, Math.abs(a-b)); return;
  }
  assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort());
  for (const k of Object.keys(b)) near(a[k], b[k]);
}
function state(m,path) {
  const a = m.geometryArtifact();
  try {
    const [p] = JSON.parse(a.toJson()).placements;
    assert.equal(p.path,path);
    return {parameters: Object.fromEntries(p.tensors.map(t => [t.name,t.values])),
            logits: Array.from(m.nextLogits('[2,5,1]'))};
  } finally { a.free(); }
}
const cases = {};
try {
  assert.equal(frozen.baseModelIdentity,ref.base);
  assert.equal(frozen.frozenLoraArtifactId,ref.frozen);
  assert.equal(frozen.loraArtifactId,undefined);
  assert.equal(frozen.frozenLoraJson(),owner.lora);
  assert.deepEqual(frozen.nextLogits('[2,5,1]'),active.nextLogits('[2,5,1]'));
  assert.throws(() => frozen.withSingleGeometryAt('x'.repeat(65)));
  assert.throws(() => active.withSingleGeometryAt('transformer.h.1.mlp'));
  assert.throws(() => frozen.withLoraJson(owner.lora,owner.lora_id));
  for (const [path,c] of Object.entries(ref.cases)) {
    const model = frozen.withSingleGeometryAt(path);
    const a = model.geometryArtifact();
    try { assert.equal(a.artifactId,c.origin); } finally { a.free(); }
    try {
      near(state(model,path),c.start);
      const s = Session.fromJson(model,ref.plan,...c.cut);
      try {
        assert.deepEqual(ck(s),c.cut); assert.equal(s.completedUpdates,1);
        const step = JSON.parse(s.stepJson());
        assert.equal(step.pre_lora_id,undefined); assert.equal(step.post_lora_id,undefined);
        assert.notEqual(step.pre_geometry_id,step.post_geometry_id);
        assert.equal(s.completedUpdates,2);
        const m = s.modelSnapshot();
        try {
          assert.equal(m.frozenLoraJson(),owner.lora);
          near({loss:step.loss,...state(m,path)},c.steps[1]);
        } finally { m.free(); }
        cases[path] = ck(s); ++n;
      } finally { s.free(); }
    } finally { model.free(); }
  }
  assert.equal(n,6);
  writeFileSync(returned, JSON.stringify({reference_sha256:pin(bytes),cases}), {flag:'wx'});
  console.log(JSON.stringify({wasm_imports:n, wasm_updates:n, maximum_absolute_difference:worst, pretrained_calls:0}));
} finally { frozen.free(); active.free(); geo.free(); art.free(); base.free(); }
