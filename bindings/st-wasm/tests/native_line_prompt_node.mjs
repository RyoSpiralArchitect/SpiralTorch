// Run the same synthetic plans in a real WASM instance. Never import a native ML runtime.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {stateRows, forwardKl} from './native_line_anchor_kl.mjs';

const [pkg, baseline] = process.argv.slice(2);
const url = pathToFileURL(resolve(pkg));
const M = await import(url);
await M.default({module_or_path: readFileSync(new URL(url.href.replace(/\.js$/, '_bg.wasm')))});
const read = p => JSON.parse(readFileSync(p));
const fixture = read(new URL('./native_causal_lm_cache_fixture.json', import.meta.url));
const owners = read(new URL('./native_line_owners_bridge.json', import.meta.url));
const ref = read(baseline);
assert.equal(ref.schema, 'spiraltorch.synthetic_line_prompt_identity.v1');
assert.equal(Object.keys(ref.cases).length, 90);
const f = fixture.cases.find(x => x.name === owners.host);
const base = M.NativeCausalLm.fromGpt2(f.config, Buffer.from(fixture.weights, 'base64'));
const hash = tokens => {
  const h = createHash('sha256').update('spiraltorch.native_causal_lm.line_prompt.v1\0');
  const data = Buffer.alloc(8);
  for (const n of [tokens.length, ...tokens]) { data.writeBigUInt64LE(BigInt(n)); h.update(data); }
  return 'sha256:' + h.digest('hex');
};
const near = (a,b,tolerance) => assert.ok(Math.abs(a-b)<=tolerance, a+' vs '+b);
const ck = s => { const c = s.checkpoint(); try { return [c.payload,c.id]; } finally { c.free(); } };
let art, geo, origin, attempts=0, applied=0, refusals=0, selectors=0;
try {
  art = M.GeometryAdapterArtifact.fromJson(f.geometry, f.geometry_id, f.base_id);
  geo = base.withGeometry(art);
  origin = geo.withLoraJson(owners.lora, owners.lora_id);
  for(const e of JSON.parse(owners.cases.joint.plan).examples) {
    const g = origin.lineAdaptationGradients(Uint32Array.from(e.prompt_ids), Uint32Array.from(e.candidates.flat()),
      Uint32Array.from(e.candidates.map(x=>x.length)), e.stop_id, JSON.stringify(e.objective), 'joint', 11);
    try { assert.equal(g.linePromptId, hash(e.prompt_ids)); } finally { g.free(); }
  }
  for(const c of Object.values(ref.cases)) {
    const p = JSON.parse(c.plan), s = new M.NativeCausalLmLineTrainingSession(origin,c.plan);
    try {
      let z = stateRows(s,p);
      for(const [i,e] of c.events.entries()) {
        const pre = ck(s), a = JSON.parse(s.attemptStepJson());
        const ok = a.status === 'applied', k = ok ? a.step.anchor_kl : a.refusal;
        ++attempts;
        assert.equal(ok, e.applied);
        assert.equal(!!k.raw_miss, e.branch === 'midpoint');
        assert.deepEqual(k.trials.map(t=>t.requested_l2), e.trials);
        assert.equal(k.anchor_readouts,e.readouts);
        if(k.raw_miss) assert.deepEqual(k.raw_miss.trials.map(t=>t.requested_l2),e.raw_trials);
        assert.equal(!!k.direction, 'selected' in e);
        if(k.direction) {
          ++selectors;
          const d = k.direction;
          assert.equal(c.bank[d.peak_anchor],e.peak);
          assert.equal(c.bank[d.selected_anchor],e.selected);
          assert.equal(d.loss_prompt_id, c.choice==='loss_prompt' ? hash(p.examples[i%3].prompt_ids) : undefined);
        }
        if(!ok) {
          ++refusals;
          assert.equal(k.reason,e.reason);
          assert.deepEqual(ck(s),pre);
          break;
        }
        ++applied;
        near(a.step.loss,e.loss,3e-6);
        const next=stateRows(s,p), kl=z.map((row,j)=>forwardKl(row,next[j]));
        kl.forEach((v,j)=>{near(v,k.per_anchor_kl[j],4e-15);near(v,e.kl[c.bank[j]],2e-8);});
        const mean=kl.reduce((a,b)=>a+b,0)/3;
        assert.ok(mean>=.98/65536 && mean<=1/65536 && Math.max(...kl)<=2/65536);
        z=next;
      }
      assert.equal(s.completedUpdates,c.completed);
      z.forEach((row,j)=>row.forEach((v,col)=>near(v,c.endpoint[c.bank[j]][col],3e-6)));
    } finally { s.free(); }
  }
} finally { origin?.free(); geo?.free(); art?.free(); base.free(); }
console.log(JSON.stringify({cells:90,attempts,applied,refusals,selectors,pretrained_calls:0}));
