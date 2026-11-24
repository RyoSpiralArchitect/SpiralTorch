// Cross-runtime suffix replay of native synthetic cuts. No pretrained weights.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {forwardKl, stateRows} from './native_line_anchor_kl.mjs';

const [glue, rawPath, cutsPath] = process.argv.slice(2);
const moduleUrl = pathToFileURL(resolve(glue));
const M = await import(moduleUrl);
await M.default({module_or_path: readFileSync(new URL(moduleUrl.href.replace(/\.js$/, '_bg.wasm')))});
const read = p => JSON.parse(readFileSync(p, 'utf8'));
const fixture = read(new URL('./native_causal_lm_cache_fixture.json', import.meta.url));
const owners = read(new URL('./native_line_owners_bridge.json', import.meta.url));
const native = read(rawPath), cuts = read(cutsPath).cases;
assert.equal(native.schema,'spiraltorch.synthetic_line_secant.v1');
const needNear = (a,b,eps) => assert.ok(Number.isFinite(a) && Math.abs(a-b)<=eps, a+' vs '+b);
const f = fixture.cases.find(x=>x.name===owners.host);
const base = M.NativeCausalLm.fromGpt2(f.config,Uint8Array.from(atob(fixture.weights),c=>c.charCodeAt(0)));
const cases = {};
let art, geo, origin;
try {
  art=M.GeometryAdapterArtifact.fromJson(f.geometry,f.geometry_id,f.base_id);
  geo=base.withGeometry(art); origin=geo.withLoraJson(owners.lora,owners.lora_id);
  const malformed=JSON.parse(cuts['raw/lora'].plan);
  malformed.update.direction={kind:'loss_gradient',probe_l2:0.1};
  assert.throws(()=>new M.NativeCausalLmLineTrainingSession(origin,JSON.stringify(malformed)),/unknown field/);
  for(const [name,ref] of Object.entries(native.cases)) {
    const cut=cuts[name], plan=JSON.parse(cut.plan), selection=name.split('/')[0];
    const s=M.NativeCausalLmLineTrainingSession.fromJson(origin,cut.plan,cut.payload,cut.id);
    const snap=()=>{const c=s.checkpoint();try{return [c.id,c.payload];}finally{c.free();}};
    let resumed=0, refusals=0;
    try {
      assert.equal(s.completedUpdates,cut.cut); assert.deepEqual(snap(),[cut.id,cut.payload]);
      let z=stateRows(s,plan);
      for(let i=cut.cut;i<ref.attempts.length;i++) {
        const old=snap(), a=JSON.parse(s.attemptStepJson()), expected=ref.attempts[i];
        assert.equal(a.status,expected.applied?'applied':'unmatched');
        const k=expected.applied ? a.step.anchor_kl : a.refusal;
        assert.equal(k.trials.length,expected.trials);
        assert.equal(k.anchor_readouts,expected.readouts);
        if(selection === 'raw') assert.ok(!k.direction);
        else {
          const d=k.direction;
          assert.deepEqual(d.policy,expected.policy);
          assert.equal(d.peak_anchor,expected.peak_anchor); assert.equal(d.selected_anchor,expected.selected_anchor);
          d.probe_per_anchor_kl.forEach((v,j)=>needNear(v,expected.probe_per_anchor_kl[j],2e-8));
          needNear(d.retained_fraction,expected.retained_fraction,2e-4);
          assert.ok(Math.abs(d.projected_normal_cosine)<1e-6);
          if(expected.applied) assert.ok(d.source_loss_dot_step<0); else assert.ok(!('source_loss_dot_step' in d));
        }
        if(!expected.applied) {
          assert.equal(a.refusal.reason,cut.reason); assert.equal(a.next_update,i+1);
          assert.equal(a.example_index,i%3); assert.deepEqual(snap(),old);
          refusals++; break;
        }
        const next=stateRows(s,plan), kl=z.map((row,j)=>forwardKl(row,next[j]));
        kl.forEach((v,j)=>needNear(v,k.per_anchor_kl[j],4e-15));
        const mean=kl.reduce((x,y)=>x+y)/kl.length;
        assert.ok(mean>=0.98/65536 && mean<=1/65536 && Math.max(...kl)<=2/65536);
        needNear(a.step.loss,expected.loss,3e-6);
        z=next; resumed++;
      }
      assert.equal(s.completedUpdates,ref.completed);
      assert.equal(s.exhausted,ref.stop===null);
      const output=stateRows(s,plan);
      output.forEach((row,i)=>row.forEach((v,j)=>needNear(v,cut.last_rows[i][j],3e-6)));
      cases[name]={cut:cut.cut,completed:s.completedUpdates,resumed,refusals};
    } finally {s.free();}
  }
} finally {origin?.free();geo?.free();art?.free();base.free();}
console.log(JSON.stringify({schema:'spiraltorch.synthetic_line_secant_wasm.v1',cases,pretrained_calls:0}));
