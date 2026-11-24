// Actual WASM steps, compared to native decisions and independent JS KL/axis scores.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {stateRows,forwardKl} from './native_line_anchor_kl.mjs';

const [pkg, data, journal]=process.argv.slice(2);
const url=pathToFileURL(resolve(pkg)), M=await import(url);
await M.default({module_or_path:readFileSync(new URL(url.href.replace(/\.js$/,'_bg.wasm')))});
const read=p=>JSON.parse(readFileSync(p)), native=read(data), originals=new Map();
assert.equal(native.schema,'spiraltorch.synthetic_line_sensitivity.v1');
for(const line of readFileSync(journal,'utf8').trimEnd().split('\n')) {
  const {case:key,cursor,attempt}=JSON.parse(line);
  const rows=originals.get(key)??[];
  assert.equal(cursor,rows.length);
  rows.push(attempt);originals.set(key,rows);
}
assert.equal(originals.size,20);assert.equal(Object.keys(native.cases).length,20);
const fixture=read(new URL('./native_causal_lm_cache_fixture.json',import.meta.url));
const owners=read(new URL('./native_line_owners_bridge.json',import.meta.url));
const f=fixture.cases.find(c=>c.name===owners.host);
const base=M.NativeCausalLm.fromGpt2(f.config,Buffer.from(fixture.weights,'base64'));
const T=1/65536,H=1/256,C=2*T;
const near=(a,b,t)=>assert.ok(Math.abs(a-b)<=t,a+' vs '+b);
const ck=s=>{const c=s.checkpoint();try{return [c.payload,c.id];}finally{c.free();}};
let art,geo,origin,n=0,updates=0,refusals=0,probes=0,raw=0,maxProbeDiff=0,maxStepDiff=0;
function rank(r) {
  let best=null, at=null;
  r.axes.forEach((a,i)=>{
    const v=a.probe_per_anchor_kl;
    if(!a.gradient_l2){assert.equal(v,null);assert.equal(a.prediction,null);return;}
    assert.equal(v.length,3);
    const mean=v.reduce((x,y)=>x+y,0)/3;
    let row;
    if(mean>0){
      const d=Math.min(1/8,H*Math.sqrt(T/mean),H*Math.sqrt(C/Math.max(...v)));
      const frac=Math.min(1,mean/T*(d/H)**2), gain=d*a.gradient_l2;
      near(a.prediction.radius,d,1e-12);near(a.prediction.target_fraction,frac,1e-12);
      near(a.prediction.linear_loss_decrease,gain,1e-12);
      row=[Number(frac>=.98),frac>=.98?0:frac,gain];
    }else assert.equal(a.prediction,null);
    if(r.select==='gradient_norm')row=[0,a.gradient_l2,0];
    if(row && (!best || row[0]>best[0] || row[0]===best[0] &&
                (row[1]>best[1] || row[1]===best[1] && row[2]>best[2]))){at=i;best=row;}
  });
  assert.equal(at,r.selected_axis);
}
try {
  art=M.GeometryAdapterArtifact.fromJson(f.geometry,f.geometry_id,f.base_id);
  geo=base.withGeometry(art);origin=geo.withLoraJson(owners.lora,owners.lora_id);
  for(const [key,c] of Object.entries(native.cases)) {
    const p=JSON.parse(c.plan),s=new M.NativeCausalLmLineTrainingSession(origin,c.plan),old=originals.get(key);
    assert.equal(c.events.length,old.length);
    try {
      let z=stateRows(s,p);
      for(const [i,e] of c.events.entries()){
        ++n;
        const pre=ck(s),a=JSON.parse(s.attemptStepJson()),ok=a.status==='applied';
        const k=ok?a.step.anchor_kl:a.refusal,prior=old[i].step?.anchor_kl??old[i].refusal;
        assert.equal(a.status,old[i].status);assert.equal(ok,e.applied);
        assert.equal(!k.raw_miss,e.raw);assert.equal(k.anchor_readouts,e.readouts);
        assert.deepEqual(k.trials.map(t=>t.requested_l2),prior.trials.map(t=>t.requested_l2));
        if(k.raw_miss) {
          assert.deepEqual(k.raw_miss.trials.map(t=>t.requested_l2),prior.raw_miss.trials.map(t=>t.requested_l2));
          const r=k.sensitivity,ref=prior.sensitivity;
          assert.equal(r.selected_axis,ref.selected_axis);assert.equal(r.probe_readouts,e.probes*3);
          rank(r);probes+=e.probes;
          assert.equal(r.axes.length,ref.axes.length);
          r.axes.forEach((axis,j)=>{
            const a=ref.axes[j];
            assert.deepEqual([axis.path,axis.tensor,axis.coordinate],[a.path,a.tensor,a.coordinate]);
            assert.equal(axis.probe_per_anchor_kl===null,a.probe_per_anchor_kl===null);
            axis.probe_per_anchor_kl?.forEach((v,col)=>{
              maxProbeDiff=Math.max(maxProbeDiff,Math.abs(v-a.probe_per_anchor_kl[col]));
              near(v,a.probe_per_anchor_kl[col],2e-8);
            });
          });
          if(ok)assert.ok(r.source_loss_dot_step<0);
        }else{++raw;assert.equal(k.sensitivity,undefined);}
        if(!ok){
          ++refusals;assert.equal(k.reason,e.reason);assert.deepEqual(ck(s),pre);break;
        }
        ++updates;
        near(a.step.loss,e.loss,3e-6);
        const next=stateRows(s,p),kl=z.map((v,j)=>forwardKl(v,next[j]));
        kl.forEach((v,j)=>{
          near(v,k.per_anchor_kl[j],4e-15);near(v,e.kl[j],2e-8);
          maxStepDiff=Math.max(maxStepDiff,Math.abs(v-e.kl[j]));
        });
        const mean=kl.reduce((x,y)=>x+y,0)/3;
        assert.ok(mean>=.98*T && mean<=T && Math.max(...kl)<=C);
        z=next;
      }
      assert.equal(s.completedUpdates,c.completed);
      z.forEach((row,j)=>row.forEach((v,col)=>near(v,c.endpoint[j][col],3e-6)));
    }finally{s.free();}
  }
}finally{origin?.free();geo?.free();art?.free();base.free();}
console.log(JSON.stringify({cells:20,attempts:n,updates,refusals,raw,probes,maxProbeDiff,maxStepDiff,pretrained_calls:0}));
