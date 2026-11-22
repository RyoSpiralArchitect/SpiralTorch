import {forwardKl, rows, stateRows} from './native_line_anchor_kl.mjs';

const T=1/65536, C=2*T;
const need=(ok,msg)=>{if(!ok)throw Error('anchor envelope: '+msg);};
const near=(a,b,eps=4e-15)=>need(Number.isFinite(a) && Math.abs(a-b)<=eps,'numeric disagreement '+a+' vs '+b);
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const observed=(a,b)=>a.map((row,i)=>forwardKl(row,b[i]));
const words=xs=>Uint32Array.from(xs);
function position(rule,m,peak) {
  const [metric,target]=rule==='peak'?[peak,C]:[m,T];
  return metric>target || (rule==='capped' && peak>C) ? 'above' : metric/target>=.98?'matched':'below';
}
function snap(s) {
  const c=s.checkpoint();
  try{return {id:c.id,payload:c.payload,cut:s.completedUpdates};}finally{c.free();}
}
function trial(t,by) {
  const m=mean(by), peak=Math.max(...by);
  near(m,t.mean_kl); near(peak,t.peak.kl);
  need(t.peak.anchor_index===by.indexOf(peak),'limiting anchor');
}
function candidateAudit(s,p,rule,before,r) {
  const m=s.modelSnapshot(), e=p.examples[s.completedUpdates%p.examples.length];
  let g;
  try {
    g=m.lineAdaptationGradients(words(e.prompt_ids),words(e.candidates.flat()),
      words(e.candidates.map(x=>x.length)),e.stop_id,JSON.stringify(e.objective),p.mode,p.max_tile_values);
    for(const t of r.trials){
      const reference=m.stepLineGradientAtDistance(g,t.requested_l2).intoModel();
      try{
        const by=observed(before,rows(reference,p));
        trial(t,by);
        need(position(rule,mean(by),Math.max(...by))!=='matched','refused a measured admissible trial');
      }finally{reference.free();}
    }
  }finally{g?.free();m.free();}
}

export function checkLineEnvelope(M,fixture,owners,native) {
  const S=M.NativeCausalLmLineTrainingSession, cases={};
  const f=fixture.cases.find(x=>x.name===owners.host);
  const base=M.NativeCausalLm.fromGpt2(f.config,Uint8Array.from(atob(fixture.weights),c=>c.charCodeAt(0)));
  let art,geo,origin,accepted=0,audited=0,resumed=0,refusals=0,resumedRefusals=0;
  try {
    art=M.GeometryAdapterArtifact.fromJson(f.geometry,f.geometry_id,f.base_id);
    geo=base.withGeometry(art);origin=geo.withLoraJson(owners.lora,owners.lora_id);
    for(const rule of ['capped','peak']) for(const mode of ['lora','geometry','joint']){
      const name=rule+'/'+mode,p=JSON.parse(owners.cases[mode].plan);
      delete p.learning_rate;p.schema='spiraltorch.native_causal_lm_line_plan.v3';
      p.update={kind:'anchor_forward_kl',target_kl:rule==='peak'?C:T,max_parameter_l2:1/8,anchor_examples:[0,1,2],
                criterion:rule==='peak'?{kind:'peak'}:{kind:'mean_with_cap',max_anchor_kl:C}};
      const plan=JSON.stringify(p);
      let live,restored;
      try {
        live=new S(origin,plan);
        let z=rows(origin,p),saved=snap(live),reason=null,peakIndex=null,work=0;
        const steps=[];
        for(let i=0;i<6;i++) {
          const pre=snap(live),a=JSON.parse(live.attemptStepJson());
          const k=a.status==='applied'?a.step.anchor_kl:a.refusal;
          need(k.trials.length>=1 && k.trials.length<=20 && k.anchor_readouts===3*(1+k.trials.length),'trial/readout bound');
          work+=k.anchor_readouts;
          if(a.status==='unmatched'){
            const after=snap(live);
            need(pre.id===after.id && pre.payload===after.payload && after.cut===i,'refusal mutated state');
            need(a.next_update===i+1 && a.example_index===i%3 && a.plan_id===live.planId,'refusal cursor');
            candidateAudit(live,p,rule,z,k);
            audited+=k.trials.length;reason=k.reason;peakIndex=k.trials.at(-1).peak.anchor_index;
            refusals++;break;
          }
          need(a.status==='applied' && a.step.mode===mode && a.step.update===i+1,'wrong owner or cursor');
          const nz=stateRows(live,p),by=observed(z,nz);
          trial(k.trials.at(-1),by);
          by.forEach((v,j)=>near(v,k.per_anchor_kl[j]));
          need(position(rule,mean(by),Math.max(...by))==='matched','outside actual limits');
          steps.push({loss:a.step.loss, mean_kl:mean(by),peak_kl:Math.max(...by)});
          z=nz;accepted++;
          if(i+1<=2)saved=snap(live);
        }
        if(!reason)need(live.exhausted && live.attemptStepJson()===undefined,'missing exhaustion');
        const ref=native?.cases[name];
        if(native)need(ref && ref.completed===steps.length && ref.refusal===reason,'native terminal mismatch');
        const cut=ref??saved;
        restored=S.fromJson(origin,ref?.plan??plan,cut.payload,cut.id);
        need(restored.completedUpdates===cut.cut && snap(restored).id===cut.id,'imported cut mismatch');
        for(let i=cut.cut;i<steps.length;i++){
          const a=JSON.parse(restored.attemptStepJson());
          need(a.status==='applied' && a.step.update===i+1,'resumed step mismatch');
          near(a.step.loss,steps[i].loss,3e-6);
          if(ref)near(a.step.loss,ref.steps[i],3e-6);
          resumed++;
        }
        if(reason){
          const pre=snap(restored),a=JSON.parse(restored.attemptStepJson());
          need(a.status==='unmatched' && a.refusal.reason===reason && snap(restored).id===pre.id,'resumed refusal mismatch');
          resumedRefusals++;
        }else need(restored.exhausted && restored.attemptStepJson()===undefined,'restored cursor');
        stateRows(restored,p).forEach((row,i)=>row.forEach((x,j)=>near(x,z[i][j],3e-6)));
        if(ref)z[0].forEach((x,j)=>near(x,ref.endpoint.logits[j],3e-6));
        cases[name]={steps,reason,limiting_anchor_at_stop:peakIndex,anchor_readouts:work};
      } finally {restored?.free();live?.free();}
    }
  } finally {origin?.free();geo?.free();art?.free();base.free();}
  return {cases,accepted,verified_rejected_candidates:audited,refusals,resumed,resumed_refusals:resumedRefusals,
          checkpoint_source:native?'native':'wasm',pretrained_calls:0};
}
