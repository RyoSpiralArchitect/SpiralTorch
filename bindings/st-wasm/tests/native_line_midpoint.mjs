import {forwardKl, stateRows} from './native_line_anchor_kl.mjs';

const T=1/65536;
const check=(ok,why)=>{if(!ok)throw Error('raw-first midpoint: '+why);};
const near=(a,b,tol)=>check(Number.isFinite(a)&&Math.abs(a-b)<=tol,a+' vs '+b);

function snapshot(s) {
  const c=s.checkpoint();
  try {return [c.id,c.payload];} finally {c.free();}
}

function trajectory(s,p,selection,ref) {
  let z=stateRows(s,p), raw=0, rescued=0, reason=null;
  const events=[];
  for(let i=s.completedUpdates;i<6;i++) {
    const pre=snapshot(s),a=JSON.parse(s.attemptStepJson()),ok=a.status==='applied';
    check(ok||a.status==='unmatched','unknown attempt');
    const k=ok?a.step.anchor_kl:a.refusal;
    const first=k.raw_miss, shaped=!!first, count=k.trials.length;
    const exp=ref?.attempts[i];
    if(ref)check(exp && exp.applied===ok && exp.raw===!shaped && exp.readouts===k.anchor_readouts &&
                 exp.trials===count,'native stage, readout or cursor');
    if(shaped) {
      check(first.trials.length>=1 && first.trials.length<=20 &&
            first.anchor_readouts===3*(1+first.trials.length),'invalid raw miss');
      check(k.anchor_readouts===first.anchor_readouts+3*(1+count),'duplicate origin bank');
      if(ref)check(first.reason===exp.raw_reason && first.trials.length===exp.raw_trials,'native raw miss');
      if(k.direction) {
        const d=k.direction,m=d.midpoint,idx=selection==='largest'?d.peak_anchor:(d.peak_anchor+1)%3;
        check(d.policy.anchor===selection && d.selected_anchor===idx && count>=1 && count<=20,'invalid midpoint');
        near(m.source_cosine,m.projected_cosine,1e-6);
        if(ref) {check(d.selected_anchor===exp.anchor,'native anchor');near(m.source_cosine,exp.source_cosine,2e-4);}
        if(ok)check(d.source_loss_dot_step<0 && a.step.parameter_ray.method==='selected_parameter_secant_midpoint_ray_f32.v1','no source descent');
      } else check(!ok && count===0,'unresolved rescue accepted or measured');
    } else check(ok && !k.direction && count>=1 && count<=20 && k.anchor_readouts===3*(1+count),'raw success did extra work');
    if(!ok) {
      const after=snapshot(s);
      check(a.next_update===i+1 && a.example_index===i%3 && s.completedUpdates===i &&
            pre[0]===after[0] && pre[1]===after[1],'refusal changed state');
      reason=k.reason;
      if(ref)check(reason===exp.reason,'native terminal reason');
      events.push({i:i+1,kind:'refused'});break;
    }
    const next=stateRows(s,p),kl=z.map((v,j)=>forwardKl(v,next[j]));
    kl.forEach((v,j)=>near(v,k.per_anchor_kl[j],4e-15));
    const mean=kl.reduce((x,y)=>x+y)/3,peak=Math.max(...kl);
    check(mean>=0.98*T && mean<=T && peak<=2*T,'actual distributions outside envelope');
    if(ref) {near(a.step.loss,exp.loss,3e-6);near(mean,exp.mean_kl,2e-8);}
    if(shaped)rescued++;else raw++;
    events.push({i:i+1,kind:shaped?'midpoint':'raw',mean_over_target:mean/T,peak_over_target:peak/T});
    z=next;
  }
  check(s.exhausted===(reason===null),'wrong terminal cursor');
  return {raw,rescued,reason,completed:s.completedUpdates,events,endpoint:z};
}

export function checkLineMidpoint(M,fixture,owners,native,cuts) {
  const S=M.NativeCausalLmLineTrainingSession;
  const f=fixture.cases.find(x=>x.name===owners.host);
  const base=M.NativeCausalLm.fromGpt2(f.config,Uint8Array.from(atob(fixture.weights),c=>c.charCodeAt(0)));
  const cases={};
  let geom,art,origin,suffixUpdates=0,suffixRefusals=0;
  try {
    art=M.GeometryAdapterArtifact.fromJson(f.geometry,f.geometry_id,f.base_id);
    geom=base.withGeometry(art);
    origin=geom.withLoraJson(owners.lora,owners.lora_id);
    for(const selection of ['largest','successor'])for(const mode of ['lora','geometry','joint']) {
      const name=selection+'/'+mode,ref=native?.cases[name],cut=cuts?.cases[name];
      if(native)check(ref && cut,'missing native result');
      const p=JSON.parse(owners.cases[mode].plan);
      delete p.learning_rate; p.schema='spiraltorch.native_causal_lm_line_plan.v3';
      p.update={kind:'anchor_forward_kl',target_kl:T,max_parameter_l2:1/8,anchor_examples:[0,1,2],
                criterion:{kind:'mean_with_cap',max_anchor_kl:2*T},
                direction:{kind:'raw_then_secant_midpoint',probe_l2:1/256,anchor:selection}};
      const text=cut?.plan??JSON.stringify(p);
      const live=new S(origin,text);
      try {
        const r=trajectory(live,p,selection,ref);
        if(ref) {
          check(r.completed===ref.completed && r.reason===ref.stop,'native terminal state');
          r.endpoint.forEach((row,i)=>row.forEach((v,j)=>near(v,cut.last_rows[i][j],3e-6)));
          const restored=S.fromJson(origin,text,cut.payload,cut.id);
          try {
            check(restored.completedUpdates===cut.cut && snapshot(restored)[0]===cut.id,'native cut rejected');
            const back=trajectory(restored,p,selection,ref);
            check(back.completed===r.completed && back.reason===r.reason,'restored terminal state');
            back.endpoint.forEach((row,i)=>row.forEach((v,j)=>near(v,r.endpoint[i][j],3e-6)));
            suffixUpdates+=back.raw+back.rescued;suffixRefusals+=Number(back.reason!==null);
          } finally {restored.free();}
        }
        cases[name]=r;
      } finally {live.free();}
    }
  } finally {origin?.free();geom?.free();art?.free();base.free();}
  return {cases,suffixUpdates,suffixRefusals,pretrained_calls:0};
}
