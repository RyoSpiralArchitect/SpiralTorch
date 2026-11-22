// Rust chooses each proposal. JS independently measures full-vocabulary KL
// from actual pre/post model snapshots, not a metric reconstructed from receipts.
export function checkLineAnchorKl(M, fixture, owners, native) {
  const need = (ok, message) => { if (!ok) throw Error(message); };
  const near = (x,y,tol=4e-15) => need(Number.isFinite(x) && Math.abs(x-y)<=tol,'anchor KL mismatch: '+x+' vs '+y);
  const logP = row => {
    const top=Math.max(...row), z=Math.log(row.reduce((s,x)=>s+Math.exp(x-top),0));
    return row.map(x=>(x-top)-z);
  };
  const kl = (a,b) => {
    const x=logP(a), y=logP(b);
    return x.reduce((s,p,i)=>s+Math.exp(p)*(p-y[i]),0);
  };
  const rows = (m,p) => p.update.anchor_examples.map(i=>Array.from(m.nextLogits(JSON.stringify(p.examples[i].prompt_ids))));
  const stateRows = (s,p) => {
    const m=s.modelSnapshot();
    try {return rows(m,p);} finally {m.free();}
  };
  const S=M.NativeCausalLmLineTrainingSession;
  const target=1/65536, ceiling=1/8;
  const f=fixture.cases.find(x=>x.name===owners.host);
  const base=M.NativeCausalLm.fromGpt2(f.config,Uint8Array.from(atob(fixture.weights),c=>c.charCodeAt(0)));
  let art,geo,origin;
  const cases={};
  let resumed=0, independent=0;
  try {
    art=M.GeometryAdapterArtifact.fromJson(f.geometry,f.geometry_id,f.base_id);
    geo=base.withGeometry(art);
    origin=geo.withLoraJson(owners.lora,owners.lora_id);
    for (const mode of ['lora','geometry','joint']) {
      const p=JSON.parse(owners.cases[mode].plan);
      p.schema='spiraltorch.native_causal_lm_line_plan.v3';
      delete p.learning_rate;
      p.update={kind:'anchor_forward_kl',anchor_examples:[0,1,2],target_kl:target,max_parameter_l2:ceiling};
      const plan=JSON.stringify(p);
      let live, restored;
      try {
        live=new S(origin,plan);
        const z0=rows(origin,p), steps=[];
        let prior=z0, last=z0, path=0, probes=0;
        for (let i=0;i<6;i++) {
          const r=JSON.parse(live.stepJson()), k=r.anchor_kl;
          last=stateRows(live,p);
          const by=prior.map((z,j)=>kl(z,last[j]));
          by.forEach((v,j)=>near(v,k.per_anchor_kl[j]));
          const mean=by.reduce((s,x)=>s+x,0)/by.length;
          near(mean,k.mean_kl);
          need(mean<=target && mean>=0.98*target,'unmatched update');
          need(k.trials.length>0 && k.trials.length<=20 && k.trials[0].requested_l2===ceiling &&
               k.anchor_readouts===by.length*(k.trials.length+1),'unbounded anchor work');
          need(r.mode===mode && r.update===i+1 && r.parameter_ray.requested_l2===k.trials.at(-1).requested_l2,'bad KL ray or cursor');
          steps.push({loss:r.loss,mean_kl:mean});
          path+=r.parameter_ray.realized_l2; probes+=k.trials.length;
          independent++; prior=last;
          if(i===1) {
            const ref=native?.cases[mode];
            if(native) need(ref,'missing native cut');
            const checkpoint=live.checkpoint();
            try {restored=S.fromJson(origin,ref?.plan??plan,ref?.payload??checkpoint.payload,ref?.id??checkpoint.id);}
            finally {checkpoint.free();}
            need(restored.completedUpdates===2,'incorrect KL checkpoint cursor');
          } else if(i>1) {
            const c=JSON.parse(restored.stepJson());
            near(c.loss,r.loss,3e-6);
            if(native) near(c.loss,native.cases[mode].losses[i],3e-6);
            need(c.anchor_kl.mean_kl>=0.98*target && c.anchor_kl.mean_kl<=target,'restored KL out of band');
            resumed++;
          }
        }
        need(live.exhausted && restored.exhausted,'incomplete KL trajectory');
        stateRows(restored,p).forEach((z,j)=>z.forEach((x,h)=>near(x,last[j][h],3e-6)));
        const end=last.map((z,j)=>kl(z0[j],z)).reduce((s,x)=>s+x,0)/last.length;
        cases[mode]={steps,parameter_path:path,endpoint_kl:end,trials:probes};
      } finally {restored?.free();live?.free();}
    }
  } finally {origin?.free();geo?.free();art?.free();base.free();}
  return {cases, independently_checked_steps:independent, resumed_steps:resumed,
          checkpoint_source:native?'native':'wasm', pretrained_calls:0};
}
