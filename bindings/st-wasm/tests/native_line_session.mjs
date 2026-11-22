// Check the sealed Rust cursor in this engine, then accept an actual CPython checkpoint.
export function checkLineSession(M, fixture, plan, bridge, handoff) {
  const need=(v,msg)=>{if(!v) throw Error(msg);};
  const Session=M.NativeCausalLmLineTrainingSession;
  const recipe=JSON.parse(plan), words=Uint32Array.from.bind(Uint32Array);
  const specs=recipe.examples.map(e=>[words(e.prompt_ids),words(e.candidates.flat()),words(e.candidates.map(l=>l.length)),
                                      e.stop_id,JSON.stringify(e.objective),recipe.max_tile_values]);
  let directSteps=0,resumedSteps=0,importedSteps=0,refusals=0,maxLoss=0,maxEndpoint=0;
  function refuse(f) {
    try { f(); } catch { refusals++; return; }
    throw Error('invalid line session input admitted');
  }
  function close(a,b) {
    if (typeof b==='number') {const d=Math.abs(a-b);need(typeof a==='number' && d<=3e-6,'endpoint difference: '+d);return d;}
    need(a && b && Array.isArray(a)===Array.isArray(b) &&
         JSON.stringify(Object.keys(a).sort())===JSON.stringify(Object.keys(b).sort()),'endpoint shape');
    return Math.max(0,...Object.keys(b).map(k=>close(a[k],b[k])));
  }
  function snapshot(s) {
    const c=s.checkpoint();
    try {return {id:c.id,payload:c.payload};} finally {c.free();}
  }
  function geometry(m) {
    const a=m.geometryArtifact();
    try {return {id:a.artifactId,record:JSON.parse(a.toJson())};} finally {a.free();}
  }
  function endpoint(m) {
    const {record}=geometry(m);
    const parameters=Object.fromEntries(record.placements.map(p=>[p.path,Object.fromEntries(p.tensors.map(t=>[t.name,t.values]))]));
    return {parameters,logits:Array.from(m.nextLogits('[1,2]'))};
  }
  const exported={plan_id:bridge.plan_id,cases:{}};
  for(const f of fixture.cases.filter(c=>c.geometry)) {
    const base=M.NativeCausalLm.fromGpt2(f.config,Uint8Array.from(atob(fixture.weights),c=>c.charCodeAt(0)));
    const art=M.GeometryAdapterArtifact.fromJson(f.geometry,f.geometry_id,f.base_id);
    const origin=base.withGeometry(art);
    const s=new Session(origin,plan), ref=bridge.cases[f.name];
    let direct,local,foreign,saved;
    try {
      need(s.planId===bridge.plan_id && s.completedUpdates===0 && s.totalUpdates===bridge.total && !s.exhausted,'bad start');
      direct=s.modelSnapshot();
      for(let i=0;i<bridge.total;i++){
        const j=i%specs.length,rJson=s.stepJson(),r=JSON.parse(rJson);
        const grad=direct.geometryLineGradients(...specs[j]);
        try{
          const next=direct.stepLineGeometrySgd(grad,recipe.learning_rate);
          direct.free();direct=next;
          need(r.update===i+1 && r.example_index===j && r.epoch===Math.floor(i/specs.length) &&
               r.plan_id===bridge.plan_id && r.objective.kind===recipe.examples[j].objective.kind &&
               r.post_geometry_id===geometry(direct).id &&
               JSON.stringify(r.objective.scores)===JSON.stringify(JSON.parse(grad.reportJson())),'direct SGD differs');
          maxLoss=Math.max(maxLoss,close(r.loss,ref.losses[i]));
          directSteps++;
        }finally{grad.free();}
        if(local){need(local.stepJson()===rJson,'local resume differs');resumedSteps++;}
        if(i+1===bridge.cut){
          const c=snapshot(s);
          local=Session.fromJson(origin,plan,c.payload,c.id);
          need(snapshot(local).id===c.id,'local cut differs');
        }
      }
      need(s.exhausted && s.stepJson()===undefined && local.exhausted && snapshot(local).payload===snapshot(s).payload,'exhaustion');
      maxEndpoint=Math.max(maxEndpoint,close(endpoint(direct),ref.endpoint));
      foreign=Session.fromJson(origin,plan,ref.payload,ref.id);
      const incoming=snapshot(foreign);
      need(incoming.id===ref.id && incoming.payload===ref.payload && foreign.completedUpdates===bridge.cut,'native checkpoint bytes changed');
      const losses=[];
      for(let i=bridge.cut;i<bridge.total;i++){
        const r=JSON.parse(foreign.stepJson());
        maxLoss=Math.max(maxLoss,close(r.loss,ref.losses[i]));importedSteps++;
        if(i+1===bridge.return_at) saved=snapshot(foreign);
        if(i>=bridge.return_at) losses.push(r.loss);
      }
      const final=foreign.modelSnapshot();
      try {
        const end=endpoint(final);
        maxEndpoint=Math.max(maxEndpoint,close(end,ref.endpoint));
        exported.cases[f.name]={...saved,suffix_losses:losses,endpoint:end};
      } finally {final.free();}
      refuse(()=>new Session(base,plan));
      refuse(()=>new Session(origin,'{}'));
      refuse(()=>Session.fromJson(origin,plan+' ',ref.payload,ref.id));
      refuse(()=>Session.fromJson(origin,plan,ref.payload+' ',ref.id));
      refuse(()=>Session.fromJson(origin,plan,ref.payload,'sha256:bad'));
      refuse(()=>Session.fromJson(base,plan,ref.payload,ref.id));
      need(geometry(origin).id===f.geometry_id && foreign.exhausted,'origin changed');
    }finally{foreign?.free();local?.free();direct?.free();s.free();origin.free();art.free();base.free();}
  }
  need(directSteps===24 && resumedSteps===16 && importedSteps===16 && refusals===24,'incomplete resume exercise');
  handoff?.(exported);
  return {direct_steps:directSteps,local_resumed_steps:resumedSteps,native_imported_steps:importedSteps,
          refusals,max_native_loss_error:maxLoss,max_native_endpoint_error:maxEndpoint};
}
