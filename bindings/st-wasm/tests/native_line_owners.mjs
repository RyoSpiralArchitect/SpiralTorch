// Same artificial origin, examples, SGD rate and order. Only the selected gradient groups vary.
export function checkLineOwners(M, fixture, bridge, handoff) {
  const need = (v,msg) => { if (!v) throw Error(msg); };
  const modes=['lora','geometry','joint'];
  const Session=M.NativeCausalLmLineTrainingSession;
  const words=xs=>Uint32Array.from(xs);
  function close(a,b) {
    if (typeof b==='number') {
      const error=Math.abs(a-b);
      need(typeof a==='number' && error<=3e-6,'native/WASM line owner error: '+error);
      return error;
    }
    need(a && b && Array.isArray(a)===Array.isArray(b) &&
         JSON.stringify(Object.keys(a).sort())===JSON.stringify(Object.keys(b).sort()),'endpoint shape');
    return Math.max(0,...Object.keys(b).map(k=>close(a[k],b[k])));
  }
  function snap(s) {
    const c=s.checkpoint();
    try { return {payload:c.payload,id:c.id}; } finally {c.free();}
  }
  function geom(m) {
    const a=m.geometryArtifact();
    try {return {id:a.artifactId,values:JSON.parse(a.toJson()).placements[0].tensors.map(t=>t.values)};}
    finally {a.free();}
  }
  function state(m) {
    const {a,b}=JSON.parse(m.loraJson());
    return {a,b,geometry:geom(m).values,logits:Array.from(m.nextLogits('[1,2]'))};
  }
  // Observation only: recover actual f32 weights and compare native decoder outputs.
  const coords=m=>{
    const l=JSON.parse(m.loraJson());
    return [...l.a,...l.b,...geom(m).values.flat()].map(Math.fround);
  };
  const normDiff=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
  const outputRows=(m,p)=>p.examples.map(e=>Array.from(m.nextLogits(JSON.stringify(e.prompt_ids))));
  function centeredRms(before,after) {
    const squared=before.flatMap((a,i)=>{
      const d=a.map((v,j)=>after[i][j]-v);
      const mean=d.reduce((x,y)=>x+y,0)/d.length;
      return d.map(x=>(x-mean)**2);
    });
    return Math.sqrt(squared.reduce((x,y)=>x+y,0)/squared.length);
  }
  const measurements={};
  let refusals=0;
  function refuse(f) {
    try { f()?.free?.(); } catch {refusals++; return;}
    throw Error('invalid complete-line owner admitted');
  }
  const f=fixture.cases.find(x=>x.name===bridge.host);
  const base=M.NativeCausalLm.fromGpt2(f.config,Uint8Array.from(atob(fixture.weights),c=>c.charCodeAt(0)));
  const a=M.GeometryAdapterArtifact.fromJson(f.geometry,f.geometry_id,f.base_id);
  const geo=base.withGeometry(a);
  const init=geo.withLora(0,2,2);
  const origin=geo.withLoraJson(bridge.lora,bridge.lora_id);
  const returned={};
  let steps=0, imported=0, maxLoss=0, maxEndpoint=0, fixed=0, maxDistanceError=0, maxVectorError=0;
  try {
    need(init.loraArtifactId===bridge.seed_id && origin.loraArtifactId===bridge.lora_id,'initial LoRA mismatch');
    refuse(()=>geo.withLoraJson(bridge.lora,bridge.seed_id));
    for (const mode of modes) {
      const ref=bridge.cases[mode], p=JSON.parse(ref.plan);
      const live=new Session(origin,ref.plan);
      let manual=live.modelSnapshot(), foreign;
      try {
        const ray=p.update?.kind==='parameter_gradient_l2';
        const start=coords(origin), beforeRows=outputRows(origin,p);
        let last=start,path=0;
        need(live.planId===(ref.plan_id ?? JSON.parse(ref.payload).plan_id),'plan pin mismatch');
        for(let i=0;i<bridge.total;i++) {
          const e=p.examples[i%p.examples.length];
          const args=[words(e.prompt_ids),words(e.candidates.flat()),words(e.candidates.map(xs=>xs.length)),e.stop_id,JSON.stringify(e.objective)];
          const g=manual.lineAdaptationGradients(...args,mode,p.max_tile_values);
          try {
            const before=[manual.loraArtifactId,geom(manual).id];
            let next, rayReport;
            if (ray) {
              const result=manual.stepLineGradientAtDistance(g,p.update.target_l2);
              rayReport=JSON.parse(result.reportJson());
              next=result.intoModel();
              if (i===0) {
                refuse(()=>manual.stepLineGradientAtDistance(g,true));
                refuse(()=>manual.stepLineGradientAtDistance(g,1e-200));
              }
            } else next=manual.stepLineAdaptationSgd(g,p.learning_rate);
            if (i===0 && mode!=='geometry') refuse(()=>manual.stepLineGeometrySgd(g,p.learning_rate));
            manual.free(); manual=next;
            const current=coords(manual);
            path+=normDiff(last,current);
            last=current;
            const r=JSON.parse(live.stepJson());
            need(r.mode===mode && r.update===i+1 && r.pre_lora_id===before[0] && r.pre_geometry_id===before[1] &&
                 r.post_lora_id===manual.loraArtifactId && r.post_geometry_id===geom(manual).id &&
                 (r.pre_lora_id!==r.post_lora_id)===(mode!=='geometry') &&
                 (r.pre_geometry_id!==r.post_geometry_id)===(mode!=='lora'),'wrong owner update');
            need(JSON.stringify(r.objective.scores)===JSON.stringify(JSON.parse(g.reportJson())) &&
                 JSON.parse(g.trainingWorkJson()).peak_logit_values===p.max_tile_values,'manual gradient mismatch');
            if (ray) {
              need(JSON.stringify(r.parameter_ray)===JSON.stringify(rayReport),'manual ray differs from session');
              const relative=Math.abs(rayReport.realized_l2/p.update.target_l2-1);
              need(relative<=1e-4 && rayReport.relative_vector_error<=1e-4 && rayReport.gradient_dot_step<0,
                   'wrong rounded parameter displacement');
              need((rayReport.lora!==null)===(mode!=='geometry') && (rayReport.geometry!==null)===(mode!=='lora'),
                   'wrong parameter budget owners');
              maxDistanceError=Math.max(maxDistanceError,relative);
              maxVectorError=Math.max(maxVectorError,rayReport.relative_vector_error);
              fixed++;
            } else need(r.parameter_ray===undefined,'legacy SGD acquired a new policy');
            const groups=Object.keys(JSON.parse(g.parameterGradientsJson()));
            need(groups.some(k=>k.endsWith('c_proj'))===(mode!=='geometry') &&
                 groups.some(k=>k.endsWith('.attn'))===(mode!=='lora'),'wrong gradient group');
            maxLoss=Math.max(maxLoss,close(r.loss,ref.steps[i]));
            steps++;
          } finally {g.free();}
        }
        need(live.exhausted && live.stepJson()===undefined,'cursor not exhausted');
        maxEndpoint=Math.max(maxEndpoint,close(state(manual),ref.endpoint));
        const m={path_l2:path,endpoint_l2:normDiff(start,last),centered_logit_rms:centeredRms(beforeRows,outputRows(manual,p))};
        if (bridge.expected_measurements) close(m,bridge.expected_measurements[mode]);
        measurements[mode]=m;
        foreign=Session.fromJson(origin,ref.plan,ref.payload,ref.id);
        need(foreign.completedUpdates===bridge.cut && snap(foreign).id===ref.id,'native cut not byte-stable');
        let back;
        for(let i=bridge.cut;i<bridge.total;i++) {
          const r=JSON.parse(foreign.stepJson());
          need(r.mode===mode && r.update===i+1,'resumed cursor mismatch');
          maxLoss=Math.max(maxLoss,close(r.loss,ref.steps[i]));
          if(i+1===bridge.back) back=snap(foreign);
          imported++;
        }
        const end=foreign.modelSnapshot();
        try {
          const endpoint=state(end);
          maxEndpoint=Math.max(maxEndpoint,close(endpoint,ref.endpoint));
          returned[mode]={...back,endpoint};
        } finally {end.free();}
        refuse(()=>Session.fromJson(init,ref.plan,ref.payload,ref.id));
        refuse(()=>Session.fromJson(origin,ref.plan+' ',ref.payload,ref.id));
      } finally {foreign?.free();manual.free();live.free();}
    }
    need(steps===18 && imported===12 && refusals===(fixed ? 15 : 9) && (!fixed || fixed===18) && origin.loraArtifactId===bridge.lora_id,'incomplete owner comparison');
  } finally {origin.free();init.free();geo.free();a.free();base.free();}
  handoff?.(returned);
  return {measurements,matched_steps:steps,native_resumed_steps:imported,refusals,max_native_loss_error:maxLoss,max_native_endpoint_error:maxEndpoint,
          ...(fixed ? {fixed_steps:fixed,max_relative_distance_error:maxDistanceError,max_relative_vector_error:maxVectorError} : {})};
}
