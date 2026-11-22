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
  let steps=0, imported=0, maxLoss=0, maxEndpoint=0;
  try {
    need(init.loraArtifactId===bridge.seed_id && origin.loraArtifactId===bridge.lora_id,'initial LoRA mismatch');
    refuse(()=>geo.withLoraJson(bridge.lora,bridge.seed_id));
    for (const mode of modes) {
      const ref=bridge.cases[mode], p=JSON.parse(ref.plan);
      const live=new Session(origin,ref.plan);
      let manual=live.modelSnapshot(), foreign;
      try {
        need(live.planId===ref.plan_id,'plan pin mismatch');
        for(let i=0;i<bridge.total;i++) {
          const e=p.examples[i%p.examples.length];
          const args=[words(e.prompt_ids),words(e.candidates.flat()),words(e.candidates.map(xs=>xs.length)),e.stop_id,JSON.stringify(e.objective)];
          const g=manual.lineAdaptationGradients(...args,mode,p.max_tile_values);
          try {
            const before=[manual.loraArtifactId,geom(manual).id];
            const next=manual.stepLineAdaptationSgd(g,p.learning_rate);
            if (i===0 && mode!=='geometry') refuse(()=>manual.stepLineGeometrySgd(g,p.learning_rate));
            manual.free(); manual=next;
            const r=JSON.parse(live.stepJson());
            need(r.mode===mode && r.update===i+1 && r.pre_lora_id===before[0] && r.pre_geometry_id===before[1] &&
                 r.post_lora_id===manual.loraArtifactId && r.post_geometry_id===geom(manual).id &&
                 (r.pre_lora_id!==r.post_lora_id)===(mode!=='geometry') &&
                 (r.pre_geometry_id!==r.post_geometry_id)===(mode!=='lora'),'wrong owner update');
            need(JSON.stringify(r.objective.scores)===JSON.stringify(JSON.parse(g.reportJson())) &&
                 JSON.parse(g.trainingWorkJson()).peak_logit_values===p.max_tile_values,'manual gradient mismatch');
            const groups=Object.keys(JSON.parse(g.parameterGradientsJson()));
            need(groups.some(k=>k.endsWith('c_proj'))===(mode!=='geometry') &&
                 groups.some(k=>k.endsWith('.attn'))===(mode!=='lora'),'wrong gradient group');
            maxLoss=Math.max(maxLoss,close(r.loss,ref.steps[i]));
            steps++;
          } finally {g.free();}
        }
        need(live.exhausted && live.stepJson()===undefined,'cursor not exhausted');
        maxEndpoint=Math.max(maxEndpoint,close(state(manual),ref.endpoint));
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
    need(steps===18 && imported===12 && refusals===9 && origin.loraArtifactId===bridge.lora_id,'incomplete owner comparison');
  } finally {origin.free();init.free();geo.free();a.free();base.free();}
  handoff?.(returned);
  return {matched_steps:steps,native_resumed_steps:imported,refusals,max_native_loss_error:maxLoss,max_native_endpoint_error:maxEndpoint};
}
