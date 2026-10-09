// Portable plan assembly shared by browser and CPU-only tests; no tensor math.
export function componentRecords(c) {
  const width = c.input_shape[2];
  const norm = f => [{role:"gain", shape:[width], values:f.gain},
                    {role:"bias", shape:[width], values:f.bias}];
  const ln = {kind:"layer_norm", gain:0, bias:1, epsilon:1e-5};
  const pre = {schema:"spiraltorch.nn.inference_plan.v3", input_shape:c.input_shape,
               parameters:norm(c.pre), stages:[ln]};
  const f = c.feed_forward;
  const parameters = [...norm(f),
    {role:"weight", shape:[width,f.hidden], values:f.up_weight},
    {role:"bias", shape:[f.hidden], values:f.up_bias}];
  const stages = [ln, {kind:"linear",weight:2,bias:3,gelu:true}];
  if (c.topos) {
    parameters.push({role:"gate",shape:[f.hidden],values:f.gate});
    stages.push({kind:"topos_resonator",gate:4,coupling:0.2,iterations:4,
                 saturation:0.12,porosity:0.3,max_volume:c.input_shape[0]*c.input_shape[1]*f.hidden});
  }
  const index = parameters.length;
  parameters.push({role:"weight",shape:[f.hidden,width],values:f.down_weight},
                  {role:"bias",shape:[width],values:f.down_bias});
  stages.push({kind:"linear",weight:index,bias:index+1,gelu:false});
  return [pre,{schema:c.topos ? "spiraltorch.nn.inference_plan.v5" : pre.schema,
               input_shape:c.input_shape,parameters,stages}];
}

export function parts(st, c) {
  const [pre, feed] = componentRecords(c).map(p=>st.InferencePlan.fromJson(JSON.stringify(p)));
  const projections = c.projections.map(p=>st.InferencePlan.fromJson(JSON.stringify({
    schema:"spiraltorch.nn.inference_plan.v1",
    input_shape:[...c.input_shape.slice(0,2),p.weight_shape[0]],
    stages:[{inner:p.weight_shape[0],cols:p.weight_shape[1],weight:p.weight,bias:p.bias,gelu:false}],
  })));
  try {
    const attention = st.AttentionInferencePlan.fromProjectionPlans(
      ...projections,c.heads,c.causal ? 0 : undefined);
    return [pre,attention,feed];
  } finally { projections.forEach(p=>p.free()); }
}

export function makePlan(st, c) {
  const components = parts(st,c);
  try { return st.ResidualAttentionPlan.fromPlans(...components); }
  finally { components.forEach(p=>p.free()); }
}
