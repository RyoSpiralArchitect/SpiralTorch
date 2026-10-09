// CPU-only WASM still transports the Rust plan, but cannot pretend to execute it.
const assert = require("node:assert/strict");
const path = require("node:path");
const {InferencePlan, AttentionInferencePlan} = require(path.resolve(process.argv[2]));
const record = {schema:"spiraltorch.nn.inference_plan.v1", input_shape:[2,3,2],
  stages:[{inner:2, cols:2, weight:[1,0,0,1], bias:[0,0], gelu:true}]};
const payload = JSON.stringify(record);
const plan = InferencePlan.fromJson(payload);
assert.deepEqual([...plan.inputShape], [2,3,2]);
assert.deepEqual([...plan.outputShape], [2,3,2]);
assert.equal(plan.stageCount, 1);
assert.equal(plan.sourceOperationCount, 2);
assert.equal(plan.isDense, true);
const canonical = plan.toJson();
const restored = InferencePlan.fromJson(canonical, Buffer.byteLength(canonical));
assert.equal(restored.toJson(), canonical);
assert.throws(() => InferencePlan.fromJson(payload, 1));
assert.throws(() => InferencePlan.fromJson(payload, true));
assert.throws(() => InferencePlan.fromJson({length:0}));
assert.throws(() => InferencePlan.fromJson(JSON.stringify({...record, input_shape:[4294967295,2]})));
assert.throws(() => InferencePlan.fromJson(payload.replace("1,0,0,1", "1e100,0,0,1")));
assert.throws(() => plan.compileWebGpu(), /webgpu/);
assert.throws(() => plan.compileGraphWebGpu(), /webgpu/);
plan.free();
assert.equal(restored.toJson(), canonical);
restored.free();
const rich = InferencePlan.fromJson(JSON.stringify({schema:"spiraltorch.nn.inference_plan.v2",input_shape:[2,3,2],
  parameters:[],stages:[{kind:"pointwise",parameters:[],steps:[{op:"relu",rhs:null}]}]}));
assert.equal(rich.isDense, false);
assert.deepEqual([...rich.outputShape], [2,3,2]);
const richJson = rich.toJson(), richRestored = InferencePlan.fromJson(richJson);
assert.equal(richRestored.toJson(), richJson);
assert.throws(() => rich.compileWebGpu(), /webgpu/);
assert.throws(() => rich.compileTrainingWebGpu(), /webgpu/);
assert.throws(() => rich.compileGraphWebGpu(), /webgpu/);
for(const policy of ["exact","module_compatible"]) assert.throws(() => rich.compileGraphTrainingWebGpu(policy), /webgpu/);
for(const policy of [undefined,null,true,1,"","auto","EXACT"," exact","module-compatible"])
  assert.throws(() => rich.compileGraphTrainingWebGpu(policy), /gradient_policy/);
rich.free(); richRestored.free();
const linearRecord = {...record, stages:[{...record.stages[0], gelu:false}]};
const projections = Array.from({length:4}, () => InferencePlan.fromJson(JSON.stringify(linearRecord)));
const attention = AttentionInferencePlan.fromProjectionPlans(...projections, 1, 0);
assert.deepEqual([...attention.inputShape], [2,3,2]);
assert.deepEqual([...attention.outputShape], [2,3,2]);
for (const bad of [true, -1, 0.5, "1", 2**33]) {
  assert.throws(() => AttentionInferencePlan.fromProjectionPlans(...projections, bad), /heads/);
  assert.throws(() => AttentionInferencePlan.fromProjectionPlans(...projections, 1, bad), /causal_offset/);
}
assert.throws(() => AttentionInferencePlan.fromProjectionPlans(...projections, 0), /nonzero heads/);
for (let index=0; index<4; index++) {
  const gelu = InferencePlan.fromJson(payload);
  const changed = projections.slice(); changed[index] = gelu;
  assert.throws(() => AttentionInferencePlan.fromProjectionPlans(...changed, 1), /exactly one unfused Linear/);
  gelu.free();
}
projections.forEach(p => p.free());
assert.throws(() => attention.compileTrainingWebGpu(), /requires the webgpu build feature/);
attention.free();
console.log("CPU-only WASM NN transport and explicit GPU rejection passed");
