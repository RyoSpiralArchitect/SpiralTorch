// CPU-only WASM still transports the Rust plan, but cannot pretend to execute it.
const assert = require("node:assert/strict");
const path = require("node:path");
const {InferencePlan} = require(path.resolve(process.argv[2]));
const record = {schema:"spiraltorch.nn.inference_plan.v1", input_shape:[2,3,2],
  stages:[{inner:2, cols:2, weight:[1,0,0,1], bias:[0,0], gelu:true}]};
const payload = JSON.stringify(record);
const plan = InferencePlan.fromJson(payload);
assert.deepEqual([...plan.inputShape], [2,3,2]);
assert.deepEqual([...plan.outputShape], [2,3,2]);
assert.equal(plan.stageCount, 1);
assert.equal(plan.sourceOperationCount, 2);
const canonical = plan.toJson();
const restored = InferencePlan.fromJson(canonical, Buffer.byteLength(canonical));
assert.equal(restored.toJson(), canonical);
assert.throws(() => InferencePlan.fromJson(payload, 1));
assert.throws(() => InferencePlan.fromJson(payload, true));
assert.throws(() => InferencePlan.fromJson({length:0}));
assert.throws(() => InferencePlan.fromJson(JSON.stringify({...record, input_shape:[4294967295,2]})));
assert.throws(() => InferencePlan.fromJson(payload.replace("1,0,0,1", "1e100,0,0,1")));
assert.throws(() => plan.compileWebGpu(), /webgpu/);
plan.free();
assert.equal(restored.toJson(), canonical);
restored.free();
console.log("CPU-only WASM NN transport and explicit GPU rejection passed");
