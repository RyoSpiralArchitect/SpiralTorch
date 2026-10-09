const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
(async()=>{
  assert.equal(process.argv.length,3,"Pass freshly generated CPU-only WASM module");
  const st = require(path.resolve(process.argv[2]));
  const {parts,makePlan} = await import("./residual_attention_plan.mjs");
  const fixture = JSON.parse(fs.readFileSync(path.join(__dirname,"../../../crates/st-nn/tests/fixtures/residual_attention_torch.json"),"utf8"));
  assert.equal(fixture.schema,"spiraltorch.residual_attention_torch.v1");
  assert.equal(fixture.training.length,2);
  assert.deepEqual(fixture.training.map(c=>c.topos),[false,true]);
  for (const c of fixture.training) {
    const plan = makePlan(st,c);
    assert.deepEqual(Array.from(plan.inputShape),c.input_shape);
    assert.deepEqual(Array.from(plan.outputShape),c.input_shape);
    assert.throws(()=>plan.compileTrainingWebGpu(),/webgpu build feature/);
    plan.free();
    const components = parts(st,c);
    for (const index of [0,2]) {
      const record = JSON.parse(components[index].toJson());
      record.input_shape.splice(0,2,1,6);
      const changed = st.InferencePlan.fromJson(JSON.stringify(record));
      const items = components.slice();items[index]=changed;
      assert.throws(()=>st.ResidualAttentionPlan.fromPlans(...items));
      changed.free();
    }
    components.forEach(p=>p.free());
  }
  console.log("CPU-only residual plan composition and explicit GPU rejection passed");
})().catch(error=>{console.error(error);process.exitCode=1;});
