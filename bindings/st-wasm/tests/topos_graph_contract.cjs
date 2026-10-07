// Node validates the browser-facing Rust contract; this is not GPU execution.
const assert = require("node:assert/strict");
const st = require(require("node:path").resolve(process.argv[2]));
const owned = [], keep = value => (owned.push(value), value);
try {
  const kernel = keep(new st.ToposResonatorKernel(.2, 5, 1, .3, 12));
  const model = keep(new st.Sequential());
  const supplied = new Float32Array([.8, -.4, 1.1]);
  model.addToposResonator("topos", supplied, kernel);
  supplied.fill(99);
  const plan = keep(model.inferencePlan([2, 2, 3]));
  const payload = JSON.parse(plan.toJson());
  assert.equal(payload.schema, "spiraltorch.nn.inference_plan.v5");
  assert.equal(payload.stages[0].kind, "topos_resonator");
  assert.equal(payload.parameters[0].role, "gate");
  assert.deepEqual(payload.parameters[0].shape, [3]);
  assert.deepEqual(payload.parameters[0].values.map(Math.fround), Array.from(new Float32Array([.8, -.4, 1.1])));
  assert.equal(keep(st.InferencePlan.fromJson(plan.toJson())).toJson(), plan.toJson());
  for (const version of [2, 3, 4]) {
    assert.throws(() => st.InferencePlan.fromJson(JSON.stringify({
      ...payload, schema: `spiraltorch.nn.inference_plan.v${version}`,
    })));
  }
  for (const shape of [[5, 3], [2, 4]]) assert.throws(() => model.inferencePlan(shape));
  for (const gate of [[], new Float64Array([1]), new Float32Array([NaN]), null]) {
    assert.throws(() => model.addToposResonator("bad", gate, kernel));
    assert.equal(keep(model.inferencePlan([2, 2, 3])).toJson(), plan.toJson());
  }
  console.log("Topos graph v5 construction, isolation, roundtrip and admission passed (Node, no GPU)");
} finally {
  for (const value of owned.reverse()) value.free();
}
