// Native Rust/Python -> actual wasm32 Rust, without a model or GPU claim.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const wasm = require(path.resolve(process.argv[2]));
const records = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
assert.equal(records.length, 96);
for (const mode of ["Json", "Object"]) {
  const call = request => mode === "Json"
    ? JSON.parse(wasm.zspaceRepetitionObjectiveControlJson(JSON.stringify(request)))
    : wasm.zspaceRepetitionObjectiveControlObject(request);
  for (const { request, expected } of records) {
    assert.deepEqual(call(request), expected);
  }
  for (const [key, value] of [
    ["completed_update_slots", -1], ["completed_update_slots", 2 ** 53],
    ["completed_update_slots", 1.5], ["completed_update_slots", true],
    ["active_position_count", 1000001], ["base_strength", -0.1],
  ]) {
    assert.throws(() => call({ ...records[0].request, [key]: value }));
  }
  const unknown = structuredClone(records[0].request);
  unknown.config.schedule.unrecognized = true;
  assert.throws(() => call(unknown));
}
console.log("96 native/wasm32 objective controls exactly match through JSON and object APIs; invalid requests rejected");
