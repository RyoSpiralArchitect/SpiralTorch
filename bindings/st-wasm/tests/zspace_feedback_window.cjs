// Native Python -> actual wasm32 Rust. No browser GPU or learning-quality claim.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const wasm = require(path.resolve(process.argv[2]));
const fixture = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

assert.equal(fixture.schema, "spiraltorch.feedback_window_fixture.v1");
assert.equal(fixture.restart_at, 37);
assert.deepEqual(fixture.cases.map(value => value.width), [1, 4, 80]);

function invoke(operation, request, mode) {
  const fn = wasm[`zspaceOptimizerFeedback${operation}${mode}`];
  const result = mode === "Json" ? JSON.parse(fn(JSON.stringify(request))) : fn(request);
  assert.equal(result.execution_client, "wasm");
  delete result.execution_client;
  return result;
}

function checkpoint(report) {
  return { config: report.config, state: report.state_after };
}

for (const mode of ["Json", "Object"]) {
  for (const test of fixture.cases) {
    assert.equal(test.records.length, 400);
    assert.deepEqual(invoke("Init", { loss_window_observations: test.width }, mode), test.initial);
    const prefix = checkpoint(test.records[fixture.restart_at - 1].observed);
    const restored = invoke("Restore", prefix, mode);
    let current = { config: restored.config, state: restored.state };
    assert.deepEqual(current, prefix);
    for (const expected of test.records.slice(fixture.restart_at)) {
      const control = invoke("Control", {
        ...current, target_step: expected.observation.step, proposed_learning_rate_scale: 0.5,
      }, mode);
      assert.deepEqual(control, expected.control);
      const observed = invoke("Observe", {
        ...checkpoint(control), observation: expected.observation,
      }, mode);
      assert.deepEqual(observed, expected.observed);
      current = checkpoint(observed);
    }
    assert.deepEqual(current, test.final);
    if (test.width > 1) {
      for (const [field, value] of [["observations_per_window", 3], ["observations", 0],
        ["completed_windows", 99], ["mean", null]]) {
        const bad = structuredClone(prefix);
        bad.state.loss_window[field] = value;
        assert.throws(() => invoke("Restore", bad, mode));
      }
      const bad = structuredClone(prefix);
      bad.config.loss_window_observations = 1;
      assert.throws(() => invoke("Restore", bad, mode));
    }
  }
  for (const width of [0, -1, 1.5, 2 ** 53]) {
    assert.throws(() => invoke("Init", { loss_window_observations: width }, mode));
  }
}
console.log("Python -> wasm32 feedback: widths 1/4/80, exact 37/363 restart, JSON/object and rejection passed");
