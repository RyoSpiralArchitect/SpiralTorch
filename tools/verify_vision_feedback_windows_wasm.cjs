// Replay retained real-image loss observations through actual wasm32 Rust.
// This does not rerun image inference or exercise browser/WebGPU training.
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const modulePath = path.resolve(process.argv[2]);
const directory = path.resolve(process.argv[3]);
const wasm = require(modulePath);
const digest = value => crypto.createHash("sha256").update(value).digest("hex");
const summaryBytes = fs.readFileSync(path.join(directory, "summary.json"));
const summary = JSON.parse(summaryBytes);
assert.equal(summary.schema, "spiraltorch.vision.feedback_window_comparison.v1");
assert.equal(summary.status, "passed");

let streams = 0;
let observations = 0;
for (const test of summary.cases) {
  assert.equal(path.basename(test.raw.file), test.raw.file);
  const raw = fs.readFileSync(path.join(directory, test.raw.file));
  assert.equal(raw.length, test.raw.bytes);
  assert.equal(digest(raw), test.raw.sha256);
  const sequences = JSON.parse(raw);
  assert.deepEqual(Object.keys(sequences).sort(), ["constant_loss", "recorded_order", "reversed_order"]);
  for (const arms of Object.values(sequences)) {
    assert.deepEqual(Object.keys(arms).sort(), ["default", "windowed"]);
    for (const expected of Object.values(arms)) {
      const config = expected.config;
      let state = JSON.parse(wasm.zspaceOptimizerFeedbackInitJson(JSON.stringify(config))).state;
      for (const row of expected.records) {
        const control = JSON.parse(wasm.zspaceOptimizerFeedbackControlJson(JSON.stringify({
          config, state, target_step: row.step, proposed_learning_rate_scale: 0.5,
        })));
        const observed = JSON.parse(wasm.zspaceOptimizerFeedbackObserveJson(JSON.stringify({
          config, state: control.state_after, observation: { step: row.step, loss: row.loss },
        })));
        state = observed.state_after;
        assert.deepEqual({ step: row.step, loss: row.loss,
          applied_scale: control.applied_learning_rate_scale, action: observed.action,
          relative_loss_delta: observed.relative_loss_delta, state_after: state }, row);
        if (row.step === summary.recipe.restart_at) {
          const restored = JSON.parse(wasm.zspaceOptimizerFeedbackRestoreJson(JSON.stringify({ config, state })));
          assert.deepEqual(restored.state, state);
          state = restored.state;
        }
        observations += 1;
      }
      streams += 1;
    }
  }
}
const wasmBytes = fs.readFileSync(modulePath.replace(/\.js$/, "_bg.wasm"));
const result = { schema: "spiraltorch.vision.feedback_window_wasm_replay.v1", status: "passed",
  scope: "Retained-loss gate replay under wasm32, not image inference or browser GPU training",
  source_summary_sha256: digest(summaryBytes), wasm_sha256: digest(wasmBytes),
  verifier_sha256: digest(fs.readFileSync(__filename)), cases: summary.cases.length,
  streams, observations, all_states_exact: true, partial_restart_at: summary.recipe.restart_at };
fs.writeFileSync(process.argv[4], JSON.stringify(result, null, 2) + "\n", { flag: "wx" });
console.log(JSON.stringify(result));
