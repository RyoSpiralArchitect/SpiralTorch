#!/usr/bin/env node
// Reduce retained synthetic runtime receipts to a small public result. No ML dependencies.
const fs = require("node:fs"), path = require("node:path"), assert = require("node:assert/strict");
const crypto = require("node:crypto"), cp = require("node:child_process");
const [fixtureFile, browserDir, reverseFile, wheelFile, output] = process.argv.slice(2);
if (!output) throw Error("usage: summarize_vision_trainer_clients.cjs PYTHON_FIXTURE BROWSER_DIR REVERSE_REPORT WHEEL NEW_OUTPUT");
const sha = bytes => crypto.createHash("sha256").update(bytes).digest("hex");
const hashFile = file => sha(fs.readFileSync(file));
const fixture = JSON.parse(fs.readFileSync(fixtureFile)), reverse = JSON.parse(fs.readFileSync(reverseFile));
assert.equal(fixture.schema, "vision_trainer_client_fixture.v1");
assert.equal(fixture.cases.length, 2); assert.equal(reverse.length, 2);
const receipts = [], cases = [];
let assets;
for (const schedule of ["constant", "cosine"]) {
  const native = fixture.cases.find(c => c.scheduled === (schedule === "cosine"));
  assert(native); assert.equal(native.records.length, 100);
  const runs = {};
  for (const mode of ["control", "prefix", "resume", "python"]) {
    const filename = `${schedule}-${mode}.json`, file = path.join(browserDir, filename);
    const raw = fs.readFileSync(file), envelope = JSON.parse(raw), run = envelope.result;
    assert.equal(run.passed, true); assert.equal(run.schedule, schedule); assert.equal(run.mode, mode);
    assert.equal(run.records.length, mode === "control" ? 100 : mode === "prefix" ? 37 : 63);
    if (assets) assert.deepEqual(envelope.assets, assets); else assets = envelope.assets;
    runs[mode] = run;
    receipts.push({file: filename, sha256: sha(raw), bytes: raw.length});
  }
  assert.deepEqual([...runs.prefix.records, ...runs.resume.records], runs.control.records);
  assert.equal(runs.resume.initial, runs.prefix.checkpoint);
  assert.equal(runs.resume.checkpoint, runs.control.checkpoint);
  assert.equal(runs.python.initial, native.prefix);
  assert.deepEqual(runs.python.records, native.records.slice(37));
  for (const records of [native.records, runs.control.records]) {
    assert.equal(records.filter(r => r.accepted).length, 90);
    records.forEach((r, i) => { assert.equal(r.revision, i + 1); assert.equal(r.accepted, !r.labels.includes("7")); });
  }
  const actual = JSON.parse(runs.python.checkpoint), expected = JSON.parse(native.final);
  assert.deepEqual(actual.input, expected.input); assert.deepEqual(actual.trainer, expected.trainer);
  const weights = c => [...c.model.backbone.parameters, ...c.model.head];
  const a = weights(actual), b = weights(expected);
  assert.equal(a.length, 24); assert.equal(a.length, b.length);
  let maximum = 0, values = 0;
  a.forEach((p, i) => {
    assert.equal(p.name, b[i].name); assert.deepEqual(p.shape, b[i].shape);
    assert.equal(p.values.length, b[i].values.length);
    p.values.forEach((v, j) => {
      const error = Math.abs(v - b[i].values[j]) / (1 + Math.abs(b[i].values[j]));
      assert(Number.isFinite(error) && error <= 2e-4); maximum = Math.max(maximum, error); values++;
    });
  });
  assert.equal(values, 456);
  const back = reverse.find(r => r.schedule === schedule);
  assert(back); assert.equal(back.parameter_values, 456); assert.equal(back.parameter_tensors, 24);
  assert(back.max_scaled_error >= 0 && back.max_scaled_error <= 2e-4);
  assert.equal(back.browser_checkpoint_sha256, sha(runs.control.checkpoint));
  cases.push({schedule, attempts: 100, accepted: 90, rejected: 10, split: [37, 63],
    config: native.config, native_final_sha256: sha(native.final), browser_final_sha256: sha(runs.control.checkpoint),
    browser_restart_checkpoint_exact: true, cross_runtime_input_rate_and_clocks_exact: true,
    python_to_browser_max_scaled_weight_error: maximum, browser_to_python: back,
    adapter: runs.control.adapter, user_agent: runs.control.user_agent});
}
assert.equal(assets["/python.json"], hashFile(fixtureFile));
assert.equal(assets["/"], hashFile(path.join(__dirname, "../bindings/st-wasm/tests/vision_trainer_clients.html")));
const result = {schema: "spiraltorch.vision.trainer_client_result.v1",
  scope: "synthetic restart correctness; no throughput, real-image quality or cross-device claim",
  source_capture_commit: cp.execFileSync("git", ["rev-parse", "HEAD"], {encoding: "utf8"}).trim(),
  dataset_sha256: fixture.dataset_sha256, parameter_tensors: 24, parameter_values: 456,
  wheel_sha256: hashFile(wheelFile), python_fixture_sha256: hashFile(fixtureFile),
  reverse_report_sha256: hashFile(reverseFile), assets, cases, receipts};
fs.writeFileSync(output, JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
console.log("Verified 8 browser cases, both cross-runtime directions and both complete restart checkpoints.");
