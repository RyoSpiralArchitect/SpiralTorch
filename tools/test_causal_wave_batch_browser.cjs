"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

// Execute the actual page validator, without a browser or a duplicate checker.
const html = fs.readFileSync(path.join(__dirname, "../crates/st-backend-wgpu/tests/causal_zspace_wave_browser.html"), "utf8");
const source = html.match(/function validateBatchReduction[\s\S]+?\n}/);
assert.ok(source, "page validator is missing");
const validate = vm.runInNewContext(`(${source[0]})`);
function fixture() {
  const checks = [];
  for (const exponent of [24, 48, 96]) {
    for (const order of [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]) {
      for (const terminal_channel of [0, 1]) {
        const expected = terminal_channel === 0 ? Math.fround(Math.fround(.99) * .25)
          : Math.fround(Math.fround(Math.fround(.99) * .5) * Math.fround(Math.PI));
        checks.push({exponent, order, terminal_channel, expected, cpu: expected, gpu: expected, exact_f32: true, finite_family: true});
      }
    }
  }
  return {analytic_reference: true, checks, overflow: [0, 1].map(terminal_channel =>
    ({terminal_channel, opposite_partials_cancel: true, finite_family: true, same_sign_whole_family_rejected: true}))};
}

test("all 36 identities and two overflow controls pass", () => validate(fixture()));
test("duplicated or truncated identities cannot produce PASS", () => {
  for (const key of ["checks", "overflow"]) {
    const duplicate = fixture();
    duplicate[key].fill(duplicate[key][0]);
    assert.throws(() => validate(duplicate));
    const missing = fixture();
    missing[key].pop();
    assert.throws(() => validate(missing));
  }
});
test("analytic expectation is independent of reported agreement", () => {
  for (const value of [.25, 0, NaN, Infinity, "NaN", false]) {
    const report = fixture();
    Object.assign(report.checks[0], {expected: value, cpu: value, gpu: value});
    assert.throws(() => validate(report));
  }
});
test("unknown case identity or lost overflow rejection fails", () => {
  const report = fixture();
  report.checks[0].exponent = 25;
  assert.throws(() => validate(report));
  const overflow = fixture();
  overflow.overflow[0].same_sign_whole_family_rejected = false;
  assert.throws(() => validate(overflow));
});

// Optional real receipts exercise the same checker after page-only edits.
for (const key of ["ST_WAVE_NATIVE_REPORT", "ST_WAVE_BROWSER_REPORT"]) {
  if (process.env[key]) {
    test(`retained runtime receipt passes: ${key}`, () =>
      validate(JSON.parse(fs.readFileSync(process.env[key], "utf8")).batch_reduction));
  }
}
