const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");
const crypto = require("node:crypto");
const assert = require("node:assert/strict");
const read = name => JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(__dirname, `${name}.json.gz`))));
const hash = name => crypto.createHash("sha256").update(fs.readFileSync(path.resolve(__dirname, name))).digest("hex");
const median = xs => {
  xs = [...xs].sort((a, b) => a - b);
  return (xs[Math.floor((xs.length - 1) / 2)] + xs[Math.floor(xs.length / 2)]) / 2;
};
const key = c => `${c.operation}/${c.size}/${c.seed}`;
const order = ["matmul", "matmul_prepacked", "gather", "scatter"];
const groups = {};
let gates = 0;
function checkTiming(t) {
  assert.equal(t.n, 20);
  assert.equal(t.samples_ms.length, 20);
  assert.equal(t.median_ms, median(t.samples_ms));
  assert.ok(t.samples_ms.every(x => Number.isFinite(x) && x >= 0));
}
for (const group of ["baseline", "simd"]) {
  groups[group] = [read(`${group}-forward`), read(`${group}-reverse`)];
  for (const [index, report] of groups[group].entries()) {
    assert.equal(report.schema, "spiraltorch.wasm_cpu_bench.v2");
    assert.equal(report.wasm_returncode, 0);
    assert.equal(report.cases.length, 36);
    assert.equal(report.threads, 1);
    assert.equal(report.harness_sha256, hash("../../../tools/bench_wasm_vs_torch.py"));
    assert.equal(report.node_harness_sha256, hash("../../../bindings/st-wasm/tests/backend_bench.cjs"));
    assert.equal(report.wasm_sha256, groups[group][0].wasm_sha256);
    assert.deepEqual(report.operation_order, index ? [...order].reverse() : order);
    assert.equal(new Set(report.cases.map(key)).size, 36);
    const control = groups.baseline[index];
    assert.equal(report.request_sha256, control.request_sha256);
    assert.equal(report.torch, control.torch);
    assert.equal(report.host, control.host);
    assert.deepEqual(report.cases.map(c => [key(c), c.input_sha256]), control.cases.map(c => [key(c), c.input_sha256]));
    for (const c of report.cases) {
      assert.equal(c.wasm.status, "passed");
      assert.equal(c.wasm_correctness.passed, true);
      assert.equal(c.torch_correctness.passed, true);
      gates += 2;
      checkTiming(c.wasm.timings);
      checkTiming(c.torch_timings.torch_cpu);
    }
  }
}
assert.notEqual(groups.baseline[0].wasm_sha256, groups.simd[0].wasm_sha256);
const summary = {
  numerical_gates_passed: gates,
  statistic: "median of six per-seed medians across two operation orders; unpaired runtime timings",
  scope: "WASM CPU forward calls; same source, portable versus +simd128 build",
  cases: {},
};
for (const operation of order) {
  summary.cases[operation] = {};
  for (const size of [32, 64, 128]) {
    const entry = {};
    for (const [group, reports] of Object.entries(groups)) {
      const cases = reports.flatMap(r => r.cases.filter(c => c.operation === operation && c.size === size));
      entry[group] = {
        wasm_ms: median(cases.map(c => c.wasm.timings.median_ms)),
        torch_ms: median(cases.map(c => c.torch_timings.torch_cpu.median_ms)),
      };
    }
    entry.wasm_speedup = entry.baseline.wasm_ms / entry.simd.wasm_ms;
    summary.cases[operation][size] = entry;
  }
}
console.log(JSON.stringify(summary, null, 2));
