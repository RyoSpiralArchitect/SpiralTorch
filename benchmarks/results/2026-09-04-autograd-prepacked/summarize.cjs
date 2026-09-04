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
  assert.ok(t.samples_ms.every(x => Number.isFinite(x) && x >= 0));
  assert.equal(t.median_ms, median(t.samples_ms));
}
for (const [group, prefix] of [["before_cpu_route", "wasm"], ["after_cpu_route", "wasm-routed"]]) {
  groups[group] = [read(`${prefix}-forward`), read(`${prefix}-reverse`)];
  for (const [index, report] of groups[group].entries()) {
    assert.equal(report.schema, "spiraltorch.wasm_cpu_bench.v2");
    assert.equal(report.wasm_returncode, 0);
    assert.equal(report.threads, 1);
    assert.equal(report.cases.length, 36);
    assert.equal(report.harness_sha256, hash("../../../tools/bench_wasm_vs_torch.py"));
    assert.equal(report.node_harness_sha256, hash("../../../bindings/st-wasm/tests/backend_bench.cjs"));
    assert.equal(report.wasm_sha256, groups[group][0].wasm_sha256);
    assert.deepEqual(report.operation_order, index ? [...order].reverse() : order);
    assert.equal(new Set(report.cases.map(key)).size, 36);
    const baseline = groups.before_cpu_route[index];
    assert.equal(report.request_sha256, baseline.request_sha256);
    assert.equal(report.torch, baseline.torch);
    assert.equal(report.host, baseline.host);
    assert.deepEqual(report.cases.map(c => [key(c), c.input_sha256]), baseline.cases.map(c => [key(c), c.input_sha256]));
    for (const c of report.cases) {
      assert.equal(c.wasm.status, "passed");
      assert.equal(c.wasm_correctness.passed, true);
      assert.equal(c.torch_correctness.passed, true);
      gates += 2;
      checkTiming(c.wasm.timings);
      checkTiming(c.torch_timings.torch_cpu);
      if (c.operation === "matmul_prepacked") {
        assert.ok(Number.isFinite(c.wasm.prepack_ms) && c.wasm.prepack_ms >= 0);
        const plain = report.cases.find(x => x.operation === "matmul" && x.size === c.size && x.seed === c.seed);
        assert.deepEqual(c.input_sha256, plain.input_sha256);
      } else {
        assert.equal(c.wasm.prepack_ms, null);
      }
    }
  }
}
assert.notEqual(groups.before_cpu_route[0].wasm_sha256, groups.after_cpu_route[0].wasm_sha256);
const summary = {
  numerical_gates_passed: gates,
  statistic: "median of six per-seed medians across two operation orders; unpaired runtime timings",
  scope: "WASM CPU forward AutogradTensor calls, not backward or browser WebGPU; one-time packing excluded",
  cases: {},
};
for (const operation of order) {
  summary.cases[operation] = {};
  for (const size of [32, 64, 128]) {
    summary.cases[operation][size] = {};
    for (const [group, reports] of Object.entries(groups)) {
      const cases = reports.flatMap(r => r.cases.filter(c => c.operation === operation && c.size === size));
      summary.cases[operation][size][group] = {
        wasm_ms: median(cases.map(c => c.wasm.timings.median_ms)),
        torch_ms: median(cases.map(c => c.torch_timings.torch_cpu.median_ms)),
        one_time_pack_ms: operation === "matmul_prepacked" ? median(cases.map(c => c.wasm.prepack_ms)) : null,
      };
    }
  }
}
console.log(JSON.stringify(summary, null, 2));
