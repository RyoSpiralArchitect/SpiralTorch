const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const read = name => JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(__dirname, `${name}.json.gz`))));
const median = xs => {
  xs = [...xs].sort((a, b) => a - b);
  return (xs[Math.floor((xs.length - 1) / 2)] + xs[Math.floor(xs.length / 2)]) / 2;
};
const hash = name => crypto.createHash("sha256").update(fs.readFileSync(path.resolve(__dirname, name))).digest("hex");
const native = {}, wasm = {};
let gates = 0;
function timings(record) {
  assert.equal(record.samples_ms.length, 20);
  assert.ok(record.samples_ms.every(x => Number.isFinite(x) && x >= 0));
  assert.equal(record.median_ms, median(record.samples_ms));
}
for (const name of ["baseline", "candidate"]) {
  native[name] = [read(`${name}-native`), read(`${name}-native-repeat`)];
  wasm[name] = [read(`${name}-wasm-clean`), read(`${name}-wasm-repeat`)];
  for (const d of native[name]) {
    assert.equal(d.status, "passed");
    assert.equal(d.cases.length, 6);
    assert.equal(d.script_sha256, hash("../2026-09-04-tensor-out-borrow-guard/bench_out.py"));
    assert.equal(d.helper_sha256, hash("../../../tools/bench_backend_vs_torch.py"));
    assert.equal(d.native_sha256, native[name][0].native_sha256);
    assert.deepEqual(d.cases.map(c => c.input_sha256), native.baseline[0].cases.map(c => c.input_sha256));
    for (const c of d.cases) {
      assert.deepEqual(Object.keys(c.correctness).sort(), ["st_faer_out", "st_packed_out", "torch_cpu_out"]);
      for (const v of Object.values(c.correctness)) { assert.equal(v.passed, true); gates++; }
      for (const v of Object.values(c.timings)) timings(v);
    }
  }
  for (const d of wasm[name]) {
    assert.equal(d.wasm_returncode, 0);
    assert.equal(d.cases.length, 27);
    assert.equal(d.request_sha256, wasm.baseline[0].request_sha256);
    assert.equal(d.wasm_sha256, wasm[name][0].wasm_sha256);
    for (const c of d.cases) {
      assert.equal(c.wasm.status, "passed");
      assert.equal(c.wasm_correctness.passed, true);
      assert.equal(c.torch_correctness.passed, true);
      gates += 2;
      timings(c.wasm.timings);
      timings(c.torch_timings.torch_cpu);
    }
  }
}
assert.notEqual(native.baseline[0].native_sha256, native.candidate[0].native_sha256);
assert.notEqual(wasm.baseline[0].wasm_sha256, wasm.candidate[0].wasm_sha256);
const summary = { numerical_gates_passed: gates, statistic: "median of six per-seed medians across two process runs", native: {}, wasm: {} };
for (const size of [128, 256]) {
  summary.native[size] = {};
  for (const name of ["baseline", "candidate"]) {
    const cases = native[name].flatMap(d => d.cases.filter(c => c.size === size));
    summary.native[size][name] = Object.fromEntries(["st_packed_out", "st_faer_out", "torch_cpu_out"].map(k => [k, median(cases.map(c => c.timings[k].median_ms))]));
  }
}
for (const operation of ["matmul", "gather", "scatter"]) {
  summary.wasm[operation] = {};
  for (const size of [32, 64, 128]) {
    summary.wasm[operation][size] = {};
    for (const name of ["baseline", "candidate"]) {
      const cases = wasm[name].flatMap(d => d.cases.filter(c => c.operation === operation && c.size === size));
      summary.wasm[operation][size][name] = {
        wasm_ms: median(cases.map(c => c.wasm.timings.median_ms)),
        torch_ms: median(cases.map(c => c.torch_timings.torch_cpu.median_ms)),
      };
    }
  }
}
console.log(JSON.stringify(summary, null, 2));
