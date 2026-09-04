// Actual WASM execution; JSON-lines fixture transport is outside timed regions.
const fs = require("node:fs");
const { performance } = require("node:perf_hooks");
const { AutogradTensor } = require(process.argv[2]);
let failed = false;
for (const line of fs.readFileSync(0, "utf8").trim().split("\n")) {
  const owned = [];
  try {
    const r = JSON.parse(line);
    const make = (rows, cols, data) => {
      const tensor = new AutogradTensor(rows, cols, new Float32Array(data), false);
      owned.push(tensor);
      return tensor;
    };
    const a = make(r.input_rows, r.input_cols, r.values);
    let fn, prepackMs = null;
    if (r.operation === "matmul" || r.operation === "matmul_prepacked") {
      const b = make(r.input_cols, r.output_cols, r.other);
      if (r.operation === "matmul_prepacked") {
        const start = performance.now();
        const packed = b.prepackRhs();
        prepackMs = performance.now() - start;
        owned.push(packed);
        fn = () => a.matmulPrepacked(packed);
      } else {
        fn = () => a.matmul(b);
      }
    } else if (r.operation === "gather") {
      fn = () => a.gatherRows(r.ids);
    } else if (r.operation === "scatter") {
      fn = () => a.scatterAddRows(r.ids, r.output_rows);
    } else {
      throw new Error("unsupported benchmark operation");
    }
    const firstStart = performance.now();
    const first = fn();
    const firstCall = performance.now() - firstStart;
    const values = Array.from(first.values());
    first.free();
    for (let i = 0; i < r.warmup; i++) fn().free();
    const samples = [];
    for (let i = 0; i < r.iterations; i++) {
      const start = performance.now();
      const output = fn();
      samples.push(performance.now() - start);
      output.free();
    }
    console.log(JSON.stringify({ status: "passed", values, samples_ms: samples,
      first_call_ms: firstCall, prepack_ms: prepackMs, node: process.version,
      boundary: "WASM host tensors -> tensor; JS conversion and optional one-time RHS packing excluded" }));
  } catch (error) {
    failed = true;
    console.log(JSON.stringify({ status: "error", error: String(error) }));
  } finally {
    for (const tensor of owned.reverse()) tensor.free();
  }
}
if (failed) process.exitCode = 1;
