// Real WASM layout operations, including preparation and free, not browser timing.
const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { AutogradTensor } = require(process.argv[2]);
const cases = [];
for (const [rows, cols] of [[64, 64], [768, 3072], [3072, 768], [256, 1024], [137, 195], [1025, 97]]) {
  const values = Float32Array.from({ length: rows * cols }, (_, i) => (i % 17 - 8) / 256);
  const source = new AutogradTensor(rows, cols, values, false);
  const lhsValues = Float32Array.from({ length: rows }, (_, i) => (i % 13 - 6) / 64);
  const lhs = new AutogradTensor(1, rows, lhsValues, false);
  try {
    for (const operation of ["pack", "transpose"]) {
      const run = () => operation === "pack" ? source.prepackRhs() : source.transpose();
      for (let i = 0; i < 3; i++) run().free();
      const elapsed = [];
      for (let i = 0; i < 9; i++) {
        const start = process.hrtime.bigint();
        for (let j = 0; j < 2; j++) run().free();
        elapsed.push(Number(process.hrtime.bigint() - start) / 2);
      }
      const product = run();
      let output;
      try {
        const expected = new Float32Array(operation === "pack" ? cols : rows * cols);
        if (operation === "pack") {
          for (let c = 0; c < cols; c++) {
            for (let r = 0; r < rows; r++) expected[c] += lhsValues[r] * values[r * cols + c];
          }
          output = lhs.matmulPrepacked(product);
        } else {
          for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) expected[c * rows + r] = values[r * cols + c];
          }
          output = product;
        }
        const actual = output.values();
        assert.deepEqual(actual, expected);
        cases.push({ rows, cols, operation, valid: true, elapsed_ns: elapsed,
          output_sha256: createHash("sha256").update(new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength)).digest("hex") });
      } finally {
        if (output && output !== product) output.free();
        product.free();
      }
    }
  } finally {
    lhs.free();
    source.free();
  }
}
console.log(JSON.stringify({ schema: "spiraltorch.wasm_cpu_layout.v1", cases,
  warmups: 3, intervals: 9, repetitions: 2, node: process.version,
  boundary: "Node CPU WASM preparation and free included; input construction and correctness matmul/value export excluded; not browser or GPU speed" }));
