// Measured WASM client calls; construction, prepacking and output export are excluded.
const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { AutogradTensor } = require(process.argv[2]);

const shapes = [
  [8, 768, 3072], [32, 768, 3072], [64, 256, 1024],
  [128, 256, 256], [17, 137, 195], [65, 1025, 97],
];
const cases = [];
for (const [rows, inner, cols] of shapes) {
  const a = Float32Array.from({ length: rows * inner }, (_, i) => (i % 13 - 6) / 8);
  const b = Float32Array.from({ length: inner * cols }, (_, i) => (i % 17 - 8) / 8);
  const expected = new Float32Array(rows * cols);
  // All products and partial sums of these bounded dyadic fixtures are exact f32.
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      for (let k = 0; k < inner; k++) {
        expected[row * cols + col] += a[row * inner + k] * b[k * cols + col];
      }
    }
  }
  const lhs = new AutogradTensor(rows, inner, a, false);
  const rhs = new AutogradTensor(inner, cols, b, false);
  const packed = rhs.prepackRhs();
  try {
    for (const usePacked of [false, true]) {
      const run = () => usePacked ? lhs.matmulPrepacked(packed) : lhs.matmul(rhs);
      for (let i = 0; i < 3; i++) run().free();
      const elapsedNs = [];
      for (let interval = 0; interval < 9; interval++) {
        const start = process.hrtime.bigint();
        for (let repetition = 0; repetition < 2; repetition++) run().free();
        elapsedNs.push(Number(process.hrtime.bigint() - start) / 2);
      }
      const output = run();
      let hash;
      try {
        const values = output.values();
        assert.deepEqual(values, expected);
        hash = createHash("sha256").update(new Uint8Array(values.buffer, values.byteOffset, values.byteLength)).digest("hex");
      } finally {
        output.free();
      }
      cases.push({ rows, inner, cols, packed: usePacked, bitwise_equal: true,
        output_sha256: hash, elapsed_ns: elapsedNs, repetitions: 2 });
    }
  } finally {
    packed.free();
    rhs.free();
    lhs.free();
  }
}
console.log(JSON.stringify({ schema: "spiraltorch.wasm_cpu_dense.v1", cases,
  node: process.version, arch: process.arch, platform: process.platform,
  requires_grad: false, fixture: "dyadic_13_17_over_8",
  boundary: "Node WASM CPU tensor forward and output free; input construction, prepack and JS output export excluded; not a browser or GPU measurement" }));
