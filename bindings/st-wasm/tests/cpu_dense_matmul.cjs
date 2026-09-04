// Exercise the shared Rust CPU kernels through real WASM autograd, including tails.
const assert = require("node:assert/strict");
const { AutogradTensor } = require(process.argv[2]);
const rows = 17, inner = 37, cols = 29;
const a = Float32Array.from({ length: rows * inner }, (_, i) => (i % 13 - 6) / 8);
const b = Float32Array.from({ length: inner * cols }, (_, i) => (i % 17 - 8) / 8);
const seed = Float32Array.from({ length: rows * cols }, (_, i) => (i % 7 - 3) / 8);
const expected = new Float32Array(rows * cols);
const da = new Float32Array(rows * inner);
const db = new Float32Array(inner * cols);
// These dyadic fixtures are exactly representable throughout the reductions.
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    for (let k = 0; k < inner; k++) {
      expected[r * cols + c] += a[r * inner + k] * b[k * cols + c];
      da[r * inner + k] += seed[r * cols + c] * b[k * cols + c];
      db[k * cols + c] += a[r * inner + k] * seed[r * cols + c];
    }
  }
}
const owned = [];
const keep = tensor => { owned.push(tensor); return tensor; };
try {
  const lhs = keep(new AutogradTensor(rows, inner, a, true));
  const rhs = keep(new AutogradTensor(inner, cols, b, true));
  const output = keep(lhs.matmul(rhs));
  assert.deepEqual(output.values(), expected);
  const report = output.backwardWithGrad(seed);
  assert.equal(report.semantic_owner, "st-tensor");
  assert.equal(report.leaf_gradient_count, 2);
  assert.deepEqual(lhs.gradientValues(), da);
  assert.deepEqual(rhs.gradientValues(), db);
  assert.deepEqual(lhs.values(), a);
  assert.deepEqual(rhs.values(), b);
  console.log("WASM rectangular CPU matmul forward/backward and tails passed");
} finally {
  for (const tensor of owned.reverse()) tensor.free();
}
