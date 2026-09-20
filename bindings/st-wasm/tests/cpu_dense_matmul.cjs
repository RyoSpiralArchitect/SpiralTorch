// Shared Rust CPU paths through WASM autograd: direct, packed, tiles and tails.
const assert = require("node:assert/strict");
const { AutogradTensor } = require(process.argv[2]);

function check(rows, inner, cols, packed) {
  const a = Float32Array.from({ length: rows * inner }, (_, i) => (i % 13 - 6) / 8);
  const b = Float32Array.from({ length: inner * cols }, (_, i) => (i % 17 - 8) / 8);
  const seed = Float32Array.from({ length: rows * cols }, (_, i) => (i % 7 - 3) / 8);
  const expected = new Float32Array(rows * cols);
  const da = new Float32Array(rows * inner);
  const db = new Float32Array(inner * cols);
  // Dyadic fixtures remain exactly representable throughout these reductions.
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
    const output = keep(packed ? lhs.matmulPrepacked(keep(rhs.prepackRhs())) : lhs.matmul(rhs));
    assert.deepEqual(output.values(), expected);
    const report = output.backwardWithGrad(seed);
    assert.equal(report.semantic_owner, "st-tensor");
    assert.equal(report.leaf_gradient_count, 2);
    assert.deepEqual(lhs.gradientValues(), da);
    assert.deepEqual(rhs.gradientValues(), db);
    assert.deepEqual(lhs.values(), a);
    assert.deepEqual(rhs.values(), b);
    console.log(`WASM CPU matmul ${rows}x${inner}x${cols}, packed=${packed}: forward/backward passed`);
  } finally {
    for (const tensor of owned.reverse()) tensor.free();
  }
}

for (const [rows, inner, cols] of [
  [1, 31, 8], [3, 37, 13], [4, 5, 16], [8, 31, 8],
  [17, 37, 29], [33, 129, 49], [65, 257, 25],
]) {
  for (const packed of [false, true]) check(rows, inner, cols, packed);
}
