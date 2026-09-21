// The WASM autograd VJP reaches the same host Tensor backward implementation.
const assert = require("node:assert/strict");
const { AutogradTensor } = require(process.argv[2]);
for (const [rows, cols] of [[1, 1], [2, 6], [33, 195], [1, 257]]) {
  const x = Float32Array.from({ length: rows * cols }, (_, i) => (i % 131) / 16 - 4);
  const g = Float32Array.from({ length: rows * cols }, (_, i) => (i % 29) / 16 - 0.5);
  const input = new AutogradTensor(rows, cols, x, true);
  let output;
  try {
    output = input.gelu();
    const receipt = output.backwardWithGrad(g);
    assert.equal(receipt.semantic_owner, "st-tensor");
    assert.equal(receipt.leaf_gradient_count, 1);
    const values = output.values(), gradients = input.gradientValues();
    for (let i = 0; i < x.length; i++) {
      const c = Math.sqrt(2 / Math.PI), v = x[i];
      const t = Math.tanh(c * (v + 0.044715 * v * v * v));
      const y = 0.5 * v * (1 + t);
      const dx = (0.5 * (1 + t) + 0.5 * v * (1 - t * t) * c * (1 + 3 * 0.044715 * v * v)) * g[i];
      assert.ok(Number.isFinite(values[i]) && Math.abs(values[i] - y) <= 2e-6 * (1 + Math.abs(y)));
      assert.ok(Number.isFinite(gradients[i]) && Math.abs(gradients[i] - dx) <= 2e-6 * (1 + Math.abs(dx)));
    }
    assert.deepEqual(input.values(), x);
    console.log(`WASM GELU ${rows}x${cols}: independent forward/VJP checks passed`);
  } finally {
    if (output) output.free();
    input.free();
  }
}
