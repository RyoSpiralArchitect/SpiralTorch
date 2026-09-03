// Run against the real wasm-pack --target nodejs package, not a JS replacement.
const assert = require("node:assert/strict");
const { AutogradTensor } = require(process.argv[2]);
const owned = [];
const keep = (tensor) => { owned.push(tensor); return tensor; };
try {
  const x = keep(new AutogradTensor(2, 3, new Float32Array(6), true));
  const bias = keep(new AutogradTensor(1, 3, new Float32Array(3), true));
  const shifted = keep(x.addRow(bias));
  const activated = keep(shifted.gelu());
  const probabilities = keep(activated.rowSoftmax());
  const seed = new Float32Array([1, 0, 0, 1, 0, 0]);
  const vjp = probabilities.vectorJacobianProduct(x, seed);
  assert.equal(x.hasGradient(), false);
  const report = probabilities.backwardWithGrad(seed);
  assert.equal(report.semantic_owner, "st-tensor");
  assert.equal(report.leaf_gradient_count, 2);
  const expected = [1 / 9, -1 / 18, -1 / 18, 1 / 9, -1 / 18, -1 / 18];
  for (const [i, value] of Array.from(x.gradientValues()).entries()) {
    assert.ok(Math.abs(value - expected[i]) < 1e-6);
    assert.ok(Math.abs(value - vjp[i]) < 1e-6);
  }
  const biasGradient = Array.from(bias.gradientValues());
  assert.ok(Math.abs(biasGradient[0] - 2 / 9) < 1e-6);
  const relu = keep(x.relu());
  keep(relu.sum()).backward();
  assert.deepEqual(Array.from(x.values()), [0, 0, 0, 0, 0, 0]);
  const empty = keep(new AutogradTensor(0, 3, new Float32Array(), true));
  const emptyOutput = keep(keep(empty.gelu()).rowSoftmax());
  keep(emptyOutput.sum()).backward();
  assert.equal(empty.gradientValues().length, 0);
  const maximum = (2 - 2 ** -23) * 2 ** 127;
  const largeInput = keep(new AutogradTensor(3, 1, new Float32Array(3), true));
  const largeBias = keep(new AutogradTensor(1, 1, new Float32Array(1), true));
  const largeOutput = keep(largeInput.addRow(largeBias));
  const largeSeed = new Float32Array([maximum, maximum, -maximum]);
  assert.deepEqual(Array.from(largeOutput.vectorJacobianProduct(largeBias, largeSeed)), [maximum]);
  largeOutput.backwardWithGrad(largeSeed);
  assert.deepEqual(Array.from(largeBias.gradientValues()), [maximum]);
  const overflowingSeed = new Float32Array([maximum, maximum, maximum]);
  assert.throws(() => largeOutput.backwardWithGrad(overflowingSeed));
  assert.deepEqual(Array.from(largeBias.gradientValues()), [maximum]);
  assert.deepEqual(Array.from(largeInput.gradientValues()), Array.from(largeSeed));
  console.log("wasm nonlinear autograd exports, VJP, bias reduction, and empty gradients passed");
} finally {
  for (const tensor of owned.reverse()) tensor.free();
}
