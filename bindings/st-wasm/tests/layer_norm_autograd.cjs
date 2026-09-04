// Execute the generated WASM, never a JavaScript substitute for the VJP.
const assert = require("node:assert/strict");
const { AutogradTensor, AutogradSgd } = require(process.argv[2]);
const fs = require("node:fs");
const path = require("node:path");
const declarations = fs.readFileSync(path.join(__dirname, "../types/spiraltorch-wasm.d.ts"), "utf8");
const declaredClass = declarations.match(/export class AutogradTensor \{([\s\S]*?)\n    \}/)[1];
for (const method of ["addRow", "relu", "gelu", "rowSoftmax", "rowLogSoftmax", "crossEntropyWithLogits", "layerNormAffine"]) {
  assert.equal(typeof AutogradTensor.prototype[method], "function", method);
  assert.ok(declaredClass.includes(`${method}(`), `missing TypeScript method ${method}`);
}
const owned = [];
const keep = value => { owned.push(value); return value; };
const make = (rows, cols, values, trainable = true) => keep(new AutogradTensor(rows, cols, new Float32Array(values), trainable));
const near = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Number.isFinite(value) && Math.abs(value - expected[i]) < 2e-5));
};
try {
  const x = make(2, 3, [0, 1, 2, 2, 1, 0]);
  const gamma = make(1, 3, [1.5, -0.75, 0.35]);
  const beta = make(1, 3, [0.1, -0.2, 0.05]);
  const y = keep(x.layerNormAffine(gamma, beta));
  assert.equal(y.operationName(), "layer_norm_affine");
  const z = Math.sqrt(1 / (2 / 3 + Math.fround(1e-5)));
  near(y.values(), [-1.5*z+0.1, -0.2, 0.35*z+0.05, 1.5*z+0.1, -0.2, -0.35*z+0.05]);
  const seed = new Float32Array([1, 2, 3, 4, 5, 6]);
  const before = y.vectorJacobianProduct(x, seed);
  assert.equal(x.hasGradient(), false);
  const report = y.backwardWithGrad(seed);
  assert.equal(report.leaf_gradient_count, 3);
  near(x.gradientValues(), before);
  near(gamma.gradientValues(), [3*z, 0, -3*z]);
  near(beta.gradientValues(), [5, 7, 9]);
  for (const epsilon of [-1, NaN, Infinity]) assert.throws(() => x.layerNormAffine(gamma, beta, epsilon));
  const empty = make(0, 3, []);
  keep(keep(empty.layerNormAffine(gamma, beta)).sum()).backward();
  assert.equal(empty.gradientValues().length, 0);

  const maximum = (2 - 2 ** -23) * 2 ** 127;
  const narrow = make(2, 1, [1, 2]);
  const frozenGamma = make(1, 1, [maximum], false);
  const frozenBeta = make(1, 1, [0], false);
  keep(narrow.layerNormAffine(frozenGamma, frozenBeta)).backwardWithGrad(new Float32Array([maximum, maximum]));
  near(narrow.gradientValues(), [0, 0]);
  const trainBeta = make(1, 1, [0]);
  const overflow = keep(narrow.layerNormAffine(frozenGamma, trainBeta));
  assert.throws(() => overflow.backwardWithGrad(new Float32Array([maximum, maximum])));
  assert.equal(trainBeta.hasGradient(), false);
  near(narrow.gradientValues(), [0, 0]);

  const input = make(3, 3, [0.4, -0.8, 1.2, -0.3, 0.9, -1.1, 0.7, 0.1, -0.2], false);
  const target = keep(input.layerNormAffine(make(1, 3, [1.7, 0.5, -0.8], false), make(1, 3, [0.2, -0.3, 0.6], false)));
  const optimizer = keep(new AutogradSgd(0.1));
  optimizer.addParameter(make(1, 3, [1, 1, 1]));
  optimizer.addParameter(make(1, 3, [0, 0, 0]));
  let first, last;
  for (let step = 0; step < 400; step++) {
    const prediction = keep(input.layerNormAffine(keep(optimizer.parameter(0)), keep(optimizer.parameter(1))));
    const loss = keep(prediction.meanSquaredError(target));
    last = loss.item();
    if (step === 0) first = last;
    loss.backward();
    optimizer.step();
  }
  assert.ok(last < first * 1e-4, `${first} -> ${last}`);
  console.log(JSON.stringify({ layerNormAutograd: "passed", steps: 400, first, last }));
} finally {
  for (const value of owned.reverse()) value.free();
}
