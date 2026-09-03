// Real wasm-pack nodejs output is required. No JS loss/derivative implementation.
const assert = require("node:assert/strict");
const { AutogradTensor } = require(process.argv[2]);
const owned = [];
const keep = (value) => { owned.push(value); return value; };
const close = (actual, expected, tolerance = 1e-6) => {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance, `${actual[i]} != ${expected[i]}`);
  }
};

try {
  const logits = keep(new AutogradTensor(3, 2, new Float32Array(6), true));
  const labels = new Int32Array([0, -100, 1]);
  const loss = keep(logits.crossEntropyWithLogits(labels, undefined, undefined, 0.2));
  labels.fill(-100);
  close(loss.values(), [Math.log(2)]);
  const expected = [-0.2, 0.2, 0, 0, 0.2, -0.2];
  close(loss.vectorJacobianProduct(logits, new Float32Array([1])), expected);
  assert.equal(logits.hasGradient(), false);
  assert.equal(loss.backward().semantic_owner, "st-tensor");
  close(logits.gradientValues(), expected);
  const none = keep(logits.crossEntropyWithLogits(new Int32Array([0, -100, 1]), "none"));
  assert.equal(none.rows(), 3);
  close(none.values(), [Math.log(2), 0, Math.log(2)]);
  assert.throws(() => none.backward());
  assert.throws(() => logits.crossEntropyWithLogits(new Int32Array([0, 0, 0]), "avg"));
  assert.throws(() => logits.crossEntropyWithLogits(new Int32Array([0, 2, 0])));
  assert.throws(() => logits.crossEntropyWithLogits(labels));
  assert.throws(() => logits.crossEntropyWithLogits(new Int32Array([0, 0, 0]), "mean", -100, NaN));

  const tiny = keep(new AutogradTensor(1, 2, new Float32Array([0, -80]), true));
  const tinyLoss = keep(tiny.crossEntropyWithLogits(new Int32Array([0])));
  assert.ok(tinyLoss.values()[0] > 0);
  tinyLoss.backward();
  close(tiny.gradientValues(), [-tinyLoss.values()[0], tinyLoss.values()[0]], 0);
  const logProbabilities = keep(tiny.rowLogSoftmax());
  close(logProbabilities.values(), [-tinyLoss.values()[0], -80], 0);
  close(logProbabilities.vectorJacobianProduct(tiny, new Float32Array([1, 0])), [tinyLoss.values()[0], -tinyLoss.values()[0]], 0);

  const features = keep(new AutogradTensor(3, 3, new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), false));
  let weights = keep(new AutogradTensor(3, 3, new Float32Array(9), true));
  const targets = new Int32Array([0, 1, 2]);
  let initial;
  for (let step = 0; step < 180; step++) {
    const output = keep(features.matmul(weights));
    const objective = keep(output.crossEntropyWithLogits(targets));
    if (step === 0) initial = objective.values()[0];
    objective.backward();
    const values = weights.values();
    const gradient = weights.gradientValues();
    for (let index = 0; index < values.length; index++) values[index] -= 0.5 * gradient[index];
    weights = keep(new AutogradTensor(3, 3, values, true));
  }
  const finalLoss = keep(keep(features.matmul(weights)).crossEntropyWithLogits(targets)).values()[0];
  assert.ok(finalLoss < 0.03 && finalLoss < initial / 20);
  console.log(JSON.stringify({ wasmClassification: "passed", steps: 180, initial, finalLoss }));
} finally {
  for (const value of owned.reverse()) value.free();
}
