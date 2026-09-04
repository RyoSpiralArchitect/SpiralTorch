// Real wasm-pack nodejs output is required. No JS loss/derivative implementation.
const assert = require("node:assert/strict");
const { AutogradTensor, AutogradSgd } = require(process.argv[2]);
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

  const maximum = (2 - 2 ** -23) * 2 ** 127;
  const cancellation = keep(new AutogradTensor(1, 3, new Float32Array(3), true));
  const cancellationOutput = keep(cancellation.rowLogSoftmax());
  const cancellationSeed = new Float32Array([maximum, 1, -maximum]);
  const cancellationGradient = cancellationOutput.vectorJacobianProduct(cancellation, cancellationSeed);
  close([cancellationGradient[1]], [2 / 3]);
  assert.equal(cancellationGradient[0], maximum);
  assert.equal(cancellationGradient[2], -maximum);

  for (const sign of [1, -1]) {
    for (const classes of [49, 257, 50_000]) {
      const input = keep(new AutogradTensor(1, classes, new Float32Array(classes), true));
      const output = keep(input.rowLogSoftmax());
      const seed = new Float32Array(classes).fill(sign * maximum);
      close(output.vectorJacobianProduct(input, seed), new Float32Array(classes), 0);
    }
    for (const [values, seed, expected] of [
      [[0, -200], [maximum, 1], [-1, 1]],
      [[-200, 0], [1, maximum], [1, -1]],
      [[0, 0, -200], [maximum / 2, maximum / 2, 1], [-0.5, -0.5, 1]],
    ]) {
      const input = keep(new AutogradTensor(1, values.length, new Float32Array(values), true));
      const output = keep(input.rowLogSoftmax());
      close(output.vectorJacobianProduct(input, new Float32Array(seed.map(value => sign * value))),
            expected.map(value => sign * value), 0);
    }
  }

  const features = keep(new AutogradTensor(3, 3, new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), false));
  const optimizer = keep(new AutogradSgd(0.5));
  optimizer.addParameter(keep(new AutogradTensor(3, 3, new Float32Array(9), true)));
  const targets = new Int32Array([0, 1, 2]);
  let initial;
  for (let step = 0; step < 180; step++) {
    const weights = optimizer.parameter(0);
    const output = features.matmul(weights);
    const objective = output.crossEntropyWithLogits(targets);
    try {
      if (step === 0) initial = objective.values()[0];
      objective.backward();
      optimizer.step();
    } finally {
      objective.free();
      output.free();
      weights.free();
    }
  }
  const finalLoss = keep(keep(features.matmul(keep(optimizer.parameter(0)))).crossEntropyWithLogits(targets)).values()[0];
  assert.ok(finalLoss < 0.03 && finalLoss < initial / 20);
  console.log(JSON.stringify({ wasmClassification: "passed", steps: 180, initial, finalLoss }));
} finally {
  for (const value of owned.reverse()) value.free();
}
