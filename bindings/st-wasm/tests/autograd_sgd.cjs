// Exercise borrowed handles and atomic optimizer updates in the real WASM module.
const assert = require("node:assert/strict");
const { AutogradTensor, AutogradSgd } = require(process.argv[2]);
const owned = [];
const keep = value => { owned.push(value); return value; };
const variable = values => keep(new AutogradTensor(1, values.length, new Float32Array(values), true));
const values = tensor => Array.from(tensor.values());

try {
  const x = variable([1, -2]);
  const bias = variable([0]);
  const loss = keep(keep(keep(x.hadamard(x)).sum()).add(bias));
  loss.backward();
  const optimizer = keep(new AutogradSgd(0.25));
  assert.equal(optimizer.addParameter(x), 0);
  assert.equal(optimizer.addParameter(bias), 1);
  assert.equal(optimizer.parameterCount(), 2);
  optimizer.step();
  const next = keep(optimizer.parameter(0));
  assert.deepEqual(values(next), [0.5, -1]);
  assert.deepEqual(values(keep(optimizer.parameter(1))), [-0.25]);
  assert.notEqual(next.id(), x.id());
  assert.equal(next.hasGradient(), false);
  loss.backward();
  assert.deepEqual(values(x), [1, -2]);
  assert.deepEqual(Array.from(x.gradientValues()), [4, -8]);
  assert.throws(() => optimizer.step(), /no gradient/);
  assert.throws(() => optimizer.addParameter(next), /unique/);
  assert.throws(() => optimizer.addParameter(keep(x.detach())), /trainable leaves/);
  assert.throws(() => optimizer.addParameter(keep(x.scale(2))), /trainable leaves/);
  assert.throws(() => optimizer.parameter(2), /out of bounds/);
  for (const index of [-1, 0.5, NaN, Infinity, -Infinity, 2 ** 32, 2 ** 53, undefined]) {
    assert.throws(() => optimizer.parameter(index), /index/);
  }
  for (const rate of [0, -1, NaN, Infinity, -Infinity]) {
    assert.throws(() => new AutogradSgd(rate));
    assert.throws(() => optimizer.setLearningRate(rate));
    assert.equal(optimizer.learningRate(), 0.25);
  }
  optimizer.setLearningRate(0.5);
  assert.equal(optimizer.learningRate(), 0.5);
  next.backwardWithGrad(new Float32Array([1, 2]));
  optimizer.zeroGrad();
  assert.equal(next.hasGradient(), false);
  assert.equal(x.hasGradient(), true);

  const empty = keep(new AutogradSgd(1));
  assert.throws(() => empty.step(), /empty/);
  const maximum = (2 - 2 ** -23) * 2 ** 127;
  for (const missing of [true, false]) {
    const first = variable([1]);
    const last = variable([maximum]);
    const failed = keep(new AutogradSgd(1));
    failed.addParameter(first);
    failed.addParameter(last);
    first.backwardWithGrad(new Float32Array([2]));
    if (!missing) last.backwardWithGrad(new Float32Array([-maximum]));
    assert.throws(() => failed.step());
    assert.equal(keep(failed.parameter(0)).id(), first.id());
    assert.equal(keep(failed.parameter(1)).id(), last.id());
    assert.deepEqual(values(first), [1]);
    assert.deepEqual(values(last), [maximum]);
    assert.deepEqual(Array.from(first.gradientValues()), [2]);
    assert.equal(last.hasGradient(), !missing);
  }
  console.log(JSON.stringify({ wasmAutogradSgd: "passed" }));
} finally {
  for (const value of owned.reverse()) value.free();
}
