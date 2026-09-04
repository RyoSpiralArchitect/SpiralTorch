const assert = require("node:assert/strict");
const { AutogradTensor, AutogradPackedRhs, AutogradSgd } = require(process.argv[2]);
const owned = [];
const keep = x => { owned.push(x); return x; };
const node = (rows, cols, data, trainable = true) => keep(new AutogradTensor(rows, cols, new Float32Array(data), trainable));
const values = x => Array.from(x.values());
try {
  const x = node(2, 3, [1, 2, -1, 0.5, -2, 3]);
  const weight = node(3, 2, [2, -1, 0, 3, 1.5, 2]);
  const rhs = keep(weight.scale(0.5));
  const packed = keep(rhs.prepackRhs());
  assert.ok(packed instanceof AutogradPackedRhs);
  assert.equal(packed.sourceId(), rhs.id());
  assert.deepEqual([packed.rows(), packed.cols(), packed.requiresGrad()], [3, 2, true]);
  const first = keep(x.matmulPrepacked(packed));
  assert.deepEqual(values(first), values(keep(x.matmul(rhs))));
  const report = keep(keep(first.add(keep(x.matmulPrepacked(packed)))).sum()).backward();
  assert.equal(report.leaf_gradient_count, 2);
  assert.deepEqual(Array.from(x.gradientValues()), [1, 3, 3.5, 1, 3, 3.5]);
  assert.deepEqual(Array.from(weight.gradientValues()), [1.5, 1.5, 0, 0, 2, 2]);

  const input = node(1, 1, [2], false);
  const parameter = node(1, 1, [3]);
  const old = keep(parameter.prepackRhs());
  const optimizer = keep(new AutogradSgd(0.5));
  optimizer.addParameter(parameter);
  keep(input.matmulPrepacked(old)).backward();
  optimizer.step();
  const refreshed = keep(keep(optimizer.parameter(0)).prepackRhs());
  assert.notEqual(old.sourceId(), refreshed.sourceId());
  assert.equal(keep(input.matmulPrepacked(old)).item(), 6);
  assert.equal(keep(input.matmulPrepacked(refreshed)).item(), 4);
  assert.throws(() => x.matmulPrepacked(old));

  const frozen = keep(input.prepackRhs());
  assert.equal(frozen.requiresGrad(), false);
  const target = node(1, 1, [6], false);
  const learner = keep(new AutogradSgd(0.05));
  learner.addParameter(node(1, 1, [0]));
  let last;
  for (let i = 0; i < 40; i++) {
    // Fresh trainable leaves, one persistent frozen RHS, and bounded JS handles.
    const current = learner.parameter(0);
    const prediction = current.matmulPrepacked(frozen);
    const loss = prediction.meanSquaredError(target);
    try {
      last = loss.item();
      loss.backward();
      learner.step();
    } finally {
      loss.free(); prediction.free(); current.free();
    }
  }
  assert.ok(last < 1e-9);
  for (const [rows, inner, cols] of [[0, 3, 2], [2, 0, 3], [2, 3, 0]]) {
    const a = node(rows, inner, new Float32Array(rows * inner));
    const b = node(inner, cols, new Float32Array(inner * cols));
    const out = keep(a.matmulPrepacked(keep(b.prepackRhs())));
    assert.deepEqual([out.rows(), out.cols()], [rows, cols]);
    keep(out.sum()).backward();
    assert.equal(a.gradientValues().length, rows * inner);
    assert.equal(b.gradientValues().length, inner * cols);
  }
  console.log(JSON.stringify({ autogradPrepacked: "passed", frozenProjectionLoss: last }));
} finally {
  for (const x of owned.reverse()) x.free();
}
