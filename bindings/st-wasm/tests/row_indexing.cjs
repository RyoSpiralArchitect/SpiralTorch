const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { AutogradTensor, AutogradSgd } = require(process.argv[2]);
const declarations = fs.readFileSync(path.join(__dirname, "../types/spiraltorch-wasm.d.ts"), "utf8");
for (const method of ["gatherRows", "scatterAddRows"]) assert.ok(declarations.includes(`${method}(`));
const owned = [];
const keep = value => (owned.push(value), value);
const make = (rows, cols, values, trainable = true) => keep(new AutogradTensor(rows, cols, new Float32Array(values), trainable));
// Numerical parity does not require signed-zero identity across CPU targets.
const numbers = values => Array.from(values, value => value === 0 ? 0 : value);
try {
  const table = make(3, 2, [1, 2, 3, 4, 5, 6]);
  const ids = [2, 0, 2];
  const gathered = keep(table.gatherRows(ids));
  ids.fill(1);
  assert.deepEqual(numbers(gathered.values()), [5, 6, 1, 2, 5, 6]);
  keep(gathered.sum()).backward();
  assert.deepEqual(numbers(table.gradientValues()), [1, 1, 0, 0, 2, 2]);
  const scatter = keep(table.scatterAddRows(new Uint32Array([2, 0, 2]), 4));
  assert.deepEqual(numbers(scatter.values()), [3, 4, 0, 0, 6, 8, 0, 0]);
  const foreignIds = vm.runInNewContext("new Uint32Array([2, 0, 2])");
  assert.equal(foreignIds instanceof Uint32Array, false);
  assert.deepEqual(numbers(keep(table.gatherRows(foreignIds)).values()), [5, 6, 1, 2, 5, 6]);
  assert.deepEqual(numbers(keep(table.scatterAddRows(foreignIds, 4)).values()), numbers(scatter.values()));
  const spoof = new Float32Array([1]);
  Object.defineProperty(spoof, Symbol.toStringTag, { value: "Uint32Array" });
  for (const bad of [spoof, new DataView(new ArrayBuffer(4)), vm.runInNewContext("new Float32Array([1])")]) {
    assert.throws(() => table.gatherRows(bad), /Array or Uint32Array/);
  }
  for (const bad of [-1, 0.5, NaN, Infinity, true, "1", 2 ** 32, 3, null, undefined]) {
    assert.throws(() => table.gatherRows([bad]));
    assert.throws(() => table.scatterAddRows([bad, 0, 1], 3));
  }
  for (const bad of ["012", new Float32Array([1]), {}, null]) assert.throws(() => table.gatherRows(bad));
  for (const bad of [-1, 0.5, 2 ** 32, true, "3"]) assert.throws(() => table.scatterAddRows([0, 1, 2], bad));
  assert.equal(keep(table.gatherRows([])).rows(), 0);
  const before = Array.from(table.gradientValues());
  const max = (2 - 2 ** -23) * 2 ** 127;
  assert.throws(() => gathered.backwardWithGrad(new Float32Array(6).fill(max)));
  assert.deepEqual(Array.from(table.gradientValues()), before);

  const optimizer = keep(new AutogradSgd(0.15));
  optimizer.addParameter(make(4, 3, [0.2, -0.1, 0.3, -0.3, 0.2, 0.1, 0.1, 0.3, -0.2, -0.2, -0.3, 0.2]));
  optimizer.addParameter(make(3, 3, [0.2, 0.1, -0.1, -0.2, 0.3, 0.1, 0.1, -0.2, 0.2]));
  const gamma = make(1, 3, [1, 1, 1], false);
  const beta = make(1, 3, [0, 0, 0], false);
  const labels = new Int32Array([1, 2, 3, 0, 1, 2, 3, 0]);
  let initial, final, predictions;
  for (let step = 0; step <= 400; step++) {
    const start = owned.length;
    const weight = keep(optimizer.parameter(0));
    const transition = keep(optimizer.parameter(1));
    const hidden = keep(keep(keep(weight.gatherRows([0, 1, 2, 3, 0, 1, 2, 3])).matmul(transition)).gelu());
    const normalized = keep(hidden.layerNormAffine(gamma, beta));
    const logits = keep(normalized.matmul(keep(weight.transpose())));
    const loss = keep(logits.crossEntropyWithLogits(labels));
    final = loss.item();
    if (step === 0) initial = final;
    if (step < 400) { loss.backward(); optimizer.step(); }
    else {
      const values = logits.values();
      predictions = Array.from(labels, (_, row) => {
        const scores = Array.from(values.slice(row * 4, row * 4 + 4));
        return scores.indexOf(Math.max(...scores));
      });
    }
    for (const handle of owned.splice(start).reverse()) handle.free();
  }
  assert.ok(final < initial * 0.1, `${initial} -> ${final}`);
  assert.deepEqual(predictions, Array.from(labels));
  console.log(JSON.stringify({ rowIndexing: "passed", initial, final, steps: 400, predictions }));
} finally { for (const handle of owned.reverse()) handle.free(); }
