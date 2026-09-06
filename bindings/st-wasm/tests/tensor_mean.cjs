const assert = require("node:assert/strict");
const path = require("node:path");
const {TensorMeanBatch} = require(path.resolve(process.argv[2]));

for (const [rows, cols] of [[1, 1025], [17, 65], [37, 71]]) {
  for (const count of [1, 3, 17]) {
    const len = rows * cols;
    const data = Float32Array.from({length: count * len}, (_, i) => ((i * 17) % 4093 - 2046) / 64);
    const expected = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      let sum = 0;
      for (let p = 0; p < count; p++) sum += data[p * len + i];
      expected[i] = sum / count * 1.25;
    }
    const batch = new TensorMeanBatch(rows, cols, count, data);
    try {
      data.fill(NaN);
      assert.deepEqual(batch.meanScaled(1.25), expected);
      assert.throws(() => batch.meanScaled(Infinity));
      const result = batch.meanScaled(1.25);
      result.fill(0);
      assert.deepEqual(batch.meanScaled(1.25), expected);
    } finally { batch.free(); }
  }
}
for (const bad of [-1, 0.5, NaN, Infinity, 2 ** 32, "1", null, true]) {
  assert.throws(() => new TensorMeanBatch(bad, 0, 1, new Float32Array()));
  assert.throws(() => new TensorMeanBatch(0, bad, 1, new Float32Array()));
  assert.throws(() => new TensorMeanBatch(0, 0, bad, new Float32Array()));
}
assert.throws(() => new TensorMeanBatch(1, 1, 0, new Float32Array()));
assert.throws(() => new TensorMeanBatch(1, 2, 2, new Float32Array(2)));
assert.throws(() => new TensorMeanBatch(1, 1, 1, new Float32Array([NaN])));
const cancellation = new TensorMeanBatch(1, 1, 3, new Float32Array([2 ** 60, 1, -(2 ** 60)]));
try { assert.deepEqual(cancellation.meanScaled(-1), new Float32Array([-0])); }
finally { cancellation.free(); }
console.log("TensorMeanBatch WASM numerical, ownership and boundary regressions passed");
