// Validate JavaScript ingress against the real generated WASM, not a mock ABI.
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const { test } = require("node:test");
const modulePath = resolve(process.argv[2]);
const st = require(modulePath);
const U32_MAX = 2 ** 32 - 1;
const I32_MIN = -(2 ** 31);
const I32_MAX = 2 ** 31 - 1;
const invalidNumbers = [NaN, Infinity, -Infinity, -0.5, 0.5, Number.MIN_VALUE];
const nonNumbers = ["0", "1", "", false, true, [], {}, new Number(0), 0n, Symbol("index")];
const invalidU32 = [...invalidNumbers, -1, 2 ** 32, 2 ** 32 + 1, Number.MAX_SAFE_INTEGER,
  ...nonNumbers, null, undefined];
const invalidI32 = [...invalidNumbers, -100.5, 0.9, I32_MIN - 1, I32_MAX + 1,
  2 ** 32 - 100, ...nonNumbers];
const record = { rows_min: 1, rows_max: 32, cols_min: 1, cols_max: 128,
  k_min: 1, k_max: 32, wg: 64, tile_cols: 32, radix: 2, segments: 1 };

function withHandles(body) {
  const owned = [];
  const errors = [];
  const keep = value => { owned.push(value); return value; };
  try {
    body(keep);
  } catch (error) {
    errors.push(error);
  }
  for (const value of owned.reverse()) {
    try { value.free(); } catch (error) { errors.push(error); }
  }
  if (errors.length === 1) throw errors[0];
  if (errors.length > 1) throw new AggregateError(errors, "WASM test/cleanup failed");
}

test("autograd dimensions reject coercion before tensor construction", () => withHandles(keep => {
  for (const dimension of [0, 1]) {
    for (const invalid of invalidU32) {
      const shape = [1, 1];
      shape[dimension] = invalid;
      assert.throws(() => keep(new st.AutogradTensor(...shape, new Float32Array(), true)),
        /autograd (rows|cols) must be/);
    }
  }
  for (const shape of [[0, 1], [1, 0], [-0, 1], [U32_MAX, 0], [0, U32_MAX]]) {
    const tensor = keep(new st.AutogradTensor(...shape, new Float32Array(), true));
    assert.equal(tensor.rows(), shape[0] === 0 ? 0 : shape[0]);
    assert.equal(tensor.cols(), shape[1]);
  }
  const valid = keep(new st.AutogradTensor(1, 1, new Float32Array([3]), true));
  assert.deepEqual(Array.from(valid.values()), [3]);
}));

test("ignore_index rejects lossy conversions without changing gradients", () => withHandles(keep => {
  const logits = keep(new st.AutogradTensor(2, 2, new Float32Array(4), true));
  const labels = new Int32Array([0, 1]);
  keep(logits.crossEntropyWithLogits(labels)).backward();
  const expected = [-0.25, 0.25, 0.25, -0.25];
  assert.deepEqual(Array.from(logits.gradientValues()), expected);
  const summary = logits.graphSummary();
  for (const invalid of invalidI32) {
    assert.throws(() => keep(logits.crossEntropyWithLogits(labels, undefined, invalid)),
      /autograd ignore_index must be/);
    assert.deepEqual(Array.from(logits.gradientValues()), expected);
    assert.deepEqual(logits.graphSummary(), summary);
  }
  // null/undefined keep the documented default; signed sentinels remain exact.
  for (const sentinel of [undefined, null, -100, I32_MIN, I32_MAX]) {
    logits.zeroGrad();
    keep(logits.crossEntropyWithLogits(labels, undefined, sentinel)).backward();
    assert.deepEqual(Array.from(logits.gradientValues()), expected);
  }
  for (const sentinel of [I32_MIN, I32_MAX, -100, 0]) {
    logits.zeroGrad();
    keep(logits.crossEntropyWithLogits(new Int32Array([sentinel, 1]), "mean", sentinel)).backward();
    assert.deepEqual(Array.from(logits.gradientValues()), [0, 0, 0.5, -0.5]);
  }
}));

test("optimizer indices reject non-numbers as well as malformed numbers", () => withHandles(keep => {
  const parameter = keep(new st.AutogradTensor(1, 1, new Float32Array([2]), true));
  const optimizer = keep(new st.AutogradSgd(0.1));
  optimizer.addParameter(parameter);
  for (const invalid of invalidU32) {
    assert.throws(() => keep(optimizer.parameter(invalid)), /parameter index must be/);
    assert.equal(optimizer.parameterCount(), 1);
    assert.equal(keep(optimizer.parameter(0)).id(), parameter.id());
  }
  assert.equal(keep(optimizer.parameter(-0)).id(), parameter.id());
  for (const index of [1, U32_MAX]) {
    assert.throws(() => keep(optimizer.parameter(index)), /out of bounds/);
  }
}));

test("numeric transport never invokes user coercion hooks", () => withHandles(keep => {
  let coercions = 0;
  const object = { [Symbol.toPrimitive]() { coercions++; return 0; } };
  const parameter = keep(new st.AutogradTensor(1, 1, new Float32Array([2]), true));
  const optimizer = keep(new st.AutogradSgd(0.1));
  optimizer.addParameter(parameter);
  const tuner = keep(new st.WasmTuner(JSON.stringify([record])));
  assert.throws(() => keep(optimizer.parameter(object)), /parameter index must be a number/);
  assert.throws(() => tuner.removeIndex(object), /record index must be a number/);
  assert.throws(() => keep(parameter.crossEntropyWithLogits(new Int32Array([0]), undefined, object)),
    /ignore_index must be a number/);
  assert.equal(coercions, 0);
  assert.equal(tuner.len(), 1);
}));

test("tuner index mutations reject invalid input before touching records", () => withHandles(keep => {
  const tuner = keep(new st.WasmTuner(JSON.stringify([record])));
  const before = tuner.to_json();
  for (const method of ["recordAt", "removeIndex", "replaceIndex"]) {
    for (const invalid of invalidU32) {
      assert.throws(() => tuner[method](invalid, { ...record, wg: 128 }), /record index must be/);
      assert.equal(tuner.to_json(), before);
    }
  }
  let recordReads = 0;
  const unreadRecord = { get cols_min() { recordReads++; return 1; } };
  assert.throws(() => tuner.replaceIndex(NaN, unreadRecord), /record index must be/);
  assert.equal(recordReads, 0, "invalid indices must fail before record deserialization");
  assert.equal(tuner.to_json(), before);
  for (const index of [1, U32_MAX]) {
    assert.equal(tuner.recordAt(index), undefined);
    assert.equal(tuner.removeIndex(index), undefined);
    assert.equal(tuner.replaceIndex(index, record), false);
    assert.equal(tuner.to_json(), before);
  }
  assert.equal(tuner.recordAt(-0).wg, 64);
  assert.equal(tuner.replaceIndex(0, { ...record, wg: 128 }), true);
  assert.equal(tuner.removeIndex(0).wg, 128);
  assert.equal(tuner.len(), 0);
}));

const workloadMethods = ["findRecord", "removeRecord", "choose", "planFft", "planFftJson",
  "planFftObject", "planFftWithFallback", "planFftWithFallbackJson", "planFftWithFallbackObject",
  "planFftResolution", "planFftResolutionJson", "planFftResolutionObject", "planFftReport"];

test("every tuner workload route validates all three dimensions", () => withHandles(keep => {
  const tuner = keep(new st.WasmTuner(JSON.stringify([record])));
  const before = tuner.to_json();
  for (const method of [...workloadMethods, "baseChoice"]) {
    for (const dimension of [0, 1, 2]) {
      for (const invalid of invalidU32) {
        const shape = [2, 64, 4, false];
        shape[dimension] = invalid;
        assert.throws(() => {
          const result = method === "baseChoice" ? st.baseChoice(...shape) : tuner[method](...shape);
          if (result && typeof result.free === "function") keep(result);
        }, /tuner (rows|cols|k) must be/);
        assert.equal(tuner.to_json(), before);
      }
    }
  }
}));

test("valid workload wrappers retain matching and fallback behavior", () => withHandles(keep => {
  const tuner = keep(new st.WasmTuner(JSON.stringify([record])));
  const args = [2, 64, 4, false];
  for (const method of workloadMethods.filter(name => name !== "removeRecord")) {
    const result = tuner[method](...args);
    assert.notEqual(result, undefined, method);
    if (result && typeof result.free === "function") keep(result);
  }
  const plan = keep(tuner.planFft(...args));
  assert.deepEqual(JSON.parse(tuner.planFftJson(...args)), JSON.parse(plan.toJson()));
  assert.deepEqual(tuner.planFftObject(...args), plan.toObject());
  const fallback = keep(tuner.planFftWithFallback(...args));
  assert.deepEqual(JSON.parse(tuner.planFftWithFallbackJson(...args)), JSON.parse(fallback.toJson()));
  assert.deepEqual(tuner.planFftWithFallbackObject(...args), fallback.toObject());
  assert.deepEqual(tuner.planFftResolutionObject(...args), tuner.planFftReport(...args));
  for (const args of [[0, 0, 0, false], [U32_MAX, U32_MAX, U32_MAX, true]]) {
    assert.equal(tuner.findRecord(...args), undefined);
    assert.equal(tuner.removeRecord(...args), undefined);
    assert.equal(tuner.choose(...args), undefined);
    assert.equal(tuner.planFft(...args), undefined);
    assert.equal(tuner.planFftJson(...args), undefined);
    assert.equal(tuner.planFftObject(...args), undefined);
    assert.equal(typeof st.baseChoice(...args), "object");
  }
  assert.equal(tuner.removeRecord(...args).wg, 64);
  assert.equal(tuner.len(), 0);
  const originalNow = Date.now;
  let clockReads = 0;
  Date.now = () => { clockReads++; return 1_700_000_000_000; };
  try {
    const defaultPlan = keep(tuner.planFftWithFallback(...args));
    assert.equal(typeof defaultPlan.toObject(), "object");
    assert.ok(clockReads > 0, "shared heuristic telemetry must use the JS host clock");
  } finally {
    Date.now = originalNow;
  }
}));

test("generated TypeScript retains numeric and optional parameter types", () => {
  const declarations = readFileSync(modulePath.replace(/\.js$/, ".d.ts"), "utf8");
  assert.match(declarations, /constructor\(rows: number, cols: number, values: Float32Array, requires_grad: boolean\)/);
  assert.match(declarations, /ignore_index\?: number \| null, label_smoothing\?: number \| null/);
  assert.match(declarations, /parameter\(index: number\)/);
  for (const method of ["recordAt", "removeIndex", "replaceIndex"]) {
    assert.match(declarations, new RegExp(`${method}\\(index: number[,)]`));
  }
  for (const method of [...workloadMethods, "baseChoice"]) {
    assert.match(declarations, new RegExp(`${method}\\(rows: number, cols: number, k: number, subgroup: boolean\\)`));
  }
});
