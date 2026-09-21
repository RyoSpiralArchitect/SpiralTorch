"use strict";
// Test-only shim: owning host results and JS call/free are timed, copies are not.
const assert = require("node:assert/strict");
const { performance } = require("node:perf_hooks");
const api = require(process.argv[2]);
const bitsOf = values => Array.from(new Uint32Array(values.buffer, values.byteOffset, values.length));
const fromBits = bits => new Float32Array(new Uint32Array([bits >>> 0]).buffer)[0];
const reference = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
const check = (actual, values) => {
  assert.equal(actual.length, values.length);
  actual.forEach((a, i) => {
    const b = reference(values[i]);
    assert.ok(Number.isFinite(a) && Math.abs(a - b) <= 2e-6 * (1 + Math.abs(b)), String(i));
  });
};
const contracts = [];
const bound = Math.fround(1e12), boundBits = bitsOf(new Float32Array([bound]))[0];
const safe = [-0, 0, fromBits(1), -fromBits(1), fromBits(0x800000), -fromBits(0x800000),
              fromBits(boundBits - 1), -fromBits(boundBits - 1), bound, -bound];
let state = 0x6a09e667;
while (safe.length < 12288) {
  state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
  const value = fromBits(state);
  if (Number.isFinite(value) && Math.abs(value) <= bound) safe.push(value);
}
for (const outliers of [false, true]) {
  const values = new Float32Array(outliers
    ? [...safe, fromBits(boundBits + 1), -fromBits(boundBits + 1), 2e12, -2e12, 5e12, -5e12] : safe);
  let rowBits;
  for (const layout of [0, 1, 2]) {
    const fixture = new api.HostGeluFixture(values.length / 6, 6, values, layout);
    const result = fixture.run(), output = result.values(), bits = bitsOf(output);
    check(output, values);
    assert.equal(bits[0], 0x80000000);
    assert.equal(bits[1], 0);
    if (layout === 0) rowBits = bits;
    else assert.deepEqual(bits, rowBits);
    contracts.push({ outliers, layout, valid: true, output_bits: bits });
    result.free(); fixture.free();
  }
}
let errorsChecked = 0;
for (const bad of [NaN, Infinity, -Infinity]) {
  for (const layout of [0, 1, 2]) {
    const fixture = new api.HostGeluFixture(2, 3, new Float32Array([1, fromBits(0x7f7fffff), 2, 1e14, 0, bad]), layout);
    assert.throws(() => fixture.run(), e => String(e).includes("gelu_input"));
    fixture.free(); errorsChecked++;
  }
}
for (const [value, label] of [[fromBits(0x7f7fffff), "gelu_square"], [1e14, "gelu_cubic"]]) {
  const fixture = new api.HostGeluFixture(1, 1, new Float32Array([value]), 0);
  assert.throws(() => fixture.run(), e => String(e).includes(label));
  fixture.free(); errorsChecked++;
}
for (const [rows, cols] of [[0, 6], [3, 0]]) {
  const fixture = new api.HostGeluFixture(rows, cols, new Float32Array(), 1);
  const result = fixture.run();
  assert.equal(result.values().length, 0);
  result.free(); fixture.free();
}
const cases = [];
for (const [rows, cols] of [[1, 1], [1, 8], [1, 32], [1, 64], [8, 3072], [32, 3072], [64, 1024], [17, 195], [65, 97]]) {
  const values = Float32Array.from({length: rows * cols}, (_, i) => (i % 257) / 32 - 4);
  const fixture = new api.HostGeluFixture(rows, cols, values, 0);
  const repetitions = values.length <= 64 ? 512 : 8;
  for (let i = 0; i < 3; i++) fixture.run().free();
  const elapsed_ns = [];
  for (let i = 0; i < 15; i++) {
    const start = performance.now();
    for (let j = 0; j < repetitions; j++) fixture.run().free();
    elapsed_ns.push((performance.now() - start) * 1e6 / repetitions);
  }
  const result = fixture.run(), output = result.values();
  check(output, values);
  cases.push({rows, cols, backward: false, valid: true, repetitions, elapsed_ns, output_bits: bitsOf(output)});
  result.free(); fixture.free();
}
console.log(JSON.stringify({schema: "spiraltorch.checked_gelu_wasm_shim.v1", warmups: 3, intervals: 15,
  errors_checked: errorsChecked, empty_shapes_checked: 2, contracts, cases,
  boundary: "Node CPU WASM test shim of Tensor.try_gelu; no public-client, browser or WebGPU performance claim"}));
