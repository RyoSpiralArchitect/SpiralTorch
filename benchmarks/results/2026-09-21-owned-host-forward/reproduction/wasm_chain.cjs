"use strict";
const assert = require("node:assert/strict");
const {performance} = require("node:perf_hooks");
const api = require(process.argv[2]);
const bits = values => Array.from(new Uint32Array(values.buffer, values.byteOffset, values.length));
function scalar(x) {
  const c = Math.sqrt(2 / Math.PI), t = Math.tanh(c * (x + 0.044715 * x * x * x));
  return [0.5 * x * (1 + t), 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)];
}
function expected(values, seeds, depth) {
  const forward = [], backward = [];
  for (let i = 0; i < values.length; i++) {
    let x = values[i], g = seeds ? seeds[i] : 1;
    for (let n = 0; n < depth; n++) {
      const [y, d] = scalar(x);
      x = y; g *= d;
    }
    forward.push(x); backward.push(g);
  }
  return {forward, backward};
}
function check(a, b) {
  assert.equal(a.length, b.length);
  a.forEach((v, i) => assert.ok(Number.isFinite(v) && Math.abs(v - b[i]) <= 2e-6 * (1 + Math.abs(b[i])), String(i)));
}
const contracts = [];
for (const [rows, cols] of [[2, 6], [33, 195]]) {
  const values = Float32Array.from({length: rows * cols}, (_, i) => (i % 131) / 16 - 4);
  const seeds = Float32Array.from({length: rows * cols}, (_, i) => (i % 29) / 16 - 0.5);
  for (const depth of [1, 4]) for (const nested of [false, true]) for (const layout of [0, 1, 2]) {
    const fixture = new api.HostChain(rows, cols, values, depth, layout, nested);
    const ref = expected(values, seeds, depth);
    const output = fixture.run(), gradient = fixture.backward(seeds), again = fixture.run();
    const y = output.values(), dx = gradient.values();
    check(y, ref.forward); check(dx, ref.backward);
    assert.deepEqual(bits(y), bits(again.values()));
    assert.deepEqual(bits(fixture.input_values()), bits(values));
    contracts.push({rows, cols, depth, nested, layout, valid: true, output_bits: bits(y), gradient_bits: bits(dx)});
    output.free(); gradient.free(); again.free(); fixture.free();
  }
}
let error_cases = 0;
for (const [values, label] of [[[3e38, NaN], "gelu_input"], [[3e38], "gelu_square"], [[1e14], "gelu_cubic"]]) {
  const fixture = new api.HostChain(1, values.length, new Float32Array(values), 4, 0, true);
  assert.throws(() => fixture.run(), e => String(e).includes(label));
  fixture.free(); error_cases++;
}
for (const [rows, cols] of [[0, 6], [3, 0]]) {
  const fixture = new api.HostChain(rows, cols, new Float32Array(), 4, 1, true);
  const output = fixture.run();
  assert.equal(output.values().length, 0);
  output.free(); fixture.free();
}
const zero = new api.HostChain(1, 2, new Float32Array([-0, 0]), 4, 0, true);
const zeroResult = zero.run();
assert.deepEqual(bits(zeroResult.values()), [0x80000000, 0]);
zeroResult.free(); zero.free();
const cases = [];
for (const [rows, cols] of [[1, 1], [1, 8], [1, 32], [1, 64], [8, 3072], [32, 3072], [64, 1024], [17, 195], [65, 97]]) {
  const values = Float32Array.from({length: rows * cols}, (_, i) => (i % 257) / 32 - 4);
  for (const depth of [1, 4, 16]) {
    const fixture = new api.HostChain(rows, cols, values, depth, 0, false);
    const ref = expected(values, null, depth).forward;
    const repetitions = values.length <= 64 ? 512 : 8;
    for (let i = 0; i < 3; i++) fixture.run().free();
    const elapsed_ns = [];
    for (let i = 0; i < 15; i++) {
      const start = performance.now();
      for (let j = 0; j < repetitions; j++) fixture.run().free();
      elapsed_ns.push((performance.now() - start) * 1e6 / repetitions);
    }
    const result = fixture.run(), output = result.values();
    check(output, ref);
    assert.deepEqual(bits(fixture.input_values()), bits(values));
    cases.push({rows, cols, depth, repetitions, valid: true, elapsed_ns, output_bits: bits(output)});
    result.free(); fixture.free();
  }
}
console.log(JSON.stringify({schema: "spiraltorch.host_chain_wasm_shim.v1", warmups: 3, intervals: 15,
  error_cases, empty_shapes: 2, signed_zero: true, contracts, cases,
  boundary: "Node CPU WASM test adapter of Rust Sequential; JS call/free and output allocation timed, copies/setup excluded; not browser, public-client or training throughput"}));
