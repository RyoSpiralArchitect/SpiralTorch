import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { resolve } from "node:path";

const require = createRequire(import.meta.url);
const wasm = require(resolve(process.argv[2]));
const shapes = [[2, 5, 63], [2, 5, 64], [2, 5, 65], [1, 3, 129], [2, 3, 4, 5]];
let cases = 0;

function equalBits(actual, expected) {
  assert.deepEqual(new Uint32Array(actual.buffer, actual.byteOffset, actual.length),
    new Uint32Array(expected.buffer, expected.byteOffset, expected.length));
}

for (const shape of shapes) {
  const size = shape.reduce((a, b) => a * b, 1);
  const input = Float32Array.from({ length: size }, (_, i) => (i * 37 % 127) / 97 - .65);
  const upstream = Float32Array.from({ length: size }, (_, i) => (i * 13 % 31) / 31 - .5);
  for (let axis = 0; axis < shape.length; axis++) {
    const inner = shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const time = shape[axis];
    for (const alpha of [.5, 2]) for (const step of [.7, 1.4]) for (const history of [false, true]) {
      const kernel = new wasm.FractionalGlKernel(8, step, size, size * 8);
      const operation = history ? "forward_history" : "forward";
      const saved = kernel[operation](input, Uint32Array.from(shape), axis, alpha);
      try {
        const expectedOutput = new Float32Array(size);
        const expectedDerivative = new Float32Array(size);
        const expectedInputGradient = new Float32Array(size);
        for (let base = 0; base < size; base += time * inner) {
          for (let feature = 0; feature < inner; feature++) {
            const lane = Float32Array.from({ length: time }, (_, t) => input[base + t * inner + feature]);
            const direction = Float32Array.from({ length: time }, (_, t) => upstream[base + t * inner + feature]);
            const scalar = kernel[operation](lane, Uint32Array.of(time), 0, alpha);
            try {
              const output = scalar.output;
              const derivative = scalar.jvp(new Float32Array(time), 1);
              const gradient = scalar.vjp_input(direction);
              for (let t = 0; t < time; t++) {
                const index = base + t * inner + feature;
                expectedOutput[index] = output[t];
                expectedDerivative[index] = derivative[t];
                expectedInputGradient[index] = gradient[t];
              }
            } finally {
              scalar.free();
            }
          }
        }
        equalBits(saved.output, expectedOutput);
        equalBits(saved.jvp(new Float32Array(size), 1), expectedDerivative);
        equalBits(saved.vjp_input(upstream), expectedInputGradient);
        cases++;
      } finally {
        saved.free();
        kernel.free();
      }
    }
  }
}
assert.equal(cases, 128);
console.log(JSON.stringify({ status: "passed", cases, output_derivative_input_vjp_bits_equal: true,
  scope: "Actual wasm32 ND/tiled versus scalar-lane composition; not a browser or GPU timing result." }));
