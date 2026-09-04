const assert = require("node:assert/strict");
const { test } = require("node:test");
const st = require(process.argv[2]);

function signal(n) {
  return Float32Array.from(Array.from({ length: n }, (_, j) =>
    [((j * 13) % 31 - 15) / 16, ((j * 7) % 19 - 9) / 8]).flat());
}

function dft(input, inverse) {
  const n = input.length / 2;
  const sign = inverse ? 1 : -1;
  const scale = inverse ? 1 / n : 1;
  const result = new Array(input.length).fill(0);
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      const phase = sign * 2 * Math.PI * j * k / n;
      result[2*k] += scale * (input[2*j] * Math.cos(phase) - input[2*j+1] * Math.sin(phase));
      result[2*k+1] += scale * (input[2*j] * Math.sin(phase) + input[2*j+1] * Math.cos(phase));
    }
  }
  return result;
}

function assertClose(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance,
      `component ${i}: ${actual[i]} != ${expected[i]}`);
  }
}

test("forward and inverse WASM FFT match an independent DFT", () => {
  for (const n of [1, 2, 4, 8, 16, 32, 64, 128]) {
    for (const inverse of [false, true]) {
      const input = signal(n);
      const before = input.slice();
      const result = inverse ? st.fft_inverse(input) : st.fft_forward(input);
      assertClose(result, dft(input, inverse), 3e-6 * n);
      assert.deepEqual(input, before);
    }
  }
});

test("four-point ramp keeps standard bin order and sign", () => {
  assert.deepEqual(Array.from(st.fft_forward(new Float32Array([1,0,2,0,3,0,4,0]))),
    [10,0,-2,2,-2,0,-2,-2]);
});

test("in-place FFTs match both direct transforms and arbitrary roundtrips", () => {
  for (const n of [1, 2, 4, 8, 16, 32, 128, 1024]) {
    const original = signal(n);
    for (const inverse of [false, true]) {
      const values = original.slice();
      const expected = inverse ? st.fft_inverse(values) : st.fft_forward(values);
      const buffer = values.buffer;
      if (inverse) st.fft_inverse_in_place(values); else st.fft_forward_in_place(values);
      assert.equal(values.buffer, buffer);
      assert.deepEqual(values, expected);
    }
    const values = original.slice();
    st.fft_forward_in_place(values);
    st.fft_inverse_in_place(values);
    assertClose(values, original, 5e-6);
  }
});

test("frequency-domain filtering agrees with time-domain circular convolution", () => {
  const original = signal(32);
  const kernel = new Float32Array(original.length);
  kernel[0] = 0.5;
  kernel[3] = 0.25;
  kernel[4] = -0.125;
  const spectrum = st.fft_forward(original);
  const response = st.fft_forward(kernel);
  const product = new Float32Array(spectrum.length);
  for (let i = 0; i < product.length; i += 2) {
    product[i] = spectrum[i] * response[i] - spectrum[i+1] * response[i+1];
    product[i+1] = spectrum[i] * response[i+1] + spectrum[i+1] * response[i];
  }
  const expected = new Float32Array(original.length);
  const n = original.length / 2;
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < 3; j++) {
      const i = 2 * ((k + n - j) % n);
      expected[2*k] += original[i] * kernel[2*j] - original[i+1] * kernel[2*j+1];
      expected[2*k+1] += original[i] * kernel[2*j+1] + original[i+1] * kernel[2*j];
    }
  }
  assertClose(st.fft_inverse(product), expected, 5e-6);
});

test("invalid interleaved buffers fail before in-place mutation", () => {
  for (const length of [0, 1, 3, 6, 10, 12, 30]) {
    for (const operation of [st.fft_forward_in_place, st.fft_inverse_in_place,
      st.fft_forward, st.fft_inverse]) {
      const input = Float32Array.from({ length }, (_, i) => i + 0.5);
      const before = input.slice();
      assert.throws(() => operation(input), /empty|power of two|interleaved/);
      assert.deepEqual(input, before);
    }
  }
});
