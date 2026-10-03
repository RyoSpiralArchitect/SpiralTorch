import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const { FractionalGlKernel } = createRequire(import.meta.url)(resolve(process.argv[2]));
const f32 = values => new Float32Array(values);
const shape = new Uint32Array([2, 4, 3]);
const x = f32(Array.from({length: 24}, (_, i) => Math.sin(i * .3)));
const upstream = f32(x.map(v => v * .4));
let cases = 0;
for (const step of [.4, 1, 1.4]) {
    const kernel = new FractionalGlKernel(5, step, 24, 120);
    for (const alpha of [.45, 1, 1.5]) for (const axis of [0, 1, 2]) {
        for (const method of ['forward', 'forward_history']) {
            const batch = kernel[method](x, shape, axis, alpha);
            const joint = batch.vjp(upstream);
            assert.deepEqual(batch.vjp_input(upstream), joint.input);
            assert.equal(batch.vjp_alpha(upstream), joint.alpha);
            for (const name of ['vjp_input', 'vjp_alpha']) {
                for (const bad of [f32([]), f32([1]), f32(Array(24).fill(NaN)), f32(Array(24).fill(Infinity))]) {
                    assert.throws(() => batch[name](bad));
                }
            }
            joint.free(); batch.free(); cases++;
        }
    }
    kernel.free();
}
const maximum = 3.4028234663852886e38;
for (const method of ['forward', 'forward_history']) {
    const kernel = new FractionalGlKernel(2, 1, 2, 4);
    const dims = new Uint32Array([2]), up = f32([0, maximum]);
    const alphaOnly = kernel[method](f32([0, 0]), dims, 0, 2);
    assert.throws(() => alphaOnly.vjp(up));
    assert.throws(() => alphaOnly.vjp_input(up));
    assert.equal(alphaOnly.vjp_alpha(up), 0);
    const inputOnly = kernel[method](f32([maximum * .5, 0]), dims, 0, 1);
    assert.throws(() => inputOnly.vjp(up));
    assert.throws(() => inputOnly.vjp_alpha(up));
    assert.deepEqual(inputOnly.vjp_input(up), f32([-maximum, method === 'forward' ? maximum : 0]));
    alphaOnly.free(); inputOnly.free(); kernel.free();
}
for (const [k, dims] of [[1, [2, 3]], [8, [6, 1]]]) {
    const kernel = new FractionalGlKernel(k, .1, 6, 48);
    const values = f32(Array(6).fill(maximum));
    const batch = kernel.forward_history(values, new Uint32Array(dims), 1, 100);
    assert.deepEqual(batch.vjp_input(values), f32(Array(6).fill(0)));
    assert.equal(batch.vjp_alpha(values), 0);
    for (const method of ['vjp_input', 'vjp_alpha']) {
        assert.throws(() => batch[method](f32([])));
        assert.throws(() => batch[method](f32(Array(6).fill(NaN))));
    }
    batch.free(); kernel.free();
}
const kernel = new FractionalGlKernel(4, 1, 4, 16);
const integer = kernel.forward_history(f32([1, 0, 0, 0]), new Uint32Array([4]), 0, 1);
assert.equal(integer.output[2], 0);
assert.equal(integer.vjp_alpha(f32([0, 0, 1, 0])), .5);
integer.free(); kernel.free();
console.log(JSON.stringify({schema:'spiraltorch.fractional_selective_vjp_wasm.v1', parity_cases:cases,
    selective_overflow_guards:true, empty_history:true, integer_order_tail_derivative:true,
    scope:'Actual wasm32 selective/joint component equality, not a timing or pretrained quality claim'}));
