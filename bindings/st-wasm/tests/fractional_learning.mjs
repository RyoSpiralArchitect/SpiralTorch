import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const { FractionalGlKernel } = createRequire(import.meta.url)(resolve(process.argv[2]));
const f32 = values => new Float32Array(values);
const shape = new Uint32Array([2, 4, 3]);
const x = f32(Array.from({length: 24}, (_, i) => Math.sin(i * .3)));
const kernel = new FractionalGlKernel(5, .7, 128, 640);
const snapshot = kernel.forward(x, shape, 1, .6);
const upstream = f32(x.map(v => v * .4));
const pullback = snapshot.vjp(upstream);
const tangent = snapshot.jvp(x, .2);
const dot = (x, y) => Array.from(x).reduce((s, v, i) => s + v*y[i], 0);
assert.ok(Math.abs(dot(tangent, upstream) - dot(x, pullback.input) - .2*pullback.alpha) < 2e-5);
const changed = x.slice(); changed.fill(99, 6);
const later = kernel.forward(changed, shape, 1, .6);
assert.deepEqual(later.output.slice(0, 6), snapshot.output.slice(0, 6));
later.free(); pullback.free(); snapshot.free();
for (const bad of [NaN, Infinity, 0, -1, '.6']) assert.throws(() => kernel.forward(x, shape, 1, bad));
for (const bad of [-1, .5, 3, '1']) assert.throws(() => kernel.forward(x, shape, bad, .6));
const invalidUpstream = kernel.forward(x, new Uint32Array([24]), 0, .6);
assert.throws(() => invalidUpstream.vjp(f32([1])));
invalidUpstream.free();

const targetBatch = kernel.forward(x, shape, 1, .65);
const target = targetBatch.output;
targetBatch.free();
let alpha = 1.15;
const losses = [];
for (let step = 0; step <= 100; step++) {
    const batch = kernel.forward(x, shape, 1, alpha);
    const residual = batch.output.map((v, i) => v - target[i]);
    losses.push(dot(residual, residual) / x.length);
    if (step === 100) { batch.free(); break; }
    const alphaGradient = batch.vjp_alpha(f32(residual.map(v => 2*v/x.length)));
    alpha -= .05 * alphaGradient;
    assert.ok(Number.isFinite(alpha) && alpha > 0);
    batch.free();
}
kernel.free();
assert.ok(losses.every(Number.isFinite) && losses.at(-1) < losses[0] * .01);
console.log(JSON.stringify({schema:'spiraltorch.fractional_learning_wasm.v1', updates:100,
    initial_loss:losses[0], final_loss:losses.at(-1), learned_alpha:alpha,
    scope:'Synthetic alpha learning, no pretrained quality or speed claim'}));
