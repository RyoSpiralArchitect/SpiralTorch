import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const { FractionalGlKernel } = createRequire(import.meta.url)(resolve(process.argv[2]));
const f32 = values => new Float32Array(values);
const shape = new Uint32Array([2, 12, 3]);
const x = f32(Array.from({length: 72}, (_, i) => Math.sin(i * .31) + .3*Math.cos(i * .17)));
const kernel = new FractionalGlKernel(5, .7, 128, 640);
const history = kernel.forward_history(x, shape, 1, .6);
const upstream = f32(x.map(v => v * .4));
const gradient = history.vjp(upstream);
const tangent = history.jvp(x, .2);
const dot = (a, b) => Array.from(a).reduce((sum, v, i) => sum + v*b[i], 0);
assert.ok(Math.abs(dot(tangent, upstream) - dot(x, gradient.input) - .2*gradient.alpha) < 2e-5);
assert.deepEqual(Array.from(history.output.slice(0, 3)), [0, 0, 0]);
const changed = x.slice(); changed.fill(99, 3);
const later = kernel.forward_history(changed, shape, 1, .6);
assert.deepEqual(later.output.slice(0, 6), history.output.slice(0, 6));
later.free(); gradient.free(); history.free();
for (const bad of [NaN, Infinity, 0, -1, '.6']) assert.throws(() => kernel.forward_history(x, shape, 1, bad));
for (const bad of [-1, .5, 3, '1']) assert.throws(() => kernel.forward_history(x, shape, bad, .6));

for (const [k, dims] of [[1, [2, 3]], [8, [6, 1]]]) {
    for (const [step, alpha] of [[.1, 38.5], [.1, 100], [1, 1e38]]) {
        const emptyKernel = new FractionalGlKernel(k, step, 6, 48);
        const values = f32(Array(6).fill(1e38));
        const batch = emptyKernel.forward_history(values, new Uint32Array(dims), 1, alpha);
        assert.deepEqual(Array.from(batch.output), Array(6).fill(0));
        const pullback = batch.vjp(values);
        assert.deepEqual(Array.from(pullback.input), Array(6).fill(0));
        assert.equal(pullback.alpha, 0);
        assert.deepEqual(Array.from(batch.jvp(values, 1e38)), Array(6).fill(0));
        for (const bad of [NaN, Infinity, 0, -1]) {
            assert.throws(() => emptyKernel.forward_history(values, new Uint32Array(dims), 1, bad));
        }
        assert.throws(() => batch.vjp(f32(Array(6).fill(NaN))));
        assert.throws(() => batch.jvp(values, NaN));
        pullback.free(); batch.free(); emptyKernel.free();
    }
}

const targetBatch = kernel.forward_history(x, shape, 1, .65);
const targetHistory = targetBatch.output;
const targetLocal = [.2, -.25, .15], targetGate = [-.35, .3, .4];
const target = x.map((v, i) => v + Math.tanh(targetLocal[i%3])*v
    + Math.tanh(targetGate[i%3])*targetHistory[i]);
targetBatch.free();
const local = [0, 0, 0], gate = [0, 0, 0];
let logAlpha = Math.log(1.1), nonzeroOrderSteps = 0;
const losses = [];
for (let step = 0; step <= 500; step++) {
    const alpha = Math.exp(logAlpha);
    const batch = kernel.forward_history(x, shape, 1, alpha);
    const h = batch.output;
    const output = x.map((v, i) => v + Math.tanh(local[i%3])*v + Math.tanh(gate[i%3])*h[i]);
    if (step === 0) assert.deepEqual(output, x);
    const residual = output.map((v, i) => v - target[i]);
    losses.push(dot(residual, residual) / x.length);
    if (step === 500) { batch.free(); break; }
    const localGradient = [0, 0, 0], gateGradient = [0, 0, 0];
    const historyUpstream = f32(residual.map((v, i) => {
        const f = i%3, g = 2*v/x.length;
        localGradient[f] += g*x[i]*(1-Math.tanh(local[f])**2);
        gateGradient[f] += g*h[i]*(1-Math.tanh(gate[f])**2);
        return g*Math.tanh(gate[f]);
    }));
    const pullback = batch.vjp(historyUpstream);
    const orderGradient = pullback.alpha * alpha;
    if (step === 0) {
        assert.equal(orderGradient, 0);
        assert.ok(localGradient.some(v => v !== 0) && gateGradient.some(v => v !== 0));
    }
    if (orderGradient !== 0) nonzeroOrderSteps++;
    for (let f = 0; f < 3; f++) {
        local[f] -= .2*localGradient[f];
        gate[f] -= .2*gateGradient[f];
    }
    logAlpha -= .2*orderGradient;
    assert.ok([...local, ...gate, logAlpha, Math.exp(logAlpha)].every(Number.isFinite));
    pullback.free(); batch.free();
}
kernel.free();
assert.ok(losses.every(Number.isFinite) && losses.at(-1) < losses[0] * .01);
assert.ok(local.some(v => v !== 0) && gate.some(v => v !== 0) && nonzeroOrderSteps > 0);
assert.notEqual(logAlpha, Math.log(1.1));
console.log(JSON.stringify({schema:'spiraltorch.fractional_history_wasm.v1',updates:500,
    initial_loss:losses[0],final_loss:losses.at(-1),learned_alpha:Math.exp(logAlpha),
    nonzero_order_steps:nonzeroOrderSteps,
    scope:'Synthetic joint gate/order learning, not pretrained quality, unique parameter recovery or speed'}));
