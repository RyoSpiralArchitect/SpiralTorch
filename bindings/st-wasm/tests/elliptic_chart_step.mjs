import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const { EllipticWarpKernel } = createRequire(import.meta.url)(resolve(process.argv[2]));
const f32 = x => new Float32Array(x);
const dot = (x, y) => Array.from(x).reduce((s, v, i) => s + v * y[i], 0);
const kernel = new EllipticWarpKernel(1.3, 3, 2, 2);
const batch = kernel.forward(f32([1, 16, 5, 1, -12, 3]));
const proposal = f32([.01, -.03, .02, .04, -.02, .01]);
const result = batch.chartStep(proposal, .1);
const jx = batch.jvp(f32([0, 1, 0, 0, 1, 0]));
const jy = batch.jvp(f32([0, 0, 1, 0, 0, 1]));
const expected = [dot(jx, jx) / 2, dot(jx, jy) / 2, dot(jy, jx) / 2, dot(jy, jy) / 2];
result.metric.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-12));
assert.ok(Math.abs(result.stepL2 / result.proposalL2 - 1) < 1e-7);
assert.ok(result.cosine > 0 && result.cosine < 1);
assert.ok(result.dampedCondition >= 1 && result.dampedCondition <= 21.000001);
for (const bad of [NaN, Infinity, 0, 1e-7, 1.1, '.1']) assert.throws(() => batch.chartStep(proposal, bad));
for (const bad of [[], [1], [NaN, 0]]) assert.throws(() => batch.chartStep(f32(bad), .1));
const zero = batch.chartStep(f32([0, 0]), .1);
assert.equal(zero.cosine, undefined);
assert.equal(zero.stepL2, 0);
zero.free(); result.free(); batch.free();

const targetSnapshot = kernel.forward(f32([1, .35, -.2]));
const target = targetSnapshot.features;
targetSnapshot.free();
let x = f32([1, .8, -.6]);
const losses = [];
for (let i = 0; i <= 100; i++) {
    const snapshot = kernel.forward(x);
    const residual = snapshot.features.map((v, j) => v - target[j]);
    losses.push(.5 * dot(residual, residual));
    if (i === 100) { snapshot.free(); break; }
    const gradient = snapshot.vjp(residual);
    const step = snapshot.chartStep(f32([-.1 * gradient[1], -.1 * gradient[2]]), .1);
    const values = step.values;
    x = f32([1, x[1] + values[0], x[2] + values[1]]);
    step.free(); snapshot.free();
}
kernel.free();
assert.ok(losses.every(Number.isFinite));
assert.ok(losses.at(-1) < losses[0] * .01);
console.log(JSON.stringify({schema:'spiraltorch.elliptic_chart_step_wasm.v1', status:'passed', scope:'Synthetic update path, not pretrained quality or speed', updates:100, initial_loss:losses[0], final_loss:losses.at(-1)}));
