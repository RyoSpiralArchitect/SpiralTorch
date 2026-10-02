// Run with a wasm-bindgen --target nodejs module; no JavaScript map fallback.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

const { EllipticWarpKernel } = createRequire(import.meta.url)(resolve(process.argv[2]));
const f32 = values => new Float32Array(values);
const dot = (a, b) => Array.from(a).reduce((sum, v, i) => sum + v * b[i], 0);
const x = f32([1, .3, -.4, 1, -.2, .5]);
const dx = f32([0, .2, .3, 0, -.4, .1]);
const upstream = f32(Array.from({ length: 18 }, (_, i) => Math.sin(i * .3)));
const warp = new EllipticWarpKernel(1.3, 3, 2, 2);
const local = warp.forward(x);
let maxDifference = 0, maxAdjointError = 0;
for (const gate of [-30, -.7, 0, .4, 30]) {
    const snapshot = warp.forwardAnchored(x, gate);
    const actual = snapshot.jvp(dx, -.35);
    const reverse = snapshot.vjp(upstream);
    const error = Math.abs(dot(actual, upstream) - dot(dx, reverse.orientations) + .35 * reverse.rawMix);
    maxAdjointError = Math.max(maxAdjointError, error);
    assert.ok(error < 1e-6);
    const plus = warp.forwardAnchored(x.map((v, i) => v + .001 * dx[i]), gate - .00035);
    const minus = warp.forwardAnchored(x.map((v, i) => v - .001 * dx[i]), gate + .00035);
    const high = plus.features, low = minus.features;
    for (let i = 0; i < actual.length; i++) {
        const difference = Math.abs(actual[i] - (high[i] - low[i]) / .002);
        maxDifference = Math.max(maxDifference, difference);
        assert.ok(difference < .002);
    }
    assert.throws(() => snapshot.jvp(f32([]), 0));
    assert.throws(() => snapshot.jvp(dx, NaN));
    assert.throws(() => snapshot.jvp(dx, '0'));
    if (gate === 0) assert.deepEqual(snapshot.jvp(dx, 0), local.jvp(dx));
    snapshot.free(); reverse.free(); plus.free(); minus.free();
}
const jv = local.jvp(dx);
const gram = local.vjp(jv);
assert.ok(Math.abs(dot(dx, gram) - dot(jv, jv)) < 1e-6);
assert.ok(dot(dx, gram) > 0);
assert.throws(() => local.jvp(f32([NaN, 0, 0, 0, 0, 0])));
assert.throws(() => local.jvp(f32([])));
const empty = warp.forward(f32([]));
assert.deepEqual(empty.jvp(f32([])), f32([]));
empty.free(); local.free();

// Use JVP curvature in a real, bounded update, with no JS geometric derivative.
const targetSnapshot = warp.forward(f32([1, .35, -.2, 1, -.4, .6]));
const target = targetSnapshot.features;
targetSnapshot.free();
const residual = snapshot => snapshot.features.map((v, i) => v - target[i]);
let current = f32([1, 3, -2, 1, -2, 2]);
const losses = [];
for (let iteration = 0; iteration <= 100; iteration++) {
    const snapshot = warp.forward(current);
    const r = residual(snapshot), loss = .5 * dot(r, r);
    losses.push(loss);
    if (loss < 1e-10 || iteration === 100) { snapshot.free(); break; }
    const gradient = snapshot.vjp(r);
    gradient[0] = gradient[3] = 0;
    const slope = snapshot.jvp(gradient), energy = dot(gradient, gradient);
    const rate = energy / (dot(slope, slope) + .001 * energy);
    assert.ok(Number.isFinite(rate) && rate > 0);
    let accepted = false;
    for (let backtrack = 0; backtrack < 16; backtrack++) {
        const candidate = current.map((v, i) => v - rate * 2 ** -backtrack * gradient[i]);
        const proposal = warp.forward(candidate), proposalResidual = residual(proposal);
        const proposalLoss = .5 * dot(proposalResidual, proposalResidual);
        proposal.free();
        if (proposalLoss < loss) { current = candidate; accepted = true; break; }
    }
    snapshot.free();
    assert.ok(accepted, 'no decreasing update; do not claim success');
}
warp.free();
assert.ok(losses.at(-1) < losses[0] * 1e-4);
console.log(JSON.stringify({ schema: 'spiraltorch.elliptic_directional_wasm.v1', status: 'passed', maxDifference, maxAdjointError,
    learning: { scope: 'Synthetic feature fitting, not a quality or speed benchmark', updates: losses.length - 1, initial_loss: losses[0], final_loss: losses.at(-1), losses }
}));
