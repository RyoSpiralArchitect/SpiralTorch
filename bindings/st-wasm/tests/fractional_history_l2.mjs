import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const { FractionalGlKernel } = createRequire(import.meta.url)(resolve(process.argv[2]));
const f32 = values => new Float32Array(values);
const dims = values => new Uint32Array(values);
const dot = (a, b) => Array.from(a).reduce((sum, v, i) => sum + v*b[i], 0);
const near = (a, b, tolerance=3e-6) => assert.ok(Math.abs(a-b) <= tolerance, `${a} != ${b}`);
const x = f32(Array.from({length:72}, (_,i) => Math.sin(i*.31) + .3*Math.cos(i*.17)));
const shape = dims([2,12,3]);
const kernel = new FractionalGlKernel(5, .7, 128, 640);
const batch = kernel.forward_history_l2(x, shape, 1, 2, 1.5);
const upstream = f32(x.map(v => .4*v)), dx = f32(x.map(v => Math.cos(v)));
const gradient = batch.vjp(upstream), tangent = batch.jvp(dx, .2);
near(dot(tangent, upstream), dot(dx, gradient.input) + .2*gradient.alpha, 2e-5);
assert.deepEqual(batch.vjp_input(upstream), gradient.input);
assert.equal(batch.vjp_alpha(upstream), gradient.alpha);
const epsilon = .001;
const plus = kernel.forward_history_l2(f32(x.map((v,i) => v+epsilon*dx[i])), shape, 1, 2+epsilon*.2, 1.5);
const minus = kernel.forward_history_l2(f32(x.map((v,i) => v-epsilon*dx[i])), shape, 1, 2-epsilon*.2, 1.5);
const yp=plus.output, ym=minus.output;
for (let i=0;i<x.length;i++) near(tangent[i], (yp[i]-ym[i])/(2*epsilon), 3e-4);
const changed = x.slice(); changed.fill(99, 3);
const later = kernel.forward_history_l2(changed, shape, 1, 2, 1.5);
assert.deepEqual(later.output.slice(0,6), batch.output.slice(0,6));
for (const handle of [plus,minus,later,gradient,batch]) handle.free();

for (const bad of [NaN, Infinity, 0, -1, '1', true, 1e-50, 1e50]) {
    assert.throws(() => kernel.forward_history_l2(x, shape, 1, 2, bad));
    assert.throws(() => kernel.forward_history_l2(x, shape, 1, bad, 1));
}
assert.throws(() => kernel.forward_history_l2(x, shape, .5, 2, 1));
assert.throws(() => kernel.forward_history_l2(x, dims([1]), 0, 2, 1));

let stepReference;
for (const step of [1e-40,.7,1,1e38]) {
    const k = new FractionalGlKernel(5,step,5,25);
    const h = k.forward_history_l2(f32([1,0,0,0,0]),dims([5]),0,4,1.5);
    const p = k.forward_history_l2(f32([1,0]),dims([2]),0,4,1.5);
    near(dot(h.output,h.output), 1.5**2);
    assert.deepEqual(p.output,h.output.slice(0,2));
    const current = [h.output,h.jvp(f32([0,0,0,0,0]),1)];
    if (stepReference) assert.deepEqual(current,stepReference);
    stepReference=current;
    h.free();p.free();k.free();
}
for (const alpha of [2**-149,1e-30,.1,1,4,1e38]) {
    const k = new FractionalGlKernel(2,1e-40,3,6);
    const h = k.forward_history_l2(f32([1,2,3]),dims([3]),0,alpha,1);
    assert.deepEqual(Array.from(h.output),[0,-1,-2]);
    assert.equal(h.vjp_alpha(f32([1,1,1])),0);
    h.free();k.free();
}
for (const [length,shape] of [[1,[2,3]],[8,[6,1]]]) {
    const k = new FractionalGlKernel(length,.1,6,48);
    const h = k.forward_history_l2(f32(Array(6).fill(1e38)),dims(shape),1,1e38,1);
    assert.deepEqual(Array.from(h.output),Array(6).fill(0));
    assert.equal(h.vjp_alpha(f32(Array(6).fill(1))),0);
    assert.throws(() => h.vjp_input(f32([1])));
    assert.throws(() => h.vjp_alpha(f32(Array(6).fill(NaN))));
    assert.throws(() => k.forward_history_l2(f32(Array(6).fill(1)),dims(shape),1,1,0));
    h.free();k.free();
}

// Real WASM order VJP in a synthetic optimizer loop; JS only owns gates/SGD.
const targetBatch = kernel.forward_history_l2(x,shape,1,.65,1.5);
const targetHistory = targetBatch.output;
const targetLocal = [.2,-.25,.15], targetGate = [-.35,.3,.4];
const target = x.map((v,i) => v+Math.tanh(targetLocal[i%3])*v
    +Math.tanh(targetGate[i%3])*targetHistory[i]);
targetBatch.free();
const local=[0,0,0], gate=[0,0,0];
let logAlpha=Math.log(1.1), nonzeroOrderSteps=0;
const losses=[];
for (let step=0;step<=500;step++) {
    const alpha=Math.exp(logAlpha), h=kernel.forward_history_l2(x,shape,1,alpha,1.5);
    const history=h.output;
    const y=x.map((v,i) => v+Math.tanh(local[i%3])*v+Math.tanh(gate[i%3])*history[i]);
    if (step===0) assert.deepEqual(y,x);
    const residual=y.map((v,i)=>v-target[i]);
    losses.push(dot(residual,residual)/x.length);
    if (step===500) {h.free();break;}
    const localGradient=[0,0,0], gateGradient=[0,0,0];
    const direction=f32(residual.map((v,i)=>{
        const f=i%3,g=2*v/x.length;
        localGradient[f]+=g*x[i]*(1-Math.tanh(local[f])**2);
        gateGradient[f]+=g*history[i]*(1-Math.tanh(gate[f])**2);
        return g*Math.tanh(gate[f]);
    }));
    const orderGradient=h.vjp_alpha(direction)*alpha;
    if (step===0) assert.equal(orderGradient,0);
    if (orderGradient!==0) nonzeroOrderSteps++;
    for (let f=0;f<3;f++) {local[f]-=.2*localGradient[f];gate[f]-=.2*gateGradient[f];}
    logAlpha-=.2*orderGradient;
    assert.ok([...local,...gate,logAlpha,Math.exp(logAlpha)].every(Number.isFinite));
    h.free();
}
kernel.free();
assert.ok(losses.every(Number.isFinite) && losses.at(-1)<losses[0]*.01);
assert.ok(local.some(v=>v!==0) && gate.some(v=>v!==0) && nonzeroOrderSteps>0);
assert.notEqual(logAlpha,Math.log(1.1));
console.log(JSON.stringify({schema:'spiraltorch.fractional_history_l2_wasm.v1',updates:500,
    initial_loss:losses[0],final_loss:losses.at(-1),learned_alpha:Math.exp(logAlpha),
    nonzero_order_steps:nonzeroOrderSteps,
    scope:'Synthetic learning through compiled WASM, not pretrained quality, unique order recovery, browser or GPU evidence'}));
