import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const { FractionalGlKernel } = createRequire(import.meta.url)(resolve(process.argv[2]));
const f32 = xs => new Float32Array(xs);
const dims = xs => new Uint32Array(xs);
const dot = (xs, ys) => Array.from(xs).reduce((sum, x, i) => sum+x*ys[i], 0);
const near = (a, b, tolerance=3e-6) => assert.ok(Math.abs(a-b)<=tolerance, `${a} != ${b}`);
const x = f32(Array.from({length:72}, (_,i) => Math.sin(.31*i)+.3*Math.cos(.17*i)));
const shape = dims([2,12,3]), kernel = new FractionalGlKernel(5,.7,128,640);
const h = kernel.forward_history_log_gain(x,shape,1,2,.3);
assert.equal(FractionalGlKernel.gain_from_log_gain(.3),h.gain);
const fixed = kernel.forward_history_l2(x,shape,1,2,h.gain);
assert.deepEqual(h.output, fixed.output);
const upstream=f32(x.map(v=>v*.3-.2)), dx=f32(x.map(v=>Math.cos(v)));
const g=h.vjp(upstream), dy=h.jvp(dx,.2,-.4);
assert.deepEqual(h.vjp_input(upstream),g.input);
assert.equal(h.vjp_alpha(upstream),g.alpha);
assert.equal(h.vjp_log_gain(upstream),g.log_gain);
assert.deepEqual(Array.from(h.vjp_parameters(upstream)),[g.alpha,g.log_gain]);
near(dot(upstream,dy),dot(dx,g.input)+.2*g.alpha-.4*g.log_gain,3e-5);
const eps=.001;
const plus=kernel.forward_history_log_gain(f32(x.map((v,i)=>v+eps*dx[i])),shape,1,2+eps*.2,.3-eps*.4);
const minus=kernel.forward_history_log_gain(f32(x.map((v,i)=>v-eps*dx[i])),shape,1,2-eps*.2,.3+eps*.4);
const yp=plus.output, ym=minus.output;
for(let i=0;i<x.length;i++)near(dy[i],(yp[i]-ym[i])/(2*eps),4e-4);
const owned=h.output; owned.fill(99);
assert.deepEqual(h.output,fixed.output);
for(const handle of [h,fixed,g,plus,minus])handle.free();
for(const bad of [NaN,Infinity,-Infinity,100,-200,'0',true,1e50]) {
    assert.throws(()=>FractionalGlKernel.gain_from_log_gain(bad));
    assert.throws(()=>kernel.forward_history_log_gain(x,shape,1,2,bad));
}
assert.throws(()=>kernel.forward_history_log_gain(x,shape,.5,2,0));
const empty=new FractionalGlKernel(1,.7,72,72);
const zero=empty.forward_history_log_gain(x,shape,1,1,0);
assert.equal(zero.vjp_log_gain(upstream),0);
assert.deepEqual(Array.from(zero.vjp_parameters(upstream)),[0,0]);
assert.throws(()=>zero.jvp(dx,0,NaN));
assert.throws(()=>zero.vjp_parameters(f32([1])));
assert.throws(()=>empty.forward_history_log_gain(x,shape,1,1,100));
zero.free();empty.free();

// Actual compiled-WASM scalar VJPs; JS owns only the optimizer and log-alpha chart.
const desired=kernel.forward_history_log_gain(x,shape,1,.65,Math.log(1.5));
const target=desired.output;
desired.free();
let logAlpha=Math.log(1.1), logGain=Math.log(.7), nonzeroAlpha=0, nonzeroGain=0;
const losses=[];
let finalGain;
for(let step=0;step<=600;step++) {
    const batch=kernel.forward_history_log_gain(x,shape,1,Math.exp(logAlpha),logGain);
    const residual=batch.output.map((v,i)=>v-target[i]);
    losses.push(dot(residual,residual)/x.length);
    finalGain=batch.gain;
    if(step===600){batch.free();break;}
    const gradient=batch.vjp_parameters(residual.map(v=>2*v/x.length));
    const da=gradient[0]*Math.exp(logAlpha), dg=gradient[1];
    if(da!==0)nonzeroAlpha++;
    if(dg!==0)nonzeroGain++;
    logAlpha-=.1*da;
    logGain-=.1*dg;
    assert.ok([logAlpha,logGain,Math.exp(logAlpha)].every(Number.isFinite));
    batch.free();
}
kernel.free();
assert.ok(losses.every(Number.isFinite) && losses.at(-1)<losses[0]*.01);
assert.ok(nonzeroAlpha>0 && nonzeroGain>0);
assert.notEqual(logAlpha,Math.log(1.1));
assert.notEqual(logGain,Math.log(.7));
console.log(JSON.stringify({schema:'spiraltorch.fractional_log_gain_wasm.v1',updates:600,
    initial_loss:losses[0],final_loss:losses.at(-1),learned_alpha:Math.exp(logAlpha),
    learned_gain:finalGain,nonzero_order_steps:nonzeroAlpha,nonzero_gain_steps:nonzeroGain,
    scope:'Synthetic compiled-WASM learning, not unique parameter recovery, pretrained quality, browser, GPU or speed evidence'}));
