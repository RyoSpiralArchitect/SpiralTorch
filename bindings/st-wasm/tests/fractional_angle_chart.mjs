import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const { FractionalGlAngleChart, FractionalGlKernel } = createRequire(import.meta.url)(resolve(process.argv[2]));
const f32 = xs => new Float32Array(xs);
const dot = (a,b) => Array.from(a).reduce((sum,x,i)=>sum+x*b[i],0);
const near = (a,b,eps=3e-6) => assert.ok(Math.abs(a-b)<eps, `${a} != ${b}`);
for(const angle of [-.4,-.24,0,.4636476,1.2]) {
    const chart=new FractionalGlAngleChart(angle);
    near(chart.alpha,1+2*Math.tan(Math.fround(angle)));
    near(chart.vjp(.3),chart.alpha_derivative*Math.fround(.3));
    assert.equal(chart.vjp(.3),chart.jvp(.3));
    for(const bad of [NaN,Infinity,-Infinity,1e50,true,'0']) {
        assert.throws(()=>chart.vjp(bad));assert.throws(()=>chart.jvp(bad));
    }
    chart.free();
}
for(const bad of [NaN,Infinity,-Infinity,-.5,1.6,6.3,true,'0'])
    assert.throws(()=>new FractionalGlAngleChart(bad));

const x=f32(Array.from({length:72},(_,i)=>Math.sin(.31*i)+.3*Math.cos(.17*i)));
const upstream=f32(x.map(v=>v*.3-.2)), shape=new Uint32Array([2,12,3]);
const short=new FractionalGlKernel(3,1,128,1024);
for(const angle of [-.4,-.24,0,.4636476,1.2]) {
    const chart=new FractionalGlAngleChart(angle);
    const batch=short.forward_history_log_gain(x,shape,1,chart.alpha,.3);
    const sin=Math.sin(chart.angle),cos=Math.cos(chart.angle),gain=batch.gain;
    const c1=Math.fround(-gain*cos),c2=Math.fround(gain*sin), y=batch.output;
    let expectedGradient=0;
    for(let i=0;i<x.length;i++){
        const time=Math.floor((i%36)/3),first=time>=1?x[i-3]:0,second=time>=2?x[i-6]:0;
        near(y[i],Math.fround(c1*first+c2*second));
        expectedGradient+=upstream[i]*gain*(sin*first+cos*second);
    }
    near(chart.vjp(batch.vjp_alpha(upstream)),expectedGradient,8e-6);
    batch.free();chart.free();
}
short.free();

// JS owns SGD only; Rust owns the chart, GL coefficients and both scalar VJPs.
const kernel=new FractionalGlKernel(8,1,128,1024);
const desired=kernel.forward_history_log_gain(x,shape,1,.65,Math.log(1.5));
const target=desired.output;desired.free();
let angle=Math.fround(.4636476090008061),logGain=Math.fround(Math.log(.7));
let nonzeroAngle=0,nonzeroGain=0,finalAlpha,finalGain;
const losses=[];
for(let step=0;step<=1000;step++) {
    const chart=new FractionalGlAngleChart(angle);
    const batch=kernel.forward_history_log_gain(x,shape,1,chart.alpha,logGain);
    const residual=batch.output.map((v,i)=>v-target[i]);
    losses.push(dot(residual,residual)/x.length);
    finalAlpha=chart.alpha;finalGain=batch.gain;
    if(step<1000){
        const gradients=batch.vjp_parameters(residual.map(v=>2*v/x.length));
        const angleGradient=chart.vjp(gradients[0]);
        if(angleGradient!==0)nonzeroAngle++;
        if(gradients[1]!==0)nonzeroGain++;
        angle=Math.fround(angle-.02*angleGradient);
        logGain=Math.fround(logGain-.02*gradients[1]);
    }
    chart.free();batch.free();
}
kernel.free();
assert.ok(losses.every(Number.isFinite)&&losses.at(-1)<losses[0]*.01);
assert.ok(nonzeroAngle>0&&nonzeroGain>0);
console.log(JSON.stringify({schema:'spiraltorch.fractional_angle_wasm.v1',updates:1000,
    initial_loss:losses[0],final_loss:losses.at(-1),final_angle:angle,final_alpha:finalAlpha,
    final_gain:finalGain,nonzero_angle_steps:nonzeroAngle,nonzero_gain_steps:nonzeroGain,
    scope:'Synthetic compiled-WASM learning; not a pretrained-quality, unique-recovery, browser, GPU or speed claim'}));
