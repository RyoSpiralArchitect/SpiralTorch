import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const { FractionalGlKernel, FractionalGlAngleChart } = createRequire(import.meta.url)(resolve(process.argv[2]));
const x = Float32Array.from({length:72}, (_,i)=>Math.sin(.31*i)+.3*Math.cos(.17*i));
const shape = new Uint32Array([2,12,3]);
const kernel = new FractionalGlKernel(8,.7,72,576);
const upstream = x.map(v=>v*.3-.2), dx = x.map(v=>Math.cos(v));
const near = (a,b,tol=3e-6)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.abs(a)),`${a} != ${b}`);
for(const alpha of [.08,.7,1,2,2.6]) {
    const full=kernel.forward_history_log_gain(x,shape,1,alpha,.3);
    const all=kernel.forward_history_log_gain_window(x,shape,1,alpha,.3,1,8);
    const short=kernel.forward_history_log_gain_window(x,shape,1,alpha,.3,1,3);
    const tail=kernel.forward_history_log_gain_window(x,shape,1,alpha,.3,3,8);
    assert.deepEqual(all.output,full.output);
    assert.deepEqual(all.vjp_parameters(upstream),full.vjp_parameters(upstream));
    assert.deepEqual(all.jvp(dx,.2,-.4),full.jvp(dx,.2,-.4));
    const outputs=[full,short,tail].map(b=>b.output);
    const gradients=[full,short,tail].map(b=>b.vjp(upstream));
    const inputs=gradients.map(g=>g.input);
    const tangents=[full,short,tail].map(b=>b.jvp(dx,.2,-.4));
    for(let i=0;i<x.length;i++) {
        near(outputs[0][i],outputs[1][i]+outputs[2][i]);
        near(inputs[0][i],inputs[1][i]+inputs[2][i]);
        near(tangents[0][i],tangents[1][i]+tangents[2][i]);
    }
    near(gradients[0].alpha,gradients[1].alpha+gradients[2].alpha);
    near(gradients[0].log_gain,gradients[1].log_gain+gradients[2].log_gain);
    if(alpha===2) {
        assert.ok(outputs[2].every(v=>v===0));
        assert.notEqual(gradients[2].alpha,0);
    }
    for(const handle of [full,all,short,tail,...gradients])handle.free();
}
for(const [start,end] of [[0,3],[4,3],[1,9],[-1,3],[.5,3],[true,3],['1',3],[1,Infinity],[1,NaN]]) {
    assert.throws(()=>kernel.validate_history_window(start,end));
    assert.throws(()=>kernel.forward_history_log_gain_window(x,shape,1,.7,0,start,end));
}
const zero=kernel.forward_history_log_gain_window(x,shape,1,3e38,0,3,3);
assert.ok(zero.output.every(v=>v===0));
assert.deepEqual(Array.from(zero.vjp_parameters(upstream)),[0,0]);
assert.throws(()=>zero.jvp(dx,NaN,0));
assert.throws(()=>kernel.forward_history_log_gain_window(x,shape,1,.7,100,3,3));
zero.free();

// Rust supplies masked normalized coefficients and both scalar pullbacks.
// JS only owns the fixed SGD loop and the two scalar parameter values.
const runs=[];
for(const [start,end] of [[1,3],[3,8]]) {
    const desired=kernel.forward_history_log_gain_window(x,shape,1,.65,Math.log(1.5),start,end);
    const target=desired.output; desired.free();
    let angle=Math.atan((.9-1)/2),logGain=Math.log(.7),angleUpdates=0,gainUpdates=0;
    const losses=[];
    for(let step=0;step<=100;step++) {
        const chart=new FractionalGlAngleChart(angle);
        const batch=kernel.forward_history_log_gain_window(x,shape,1,chart.alpha,logGain,start,end);
        const residual=batch.output.map((v,i)=>v-target[i]);
        losses.push(Array.from(residual).reduce((sum,v)=>sum+v*v,0)/x.length);
        if(step<100) {
            const g=batch.vjp_parameters(residual.map(v=>2*v/x.length));
            const da=chart.vjp(g[0]);
            if(da!==0)angleUpdates++;
            if(g[1]!==0)gainUpdates++;
            angle-=.05*da;
            logGain-=.05*g[1];
        }
        batch.free();chart.free();
    }
    assert.ok(losses.every(Number.isFinite) && losses.at(-1)<losses[0]);
    assert.ok(angleUpdates>0 && gainUpdates>0);
    runs.push({lag_window:[start,end],updates:100,initial_loss:losses[0],final_loss:losses.at(-1),
               nonzero_angle_steps:angleUpdates,nonzero_gain_steps:gainUpdates});
}
kernel.free();
console.log(JSON.stringify({schema:'spiraltorch.fractional_history_window_wasm.v1',runs,
    scope:'Synthetic compiled-WASM learning only; not pretrained quality, browser/GPU residency, speed or unique parameter recovery'}));
