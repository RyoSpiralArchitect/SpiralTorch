// Compare two separately built modules; no browser/GPU or model speed claim.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';

const [beforePath, afterPath, outputPath] = process.argv.slice(2);
assert.ok(beforePath && afterPath && outputPath, 'before.js after.js NEW-output.json required');
assert.ok(!existsSync(outputPath), 'output already exists');
const require = createRequire(import.meta.url);
const paths = [beforePath, afterPath].map(p => resolve(p));
const apis = paths.map(p => require(p));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const hash = values => digest(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
const wasmHashes = paths.map(p => digest(readFileSync(p.replace(/\.js$/, '_bg.wasm'))));
const makeInput = n => Float32Array.from({length:n}, (_, i) => Math.fround((i*73%997)/997)-.5);
const makeGradient = n => Float32Array.from({length:n}, (_, i) => Math.fround((i*17%991)/991)-.5);

function capture(kernel, input, shape, axis, alpha, window, upstream, all) {
    const batch = kernel.forward_history_log_gain_window(input, shape, axis, alpha, Math.log(5), ...window);
    try {
        const result = {output: batch.output, parameters: batch.vjp_parameters(upstream)};
        if (all) {
            const gradient = batch.vjp(upstream);
            try {
                result.input = gradient.input;
                // Baseline joint/selective scalar reductions can differ in zero sign.
                // Preserve and compare EACH route's bits across builds, not across APIs.
                result.joint_parameters = new Float32Array([gradient.alpha, gradient.log_gain]);
                result.selective_input = batch.vjp_input(upstream);
                result.jvp = batch.jvp(upstream, .2, -.3);
            } finally { gradient.free(); }
        }
        for (const values of Object.values(result)) assert.ok(values.every(Number.isFinite));
        return result;
    } finally { batch.free(); }
}
const fingerprints = result => Object.fromEntries(Object.entries(result).map(([key, value]) => [key, hash(value)]));
const correctness = [];
for (const dimensions of [[9], [2,7,65], [2,128,768]]) {
    const n = dimensions.reduce((a,b)=>a*b, 1), shape = new Uint32Array(dimensions);
    const input = makeInput(n), upstream = makeGradient(n);
    const axes = dimensions[1] === 128 ? [1] : dimensions.map((_,i)=>i);
    const kernels = apis.map(api=>new api.FractionalGlKernel(32,.7,n,n*32));
    try {
        for(const axis of axes) for(const alpha of [.09,.55,2]) for(const window of [[1,3],[3,32],[1,32],[32,32]]) {
            const observations = kernels.map(k=>fingerprints(capture(k,input,shape,axis,alpha,window,upstream,true)));
            assert.deepEqual(observations[1], observations[0]);
            correctness.push({shape:dimensions,axis,alpha,window,sha256:observations[0]});
        }
    } finally { kernels.forEach(k=>k.free()); }
}

const dimensions=[2,128,768], shape=new Uint32Array(dimensions), n=2*128*768;
const input=makeInput(n), upstream=makeGradient(n);
const kernels=apis.map(api=>new api.FractionalGlKernel(32,.7,n,n*32));
const cases=[], rounds=12, warmup=2;
try {
    for(const [alpha,window] of [[.55,[1,3]],[.09,[1,32]],[2,[1,3]],[2,[3,32]]]) {
        const run = route=>capture(kernels[route],input,shape,1,alpha,window,upstream,false);
        const expected=fingerprints(run(0));
        for(let i=0;i<warmup;i++) for(const route of [0,1]) assert.deepEqual(fingerprints(run(route)),expected);
        const samples=[[],[]], order=[];
        for(let i=0;i<rounds;i++) {
            const routes=i%2 ? [1,0] : [0,1]; order.push(routes);
            for(const route of routes) {
                const start=performance.now(), result=run(route), elapsed=performance.now()-start;
                assert.deepEqual(fingerprints(result),expected);
                samples[route].push(elapsed);
            }
        }
        const median = values=>{const x=values.toSorted((a,b)=>a-b); return (x[5]+x[6])/2;};
        cases.push({shape:dimensions,axis:1,alpha,log_gain:Math.log(5),window,kernel_len:32,
            warmup_per_route:warmup,round_order:order,measurements_ms:samples,median_ms:samples.map(median),sha256:expected});
    }
} finally { kernels.forEach(k=>k.free()); }
const report={schema:'spiraltorch.fractional_active_support_wasm.v1',status:'passed',
    node:process.version,machine:process.arch,wasm_sha256:wasmHashes,
    script_sha256:digest(readFileSync(new URL(import.meta.url))),
    correctness,cases,
    scope:'Compiled release WASM in Node, before/after exact same windows. Bitwise forward, parameter/input VJP and JVP parity. Timing includes forward, parameter VJP, output copy and snapshot free; hashing excluded. Not native-Python/Torch timing, browser/WebGPU, model throughput or quality evidence.'};
writeFileSync(outputPath,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({status:report.status,parity_cases:correctness.length,cases:cases.map(({alpha,window,median_ms})=>({alpha,window,median_ms}))}));
