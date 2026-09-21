"use strict";
const assert = require("node:assert/strict");
const {performance} = require("node:perf_hooks");
globalThis.crypto ??= require("node:crypto").webcrypto;
const api = require(process.argv[2]);
const bits = a => Array.from(new Uint32Array(a.buffer,a.byteOffset,a.length));
function expected(rows,cols,bands,residual) {
  const result=[];
  for(let row=0;row<rows;row++) {
    const values=Array.from({length:cols},(_,c)=>((row*cols+c)%257)/64-2);
    if(residual) result.push(...values);
    for(let band=0;band<bands;band++) for(const value of values) {
      const phase=value*2**band; result.push(Math.sin(phase),Math.cos(phase));
    }
  }
  return result;
}
const valid=(a,b)=>a.length===b.length&&a.every((v,i)=>Number.isFinite(v)&&Math.abs(v-b[i])<=2e-6);
const contracts=JSON.parse(api.contract_report());
const layouts=[];
for(const [rows,cols] of [[2,3],[3,6]]) for(const bands of [0,4,10]) for(const residual of [false,true]) {
  const reference=expected(rows,cols,bands,residual);
  for(const layout of [0,1,2]) {
    const fixture=new api.EncodingCase(rows,cols,bands,residual,layout);
    const output=fixture.run();const data=output.values();
    layouts.push({rows,cols,bands,residual,layout,valid:valid(data,reference),output_bits:bits(data)});
    output.free();fixture.free();
  }
}
const guards=[];
for(const [values,bands,label] of [[[3e38,NaN],2,"nerf_input"],[[3e38],2,"nerf_phase"],[[0],129,"nerf_frequency_count"]]) {
  let fixture,output,error=null;
  try { fixture=api.EncodingCase.with_values(1,values.length,bands,true,0,new Float32Array(values));output=fixture.run(); }
  catch(e) { error=String(e); }
  finally { output?.free();fixture?.free(); }
  guards.push({label,correct:error!==null&&error.includes(label)});
}
const empty=new api.EncodingCase(0,3,4,true,0),emptyOutput=empty.run();
assert.equal(emptyOutput.values().length,0);emptyOutput.free();empty.free();
const zero=api.EncodingCase.with_values(1,2,1,true,0,new Float32Array([-0,0]));
const zeroOutput=zero.run();assert.deepEqual(bits(zeroOutput.values()),[0x80000000,0,0x80000000,0x3f800000,0,0x3f800000]);zeroOutput.free();zero.free();
const cases=[];
for(const rows of [1,32,1024]) for(const cols of [3,6]) for(const bands of [0,4,10]) for(const residual of [false,true]) {
  const fixture=new api.EncodingCase(rows,cols,bands,residual,0);
  const reference=expected(rows,cols,bands,residual);
  for(let i=0;i<3;i++) fixture.run().free();
  const elapsed_ns=[],repetitions=rows===1?512:8;
  for(let i=0;i<15;i++) {
    const start=performance.now();for(let j=0;j<repetitions;j++) fixture.run().free();
    elapsed_ns.push((performance.now()-start)*1e6/repetitions);
  }
  const output=fixture.run(),data=output.values();assert.ok(valid(data,reference));
  cases.push({rows,cols,bands,residual,repetitions,valid:true,elapsed_ns,output_bits:bits(data)});
  output.free();fixture.free();
}
console.log(JSON.stringify({schema:"spiraltorch.positional_encoding_wasm.v1",contracts,layouts,guards,empty:true,signed_zero:true,
  cases,warmups:3,intervals:15,boundary:"Actual st-vision CPU WASM through a test adapter; JS call/free timed, setup/export excluded; no deployed browser or GPU claim"}));
