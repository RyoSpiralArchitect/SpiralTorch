const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const os = require("node:os");
const {spawnSync} = require("node:child_process");
assert.equal(process.argv.length,3,"Pass freshly generated WebGPU WASM module");
function syntax(files) {
  const result=spawnSync("tsc",["--noEmit","--skipLibCheck","--target","ES2022",
    "--lib","ESNext,DOM",...files],{encoding:"utf8"});
  assert.ifError(result.error);
  return result;
}
const generated=path.resolve(process.argv[2]).replace(/\.js$/,".d.ts");
const shipped=path.join(__dirname,"../types/spiraltorch-wasm.d.ts");
// Member regexes cannot detect malformed surrounding declarations.
const parsed=syntax([generated,shipped]);
assert.equal(parsed.status,0,parsed.stdout+parsed.stderr);
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),"spiraltorch-types-"));
try {
  const broken=path.join(temporary,"broken.d.ts");
  fs.writeFileSync(broken,'declare module "broken" { export class Open { export class Next {} }');
  const rejected=syntax([broken]);
  assert.notEqual(rejected.status,0,"Unclosed class must fail syntax checking");
  assert.match(rejected.stdout,/error TS1068:/);
} finally {
  fs.rmSync(temporary,{recursive:true,force:true});
}
function check(types,label) {
  const get = name=>{
    const d=types.match(new RegExp("^( *)export class "+name+" \\{[\\s\\S]*?^\\1\\}","m"))?.[0];
    assert.ok(d,label+" must export "+name);
    return d;
  };
  const plan=get("ResidualAttentionPlan");
  assert.match(plan,/private constructor\(\)/);
  assert.match(plan,/static fromPlans\(pre: InferencePlan, attention: AttentionInferencePlan, feed_forward: InferencePlan\): ResidualAttentionPlan;/);
  assert.match(plan,/compileTrainingWebGpu\(tile_mnk\?: (?:number\[\]|Array<any>) \| null, kernel\?: string \| null, accumulation\?: string \| null\): Promise<ResidentResidualAttentionTraining>;/);
  const owner=get("ResidentResidualAttentionTraining");
  assert.match(owner,/private constructor\(\)/);
  for (const d of [plan,owner]) {
    assert.match(d,/readonly inputShape: Uint32Array;/);
    assert.match(d,/readonly outputShape: Uint32Array;/);
  }
  assert.match(owner,/readonly attemptedUpdates: bigint;/);
  assert.match(owner,/parameterTensors\(\): WgpuTensor\[\];/);
  assert.match(owner,/tensorDevice\(\): WgpuTensorDevice;/);
  assert.match(owner,/forward\(input: WgpuTensor, biases: WgpuAttentionBiases\): ResidualAttentionForward;/);
  assert.match(owner,/backward\(forward: ResidualAttentionForward, cotangent: WgpuTensor\): ResidualAttentionGradients;/);
  assert.match(owner,/sgd\(gradients: ResidualAttentionGradients, rate: number\): ResidentParameterUpdate;/);
  assert.match(get("ResidualAttentionForward"),/predictionTensor\(\): WgpuTensor;/);
  assert.match(get("ResidualAttentionForward"),/readonly parameterRevision: bigint;/);
  const gradients=get("ResidualAttentionGradients");
  assert.match(gradients,/inputGradientTensor\(\): WgpuTensor;/);
  assert.match(gradients,/zBiasGradientTensor\(\): WgpuTensor \| undefined;/);
  assert.match(gradients,/pairBiasGradientTensor\(\): WgpuTensor \| undefined;/);
  assert.match(gradients,/parameterGradientTensors\(\): WgpuTensor\[\];/);
  const sequence=get("Sequential");
  assert.match(sequence,/addLayerNorm\(name: string, features: number, curvature: number, epsilon: number\): void;/);
  assert.match(sequence,/addToposResonator\(name: string, gate: Float32Array, kernel: ToposResonatorKernel\): void;/);
  assert.match(get("ToposResonatorKernel"),/constructor\(coupling: number, iterations: number, saturation: number, porosity: number, max_values: number\);/);
  console.log(label+" residual attention and module TypeScript contract passed");
}
check(fs.readFileSync(generated,"utf8"),"generated");
check(fs.readFileSync(shipped,"utf8"),"shipped");
