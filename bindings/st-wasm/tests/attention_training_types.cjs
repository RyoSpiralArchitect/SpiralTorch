const fs = require("node:fs");
const assert = require("node:assert/strict");
const path = require("node:path");

function checkAttentionTrainingContract(types, label) {
  const get = name => {
    const d = types.match(new RegExp("^( *)export class " + name + " \\{[\\s\\S]*?^\\1\\}", "m"))?.[0];
    assert.ok(d, label + " must export " + name);
    assert.match(d, /private constructor\(\)/);
    assert.match(d, /free\(\): void/);
    return d;
  };
  const plan = get("AttentionInferencePlan");
  assert.match(plan, /static fromProjectionPlans\(query: InferencePlan, key: InferencePlan, value: InferencePlan, output: InferencePlan, heads: number, causal_offset\?: number \| null\): AttentionInferencePlan;/);
  assert.match(plan, /compileTrainingWebGpu\(tile_mnk\?: (?:number\[\]|Array<any>) \| null, kernel\?: string \| null, accumulation\?: string \| null\): Promise<ResidentAttentionTraining>;/);
  const owner = get("ResidentAttentionTraining");
  for (const d of [plan, owner]) {
    assert.match(d, /readonly inputShape: Uint32Array;/);
    assert.match(d, /readonly outputShape: Uint32Array;/);
  }
  assert.match(owner, /readonly attemptedUpdates: bigint;/);
  assert.match(owner, /parameterTensors\(\): WgpuTensor\[\];/);
  assert.match(owner, /tensorDevice\(\): WgpuTensorDevice;/);
  assert.match(owner, /forward\(input: WgpuTensor, biases: WgpuAttentionBiases\): AttentionForward;/);
  assert.match(owner, /backward\(forward: AttentionForward, cotangent: WgpuTensor\): AttentionGradients;/);
  assert.match(owner, /sgd\(gradients: AttentionGradients, rate: number\): ResidentParameterUpdate;/);
  const forward = get("AttentionForward");
  assert.match(forward, /readonly parameterRevision: bigint;/);
  assert.match(forward, /predictionTensor\(\): WgpuTensor;/);
  const gradients = get("AttentionGradients");
  assert.match(gradients, /inputGradientTensor\(\): WgpuTensor;/);
  assert.match(gradients, /parameterGradientTensors\(\): WgpuTensor\[\];/);
  for (const bias of ["z", "pair"])
    assert.match(gradients, new RegExp(bias + "BiasGradientTensor\\(\\): WgpuTensor \\| undefined;"));
  const update = get("ResidentParameterUpdate");
  assert.match(update, /readonly attemptedRevision: bigint;/);
  assert.match(update, /read\(\): Promise<bigint>;/);
  console.log(label + " resident attention training TypeScript contract passed");
}
module.exports = {checkAttentionTrainingContract};
if (require.main === module) {
  assert.equal(process.argv.length, 3, "Pass the freshly generated spiraltorch_wasm.js path");
  checkAttentionTrainingContract(fs.readFileSync(path.resolve(process.argv[2]).replace(/\.js$/, ".d.ts"), "utf8"), "generated");
  checkAttentionTrainingContract(fs.readFileSync(path.join(__dirname, "../types/spiraltorch-wasm.d.ts"), "utf8"), "shipped");
}
