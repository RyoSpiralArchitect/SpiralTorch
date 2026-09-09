const fs = require("node:fs");
const assert = require("node:assert/strict");
const path = require("node:path");

const modulePath = path.resolve(process.argv[2]);
const types = fs.readFileSync(modulePath.replace(/\.js$/, ".d.ts"), "utf8");
function checkMatmulContract(types, label) {
  const declaration = types.match(/^( *)export class WgpuMatmul \{[\s\S]*?^\1\}/m)?.[0];
  assert.ok(declaration, label + " must export its workspace type");
  assert.match(declaration, /private constructor\(\)/);
  assert.match(declaration, /create\(rows: number, inner: number, cols: number\): Promise<WgpuMatmul>/);
  assert.match(declaration, /createWithTile\(rows: number, inner: number, cols: number, tile_m: number, tile_n: number, tile_k: number\): Promise<WgpuMatmul>/);
  assert.match(declaration, /tileMNK\(\): Uint32Array/);
  assert.match(declaration, /createWithKernel\(rows: number, inner: number, cols: number, tile_m: number, tile_n: number, tile_k: number, kernel: string\): Promise<WgpuMatmul>/);
  assert.match(declaration, /readonly kernel: string/);
  assert.match(declaration, /readonly accumulation: string/);
  assert.match(declaration, /createWithOptions\(rows: number, inner: number, cols: number, tile_m: number, tile_n: number, tile_k: number, kernel: string, accumulation: string\): Promise<WgpuMatmul>/);
  assert.match(declaration, /workgroupSize\(\): Uint32Array/);
  assert.match(declaration, /outputsPerThread\(\): Uint32Array/);
  assert.match(declaration, /upload\(lhs: Float32Array, rhs: Float32Array\): void/);
  assert.match(declaration, /dispatch\(repetitions\?: number(?: \| null)?\): bigint/);
  assert.match(declaration, /readback\(\): Promise<Float32Array>/);
  assert.match(declaration, /synchronize\(\): Promise<void>/);
  assert.match(declaration, /readonly generation: bigint/);
  assert.match(declaration, /readonly outputIsCurrent: boolean/);
  assert.match(declaration, /uploadRhs\(rhs: Float32Array\): void/);
  assert.match(declaration, /setLhsFrom\(source: WgpuMatmul\): void/);
  console.log(label + " resident WebGPU TypeScript contract passed");
}

function checkRankContract(types, label) {
  const rank = types.match(/^( *)export class WgpuRank \{[\s\S]*?^\1\}/m)?.[0];
  assert.ok(rank, label + " must export resident rank");
  assert.match(rank, /private constructor\(\)/);
  assert.match(rank, /create\(kind: (?:string|"topk" \| "midk" \| "bottomk"), rows: number, cols: number, k: number, tile_cols\?: number(?: \| null)?, timestamp_queries\?: boolean(?: \| null)?\): Promise<WgpuRank>/);
  assert.match(rank, /createFromAdaptation\(session: RankAdaptationSession, candidate_index: number, timestamp_queries\?: boolean(?: \| null)?\): Promise<WgpuRank>/);
  assert.match(rank, /profile\(repetitions\?: number(?: \| null)?\): Promise<Record<string, unknown>>/);
  assert.match(rank, /readonly timestampQueriesEnabled: boolean/);
  assert.match(rank, /readonly outputIsCurrent: boolean/);
  assert.match(rank, /upload\(input: Float32Array\): void/);
  assert.match(rank, /setInputFromMatmul\(source: WgpuMatmul\): void/);
  assert.match(rank, /dispatchFromMatmul\(source: WgpuMatmul, repetitions\?: number(?: \| null)?\): bigint/);
  assert.match(rank, /dispatch\(repetitions\?: number(?: \| null)?\): bigint/);
  assert.match(rank, /readback\(\): Promise<\{values: Float32Array; indices: Int32Array; generation: bigint\}>/);
  console.log(label + " resident rank/profile TypeScript contract passed");
}

const shipped = fs.readFileSync(path.join(__dirname, "../types/spiraltorch-wasm.d.ts"), "utf8");
function checkNnContract(types, label) {
  const get = name => {
    const declaration = types.match(new RegExp("^( *)export class " + name + " \\{[\\s\\S]*?^\\1\\}", "m"))?.[0];
    assert.ok(declaration, label + " must export " + name);
    assert.match(declaration, /private constructor\(\)/);
    return declaration;
  };
  const plan = get("InferencePlan"), gpu = get("ResidentInference"), snapshot = get("InferenceSnapshot");
  assert.match(plan, /static fromJson\(payload: string, max_bytes\?: number(?: \| null)?\): InferencePlan/);
  assert.match(plan, /compileWebGpu\([^\n]*\): Promise<ResidentInference>/);
  for (const declaration of [plan, gpu]) {
    assert.match(declaration, /readonly inputShape: Uint32Array/);
    assert.match(declaration, /readonly outputShape: Uint32Array/);
    assert.match(declaration, /readonly stageCount: number/);
  }
  assert.match(gpu, /upload\(values: Float32Array\): void/);
  assert.match(gpu, /snapshot\(\): InferenceSnapshot/);
  assert.match(gpu, /dispatch\(\): bigint/);
  assert.match(gpu, /adapterInfo\(\): \{ name: string; backend: string; device_type: string \}/);
  assert.match(snapshot, /readonly shape: Uint32Array/);
  assert.match(snapshot, /readonly generation: bigint/);
  assert.match(snapshot, /readValues\(\): Promise<Float32Array>/);
  assert.match(plan, /compileTrainingWebGpu\([^\n]*\): Promise<ResidentTraining>/);
  const training=get("ResidentTraining"), loss=get("TrainingLossSnapshot"),
    trainingSnapshot=get("TrainingSnapshot"), parameters=get("TrainingParametersSnapshot"), state=get("TrainingState");
  assert.match(training, /uploadBatch\(input: Float32Array, target: Float32Array\): void/);
  assert.match(training, /step\(learning_rate: number\): bigint/);
  assert.match(training, /readonly submittedSteps: bigint/);
  assert.match(training, /stateSnapshot\(\): TrainingSnapshot/);
  assert.match(training, /lossSnapshot\(\): TrainingLossSnapshot/);
  assert.match(training, /parameterSnapshot\(\): TrainingParametersSnapshot/);
  assert.match(loss, /read\(\): Promise<number>/);
  assert.match(trainingSnapshot, /readState\(\): Promise<TrainingState>/);
  assert.match(parameters, /readPlan\(\): Promise<InferencePlan>/);
  for(const declaration of [loss, trainingSnapshot, state]) {
    assert.match(declaration, /readonly submittedStep: bigint/);
    assert.match(declaration, /readonly batchGeneration: bigint/);
  }
  assert.match(state, /toPlan\(\): InferencePlan/);
  assert.match(state, /readonly loss: number/);
  for(const method of ["weightValues","biasValues","weightGradientValues","biasGradientValues"])
    assert.match(state, new RegExp(method+"\\(stage: number\\): Float32Array"));
  assert.match(plan, /readonly isDense: boolean/);
  assert.match(plan, /compileGraphTrainingWebGpu\(gradient_policy: string, [^\n]*\): Promise<ResidentGraphTraining>/);
  const graph=get("ResidentGraphTraining"), graphSnapshot=get("GraphTrainingSnapshot"),
    graphParameters=get("GraphTrainingParametersSnapshot"), graphState=get("GraphTrainingState");
  for(const declaration of [graph, graphSnapshot, graphState]) {
    assert.match(declaration, /readonly gradientPolicy: string/);
    assert.match(declaration, /readonly inputShape: Uint32Array/);
    assert.match(declaration, /readonly outputShape: Uint32Array/);
    assert.match(declaration, /readonly batchGeneration: bigint/);
  }
  for(const declaration of [graph, graphState]) {
    assert.match(declaration, /readonly parameterCount: number/);
    assert.match(declaration, /readonly stageCount: number/);
  }
  for(const declaration of [graphSnapshot, graphState]) assert.match(declaration, /readonly submittedStep: bigint/);
  assert.match(graph, /readonly submittedSteps: bigint/);
  assert.match(graph, /adapterInfo\(\): \{ name: string; backend: string; device_type: string \}/);
  assert.match(graph, /uploadBatch\(input: Float32Array, target: Float32Array\): void/);
  assert.match(graph, /step\(learning_rate: number\): bigint/);
  assert.match(graph, /stateSnapshot\(\): GraphTrainingSnapshot/);
  assert.match(graph, /lossSnapshot\(\): TrainingLossSnapshot/);
  assert.match(graph, /parameterSnapshot\(\): GraphTrainingParametersSnapshot/);
  assert.match(graphSnapshot, /readState\(\): Promise<GraphTrainingState>/);
  assert.match(graphParameters, /readPlan\(\): Promise<InferencePlan>/);
  assert.match(graphState, /readonly loss: number/);
  assert.match(graphState, /parameterRole\(parameter: number\): string/);
  assert.match(graphState, /parameterShape\(parameter: number\): Uint32Array/);
  for(const method of ["parameterValues","parameterGradientValues","effectiveGradientValues"])
    assert.match(graphState, new RegExp(method+"\\(parameter: number\\): Float32Array"));
  for(const method of ["predictionValues","inputGradientValues"])
    assert.match(graphState, new RegExp(method+"\\(\\): Float32Array"));
  assert.match(graphState, /toPlan\(\): InferencePlan/);
  assert.match(plan, /compileGraphWebGpu\([^\n]*\): Promise<ResidentGraphInference>/);
  const forward=get("ResidentGraphInference"), forwardSnapshot=get("GraphInferenceSnapshot"),
    tensor=get("WgpuTensor"), tensorDevice=get("WgpuTensorDevice"), tensorSnapshot=get("WgpuTensorSnapshot");
  assert.match(forward, /setInputTensor\(input: WgpuTensor\): void/);
  assert.match(forward, /outputTensor\(\): WgpuTensor/);
  assert.match(forward, /tensorDevice\(\): WgpuTensorDevice/);
  assert.match(forward, /snapshot\(\): GraphInferenceSnapshot/);
  assert.match(forward, /readonly submittedDispatches: bigint/);
  assert.match(forwardSnapshot, /readonly submittedDispatch: bigint/);
  for (const value of [forwardSnapshot,tensorSnapshot]) assert.match(value, /readValues\(\): Promise<Float32Array>/);
  assert.match(tensorDevice, /create\(\): Promise<WgpuTensorDevice>/);
  assert.match(tensorDevice, /upload\(shape: number\[\], data: Float32Array\): WgpuTensor/);
  assert.match(tensor, /readonly strides: Uint32Array/);
  assert.match(tensor, /narrow\(axis: number, start: number, length: number\): WgpuTensor/);
  assert.match(tensor, /snapshot\(\): WgpuTensorSnapshot/);
  assert.match(gpu, /setInputTensor\(input: WgpuTensor\): void/);
  assert.match(gpu, /tensorSnapshot\(device: WgpuTensorDevice\): WgpuTensor/);
  assert.match(graph, /uploadBatchTensors\(input: WgpuTensor, target: WgpuTensor\): void/);
  assert.match(graph, /predictionTensor\(\): WgpuTensor/);
  assert.match(graph, /inputGradientTensor\(\): WgpuTensor/);
  console.log(label + " resident NN TypeScript contract passed");
}

for (const [source, label] of [[types, "generated"], [shipped, "shipped"]]) {
  checkMatmulContract(source, label);
  checkRankContract(source, label);
  checkNnContract(source, label);
}
