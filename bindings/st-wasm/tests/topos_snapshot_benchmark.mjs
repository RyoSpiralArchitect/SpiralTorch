// Browser/Node scalar-WASM timings, not WebGPU or a Torch speed comparison.
export const TOPOS_SNAPSHOT_CASES = [[8, 3], [64, 128], [256, 768]].flatMap(
  ([rows, features]) => [1, 5, 16].map(iterations => ({id: `${rows}x${features}-k${iterations}`, rows, features, iterations})));
const require = (ok, label) => { if (!ok) throw Error(label); };
const f32 = Math.fround;
function same(a, b) {
  if (a.length !== b.length) return false;
  const left = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const right = new Uint32Array(b.buffer, b.byteOffset, b.length);
  return left.every((v, i) => v === right[i]);
}
const finite = values => values.every(Number.isFinite);
async function hash(values) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  const result = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(result, v => v.toString(16).padStart(2, "0")).join("");
}
async function hashes(values) {
  const result = [];
  for (const value of values) result.push(await hash(value));
  return result;
}
function make(length, multiplier, modulo, scale, shift) {
  return Float32Array.from({length}, (_, i) => f32(f32((i * multiplier % modulo) * f32(scale)) - f32(shift)));
}
function gradient(output, target) {
  const upstream = new Float32Array(output.length);
  let loss = 0;
  for (let i = 0; i < output.length; i++) {
    const delta = output[i] - target[i];
    loss += delta * delta / output.length;
    upstream[i] = f32(2 * delta / output.length);
  }
  return {loss, upstream};
}
function learn(kernel, input, gate, target, rows, features, learningRate) {
  const batch = kernel.captureSharedRows(input, gate, rows, features);
  let pullback;
  try {
    const output = batch.output;
    const {loss, upstream} = gradient(output, target);
    pullback = batch.vjp(upstream);
    const dx = pullback.grad_input, dg = pullback.grad_gate;
    for (let i = 0; i < gate.length; i++) gate[i] = f32(gate[i] - learningRate * dg[i]);
    return {loss, output, dx, dg};
  } finally { pullback?.free(); batch.free(); }
}

export async function runToposSnapshotBenchmark({ToposResonatorKernel}, caseOrder = "forward") {
  require(["forward", "reverse"].includes(caseOrder), "unknown case order");
  const cases = caseOrder === "reverse" ? [...TOPOS_SNAPSHOT_CASES].reverse() : TOPOS_SNAPSHOT_CASES;
  const reports = [];
  for (const config of cases) {
    const {rows, features, iterations} = config, volume = rows * features;
    const kernel = new ToposResonatorKernel(.25, iterations, 1, .2, volume);
    const input = make(volume, 17, 31, .1, 1.5);
    const initial = make(features, 7, 19, .025, .2), gate = initial.slice();
    const targetGate = make(features, 7, 19, .1, .9);
    const target = kernel.forwardSharedRows(input, targetGate, rows, features);
    const learningRate = .1 * features;
    let prepared, pulled;
    try {
      prepared = kernel.captureSharedRows(input, initial, rows, features);
      const expectedOutput = prepared.output;
      const initialState = gradient(expectedOutput, target);
      pulled = prepared.vjp(initialState.upstream);
      const expected = [expectedOutput, pulled.grad_input, pulled.grad_gate];
      const samples = {snapshots: [], learning_step: []}, roundOrder = [], blocks = [], losses = [];
      const snapshotRepeats = volume < 8192 ? 32 : 8, learningRepeats = 4;
      let updates = 0;
      for (let round = 0; round < 22; round++) {
        const order = round % 2 ? [1, 0] : [0, 1];
        if (round >= 2) roundOrder.push(order);
        for (const route of order) {
          let observed, blockLosses = [];
          const start = performance.now();
          if (route === 0) {
            for (let repeat = 0; repeat < snapshotRepeats; repeat++) {
              observed = [prepared.output, pulled.grad_input, pulled.grad_gate];
            }
          } else {
            for (let repeat = 0; repeat < learningRepeats; repeat++) {
              observed = learn(kernel, input, gate, target, rows, features, learningRate);
              blockLosses.push(observed.loss);
            }
          }
          const elapsed = performance.now() - start;
          require(Number.isFinite(elapsed) && elapsed >= 0, "invalid clock sample");
          if (round >= 2) samples[route ? "learning_step" : "snapshots"].push(elapsed / (route ? learningRepeats : snapshotRepeats));
          if (route === 0) {
            require(observed.every((a, i) => same(a, expected[i])), "snapshot value drift");
          } else {
            require(blockLosses.every(Number.isFinite) && [observed.output, observed.dx, observed.dg, gate].every(finite), "nonfinite learning update");
            updates += learningRepeats;
            losses.push(...blockLosses);
            // Hash after the timed block. Keep every loss and each block-end
            // output/gradient/gate identity; do not claim per-update VJP hashes.
            blocks.push({round, updates, sha256: await hashes([observed.output, observed.dx, observed.dg, gate])});
          }
        }
      }
      const finalOutput = kernel.forwardSharedRows(input, gate, rows, features);
      const finalLoss = gradient(finalOutput, target).loss;
      require(Number.isFinite(finalLoss) && finalLoss < initialState.loss, "learning did not reduce loss");
      reports.push({...config, coupling: .25, porosity: .2, saturation: 1, learning_rate: learningRate,
        input_sha256: await hash(input), initial_gate_sha256: await hash(initial), target_sha256: await hash(target),
        snapshot_sha256: await hashes(expected), round_order: roundOrder, measurements_ms: samples,
        repeats_per_sample: {snapshots: snapshotRepeats, learning_step: learningRepeats},
        learning: {updates, losses, blocks, initial_loss: initialState.loss, final_loss: finalLoss,
          final_gate_sha256: await hash(gate), final_output_sha256: await hash(finalOutput)}});
    } finally { pulled?.free(); prepared?.free(); kernel.free(); }
  }
  return {schema: "spiraltorch.topos_snapshot_benchmark.v1", status: "passed", backend: "rust_f32_wasm",
    case_order: caseOrder, rounds: 20, warmup_blocks_per_route: 2, routes: ["snapshots", "learning_step"], cases: reports,
    scope: "Scalar WASM in its reported JS host. Snapshot scope reads prepared output and both VJPs. Learning scope includes capture, all three getters, JS mean-MSE/upstream, Rust VJP, JS f32 gate update, and Rust object disposal. Clock-block overhead and allocation are included; setup, checks and hashes are excluded. No optimizer speed comparison to native NN, Torch, WebGPU, or model-quality claim."};
}
