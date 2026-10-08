// Node-hosted scalar WASM parity and SGD probe, not a browser/WebGPU benchmark.
// Usage: node tools/probe_wave_gate_wasm.mjs <wasm-bindgen web module dir> <new report.json>
import {readFile, writeFile} from "node:fs/promises";
import {createHash} from "node:crypto";
import {resolve, join} from "node:path";
import {pathToFileURL, fileURLToPath} from "node:url";

const [directory, destination] = process.argv.slice(2);
if (!directory || !destination || process.argv.length !== 4) throw Error("module directory and new report required");
const assert = (ok, label) => { if (!ok) throw Error(label); };
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const bytes = values => Buffer.from(values.buffer, values.byteOffset, values.byteLength);
const wasm = await readFile(join(directory, "spiraltorch_wasm_bg.wasm"));
const modulePath = resolve(directory, "spiraltorch_wasm.js");
const {initSync, WaveGateKernel} = await import(pathToFileURL(modulePath).href);
initSync({module: wasm});
const cases = [];
for (const [rows, features] of [[0, 3], [1, 1], [32, 65], [256, 768]]) {
  for (const radius of [null, -2, 0, 2]) {
    const input = Float32Array.from({length: rows * features}, (_, i) => (i * 37 % 127) / 53 - 1);
    const gate = Float32Array.from({length: features}, (_, i) => (i % 7 - 3) * .7);
    const bias = Float32Array.from({length: features}, (_, i) => (i % 5 - 2) * .1);
    const upstream = Float32Array.from(input, (_, i) => (i * 13 % 31) / 31 - .5);
    const kernel = new WaveGateKernel(-.7, 1, .2, 1_048_576);
    let batch, gradient;
    try {
      batch = radius === null ? kernel.forward(input, gate, bias, rows, features)
        : kernel.forward_with_log_radius(input, gate, bias, rows, features, radius);
      const output = batch.output;
      assert(output.length === input.length, "output shape");
      input.fill(42); gate.fill(42); bias.fill(42);
      assert(bytes(output).equals(bytes(batch.output)), "foreign mutation changed snapshot");
      gradient = radius === null ? batch.vjp(upstream) : batch.vjp_with_log_radius(upstream);
      const tensors = {output, grad_input: gradient.grad_input, grad_gate: gradient.grad_gate,
        grad_bias: gradient.grad_bias};
      if (radius !== null) tensors.grad_log_radius = Float32Array.of(gradient.grad_log_radius);
      assert(tensors.grad_input.length === output.length && tensors.grad_gate.length === features
        && tensors.grad_bias.length === features, "gradient shape");
      const record = {rows, features, log_radius: radius, sha256: {}};
      for (const [name, value] of Object.entries(tensors)) {
        assert(value.every(Number.isFinite), "nonfinite " + name);
        record.sha256[name] = hash(bytes(value));
      }
      cases.push(record);
    } finally { gradient?.free(); batch?.free(); kernel.free(); }
  }
}

const kernel = new WaveGateKernel(-.7, 1, .2, 64);
const input = Float32Array.of(.2, -.3, 1.5, .5);
const gate = Float32Array.of(1.4, -1.1), bias = Float32Array.of(.15, -.05);
let learning;
try {
  const teacher = kernel.forward_with_log_radius(input, gate, bias, 2, 2, .7);
  const target = teacher.output;
  teacher.free();
  function step(state, update) {
    const batch = kernel.forward_with_log_radius(input, gate, bias, 2, 2, state.log_radius);
    let gradient;
    try {
      const output = batch.output;
      const loss = output.reduce((sum, y, i) => sum + (y - target[i]) ** 2 / output.length, 0);
      if (update) {
        gradient = batch.vjp_with_log_radius(Float32Array.from(output, (y, i) => 2 * (y - target[i]) / output.length));
        state.log_radius = Math.fround(state.log_radius - 2 * gradient.grad_log_radius);
      }
      return loss;
    } finally { gradient?.free(); batch.free(); }
  }
  const state = {log_radius: 0}, initial = step(state, false);
  for (let i = 0; i < 400; i++) step(state, true);
  const final = step(state, false), restored = JSON.parse(JSON.stringify(state));
  const learned = state.log_radius;
  step(state, true); step(restored, true);
  assert(final < initial * .01, "radius SGD did not learn");
  assert(state.log_radius === restored.log_radius, "SGD continuation mismatch");
  learning = {initial_loss: initial, final_loss: final, learned_log_radius: learned,
    updates: 400, next_update_equal: true};
  for (const invalid of [NaN, Infinity, -Infinity, 100, -100, "0", null, true]) {
    let rejected = false;
    try { kernel.forward_with_log_radius(input, gate, bias, 2, 2, invalid).free(); }
    catch { rejected = true; }
    assert(rejected, "invalid radius accepted");
  }
} finally { kernel.free(); }
const report = {schema: "spiraltorch.wave_gate_wasm_probe.v1", status: "passed", cases, learning,
  wasm_sha256: hash(wasm), wrapper_sha256: hash(await readFile(modulePath)),
  probe_sha256: hash(await readFile(fileURLToPath(import.meta.url))),
  node: process.version, scope: "Scalar WASM in Node; not a browser or WebGPU execution/performance claim."};
await writeFile(destination, JSON.stringify(report, null, 2) + "\n", {flag: "wx"});
console.log(JSON.stringify({status: report.status, cases: cases.length, learning}));
