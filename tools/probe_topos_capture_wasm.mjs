// Scalar WASM parity and learning in Node, not a browser/WebGPU speed claim.
import {readFile, writeFile} from "node:fs/promises";
import {createHash} from "node:crypto";
import {resolve, join} from "node:path";
import {pathToFileURL, fileURLToPath} from "node:url";

const [directory, destination, option] = process.argv.slice(2);
if (!directory || !destination || process.argv.length > 5 || (option && option !== "--require-capture"))
  throw Error("module directory, new report and optional --require-capture required");
const assert = (ok, label) => {if (!ok) throw Error(label);};
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const bytes = values => Buffer.from(values.buffer, values.byteOffset, values.byteLength);
const same = (a, b) => bytes(a).equals(bytes(b));
const wasm = await readFile(join(directory, "spiraltorch_wasm_bg.wasm"));
const modulePath = resolve(directory, "spiraltorch_wasm.js");
const loaded = await import(pathToFileURL(modulePath).href);
const {initSync, ToposResonatorKernel} = loaded.initSync ? loaded : loaded.default;
if (initSync) initSync({module: wasm});
const captured = typeof ToposResonatorKernel.prototype.capture === "function";
assert(option !== "--require-capture" || captured, "captured learning API required");
const cases = [];
let guardChecks = 0;
function rejected(call) {
  let failed = false;
  try {const result = call(); result?.free?.();} catch {failed = true;}
  assert(failed, "invalid request accepted");
  guardChecks++;
}

for (const [rows, features] of [[0, 3], [1, 1], [7, 17], [256, 768]]) {
  for (const [iterations, coupling] of [[1, 0], [4, .25], [16, .75]]) {
    for (const porosity of [0, .2]) {
      const x = Float32Array.from({length: rows * features}, (_, i) => (i * 37 % 127) / 53 - 1);
      const gate = Float32Array.from(x, (_, i) => (i % 7 - 3) * .7);
      const dy = Float32Array.from(x, (_, i) => (i * 13 % 31) / 31 - .5);
      const kernel = new ToposResonatorKernel(coupling, iterations, 1, porosity, 1_048_576);
      let batch;
      let tensors;
      let capturedAuditSha256 = null;
      try {
        const output = kernel.forward(x, gate, rows, features);
        const gradient = kernel.backward(x, gate, dy, rows, features);
        tensors = {output, grad_input: Float32Array.from(gradient.grad_input), grad_gate: Float32Array.from(gradient.grad_gate)};
        if (captured) {
          batch = kernel.capture(x, gate, rows, features);
          assert(same(output, batch.output), "captured output differs");
          const audit = batch.audit_json();
          assert(typeof JSON.parse(audit) === "object", "missing captured audit");
          capturedAuditSha256 = hash(audit);
        }
      } finally {kernel.free();}
      try {
        if (batch) {
          x.fill(NaN); gate.fill(NaN);
          assert(same(tensors.output, batch.output), "snapshot reads foreign input");
          assert(hash(batch.audit_json()) === capturedAuditSha256, "audit reads foreign input");
          rejected(() => batch.vjp(new Float32Array(dy.length + 1)));
          if (dy.length) rejected(() => batch.vjp(new Float32Array(dy.length).fill(NaN)));
          for (let repeat = 0; repeat < 2; repeat++) {
            const gradient = batch.vjp(dy);
            try {
              assert(same(tensors.grad_input, gradient.grad_input) && same(tensors.grad_gate, gradient.grad_gate), "captured VJP differs");
            } finally {gradient.free();}
          }
        }
        const record = {rows, features, iterations, coupling, porosity,
          captured_audit_sha256: capturedAuditSha256, sha256: {}};
        for (const [name, tensor] of Object.entries(tensors)) {
          assert(tensor.every(Number.isFinite), "nonfinite " + name);
          record.sha256[name] = hash(bytes(tensor));
        }
        cases.push(record);
      } finally {batch?.free();}
    }
  }
}

const kernel = new ToposResonatorKernel(.4, 5, 1, .2, 64);
let learning;
try {
  const x = Float32Array.of(.2, -.3, .6, .5);
  const target = kernel.forward(x, Float32Array.of(.6, -.4, .6, -.4), 2, 2);
  function step(gate, update, capture) {
    const expanded = Float32Array.of(...gate, ...gate);
    const batch = capture ? kernel.capture(x, expanded, 2, 2) : null;
    let pulled;
    try {
      const output = batch ? batch.output : kernel.forward(x, expanded, 2, 2);
      const loss = output.reduce((s, y, i) => s + (y - target[i]) ** 2 / output.length, 0);
      if (update) {
        const dy = Float32Array.from(output, (y, i) => 2 * (y - target[i]) / output.length);
        pulled = batch ? batch.vjp(dy) : kernel.backward(x, expanded, dy, 2, 2);
        const dg = pulled.grad_gate;
        for (let i = 0; i < 2; i++) gate[i] = Math.fround(gate[i] - .3 * Math.fround(dg[i] + dg[i + 2]));
      }
      return loss;
    } finally {pulled?.free?.(); batch?.free();}
  }
  const legacy = new Float32Array(2), candidate = new Float32Array(2);
  const initial = step(candidate, false, captured);
  for (let i = 0; i < 240; i++) {
    step(legacy, true, false); step(candidate, true, captured);
    assert(same(legacy, candidate), "learning trajectory differs");
  }
  const final = step(candidate, false, captured), restored = candidate.slice();
  step(candidate, true, captured); step(restored, true, captured);
  assert(final < initial * .01 && same(candidate, restored), "learning or continuation failed");
  learning = {updates: 240, initial_loss: initial, final_loss: final,
    gate_sha256: hash(bytes(candidate)), legacy_trajectory_exact: true, next_update_exact: true};
  if (captured) {
    for (const shape of [[1.5, 2], [-1, 2], ["2", 2], [0, 0], [2**32, 2], [1, 2]])
      rejected(() => kernel.capture(x, x, ...shape));
    rejected(() => kernel.capture(new Float32Array(65), new Float32Array(65), 1, 65));
    rejected(() => kernel.capture(Float32Array.of(Infinity), Float32Array.of(1), 1, 1));
  }
} finally {kernel.free();}
const report = {schema: "spiraltorch.topos_capture_wasm_probe.v1", status: "passed", captured, cases, learning,
  module_format: initSync ? "web" : "nodejs", capture_required: option === "--require-capture",
  guard_checks: guardChecks, wasm_sha256: hash(wasm), wrapper_sha256: hash(await readFile(modulePath)),
  probe_sha256: hash(await readFile(fileURLToPath(import.meta.url))), node: process.version,
  scope: "Node-hosted scalar WASM. Not browser/WebGPU or timing evidence."};
await writeFile(destination, JSON.stringify(report, null, 2) + "\n", {flag: "wx"});
console.log(JSON.stringify({status: report.status, captured, cases: cases.length, learning, guard_checks: guardChecks}));
