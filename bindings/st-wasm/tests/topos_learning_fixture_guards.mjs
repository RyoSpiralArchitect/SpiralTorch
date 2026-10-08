import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {pathToFileURL} from "node:url";
import {resolve} from "node:path";
import {runToposLearning} from "./topos_resonator_learning.mjs";

if (process.argv.length !== 3) throw Error("generated WASM module file required");
const url = pathToFileURL(resolve(process.argv[2]));
const loaded = await import(url.href);
const runtime = loaded.initSync ? loaded : loaded.default;
if (runtime.initSync) runtime.initSync({module: await readFile(new URL("spiraltorch_wasm_bg.wasm", url))});
const fixture = JSON.parse(await readFile(new URL("topos_resonator_learning_fixture.json", import.meta.url), "utf8")).native_fixture;
assert.equal(runToposLearning(runtime.ToposResonatorKernel, fixture).status, "passed");
let rejected = 0;
for (const field of ["output", "grad_input", "grad_gate"]) {
  for (const value of [NaN, Infinity, -Infinity, null, "Infinity", "0", {}, [], true]) {
    const malformed = structuredClone(fixture);
    malformed[field][0] = value;
    const result = runToposLearning(runtime.ToposResonatorKernel, malformed);
    assert.equal(result.status, "error", field + " accepted a malformed reference");
    assert.match(result.error, /nonfinite comparison 0/);
    rejected++;
  }
}
assert.equal(rejected, 27);
console.log(JSON.stringify({status: "passed", rejected_fixture_references: rejected}));
