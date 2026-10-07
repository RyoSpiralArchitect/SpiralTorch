// Execute the browser's shared learning contract in Node, not a browser claim.
import {readFile, writeFile} from "node:fs/promises";
import {createHash} from "node:crypto";
import {resolve, join} from "node:path";
import {pathToFileURL, fileURLToPath} from "node:url";
import {runToposLearning} from "../bindings/st-wasm/tests/topos_resonator_learning.mjs";

const [directory, fixturePath, destination] = process.argv.slice(2);
if (!directory || !fixturePath || !destination || process.argv.length !== 5)
  throw Error("module directory, native fixture and new report required");
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const wasm = await readFile(join(directory, "spiraltorch_wasm_bg.wasm"));
const modulePath = resolve(directory, "spiraltorch_wasm.js");
const loaded = await import(pathToFileURL(modulePath).href);
const runtime = loaded.initSync ? loaded : loaded.default;
if (runtime.initSync) runtime.initSync({module: wasm});
const fixtureBytes = await readFile(fixturePath);
const result = runToposLearning(runtime.ToposResonatorKernel, JSON.parse(fixtureBytes).native_fixture);
const sharedPath = new URL("../bindings/st-wasm/tests/topos_resonator_learning.mjs", import.meta.url);
const report = {
  schema: "spiraltorch.topos_browser_learning_node_probe.v1",
  scope: "Node-hosted execution of the shared browser learning contract, not rendered browser or timing evidence.",
  result,
  wasm_sha256: hash(wasm),
  wrapper_sha256: hash(await readFile(modulePath)),
  fixture_sha256: hash(fixtureBytes),
  shared_contract_sha256: hash(await readFile(sharedPath)),
  probe_sha256: hash(await readFile(fileURLToPath(import.meta.url))),
  node: process.version,
};
await writeFile(destination, JSON.stringify(report, null, 2) + "\n", {flag: "wx"});
console.log(JSON.stringify({status: result.status, guards: result.guard_checks,
  updates: result.learning_updates, final_loss: result.learning_losses?.at(-1)}));
if (result.status !== "passed") {
  console.error(result.error);
  process.exitCode = 1;
}
