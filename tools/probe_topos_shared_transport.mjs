import {readFile, writeFile} from "node:fs/promises";
import {createHash} from "node:crypto";
import {resolve, join} from "node:path";
import {pathToFileURL, fileURLToPath} from "node:url";
import {checkSharedTopos} from "../bindings/st-wasm/tests/topos_shared_transport.mjs";

const [directory, destination] = process.argv.slice(2);
if (!directory || !destination || process.argv.length !== 4)
  throw Error("usage: probe_topos_shared_transport.mjs MODULE_DIR NEW_REPORT");
const modulePath = resolve(directory, "spiraltorch_wasm.js");
const wasm = await readFile(join(directory, "spiraltorch_wasm_bg.wasm"));
const loaded = await import(pathToFileURL(modulePath).href);
const api = loaded.initSync ? loaded : loaded.default;
if (api.initSync) api.initSync({module: wasm});
const hash = data => createHash("sha256").update(data).digest("hex");
const report = {...checkSharedTopos(api), schema: "spiraltorch.topos_shared_transport.v1",
  wasm_sha256: hash(wasm), wrapper_sha256: hash(await readFile(modulePath)),
  fixture_sha256: hash(await readFile(fileURLToPath(new URL("../bindings/st-wasm/tests/topos_shared_transport.mjs", import.meta.url)))),
  node: process.version};
await writeFile(destination, JSON.stringify(report, null, 2) + "\n", {flag: "wx"});
console.log(JSON.stringify(report));
