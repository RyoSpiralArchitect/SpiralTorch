import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {assert, cpuChecks} from './byte_lm_boundaries.mjs';
import {selectionChecks} from './byte_lm_selection.mjs';

const [modulePath, fixturePath, features, selectionMode] = process.argv.slice(2);
if (!modulePath || !fixturePath || !['nn', 'webgpu'].includes(features)
    || ![undefined, 'selection'].includes(selectionMode)) {
  throw Error('usage: byte_lm_node.mjs /absolute/package.js /private/input.json nn|webgpu [selection]');
}
const imported = await import(pathToFileURL(modulePath));
// Web glue has an async initializer; Node glue initializes on import.
const web = typeof imported.default === 'function';
const M = web ? imported : (imported.default ?? imported);
if (web) await M.default({module_or_path: fs.readFileSync(modulePath.replace(/\.js$/, '_bg.wasm'))});
const gpu = features === 'webgpu';
for (const name of ['ResidentByteInference', 'ResidentByteServing', 'ResidentByteCandidate']) {
  assert((typeof M[name] === 'function') === gpu, `${name} feature mismatch`);
}
assert((typeof M.ByteCorpusModelArtifact.prototype.restoreWebGpu === 'function') === gpu, 'restore feature mismatch');
assert((typeof M.ByteCorpusStudy.prototype.advanceWebGpu === 'function') === gpu, 'study feature mismatch');
assert((typeof M.ByteCorpusServingGate.prototype.bootstrapWebGpu === 'function') === gpu, 'gate feature mismatch');
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
const checks = await cpuChecks(M, fixture);
const selection = selectionMode === 'selection' || fixture.selection
  ? await selectionChecks(M, fixture.selection) : undefined;
console.log(JSON.stringify({schema: 'spiraltorch.packaged_byte_mlops.cpu.' + (selection ? 'v2' : 'v1'),
  features, feature_matrix_passed: true, checks, count: checks.length,
  ...(selection ? {selection} : {})}));
