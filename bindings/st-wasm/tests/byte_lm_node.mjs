import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {assert, cpuChecks} from './byte_lm_boundaries.mjs';

const [modulePath, fixturePath, features] = process.argv.slice(2);
if (!modulePath || !fixturePath || !['nn', 'webgpu'].includes(features)) {
  throw Error('usage: byte_lm_node.mjs /absolute/package.js /private/input.json nn|webgpu');
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
const checks = await cpuChecks(M, JSON.parse(fs.readFileSync(fixturePath, 'utf8')));
console.log(JSON.stringify({schema: 'spiraltorch.packaged_byte_mlops.cpu.v1', features, feature_matrix_passed: true, checks, count: checks.length}));
