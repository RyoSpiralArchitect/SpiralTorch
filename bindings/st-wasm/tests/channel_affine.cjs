const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {ChannelAffineKernel, GeometryAdapterArtifact} = require(path.resolve(process.argv[2]));
const f = JSON.parse(fs.readFileSync(path.resolve(
  __dirname, '../../../crates/st-nn/tests/fixtures/channel_affine.fixture.json'), 'utf8'));
const cast = name => new Float32Array(f[name]);
const kernel = new ChannelAffineKernel(f.max_values);
const x = cast('input'), gate = cast('gate'), bias = cast('bias');
const batch = kernel.forward(x, gate, bias, f.rows, f.features);
x.fill(99); gate.fill(99); bias.fill(99);
assert.deepEqual(Array.from(batch.output), f.output);
const gradient = batch.vjp(cast('upstream'));
for (const name of ['grad_input', 'grad_gate', 'grad_bias']) {
  assert.deepEqual(Array.from(gradient[name]), f[name]);
}
for (const bad of [null, undefined, true, false, '6', {}, [], -1, .5, NaN, Infinity, 2**32, 0]) {
  assert.throws(() => new ChannelAffineKernel(bad));
}
for (const bad of [true, '2', .5, 2**32, -1, null]) {
  assert.throws(() => kernel.forward(cast('input'), cast('gate'), cast('bias'), bad, 3));
}
for (const bad of [[1], new Float64Array([1]), new Uint8Array([1]), new Float32Array([NaN])]) {
  assert.throws(() => kernel.forward(bad, new Float32Array([1]), new Float32Array([0]), 1, 1));
}
assert.throws(() => kernel.forward(new Float32Array(7), new Float32Array([1]), new Float32Array([0]), 7, 1));
assert.throws(() => batch.vjp(new Float32Array(7)));
const empty = kernel.forward(new Float32Array(), cast('gate'), cast('bias'), 0, 3);
const emptyGrad = empty.vjp(new Float32Array());
assert.deepEqual(Array.from(empty.output), []);
assert.deepEqual(Array.from(emptyGrad.grad_gate), [0, 0, 0]);
const artifact = GeometryAdapterArtifact.createJson(JSON.stringify(f.artifact));
const digest = text => 'sha256:' + createHash('sha256').update(text).digest('hex');
assert.equal(artifact.artifactId, digest(artifact.toJson()));
const loaded = GeometryAdapterArtifact.fromJson(artifact.toJson(), artifact.artifactId, f.artifact.base_model_identity);
assert.equal(loaded.layoutId, artifact.layoutId);
for (const [key, value] of [['max_values', true], ['max_values', 2], ['features', 2], ['kernel', {}]]) {
  const state = structuredClone(f.artifact);
  state.placements[0].recipe[key] = value;
  assert.throws(() => GeometryAdapterArtifact.createJson(JSON.stringify(state)));
}
console.log(JSON.stringify({schema:'spiraltorch.channel_affine_check.v1', forward_exact:true,
  vjp_exact:true, owned_snapshot:true, empty_rows:true, invalid_inputs_rejected:true,
  artifact_id:artifact.artifactId, layout_id:artifact.layoutId, hf_execution:false}));
loaded.free(); artifact.free(); emptyGrad.free(); empty.free(); gradient.free(); batch.free(); kernel.free();
