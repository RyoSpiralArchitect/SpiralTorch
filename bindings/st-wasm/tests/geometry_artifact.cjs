// Synthetic shared state; no training corpus/model acquisition or HF execution.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {GeometryAdapterArtifact} = require(path.resolve(process.argv[2]));
const input = fs.readFileSync(path.resolve(__dirname, '../../../crates/st-nn/tests/fixtures/geometry_adapter_artifact.json'), 'utf8');
const expected = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../crates/st-nn/tests/fixtures/geometry_adapter_artifact.expected.json'), 'utf8'));
const digest = text => 'sha256:' + createHash('sha256').update(text).digest('hex');
const base = JSON.parse(input).base_model_identity;
assert.equal(digest(input), expected.fixture_id);
const loaded = GeometryAdapterArtifact.fromJson(input, digest(input), base);
assert.equal(loaded.toJson(), input);
assert.equal(loaded.baseModelIdentity, base);
assert.equal(JSON.parse(loaded.stateJson()).placements.length, 13);
const issued = GeometryAdapterArtifact.createJson(input);
assert.equal(issued.artifactId, expected.canonical_artifact_id);
assert.equal(issued.layoutId, expected.layout_id);
assert.equal(issued.artifactId, digest(issued.toJson()));
const roundtrip = GeometryAdapterArtifact.fromJson(issued.toJson(), issued.artifactId, base);
assert.equal(roundtrip.toJson(), issued.toJson());
assert.equal(roundtrip.stateJson(), issued.stateJson());
assert.equal(roundtrip.layoutId, issued.layoutId);
const changed = JSON.parse(issued.toJson());
changed.placements[0].tensors[0].values[0] += 0.01;
const changedWeights = GeometryAdapterArtifact.createJson(JSON.stringify(changed));
assert.notEqual(changedWeights.artifactId, issued.artifactId);
assert.equal(changedWeights.layoutId, issued.layoutId);
changedWeights.free();
assert.throws(() => GeometryAdapterArtifact.fromJson(input + ' ', digest(input), base), /pinned identity/);
assert.throws(() => GeometryAdapterArtifact.fromJson(input, digest(input), 'sha256:' + 'b'.repeat(64)), /different base/);
for (const mutate of [
  state => state.policy = 'cached',
  state => state.placements[0].tensors[0].shape = [1, 2],
  state => state.placements[0].tensors[0].values[0] = 1e100,
  state => state.placements[1].path = state.placements[0].path,
  state => state.placements[2].tensors[2].requires_grad = true,
  state => state.placements[12].tensors[1].values[0] = 1000,
  state => state.placements[0].recipe.kernel.max_values = 2**32,
  state => state.placements[0].tensors[0].role = 'buffer',
  state => { state.placements[8].recipe.kernel.step = 1e-30; state.placements[8].tensors[1].values[0] = Math.log(2); },
  state => { state.placements[10].recipe.kernel.kernel_len = 256; state.placements[10].tensors[1].values[0] = Math.log(200); },
]) {
  const state = JSON.parse(input); mutate(state);
  assert.throws(() => GeometryAdapterArtifact.createJson(JSON.stringify(state)));
}
console.log(JSON.stringify({schema:'spiraltorch.geometry_adapter_portability_check.v1',
  fixture_id:digest(input), canonical_artifact_id:issued.artifactId, placements:13,
  layout_id:issued.layoutId, weights_independent_layout:true,
  pinned_load:true, canonical_roundtrip:true, invalid_records_rejected:true,
  hf_execution:false}));
roundtrip.free(); issued.free(); loaded.free();
