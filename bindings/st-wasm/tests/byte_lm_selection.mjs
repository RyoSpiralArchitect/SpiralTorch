import {assert, rejects} from './byte_lm_boundaries.mjs';

// All choices are made by st-nn. The fixture supplies independently expected
// indices and, for retained real runs, the exact Python artifact/decision bytes.
export async function selectionChecks(M, fixture) {
  assert(['synthetic_contract', 'retained_learning'].includes(fixture?.source), 'missing selection fixture provenance');
  assert(fixture.checkpoint_jsons.length >= 3 && Object.keys(fixture.expected_indices).length > 1,
    'selection fixture must cover multiple snapshots and cases');
  const checks = [], models = [];
  const bad = async (name, run, expected) => {
    await rejects(run, name, expected);
    checks.push(name);
  };
  const study = M.ByteCorpusStudy.fromJson(fixture.request_json);
  const checkpoints = fixture.checkpoint_jsons.map(raw => study.checkpointFromJson(raw));
  const candidates = study.checkpointCandidates();
  assert(candidates.length === 0 && candidates.capacity === 32, 'candidate bounds differ');
  const policy = 'fixed_window_frozen.v1';
  const names = Object.keys(fixture.expected_indices), name = names[0];
  await bad('empty selection', () => candidates.selectCheckpoint(name, policy), /1\.\.32/);
  for (const raw of fixture.checkpoint_jsons) candidates.appendJson(raw);
  assert(candidates.length === checkpoints.length, 'append lost a checkpoint');
  for (const caseName of names) {
    const index = fixture.expected_indices[caseName];
    const choice = candidates.selectCheckpoint(caseName, policy);
    const reportJson = choice.reportJson, report = JSON.parse(reportJson);
    const artifact = choice.artifact;
    const direct = study.caseArtifact(checkpoints[index], caseName, policy);
    assert(report.selected.candidate_index === index &&
      report.selected.completed_updates === checkpoints[index].completedUpdates,
    'selection differs from the expected saved checkpoint');
    assert(artifact.toJson() === direct.toJson() && artifact.artifactId === direct.artifactId,
      'selection changed saved weights, geometry or optimizer');
    study.validateArtifact(checkpoints[index], artifact);
    if (fixture.source === 'retained_learning') {
      assert(reportJson === fixture.expected[caseName].report_json, 'Python/WASM decision bytes differ');
      assert(artifact.toJson() === fixture.expected[caseName].artifact_json, 'Python/WASM artifact bytes differ');
    }
    choice.free();
    const imported = M.ByteCorpusModelArtifact.fromJson(artifact.toJson(), artifact.artifactId);
    assert(imported.toJson() === artifact.toJson(), 'selected artifact import changed bytes');
    assert(!reportJson.includes('"model_json"') && !reportJson.includes('"parameters"'),
      'selection report exposed model weights');
    models.push({case_name: caseName, candidate_index: index,
      completed_updates: report.selected.completed_updates, artifact_id: artifact.artifactId,
      exact_saved_artifact: true, exact_python_selection: fixture.source === 'retained_learning'});
    imported.free(); direct.free(); artifact.free();
    checks.push('saved artifact and owned selection result: ' + caseName);
  }
  let coercions = 0;
  const trap = {get length() { coercions++; return Number.MAX_SAFE_INTEGER; },
    *[Symbol.iterator]() { coercions++; while (true) yield checkpoints[0]; },
    toString() { coercions++; return name; }};
  for (const invalid of [null, [], checkpoints, trap]) {
    await bad('primitive case name only', () => candidates.selectCheckpoint(invalid, policy));
    await bad('opaque checkpoint JSON only', () => candidates.appendJson(invalid));
  }
  await bad('case coercion', () => candidates.selectCheckpoint(trap, policy));
  await bad('policy coercion', () => candidates.selectCheckpoint(name, trap));
  await bad('unknown case', () => candidates.selectCheckpoint('absent', policy));
  await bad('unknown policy', () => candidates.selectCheckpoint(name, 'train'));
  assert(coercions === 0, 'selection invoked a JS sequence or coercion hook');
  const foreign = M.ByteCorpusStudy.fromJson(fixture.request_json + ' ');
  const foreignCandidates = foreign.checkpointCandidates();
  await bad('foreign request import', () => foreignCandidates.appendJson(fixture.checkpoint_jsons[0]),
    /checkpoint schema, request identity, cursor or case count differs/);
  assert(foreignCandidates.length === 0, 'failed foreign import changed storage');
  foreignCandidates.free(); foreign.free();
  for (const indices of [[1, 0], [0, 0]]) {
    const invalid = study.checkpointCandidates();
    indices.forEach(index => invalid.appendJson(fixture.checkpoint_jsons[index]));
    await bad('chronology rejection', () => invalid.selectCheckpoint(name, policy),
      /increasing cursors and unchanged all-case history prefixes/);
    invalid.free();
  }
  const full = study.checkpointCandidates();
  for (let i = 0; i < full.capacity; i++) full.appendJson(fixture.checkpoint_jsons[0]);
  await bad('bounded append', () => full.appendJson(fixture.checkpoint_jsons[0]), /maximum 32/);
  assert(full.length === full.capacity, 'failed append changed the candidate set');
  full.free();
  const fork = JSON.parse(fixture.checkpoint_jsons.at(-1));
  fork.cases[1].validation[0].batch_losses[0] += 1;
  const divergent = study.checkpointCandidates();
  divergent.appendJson(fixture.checkpoint_jsons[0]); divergent.appendJson(JSON.stringify(fork));
  await bad('unselected case history fork', () => divergent.selectCheckpoint(name, policy),
    /increasing cursors and unchanged all-case history prefixes/);
  divergent.free();
  if (fixture.source === 'synthetic_contract') {
    const invalid = study.checkpointCandidates(); invalid.appendJson(fixture.non_boundary_json);
    await bad('non-evaluation snapshot', () => invalid.selectCheckpoint(name, policy),
      /saved evaluation-boundary model/);
    invalid.free();
    const sparse = study.checkpointCandidates();
    sparse.appendJson(fixture.checkpoint_jsons[0]); sparse.appendJson(fixture.checkpoint_jsons[2]);
    const selected = sparse.selectCheckpoint(name, policy);
    assert(JSON.parse(selected.reportJson).selected.completed_updates === 4, 'selected an unsaved minimum');
    selected.free(); sparse.free(); checks.push('unsaved minimum is ineligible');
  }
  const disposed = study.checkpointFromJson(fixture.checkpoint_jsons[0]); disposed.free();
  const guarded = study.checkpointCandidates();
  await bad('disposed handle is not JSON', () => guarded.appendJson(disposed), /bounded primitive string/);
  assert(guarded.length === 0, 'invalid disposed input poisoned collection');
  guarded.appendJson(fixture.checkpoint_jsons[0]);
  const afterRejection = guarded.selectCheckpoint(name, policy); afterRejection.free(); guarded.free();
  await bad('disposed collection', () => guarded.selectCheckpoint(name, policy), /null pointer/);
  const intrinsic = String.prototype.charCodeAt;
  let hookCalls = 0;
  try {
    String.prototype.charCodeAt = function() {
      hookCalls++; candidates.appendJson(fixture.checkpoint_jsons[0]);
      throw Error('reentrant string hook');
    };
    const unhooked = candidates.selectCheckpoint(name, policy); unhooked.free();
    const imported = study.checkpointCandidates();
    imported.appendJson(fixture.checkpoint_jsons[0]); imported.free();
  } finally { String.prototype.charCodeAt = intrinsic; }
  assert(hookCalls === 0, 'ingress invoked a mutable string prototype');
  checks.push('intrinsic string validation avoids reentrant borrow');
  const before = candidates.selectCheckpoint(name, policy);
  const expected = before.reportJson; before.free();
  checkpoints.forEach(checkpoint => checkpoint.free());
  study.free();
  const retained = candidates.selectCheckpoint(name, policy);
  assert(retained.reportJson === expected, 'freed study invalidated owned candidates');
  const artifact = retained.artifact, second = retained.artifact;
  candidates.free(); retained.free(); artifact.free();
  const imported = M.ByteCorpusModelArtifact.fromJson(second.toJson(), second.artifactId);
  assert(imported.toJson() === second.toJson(), 'freed parents invalidated selected artifact');
  imported.free(); second.free(); checks.push('request and artifact independent ownership');
  return {source: fixture.source, checks, count: checks.length, models};
}
