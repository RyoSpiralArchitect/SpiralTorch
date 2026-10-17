// Shared Node/browser checks against generated package exports, not a JS mock.
export function assert(value, message) {
  if (!value) throw Error(message);
}

export async function rejects(run, message, expectedError) {
  try { await run(); } catch (error) {
    if (expectedError) assert(expectedError.test(String(error)), `wrong rejection for ${message}: ${error}`);
    return;
  }
  throw Error(`accepted invalid input: ${message}`);
}

export async function cpuChecks(M, fixture) {
  const checks = [];
  const bad = async (name, run, expected) => { await rejects(run, name, expected); checks.push(name); };
  let coercions = 0;
  const trap = {toString() { coercions++; return fixture.request_json; }};
  for (const value of [null, true, 1, {}, trap, new String(fixture.request_json)]) {
    await bad('non-string study', () => M.ByteCorpusStudy.fromJson(value), /bounded primitive string/);
  }
  await bad('ASCII ingress budget', () => M.ByteCorpusStudy.fromJson(' '.repeat(64 * 1024 * 1024 + 1)), /bounded primitive string/);
  await bad('UTF8 ingress budget', () => M.ByteCorpusStudy.fromJson('\u3042'.repeat(24 * 1024 * 1024)), /exceeds WASM ingress budget/);
  for (const surrogate of ['\ud800', '\udfff', '\ud800x']) {
    // This is otherwise-valid JSON; lossy replacement must not admit a renamed case.
    const invalid = fixture.request_json.replace(fixture.case_names[0], fixture.case_names[0] + surrogate);
    await bad('unpaired surrogate', () => M.ByteCorpusStudy.fromJson(invalid), /unpaired UTF-16 surrogate/);
  }
  const study = M.ByteCorpusStudy.fromJson(fixture.request_json);
  const checkpoint = study.checkpointFromJson(fixture.checkpoint_json);
  const foreign = M.ByteCorpusStudy.fromJson(fixture.request_json + ' ');
  const foreignCheckpoint = /checkpoint schema, request identity, cursor or case count differs/;
  await bad('foreign checkpoint request', () => foreign.checkpointFromJson(fixture.checkpoint_json), foreignCheckpoint);
  foreign.free();
  const name = fixture.case_names[0];
  const artifact = study.caseArtifact(checkpoint, name, 'fixed_window_frozen.v1');
  study.validateArtifact(checkpoint, artifact);
  const json = artifact.toJson(), id = artifact.artifactId;
  const imported = M.ByteCorpusModelArtifact.fromJson(json, id);
  assert(imported.toJson() === json && imported.artifactId === id, 'opaque artifact bytes changed');
  assert(imported.attemptedRevision === String(checkpoint.completedUpdates), 'revision changed');
  checks.push('opaque artifact roundtrip', 'request/checkpoint binding');
  await bad('artifact whitespace tampering', () => M.ByteCorpusModelArtifact.fromJson(json + ' ', id));
  await bad('artifact wrong pin', () => M.ByteCorpusModelArtifact.fromJson(json, 'sha256:' + '0'.repeat(64)));
  await bad('coercible pin', () => M.ByteCorpusModelArtifact.fromJson(json, trap));
  await bad('unknown policy', () => study.caseArtifact(checkpoint, name, 'train'));
  await bad('unknown case', () => study.caseArtifact(checkpoint, 'absent', 'fixed_window_frozen.v1'));
  const gate = M.ByteCorpusServingGate.fromJson(fixture.serving.gate_json);
  assert(gate.toJson() === fixture.serving.gate_json, 'gate bytes changed');
  assert(gate.schemaVersion === 2 && gate.cohortNames.length >= 2, 'missing cohort API');
  await bad('gate JSON object', () => M.ByteCorpusServingGate.fromJson(JSON.parse(fixture.serving.gate_json)));
  if (typeof study.advanceWebGpu === 'function') {
    for (const value of ['1', true, 0.5, -1, NaN, Infinity, 2 ** 32 + 1, trap]) {
      const expected = typeof value === 'number'
        ? /stop_after must be a finite nonnegative u32 integer/
        : /stop_after must be a number/;
      await bad('strict study cursor', () => study.advanceWebGpu(value), expected);
    }
    await bad('cursor beyond total', () => study.advanceWebGpu(study.totalUpdates + 1),
      /stop cursor must lie between saved cursor and total updates/);
    const other = M.ByteCorpusStudy.fromJson(fixture.request_json + ' ');
    await bad('foreign typed resume', () => other.resumeWebGpu(checkpoint, checkpoint.completedUpdates), foreignCheckpoint);
    other.free();
  }
  assert(coercions === 0, 'FFI invoked a coercion hook');
  gate.free(); imported.free(); artifact.free(); checkpoint.free(); study.free();
  return checks;
}
