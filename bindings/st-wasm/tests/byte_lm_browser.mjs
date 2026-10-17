import {assert, rejects, cpuChecks} from './byte_lm_boundaries.mjs';
import * as M from '/package/spiraltorch_wasm.js';

const target = document.querySelector('#result');
const phase = new URLSearchParams(location.search).get('phase') ?? 'main';
const read = async path => {
  const response = await fetch(path, {cache: 'no-store'});
  if (!response.ok) throw Error(`missing ${path}`);
  return response.json();
};
const save = async (name, value) => {
  const response = await fetch(`/results/${name}.json`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)});
  if (!response.ok) throw Error(`save ${name}: ${response.status}`);
};
const digest = async value => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
  typeof value === 'string' ? new TextEncoder().encode(value) : value)), x => x.toString(16).padStart(2, '0')).join('');
const logitHash = values => {
  const buffer = new ArrayBuffer(values.length * 4), view = new DataView(buffer);
  values.forEach((value, i) => view.setFloat32(i * 4, value, true));
  return digest(buffer);
};
const summary = receipt => ({schema: receipt.schema, checks: receipt.count,
  models: receipt.models, training_updates: receipt.training?.completed_updates,
  training_cases: receipt.training?.cases.length, adapter_requests: receipt.adapter_requests,
  cohort_gate_accepted: receipt.assessment?.accepted,
  negative_control_rejected: receipt.failed_assessment?.accepted === false,
  scope: 'Actual packaged runtime qualification, not a language-quality claim'});

try {
  await M.default();
  const fixture = await read('/fixtures/input.json');
  if (phase === 'resume') {
    const saved = await read('/results/training.json');
    const study = M.ByteCorpusStudy.fromJson(fixture.learning_request_json);
    const partial = study.checkpointFromJson(saved.partial_checkpoint);
    const pending = study.resumeWebGpu(partial, study.totalUpdates);
    partial.free(); study.free();
    const resumed = await pending, checkpoint = resumed.checkpoint;
    assert(resumed.reportJson === saved.report_json, 'fresh-page resume report differs');
    assert(checkpoint.toJson() === saved.checkpoint_json, 'fresh-page resume checkpoint differs');
    const receipt = {schema: 'spiraltorch.packaged_byte_mlops.resume.v1',
      exact_report: true, exact_checkpoint: true, completed_updates: checkpoint.completedUpdates,
      checkpoint_sha256: await digest(checkpoint.toJson()), report_sha256: await digest(resumed.reportJson)};
    checkpoint.free(); resumed.free();
    await save('resume', receipt);
    target.textContent = JSON.stringify(receipt, null, 2); target.dataset.status = 'passed';
  } else {
    let requests = 0;
    const requestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter = (...args) => { requests++; return requestAdapter(...args); };
    const checks = await cpuChecks(M, fixture);
    assert(requests === 0, 'CPU preflight acquired a device');
    const record = async (name, fn) => { await fn(); checks.push(name); };
    const bad = (name, fn, expected) => record(name, () => rejects(fn, name, expected));
    const shape = JSON.parse(fixture.request_json).config;
    const study = M.ByteCorpusStudy.fromJson(fixture.request_json);
    const checkpoint = study.checkpointFromJson(fixture.checkpoint_json);
    const artifacts = fixture.case_names.map(name => study.caseArtifact(checkpoint, name, 'fixed_window_frozen.v1'));
    const models = [];
    for (const artifact of artifacts) {
      const id = artifact.artifactId, name = artifact.caseName;
      study.validateArtifact(checkpoint, artifact);
      const pending = artifact.restoreWebGpu();
      artifact.free();
      const inference = await pending;
      const windows = Uint8Array.from(fixture.windows);
      const forward = JSON.parse(await inference.forward(windows));
      assert(forward.artifact_id === id && forward.revision === String(checkpoint.completedUpdates)
        && forward.logits.length === shape.batch * shape.steps * 256
        && forward.logits.every(Number.isFinite) && Number.isFinite(forward.loss), 'invalid trained forward');
      let coercions = 0;
      const trap = {valueOf() { coercions++; return 1; }};
      for (const count of ['1', true, 1.5, -1, 0, 257, 2 ** 32 + 1, trap]) {
        await bad('strict generation count', () => inference.generateGreedy(Uint8Array.from(fixture.prompts), count));
      }
      for (const input of [fixture.windows, new Int8Array(windows), new Uint16Array(windows),
        {length: windows.length, [Symbol.toStringTag]: 'Uint8Array'}, windows.subarray(1)]) {
        await bad('strict byte window', () => inference.forward(input));
      }
      const detached = windows.slice(); structuredClone(detached.buffer, {transfer: [detached.buffer]});
      await bad('detached bytes', () => inference.forward(detached));
      assert(typeof SharedArrayBuffer === 'function', 'serve with COOP/COEP to test shared storage');
      await bad('shared bytes', () => inference.forward(new Uint8Array(new SharedArrayBuffer(windows.length))));
      const frame = document.createElement('iframe'); document.body.append(frame);
      const foreign = new frame.contentWindow.Uint8Array(windows);
      const cross = JSON.parse(await inference.forward(foreign)); frame.remove();
      assert(await logitHash(cross.logits) === await logitHash(forward.logits), 'cross-realm bytes changed');
      const shadowed = windows.slice();
      Object.defineProperty(shadowed, 'length', {value: 1});
      Object.defineProperty(shadowed, 'buffer', {get() { throw Error('shadowed buffer invoked'); }});
      const shadowForward = JSON.parse(await inference.forward(shadowed));
      assert(await logitHash(shadowForward.logits) === await logitHash(forward.logits)
        && shadowForward.loss === forward.loss, 'intrinsic byte snapshot changed');
      assert(coercions === 0, 'generation coerced a JS object');
      const prompt = Uint8Array.from(fixture.prompts);
      const generatedPending = inference.generateGreedy(prompt, 4);
      prompt.fill(0);
      await bad('overlapping inference', () => inference.forward(windows), /operation in progress/);
      inference.free();
      const generation = JSON.parse(await generatedPending);
      assert(generation.artifact_id === id && generation.generated_bytes.length === shape.batch
        && generation.prompt_sha256.length === shape.batch && generation.next_logits_f32le_sha256.length === 4
        && generation.generated_bytes.every(row => row.length === 4
          && row.every(byte => Number.isInteger(byte) && byte >= 0 && byte < 256)), 'freed-owner generation failed');
      const steps = fixture.prompts.length / generation.generated_bytes.length;
      for (let row = 0; row < generation.generated_bytes.length; row++) {
        const expectedPrompt = Uint8Array.from(fixture.prompts.slice(row * steps, (row + 1) * steps));
        assert(generation.prompt_sha256[row] === await digest(expectedPrompt), 'prompt was not snapshotted before the promise');
        const logits = forward.logits.slice(((row + 1) * steps - 1) * 256, (row + 1) * steps * 256);
        const greedy = logits.reduce((best, value, i) => value > logits[best] ? i : best, 0);
        assert(generation.generated_bytes[row][0] === greedy, 'first generated byte differs from actual full-forward argmax');
      }
      models.push({name, artifact_id: id, revision: forward.revision, loss: forward.loss,
        logits_f32le_sha256: await logitHash(forward.logits), generation_sha256: await digest(JSON.stringify(generation))});
    }
    checkpoint.free(); study.free();
    assert(new Set(models.map(m => m.logits_f32le_sha256)).size === models.length, 'geometric artifact outputs are not all distinct');
    checks.push('trained artifact inference and free-while-pending', 'learned/frozen/ordinary outputs remain distinct');
    const warmRequests = requests;

    const source = fixture.serving;
    const incumbent = M.ByteCorpusModelArtifact.fromJson(source.incumbent.json, source.incumbent.id);
    const candidateArtifact = M.ByteCorpusModelArtifact.fromJson(source.candidate.json, source.candidate.id);
    const gate = M.ByteCorpusServingGate.fromJson(source.gate_json);
    const pendingSlot = gate.bootstrapWebGpu(incumbent);
    gate.free(); incumbent.free();
    const slot = await pendingSlot;
    const original = JSON.parse(await slot.forward(Uint8Array.from(source.windows)));
    const bootstrap = JSON.parse(slot.bootstrapJson());
    const prepare = slot.prepareCandidate(candidateArtifact);
    await bad('busy owner getter', () => slot.generation, /operation in progress/);
    await bad('busy owner rollback', () => slot.rollback('0'), /operation in progress/);
    const permit = await prepare;
    const assessment = JSON.parse(permit.toJson());
    assert(permit.accepted && assessment.accepted, 'pinned mechanics candidate rejected; do not relax gate');
    assert(assessment.adapter.backend === 'BrowserWebGpu' && assessment.adapter.device_type !== 'Cpu', 'expected actual BrowserWebGpu runtime');
    assert(await logitHash(original.logits) === assessment.cohorts[0].incumbent.logits_f32le_sha256[0], 'live incumbent differs from measured owner');
    const stale = await slot.prepareCandidate(candidateArtifact);
    const busyForward = slot.forward(Uint8Array.from(source.windows));
    await bad('busy activation keeps permit', () => slot.activate(permit), /operation in progress/);
    assert(!permit.consumed, 'busy activation consumed permit');
    await busyForward;
    const gate2 = M.ByteCorpusServingGate.fromJson(source.gate_json);
    const initial2 = M.ByteCorpusModelArtifact.fromJson(source.incumbent.json, source.incumbent.id);
    const other = await gate2.bootstrapWebGpu(initial2); gate2.free(); initial2.free();
    const foreignPermit = await other.prepareCandidate(candidateArtifact);
    await bad('foreign slot permit', () => slot.activate(foreignPermit));
    assert(foreignPermit.consumed, 'failed foreign permit was not consumed'); foreignPermit.free(); other.free();
    await bad('JSON is not a permit', () => slot.activate(assessment));
    slot.activate(permit);
    assert(permit.consumed && slot.generation === '1', 'activation ownership failed');
    await bad('consumed permit', () => slot.activate(permit)); permit.free();
    const changed = JSON.parse(await slot.forward(Uint8Array.from(source.windows)));
    assert(changed.artifact_id === source.candidate.id && await logitHash(changed.logits) !== await logitHash(original.logits), 'activation did not select actual weights');
    assert(await logitHash(changed.logits) === assessment.cohorts[0].candidate.logits_f32le_sha256[0], 'live candidate differs from measured owner');
    await bad('async rejection releases lease', () => slot.prepareCandidate(candidateArtifact), /candidate is already active/);
    assert(slot.generation === '1' && slot.artifactId === source.candidate.id, 'async rejection lost owner');
    for (const value of [1, true, '01', '-1', '1.0', '18446744073709551616']) {
      await bad('strict generation identity', () => slot.rollback(value));
    }
    await bad('stale generation', () => slot.rollback('0'));
    slot.rollback('1');
    const rolled = JSON.parse(await slot.forward(Uint8Array.from(source.windows)));
    assert(rolled.artifact_id === source.incumbent.id && rolled.generation === '2'
      && await logitHash(rolled.logits) === await logitHash(original.logits) && rolled.loss === original.loss, 'rollback did not restore exact outputs');
    await bad('ABA permit', () => slot.activate(stale)); stale.free();
    await bad('rollback consumed', () => slot.rollback('2'));
    const rejectedRequest = JSON.parse(source.gate_json);
    rejectedRequest.cohorts[0].limits.max_mean_ce_regression = -100;
    const rejectedGate = M.ByteCorpusServingGate.fromJson(JSON.stringify(rejectedRequest));
    const initial3 = M.ByteCorpusModelArtifact.fromJson(source.incumbent.json, source.incumbent.id);
    const guarded = await rejectedGate.bootstrapWebGpu(initial3); rejectedGate.free(); initial3.free();
    const rejected = await guarded.prepareCandidate(candidateArtifact);
    assert(!rejected.accepted, 'negative control must fail');
    await bad('rejected candidate activation', () => guarded.activate(rejected));
    assert(guarded.artifactId === source.incumbent.id && rejected.consumed, 'failed candidate changed live owner');
    const failedAssessment = JSON.parse(rejected.toJson()); rejected.free(); guarded.free();
    const servingShape = slot.windowShape;
    const prompts = Uint8Array.from(source.windows.filter((_, i) => i % (servingShape[1] + 1) < servingShape[1]));
    const pendingGeneration = slot.generateGreedy(prompts, 4, '2');
    await bad('activation during generation', () => slot.prepareCandidate(candidateArtifact), /operation in progress/);
    candidateArtifact.free(); slot.free();
    const servedGeneration = JSON.parse(await pendingGeneration);
    assert(servedGeneration.serving.generation === '2' && servedGeneration.artifact_id === source.incumbent.id
      && servedGeneration.generated_bytes.length === servingShape[0]
      && servedGeneration.generated_bytes.every(row => row.length === 4), 'serving generation identity changed');
    assert(requests === warmRequests, 'packaged clients did not reuse the warm runtime');
    checks.push('measured cohort activation and exact rollback', 'warm runtime reuse', 'freed serving owner generation');

    target.textContent = 'Serving passed. Running actual packaged AdamW study and checkpoint capture...';
    const learning = M.ByteCorpusStudy.fromJson(fixture.learning_request_json);
    const learningRequest = JSON.parse(fixture.learning_request_json);
    const stop = 5, total = learning.totalUpdates;
    assert(total === learningRequest.train_batches.length && learningRequest.cases.length > 0, 'unexpected study dimensions');
    const partial = await learning.advanceWebGpu(stop), partialCheckpoint = partial.checkpoint;
    assert(partialCheckpoint.completedUpdates === stop, 'partial checkpoint did not reach stop');
    const partialJson = partialCheckpoint.toJson(); partialCheckpoint.free(); partial.free();
    const fullPromise = learning.advanceWebGpu(total); learning.free();
    const full = await fullPromise, fullCheckpoint = full.checkpoint;
    assert(fullCheckpoint.completedUpdates === total, 'full checkpoint did not reach stop');
    const fullJson = fullCheckpoint.toJson(), reportJson = full.reportJson;
    await save('training', {partial_checkpoint: partialJson, checkpoint_json: fullJson, report_json: reportJson});
    fullCheckpoint.free(); full.free();
    const report = JSON.parse(reportJson);
    assert(report.cases.length === learningRequest.cases.length && report.cases.every((c, i) =>
      c.name === learningRequest.cases[i].name && c.seed === learningRequest.cases[i].seed
      && c.training.length === total && c.training.every((p, j) => p.revision === j + 1 && Number.isFinite(p.ce))
      && c.validation.at(-1)?.revision === total), 'learning did not finish every requested case');
    const receipt = {schema: 'spiraltorch.packaged_byte_mlops.browser.v1', checks, count: checks.length,
      adapter_requests: requests, physical_hardware_verified: false, models, bootstrap, assessment,
      failed_assessment: failedAssessment, serving_generation_sha256: await digest(JSON.stringify(servedGeneration)),
      training: {request_sha256: await digest(fixture.learning_request_json), completed_updates: total,
        cases: report.cases.map(c => ({name: c.name, training: c.training, validation: c.validation})),
        partial_checkpoint_sha256: await digest(partialJson), checkpoint_sha256: await digest(fullJson), report_sha256: await digest(reportJson)}};
    await save('main', receipt);
    target.textContent = JSON.stringify(summary(receipt), null, 2); target.dataset.status = 'passed';
  }
} catch (error) {
  const failure = {schema: 'spiraltorch.packaged_byte_mlops.failure.v1', phase, error: String(error), stack: error?.stack};
  target.textContent = JSON.stringify(failure, null, 2); target.dataset.status = 'failed';
  await save(`failure-${phase}`, failure).catch(() => {});
}
