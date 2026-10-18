import * as M from '/package/spiraltorch_wasm.js';
import {assert, rejects} from './byte_lm_boundaries.mjs';

const target = document.querySelector('#result');
const phase = new URLSearchParams(location.search).get('phase') ?? 'main';
const read = async path => {
  const response = await fetch(path, {cache: 'no-store'});
  if (!response.ok) throw Error(`read ${path}: ${response.status}`);
  return response.json();
};
const save = async (name, value) => {
  const response = await fetch(`/results/${name}.json`, {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)});
  assert(response.ok, `save ${name}: ${response.status}`);
};
const hash = async text => Array.from(new Uint8Array(await crypto.subtle.digest(
  'SHA-256', new TextEncoder().encode(text))), n => n.toString(16).padStart(2, '0')).join('');

function checkLearning(request, report) {
  assert(report.cases.length === 3, 'expected ordinary / learned / frozen arms');
  for (let i = 0; i < request.cases.length; i++) {
    const spec = request.cases[i], row = report.cases[i];
    assert(spec.name === row.name && row.training.length === 2, 'case or update mismatch');
    let geometryChanged = false;
    spec.parameters.forEach((parameter, slot) => {
      const values = row.final_parameters[slot];
      assert(values.length === parameter.values.length && values.every(Number.isFinite), 'invalid parameter');
      const geometric = parameter.name.startsWith('geometry.');
      const frozen = geometric && spec.geometry_update === 'frozen';
      const same = values.every((v, k) => Object.is(v, parameter.values[k]));
      if (frozen) assert(same, 'frozen geometry changed');
      if (geometric && !same) geometryChanged = true;
      if (row.optimizer_state !== null) {
        const state = row.optimizer_state.slots[slot];
        assert(state.steps === (frozen ? 0 : 2), 'optimizer clock differs');
        if (frozen) assert([...state.first, ...state.second].every(v => Object.is(v, 0)), 'frozen moments changed');
      }
    });
    if (spec.geometry) assert(geometryChanged === (spec.geometry_update === 'train'), 'geometry did not follow update policy');
  }
}

try {
  await M.default();
  const fixture = await read('/fixtures/input.json');
  assert(fixture.requests.map(r => r.name).join(',') === 'sgd-v9,sgd-off,sgd-active,adamw-v9,adamw-off,adamw-active', 'unexpected study set');
  if (phase === 'resume') {
    const saved = await read('/results/training.json');
    const rows = [];
    for (const input of fixture.requests.filter(r => r.name.endsWith('-active'))) {
      const expected = saved.runs.find(r => r.name === input.name);
      const study = M.ByteCorpusStudy.fromJson(input.request_json);
      const checkpoint = study.checkpointFromJson(expected.partial_checkpoint);
      const pending = study.resumeWebGpu(checkpoint, study.totalUpdates);
      checkpoint.free(); study.free();
      const result = await pending, after = result.checkpoint;
      assert(result.reportJson === expected.report_json, 'fresh-page full report differs');
      assert(after.toJson() === expected.checkpoint_json, 'fresh-page checkpoint differs');
      rows.push({name: input.name, exact_report: true, exact_checkpoint: true,
        report_sha256: await hash(result.reportJson), checkpoint_sha256: await hash(after.toJson())});
      after.free(); result.free();
    }
    const receipt = {schema: 'spiraltorch.byte_clipping.browser_resume.v1', rows};
    await save('resume', receipt);
    target.textContent = JSON.stringify(receipt, null, 2);
  } else {
    const runs = [], rows = [];
    for (const input of fixture.requests) {
      const request = JSON.parse(input.request_json);
      const study = M.ByteCorpusStudy.fromJson(input.request_json);
      const policy = JSON.parse(study.gradientClipRecipeJson);
      assert(policy.kind === (request.gradient_clip?.kind ?? 'disabled.v1'), 'clipping getter differs');
      if (policy.kind !== 'disabled.v1') assert(policy.max_norm === Math.fround(.01), 'clipping limit differs');
      const result = await study.advanceWebGpu(study.totalUpdates), checkpoint = result.checkpoint;
      const report = JSON.parse(result.reportJson);
      if (request.gradient_clip) assert(JSON.stringify(report.gradient_clip) === study.gradientClipRecipeJson, 'report clipping differs');
      checkLearning(request, report);
      const record = {name: input.name, report_json: result.reportJson, checkpoint_json: checkpoint.toJson()};
      if (input.name.endsWith('-active')) {
        const partial = await study.advanceWebGpu(1), before = partial.checkpoint;
        record.partial_checkpoint = before.toJson();
        const offInput = fixture.requests.find(r => r.name === input.name.replace('-active', '-off'));
        const offStudy = M.ByteCorpusStudy.fromJson(offInput.request_json);
        await rejects(() => offStudy.checkpointFromJson(record.partial_checkpoint), 'changed clipping cannot resume');
        offStudy.free(); before.free(); partial.free();
      }
      runs.push(record);
      rows.push({name: input.name, request_sha256: await hash(input.request_json), gradient_clip: policy,
        report_sha256: await hash(record.report_json), checkpoint_sha256: await hash(record.checkpoint_json),
        adapter: report.adapter, geometry_updated_and_frozen: true,
        cases: report.cases.map(c => ({name: c.name, updates: c.training.length,
          initial_bpb: c.validation[0].bits_per_byte, final_bpb: c.validation.at(-1).bits_per_byte}))});
      checkpoint.free(); result.free(); study.free();
    }
    for (const optimizer of ['sgd', 'adamw']) {
      const cases = suffix => JSON.parse(runs.find(r => r.name === optimizer + suffix).report_json).cases;
      assert(JSON.stringify(cases('-v9')) === JSON.stringify(cases('-off')), 'v10 disabled changed v9 learning');
      assert(JSON.stringify(cases('-off')[0].final_parameters) !== JSON.stringify(cases('-active')[0].final_parameters), 'active clipping did not affect learning');
    }
    await save('training', {runs});
    const receipt = {schema: 'spiraltorch.byte_clipping.browser.v1', fixture_sha256: await hash(JSON.stringify(fixture)),
      legacy_disabled_exact: true, clipping_changes_learning: true, foreign_policy_rejected: true, rows,
      scope: 'Two-update synthetic mechanism qualification; no quality or speed claim'};
    await save('main', receipt);
    target.textContent = JSON.stringify(receipt, null, 2);
  }
  target.dataset.status = 'passed';
} catch (error) {
  target.dataset.status = 'failed'; target.textContent = String(error.stack ?? error);
  await save('failure-' + phase, {error: String(error.stack ?? error)}).catch(() => {});
}
