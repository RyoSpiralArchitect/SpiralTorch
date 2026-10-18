const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {CausalDocumentWindows} = require(require('node:path').resolve(process.argv[2]));
const fixture = JSON.parse(require('node:fs').readFileSync(require('node:path').resolve(
  __dirname, '../../../crates/st-nn/tests/fixtures/causal_document_windows.fixture.json'), 'utf8'));
const plan = CausalDocumentWindows.createJson(JSON.stringify(fixture.recipe));
assert.equal(plan.planId, fixture.plan_id);
assert.deepEqual(Array.from({length:plan.length}, (_,i) => Array.from(plan.window(i))), fixture.windows);
assert.equal(plan.length, 4);
assert.deepEqual(Array.from(plan.window(0)), [2,0,4]);
assert.deepEqual(Array.from(plan.window(3)), [3,8,4]);
assert.equal(plan.planId, 'sha256:' + createHash('sha256').update(plan.toJson()).digest('hex'));
assert.equal(JSON.parse(plan.toJson()).dropped_tokens, 4);
assert.equal(JSON.parse(plan.toJson()).target_tokens, 12);
for (const index of [-1, .5, NaN, Infinity, 4, 2**32, null, undefined, [], {}, true, false, '1']) {
  assert.throws(() => plan.window(index));
}
for (const recipe of [
  {document_lengths:[3], context_length:1},
  {document_lengths:[2**32-1], context_length:2},
  {document_lengths:[-1], context_length:4},
  {document_lengths:[4.5], context_length:4},
  {document_lengths:[4], context_length:true},
]) assert.throws(() => CausalDocumentWindows.createJson(JSON.stringify(recipe)));
const short = CausalDocumentWindows.createJson('{"document_lengths":[3,3],"context_length":4}');
assert.equal(short.length, 0);
console.log(JSON.stringify({schema:'spiraltorch.causal_windows_check.v1',plan_id:plan.planId,
  windows:plan.length, no_cross_document_packing:true, invalid_inputs_rejected:true}));
short.free(); plan.free();
