const assert = require('node:assert/strict');
const artifact = require('node:path').resolve(process.argv[2]);
const {greedyTokenIdF32: id, greedyTokenDecisionJsonF32: report} = require(artifact);
const declarations = require('node:fs').readFileSync(artifact.replace(/\.js$/, '.d.ts'), 'utf8');
for (const [name, result] of [['greedyTokenIdF32','number'], ['greedyTokenDecisionJsonF32','string']]) {
  assert.ok(declarations.includes(`function ${name}(logits: Float32Array, history: Uint32Array, config_json: string): ${result};`));
}
const empty = new Uint32Array();
function check(scores, history, config, expected) {
  const logits = Float32Array.from(scores), ids = Uint32Array.from(history), cfg = JSON.stringify(config);
  const r = JSON.parse(report(logits, ids, cfg));
  assert.equal(id(logits, ids, cfg), expected);
  assert.equal(r.token_id, expected);
  assert.ok(r.probability > 0 && r.probability <= 1 && Number.isFinite(r.entropy));
  return r;
}
assert.equal(check([-0,0,0,-1], [], {top_k:2}, 0).probability, 0.5);
assert.equal(check([-15000,0,1], [], {top_p:0.01}, 2).candidates, 1);
assert.equal(check([1,0,1,1], [], {top_k:2,top_p:1}, 0).candidates, 2);
const z = {top_k:2,distribution:{kind:'zspace',config:{curvature:-1,repression_strength:10,last_token_repression:0}}};
const controlled = check([3,2,1], [0], z, 1);
assert.equal(controlled.raw_top_token, 0);
assert.ok(controlled.zspace_contract);
for (const fn of [id, report]) {
  for (const [x,h,c] of [
    [new Float64Array(2),empty,{}], [new Float32Array(2),[],{}], [new Float32Array(),empty,{}],
    [new Float32Array([1,NaN]),empty,{top_k:1}], [new Float32Array(2),Uint32Array.of(2),{}],
    [new Float32Array(2),empty,{selection:'sample'}], [new Float32Array(2),empty,{top_k:3}],
    [new Float32Array(1_048_577),empty,{}], [new Float32Array(2),new Uint32Array(16_385),{}],
  ]) assert.throws(() => fn(x,h,JSON.stringify(c)));
  assert.throws(() => fn(new Float32Array(2),empty,' '.repeat(65_537)));
}
console.log(JSON.stringify({plain_and_controlled_id_report_parity:true, invalid_inputs_rejected:true, typed_declarations:true, gpu:false}));
