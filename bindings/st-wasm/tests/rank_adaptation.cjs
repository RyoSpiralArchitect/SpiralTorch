const assert = require("node:assert/strict");

const modulePath = process.argv[2];
if (!modulePath) {
  throw new Error("usage: node rank_adaptation.cjs /path/to/spiraltorch_wasm.js");
}

const wasm = require(modulePath);
const session = new wasm.RankAdaptationSession({
  rank_plan: {
    kind: "topk",
    rows: 2,
    cols: 256,
    k: 8,
    backend: "wgpu",
  },
  scripts: ["u2: false;", "u2: true;"],
  policy: "ucb",
  seed: 17,
});

const first = session.choose();
assert.equal(first.contract_version, "spiraltorch.rank_adaptation.v1");
assert.equal(first.semantic_owner, "st-core::runtime::rank_adaptation");
assert.equal(first.execution_client, "wasm");
assert.equal(first.policy, "upper_confidence_bound");
assert.equal(first.decision.mode, "upper_confidence_bound");
assert.equal(first.decision.forced_exploration, true);
assert.equal(
  first.decision.arms.reduce((sum, arm) => sum + arm.selection_attempts, 0),
  1,
);
assert.equal(first.plan.execution_client, "wasm");
assert.equal(first.plan.requested_backend, "wgpu");
assert.match(first.execution_signature, /backend=wgpu/);
assert.equal(first.selection_id, 1);
assert.equal(session.pendingSelectionId(), 1);
assert.throws(() => session.choose(), /still awaiting observation/);

const rejected = session.observe(first.selection_id, 1.25, false);
assert.equal(rejected.credited, false);
assert.equal(rejected.candidate_quarantined, true);
assert.equal(rejected.reward, null);
assert.deepEqual(rejected.quarantined_choices.rank_plan_variant, [
  String(first.candidate_index),
]);
assert.equal(session.pendingSelectionId(), undefined);
assert.equal(
  Object.values(session.snapshot().observation_counts.rank_plan_variant).reduce(
    (sum, value) => sum + value,
    0,
  ),
  0,
);

const second = session.choose();
assert.notEqual(second.candidate_index, first.candidate_index);
assert.equal(second.decision.forced_exploration, true);
assert.equal(
  second.decision.arms.reduce((sum, arm) => sum + arm.selection_attempts, 0),
  2,
);
const credited = session.observe(second.selection_id, 4.0, true);
assert.equal(credited.credited, true);
assert.equal(credited.reward, 0.2);
assert.equal(
  Object.values(session.snapshot().observation_counts.rank_plan_variant).reduce(
    (sum, value) => sum + value,
    0,
  ),
  1,
);
assert.throws(() => session.abandon(second.selection_id), /no pending selection/);
const followUp = session.choose();
assert.equal(followUp.candidate_index, second.candidate_index);
assert.equal(followUp.decision.arms[first.candidate_index].quarantined, true);
session.abandon(followUp.selection_id);
assert.throws(
  () => new wasm.RankAdaptationSession({
    rank_plan: { kind: "topk", rows: 2, cols: 256, k: 8, backend: "wgpu" },
    scripts: ["u2: false;", " u2: false; "],
    seed: 1,
  }),
  /duplicates source/,
);

session.free();

const thompson = new wasm.RankAdaptationSession({
  rank_plan: {
    kind: "topk",
    rows: 2,
    cols: 256,
    k: 8,
    backend: "wgpu",
  },
  scripts: ["u2: false;", "u2: true;"],
  policy: "thompson_sampling",
  seed: Number.MAX_SAFE_INTEGER,
});
const sampled = thompson.choose();
assert.equal(sampled.decision.sampling_applied, true);
assert.match(sampled.decision.rng_stream_seed, /^[0-9]+$/);
assert.equal(thompson.pendingSelectionId(), sampled.selection_id);
thompson.abandon(sampled.selection_id);
assert.equal(thompson.pendingSelectionId(), undefined);
thompson.free();

console.log("rank adaptation wasm tests passed");
