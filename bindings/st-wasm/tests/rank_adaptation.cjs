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
    backend: "cpu",
  },
  scripts: ["u2: false;", "u2: true;"],
  policy: "ucb",
  seed: 17,
});

const first = session.choose();
assert.equal(first.contract_version, "spiraltorch.rank_adaptation.v1");
assert.equal(first.semantic_owner, "st-core::runtime::rank_adaptation");
assert.equal(first.execution_client, "wasm");
assert.equal(first.plan.execution_client, "wasm");
assert.equal(first.plan.requested_backend, "cpu");
assert.equal(first.selection_id, 1);
assert.equal(session.pendingSelectionId(), 1);
assert.throws(() => session.choose(), /still awaiting observation/);

const rejected = session.observe(first.selection_id, 1.25, false);
assert.equal(rejected.credited, false);
assert.equal(rejected.reward, null);
assert.equal(session.pendingSelectionId(), undefined);
assert.equal(
  Object.values(session.snapshot().observation_counts.rank_plan_variant).reduce(
    (sum, value) => sum + value,
    0,
  ),
  0,
);

const second = session.choose();
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
assert.throws(
  () => new wasm.RankAdaptationSession({
    rank_plan: { kind: "topk", rows: 2, cols: 256, k: 8, backend: "cpu" },
    scripts: ["u2: false;", " u2: false; "],
    seed: 1,
  }),
  /duplicates source/,
);

session.free();
console.log("rank adaptation wasm tests passed");
