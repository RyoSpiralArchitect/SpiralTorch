# A finite-checked geometry update in WASM

Build the `st-wasm` web package with the `nn` feature, for example
`wasm-pack build bindings/st-wasm --target web --dev --features nn` from the repository root.
Serve the resulting `bindings/st-wasm` directory over loopback HTTP and open
[`tests/native_causal_lm.html`](native_causal_lm.html). For Node, run
`node bindings/st-wasm/tests/native_causal_lm_node.mjs bindings/st-wasm/pkg/spiraltorch_wasm.js`.
Both execute the same four artificial controls and compare against native Python results.

## Cached continuation

For a known sequence (for example, a document suffix or one answer candidate),
reuse a prefix rather than running the full decoder for every token:

```js
const prefix = model.prefill(Uint32Array.of(1, 2), 1024);
try {
  const branch = prefix.withSuffix(Uint32Array.of(3, 4, 5));
  try { console.log(branch.nextLogits(), branch.tokenCount); } finally { branch.free(); }
} finally { prefix.free(); }
```

The same operation is `prefix.with_suffix(&ids)?` in Rust and
`ctx.with_suffix(ids).logits().values()` in Python. In WASM, pass an ordinary,
non-shared `Uint32Array`; the explicit cache budget is at most 1,048,576
persistent f32 values per context, excluding transient storage and other forks.
The source remains usable after either success or failure. The child owns its
model and may outlive the JS handles for the source and model.

Only the last suffix position gets vocabulary logits. Rust applies the absolute
position offset and causal mask to all new queries; existing decoder keys and
values are copied into the fork, not reprojected. A causal geometry site may
still replay its retained input history. The attention budget charges the
rectangular query-by-key shape, including masked slots. A large chunk can
exceed it even if separate single-token calls fit. This API neither guesses
future tokens nor scores intermediate ones: extend by one when each next
token depends on the prediction. A successful chunk does not certify the
vocabulary logits or loss of any skipped intermediate row.

The browser and Node controls also include six nonconstant models (ordinary
and key-lag attention; bare, trained WaveGate and trained fractional history).
Their fixture was generated with the original full-prefix and one-token Python
interfaces using `native_causal_lm_fixture.py --cache OUTPUT`.

## Score complete answers

To compare answers after the same prompt, score their content **and the stop**
instead of reading just the last logits of a fork:

```js
const ctx = model.prefill(Uint32Array.of(1, 2), 1024);
try {
  // Two contents [1] and [1, 2], packed without the common stop ID (0).
  const r = JSON.parse(ctx.scoreLinesJson(Uint32Array.of(1, 1, 2), Uint32Array.of(1, 2), 0));
  console.log(r.scores.winner_index, r.scores.token_nlls, r.decoded_content_tokens);
} finally { ctx.free(); }
```

Rust: `ctx.score_lines(&[vec![1], vec![1, 2]], 0)?`. Python:
`ctx.score_lines([[1], [1, 2]], stop_id=0)`. The nested `scores` uses the
same scoring rules as the native decoder's padded `score_lines`: per-token
full-vocabulary NLLs, line totals, menu-conditional NLLs, accepted-set NLL,
and first winner on an exact tie. This is teacher forcing, not free generation.
Pass 1 to 32 distinct, nonempty, lexicographically sorted contents without
the stop ID. All contents plus their individual stops may total at most 4096
targets. Earlier positions already in the prompt are not scored.

Each distinct content prefix is decoded once; a single continuing branch
can read out many positions together. The root's retained row predicts the
first token. The stop is predicted, never fed. `decoded_content_tokens` and
`decoder_chunks` count physical content rows and suffix calls;
`scores.decoder_lanes` describes maximal candidate paths, not cached calls.
Each trie node still costs a **full-vocabulary scan**, including the root.
Rust reduces each projected row to its outgoing edge logits and the partition
of all other tokens, then drops the tile. It does not renormalize over answer
tokens alone. The cumulative model logit budget is unchanged.
`max_cache_values` also limits all simultaneously live child K/V and geometry
histories in a depth-first walk. `peak_fork_cache_values` excludes the supplied
root, frozen weights, token IDs, logits, and temporaries; it is not an RSS measure.
A rejected request does not fall back to padded execution or modify the root.

To further bound the largest new vocabulary output, pass a fourth argument
to `ctx.scoreLinesJson(content, lengths, stopId, maxTileValues)`, Python's
`max_tile_values=` keyword, or Rust's `ctx.score_lines_with_tile_limit(lines,
stop, Some(max_values))?`. Units are f32 **values**, not bytes or rows; the
default targets 262,144 values, subject to the model limit and at least one
full row. An explicit ceiling must fit one vocabulary row and not exceed the
model limit. It changes only projection tiling, not KV chunking or the
absolute/conditional score definitions. `scanned_logit_values` counts all
nodes including the borrowed root; `peak_projected_logit_values` is the
largest individual new head output, not RSS or a bound on backend temporaries.
`readout_batches` counts actual projections, and `retained_edge_logits`
counts distinct outgoing values kept for the final reverse pass.

The same six synthetic models are checked against every token cost from a
separately built native padded scorer and from individual full-prefix calls.
Use `native_cached_lines_fixture.py generate NEW_OUTPUT` with the reference
wheel, or `verify native_cached_lines_fixture.json` with the new development wheel.

## Train on complete answers

Give the decoder a trainable geometry artifact using `withGeometry` /
`with_geometry`. The following uses plain SGD on geometry only; it freezes the
base and any LoRA. The source model remains unchanged. In WASM:

```js
const g = model.geometryLineGradients(
  Uint32Array.of(5), Uint32Array.of(1, 1, 2, 3, 4), Uint32Array.of(1, 2, 2),
  0, JSON.stringify({kind: 'subset', accepted_indices: [0, 2]})
);
try {
  console.log(JSON.parse(g.reportJson()).loss, JSON.parse(g.trainingWorkJson()));
  const next = model.stepLineGeometrySgd(g, 0.01);
  try { console.log(next.nextLogits('[5]')); } finally { next.free(); }
} finally { g.free(); }
```

Likewise, with the native Python binding:

```python
g = model.geometry_line_subset_gradients([5], [[1], [1, 2], [3, 4]],
                                        accepted_indices=[0, 2], stop_id=0)
print(g.line_subset_report()["loss"], g.line_training_work())
```

Use `set` for absolute accepted-set NLL, `menu` with `gold_index` and `mix`
for the gold/ranking mixture, or `subset` as above. Each objective scores a
stop after each content line. The last optional WASM argument (Python:
`max_tile_values=`) is a vocabulary tile budget, not a vocabulary shortlist.
No held-out check or loss-decrease guarantee is implied by a plain SGD step.

### Choose which adapter learns

A composed decoder applies LoRA at the attention output projection and then the
geometry at the same site's output. Freezing an adapter keeps its transform in
the forward pass **and** in the derivative of the other adapter. One joint VJP
returns both parameter groups from one decoder forward/reverse traversal.

```python
import json

# geometry_model already owns trainable geometry at transformer.h.0.attn.
origin = geometry_model.with_lora(layer=0, rank=2, alpha=2.0)
spec = json.dumps({"kind": "subset", "accepted_indices": [0, 2]})
g = origin.line_adaptation_gradients([5], [[1], [1, 2], [3, 4]],
                                     stop_id=0, objective=spec, mode="joint")
next_model = origin.step_adaptation_sgd(g, mode="joint", learning_rate=0.01)
```

Use `lora` to freeze geometry, `geometry` to freeze LoRA, or `joint` for both.
LoRA-only does not require a geometry artifact. In Rust use
`spec.gradients_for(&model, prompt, lines, stop, cap, mode)?`. In JS use
`model.lineAdaptationGradients(prompt, flat, lengths, stop, specJson, mode, cap)`
and `model.stepLineAdaptationSgd(g, rate)`; the returned handle binds the mode.
`withLoraJson(payload, expectedId)` restores the canonical adapter pinned to its
base; use `model.with_lora_json(payload, expected_id=pin)` in Python.

## Resume complete-answer training

Use the same original adapted model and the same plan bytes on each
client. A [sample plan](native_line_session_plan.json) switches from set NLL to
a gold/menu mixture to an accepted subset, then repeats for the next epoch.

```python
from pathlib import Path
import spiraltorch as st

plan = Path("native_line_session_plan.json").read_bytes()
session = st.NativeCausalLmLineTrainingSession(origin, plan)
reports, pin = session.advance_to_checkpoint_new(2, "update-2.json")
# Retain the pin separately. A new process can re-create origin and continue.
again = st.NativeCausalLmLineTrainingSession.from_json(
    origin, plan, Path("update-2.json").read_bytes(), pin
)
step = again.step()  # update 3; returns None once exhausted
```

The same checkpoint is valid in Rust through
`NativeCausalLmLineTrainingSession::restore(origin, plan, payload, pin)`.
In the browser, keep the plan as text: parsing and reformatting it changes its pin.

```js
const plan = await (await fetch('./native_line_session_plan.json')).text();
const s = M.NativeCausalLmLineTrainingSession.fromJson(origin, plan, payload, pin);
try {
  const report = s.stepJson(); // JSON string or undefined on exhaustion
  const cut = s.checkpoint();
  try { console.log(report, cut.id); /* persist both cut.payload and cut.id */ }
  finally { cut.free(); }
} finally { s.free(); }
```

The v1 plan and its checkpoint remain geometry-only, including their original
byte pins. To select owners, change the plan schema to
`spiraltorch.native_causal_lm_line_plan.v2` and add a required `"mode": "joint"`
(or `"lora"` / `"geometry"`). Its v2 checkpoint contains the selected weights
and both initial/current adapter IDs, but no frozen weights or examples. A
different mode, frozen companion, LoRA layout, geometry recipe or arithmetic is
not a continuation. A hash checks the supplied bytes, not authorship or training.
Rust uses `step_adaptation()` and `advance_adaptation_to_checkpoint_new()` for
the owner-aware receipt; the original v1 `step()` and its non-null geometry IDs
are unchanged. Python and WASM dispatch either plan through the same session.

The [three-owner fixture](native_line_owners_fixture.py) compares six fixed
updates per mode from the same artificial model and round-trips an intermediate
checkpoint between CPython and WASM. Its losses are in-sample and measure
execution, not a pretrained language model's quality. See the [matched losses and numerical checks](../../../benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/LINE_OWNERS.md).

### Fixed parameter distance instead of a learning rate

For a v3 plan, keep `mode`, change the schema to
`spiraltorch.native_causal_lm_line_plan.v3`, **remove** the top-level
`learning_rate` and provide one explicit update policy:

```json
{"kind":"parameter_gradient_l2","target_l2":0.00390625}
```

Put this object under the `update` key. For an ordinary baseline the same key
accepts `{"kind":"sgd","learning_rate":0.125}`. A v3 checkpoint is pinned to
the particular plan bytes including its update rule; it cannot resume v1/v2,
another distance, or another rule. A positive unrepresentable step refuses
without advancing the cursor or replacing weights. Receipts for a distance step
have a `parameter_ray` with the realized f32 movement and each selected owner's share.

For a single step (no session or schedule):

```python
next_model, motion = origin.step_adaptation_gradient_at_distance(g, mode="joint", target_l2=1/256)
print(motion["realized_l2"], motion["lora"]["realized_l2"], motion["geometry"]["realized_l2"])
```

The same method in Rust returns `(model, receipt)`. In WASM use
`const step = model.stepLineGradientAtDistance(g, 1/256)`, read
`step.reportJson()` then consume `step.intoModel()` (or free the unused step).
The held parameter distance is Euclidean, not a distance on output probabilities.
The [matched six-arm comparison](../../../benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/LINE_OWNER_DISTANCE.md) gives
actual training losses and output changes. To capture its browser return add a
third output path to `native_line_session_capture.py DOM V1 V2 V3`; in Node
supply it after the v2 return path.

The demo download contains four artificial step-4 checkpoints from an actual
Python-to-WASM handoff. Resume the matching case, not an unadapted base or a
later version of its geometry. [Tests, limits and measured portability](../../../benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/LINE_SESSION.md).

### Match prediction changes instead of parameter distance

Set a v3 line plan's `update` to the following object. Python and WASM sessions
run the same Rust policy; no client-side optimizer or probability library is needed.

```json
{"kind":"anchor_forward_kl","target_kl":0.0000152587890625,"max_parameter_l2":0.125,"anchor_examples":[0,1,2]}
```

`anchor_examples` indexes distinct prompts in the plan. Each accepted update
uses 98% to 100% of the requested **mean** forward KL in nats,
`KL(next_token_before || next_token_after)`, on those full-vocabulary last
rows. Individual anchors can exceed the mean budget. Rust checks the ceiling
first and then at most 19 bisection trials along the same sealed gradient.
No match means no update or cursor advance; the ceiling is not a proof about
unexamined smaller distances. Each step exposes `anchor_kl.per_anchor_kl`,
`anchor_kl.trials`, the readout count and the accepted `parameter_ray`.
Maximum 16 prompts, 1,024 anchor token IDs and 1,048,576 reference logits.
Every trial costs one extra forward per anchor, not another backward.
Rust also exposes `NativeCausalLm::step_adaptation_at_anchor_kl` for explicit
anchor slices without a session. See the [three-owner comparison](../../../benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/LINE_ANCHOR_KL.md).

To protect each prompt as well as the mean, set the v3 update's optional
`criterion` (all quantities are full-vocabulary forward KL in nats):

```json
{"kind":"anchor_forward_kl","target_kl":0.0000152587890625,"max_parameter_l2":0.125,"anchor_examples":[0,1,2],"criterion":{"kind":"mean_with_cap","max_anchor_kl":0.000030517578125}}
```

Every prompt must satisfy the cap in addition to the existing mean band. For
a different control use `"criterion":{"kind":"peak"}` with `target_kl` set to
the desired maximum; it matches 98-100% of that peak, **not** of the mean.
Omitting the criterion preserves old plan pins, successful receipts and trajectories. Each
new trial includes `peak.kl` and `peak.anchor_index` (zero-based **bank**
position, first on a tie). A cap smaller than the mean floor is rejected at
plan admission. A changed criterion requires a new plan/checkpoint lineage.

For one scheduled update, use *either* the strict step or the attempt API:

```python
outcome = session.attempt_step()
if outcome is None:
    print("schedule exhausted")
elif outcome["status"] == "unmatched":
    failure = outcome["refusal"]
    print(outcome["next_update"], failure["reason"], failure["trials"][-1])
else:
    print(outcome["step"]["anchor_kl"]["mean_kl"])
```

WASM: `attemptStepJson()` returns the same tagged JSON, or `undefined` when
exhausted. Rust: `attempt_adaptation_step()`; for an explicit gradient and
anchor bank use `step_adaptation_at_anchor_kl_with_criterion`. The attempt
applies a successful step exactly once, and does not retry or alter the cursor
on a bounded miss. Other errors still propagate; strict `step()` in Python
and `stepJson()` in WASM still raise on a miss. See the [matched and stopped trajectories](../../../benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/LINE_ANCHOR_ENVELOPE.md).

### Experimental direction control

The default remains the unmodified loss gradient. In a v3 `anchor_forward_kl`
update, Python and WASM can opt into the same Rust implementation:

```json
{"kind":"secant_projection","probe_l2":0.00390625,"anchor":"largest"}
```

Put this object under `update.direction`; `"successor"` selects the next
anchor in cyclic bank order as a matched-cost control. From the origin, a
single uncommitted loss-gradient probe measures which prompt changed most.
For the chosen prompt the origin decoder pulls back the probe-minus-origin
probabilities, and removes the loss-gradient component along this vector in
trainable coordinates. This is a finite secant approximation, not an exact
Hessian or a guarantee that the next search succeeds. The KL acceptance
criterion, scalar trial limit and parameter ceiling are unchanged.

Each shaped attempt costs one extra bank of readouts, plus one checkpointed
origin forward and one seeded backward for the selected anchor. The readout
count includes the extra bank, not the backward. Inspect `direction` on
the match or refusal for selected bank indices, probe KLs and retained norm.
On an accepted step, `parameter_ray.gradient_dot_step` refers to the
**shaped** gradient; `anchor_kl.direction.source_loss_dot_step` measures the
original gradient against the actual displacement, not a new loss evaluation.
At least two distinct anchors, a positive probe at most the search ceiling,
and at most 1,048,576 selected trainable coordinates are required. An
unresolved normal, a retained fraction below 0.001, or a non-descent
displacement fails without advancing; it is not silently replaced by the raw
gradient. See the [synthetic comparison](../../../benchmarks/results/2026-11-23-native-wave-agreement/matched-rival/declared-set/LINE_SECANT.md)
before choosing this instead of the default.

## Finite-checked learning

```js
// M is the initialized web package. configJson and weights (Uint8Array)
// identify a local decoder. Pin the geometry JSON to both its digest and this base.
const base = M.NativeCausalLm.fromGpt2(configJson, weights);
const art = M.GeometryAdapterArtifact.fromJson(geometryJson, artifactId, base.baseModelIdentity);
const origin = base.withGeometry(art);
const batch = M.CausalLmDeclaredBatch.fromJson(JSON.stringify({
  ids: [2, 0], shape: [1, 2], contrasts: [[1, 0, [1]]],
}));
try {
  const step = origin.stepDeclaredSetGeometryAtDistance(batch, 1 / 65536);
  const receipt = JSON.parse(step.reportJson());
  const chosen = step.intoModel(); // consumes step; do not free it again
  try {
    console.log(receipt.accepted, chosen.nextLogits('[2]'));
    const saved = chosen.geometryArtifact();
    try { console.log(saved.artifactId, saved.toJson()); } finally { saved.free(); }
  } finally { chosen.free(); }
} finally { batch.free(); origin.free(); art.free(); base.free(); }
```

Every declaration is `[unshifted_flat_position, gold_id, sorted_rival_ids]`.
Optional unshifted `labels` use `-100` for ignored targets. Without a declaration
an active token contributes ordinary CE and zero coverage loss. Rust scores the
proposed weights on the **same training batch**: both mean `-log P(gold | set)`
and mean `-log P(set)` must decrease. A finite tie or regression returns the
origin; an infeasible direction throws without modifying it. `report.proposed`
can therefore describe weights **not** selected by `intoModel()`.

This is a small-model CPU client, not a WebGPU benchmark or a general HF loader.
Ingress accepts an ordinary, non-shared safetensors byte view up to 64 MiB;
total base parameters are limited to 8,388,608 elements. Each activation, attention,
logit and training-checkpoint budget is 1,048,576 elements. No tokenizer, model
download, Python, Torch or NumPy is required at runtime. IDs and geometry use
the same contracts as native Rust and Python.

To check a WASM-to-Python handoff, pass an unused output path as the Node
runner's final argument. With a native development wheel supporting the
declared-set APIs, run `native_causal_lm_fixture.py --verify-wasm FIXTURE_JSON
SELECTED_JSON` using Python. The same script with one unused output path
regenerates the tiny fixture from deterministic weights; it never opens a pretrained model.
