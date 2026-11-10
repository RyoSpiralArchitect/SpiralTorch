# Full-vocabulary answer learning with an optional binding objective

**Learning the answer vocabulary worked. Learning which answer belongs to which
question did not yet meet the behavioral gate.** To isolate the objective we
trained only the right factor (32 coordinates) of a first-block **value** projection
with one fixed left direction. This is not a K-versus-V comparison or pretrained GPT-2.

## Native API

[The Rust implementation](../../crates/st-nn/src/causal_lm/quartet_projection/answer.rs)
takes a prefix and a separately supplied next-token ID for each example. CE covers
the full vocabulary. The label is never appended as an input. Ordinary CE permits
repeated prompts with different labels. Optional quartet rows index unique prompts;
their orientation has no effect on the CE labels or weights.

```rust
let report = projection.step_answers(&prompts, &gold_ids, None, 0.08)?;
let binding = CausalLmAnswerBinding {
    ids: [standby_id, on_id], quartets: &groups, mix: 0.5,
};
let before = projection.answer_gradients(&prompts, &gold_ids, Some(binding))?;
let step = projection.step_answers(&prompts, &gold_ids, Some(binding), 0.08)?;
```

The optional objective is `(1-mix) * mean_answer_CE + mix * mean_quartet_loss`.
It uses one factual forward per prompt, with a logit-softmax CE seed and a separate
named-logit contrast seed on the same immutable checkpoints. The compact parameter
gradients are combined in f64. Both are measured even at mix 0/1 if binding was
supplied; with `None` the ordinary path uses just one reverse pass. A step is
normalized descent, not an unconstrained learning rate. The offline Rust example
can resume from exact right-factor bits bound to the original projection ID.

## Prospective result

One partly structured, two-block synthetic decoder; two related frozen geometry
starts and two training orders, not four independent seeds. We crossed opposite
halves of the prior identity pools. Eight fresh fit pairs, eight fresh pairs
of the same identities, and eight out-of-fit identity pairs were mutually disjoint
and disjoint from the prior 24 cross-pool pairs. All 24 new pairs are consumed.
They supply 64 prompts per split. These are not general language tasks.

Each of 24 lanes took 64 steps of length 0.08. Controls: CE only, interaction only,
50/50 aligned or reflected quartets, and train-scaled aligned or reflected quartets.
All had the same prompts, labels, steps and both reverse passes. The train-scaled
weight was fixed once from both fit batches at the origin, not tuned on the check.
CE gradient norms were 1.740 / 1.763, interaction norms 0.000996 / 0.001035, cosines
0.034 / 0.059. Equalizing their initial contributions required interaction weights
0.999428 / 0.999414. The symmetric control used the very same coefficient.

| Within-fit identities, AB order | Gold NLL (initial / subset-rank; lower is better) | Full-vocabulary top-1 / 64 | Two-way correct / 64 |
| --- | ---: | ---: | ---: |
| Hold | 4.252 / 4.284 | 1 / 1 | 32 / 32 |
| Answer CE | 1.044 / 1.056 | 32 / 32 | 32 / 32 |
| 50/50 aligned | 1.044 / 1.056 | 32 / 32 | 32 / 32 |
| Train-scaled aligned | 5.733 / 5.611 | 8 / 8 | 32 / 32 |
| Train-scaled symmetric | 1.041 / 1.052 | 32 / 32 | 32 / 32 |

At **both starts and both orders**, scaled aligned interaction improved the shared
binding margin on 8/8 familiar-ID check edges vs both CE and symmetric control, but
on only 4-5/8 out-of-fit edges. The 50/50 treatment did not pass the shared-margin
gate. Neither mixture passed the preregistered behavior gate anywhere. No learned
endpoint strictly ranked both positive cells above both negatives in a check quartet.
CE-only reached full-vocabulary 26/64 on the out-of-fit split, with two-way choice
still at 32/64. Occasional out-of-fit binary gains for scaled interaction were at
most two questions, not reliable transfer.

[Post-hoc loss decomposition](answer_loss_breakdown.py), using only sealed rows:
`gold NLL = -log(p(on) + p(standby)) - log(p(gold) / (p(on) + p(standby)))`.
On the familiar-ID split CE raised average mass on those two IDs from 6.2% / 7.4%
at hold to 78.5-79.3% across starts/orders. Conditional NLL also improved
(AB: 1.371 to 0.800 / 1.582 to 0.817); it is not true that *only* vocabulary
selection improved. But the conditional ranking still failed. In contrast, scaled
interaction produced only 11.9% / 12.1% mean answer-set mass in AB.

The frozen weight equalized gradients only at the origin. In the 64 recorded
scaled-aligned steps, the effective interaction/CE gradient norm ratio rose as high
as 10.6-10.8. At 37-39 steps the chosen descent direction actually pointed uphill
for **local, pre-update minibatch CE**. This is an analytic sign, not a count of
observed finite loss increases. A useful next control is to keep CE as the protected
objective and accept an interaction correction only after checking the actual
candidate CE. Do not select a checkpoint using these check or transfer pairs.

## Evidence and replay

Data directory: `2026-11-04-native-answer-quartet-learning-v1`. The previous
CPU wheel independently constructed CE adjoints by summing the old native dense
`logit[k] - logit[gold]` gradients over all 63 non-gold IDs, weighted by their
full softmax probabilities (8,064 reference pullbacks). Both components agreed
with the Rust observation. The 1,536 study updates completed. One preselected lane
additionally resumed after step 32: all 32 subsequent reports, factors and final
owner matched the uninterrupted lane exactly. Ordinary Rust, prefix-cached Rust
and independently merged wheel logits were bit-identical on all 4,992 endpoint
rows. 197 Rust causal-LM and 154 Python procedure tests passed.

The original **offline scoring** failed looking up `fit_scale_control` instead of
the recorded `fit_control`. Its receipt is retained. A versioned readback verified
that translating exactly one source line back reproduces the original pinned source,
then recomputed from the same immutable rows into a new directory. No new training,
inference, gate adjustment or model selection took place. Fresh runs include the
fix and a complete two-control readback regression test.

| Artifact | SHA-256 |
| --- | --- |
| Previous plan | `8595426d95733205cd9ec2e67d47bcba4ef7b7b0ae0ee387a6016098be4e602c` |
| Previous readback | `dda80723da2c4bded81eff4e50662d9044157e81d88b0ad99fa9202898e69c42` |
| Rust driver used | `51fdc27a6837055a94618c1f3c411190ebcca2e689e1d2f15047a575c62cad63` |
| Frozen plan | `d5e38904611f8cf2e07e17dfae30dec6d8410bbf8115483cbb3dc2cc9a09d133` |
| Calibration | `308e60f9b3240670c3e457060d7d60c41bc1467b0896670b13ff06e3f659914c` |
| Training | `dbeb567622500d3795a8c6231e3f4243bc8cec7222abe344a0868b39b6886240` |
| Raw logits inventory | `1156c96f2a598e3f552187f54b6e99bfc2465c7f0696defe7840d8635d53be0a` |
| Repaired model-free readback | `06820bf153a58efbf812f0d5578462baf06c6e96c34503f1932c90112dc061ac` |
| Post-hoc loss components | `9ad7126a7e81281a3d927ca529e4673c56c1cd54da3896b680fc3f6ab7762690` |

Build with `cargo build --locked --offline --release -p st-nn --example native_quartet_projection`.
Use the pinned prior CPU wheel, Python 3.12 with `-I -B`, `SPIRALTORCH_BACKEND=cpu`,
and a fresh private output parent. Run the [study](answer_quartet_learning.py) with
each printed SHA as the successor's input, using an external 240-second wall
bound per physical phase. No GPU, wheel rebuild, publishing or real-model evaluation.

```text
freeze    PROTOCOL PRIOR_DIR PRIOR_PLAN_SHA PRIOR_READBACK_SHA DRIVER
calibrate PROTOCOL PLAN_SHA DRIVER
train     PROTOCOL PLAN_SHA DRIVER CALIBRATION_SHA
serve     PROTOCOL PLAN_SHA DRIVER TRAIN_SHA
audit     PROTOCOL PLAN_SHA CALIBRATION_SHA TRAIN_SHA SERVE_SHA
```
