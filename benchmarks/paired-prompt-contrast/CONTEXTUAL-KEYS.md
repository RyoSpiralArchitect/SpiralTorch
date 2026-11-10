# Does a contextual key improve binding?

**Not at the tested budget.** The second block's state-cue key depends on which
entity came before it, and replacing only that key changes the eventual answer
scores. But learning K in the second block did not outperform the first block,
and none of the 32 trained arms learned to select the correct state on this task.
This is one synthetic two-block decoder, not a pretrained-language or speed result.

## Using the learned projection for inference

[NativeCausalLmRankOneProjection](../../crates/st-nn/src/causal_lm/quartet_projection.rs)
can prefill a factual prefix once and answer different questions from independent
branches. The returned prefix owns its exact model snapshot, including geometry
and LoRA. Further optimizer steps do not invalidate an existing prefix.

```rust
let facts = projection.prefill(fact_tokens, max_cache_values)?;
let a = facts.branch(first_entity)?.branch(status_token)?;
let b = facts.branch(second_entity)?.branch(status_token)?;
let first_answer_logits = a.logits();  // full vocabulary, not a two-choice score
let second_answer_logits = b.logits();
```

`key_row(layer, position)` reads one past key in head order. For a controlled
intervention, `recipient.probe_key_from(&donor, layer, position, &suffix)`
returns the final suffix logits. The two prefixes must have the same projection
identity, length and token at the selected position. The ID binds the exact
base, adapters, readout policy, layer, site and both factors. Only the selected
past key is replaced when the next token is appended. Past values and prior
geometry history are not replaced; subsequent computations may change. No edited
cache escapes and neither prefix is mutated. No arithmetic is delegated to Python.
This is a Rust API, not a class in the old reference Python wheel.

## Experiment and result

We crossed two disjoint pools of known entity IDs, so none of these pairs appeared
in the previous binding tasks. Eight identities produced eight training pairs and
eight different within-fit-ID check pairs. Eight other, out-of-fit identities
produced eight transfer pairs. Each edge supplies both assignments, both fact
orders, and both questions. All 24 pairs have now been used. The two geometry
starts share a base and LoRA and are not independent seeds.

Both depths trained on the same native quartet **interaction loss**: 2 geometry
starts x 2 layers x Q/K x 2 fixed left factors x aligned/symmetric supervision.
Only 32 right-factor parameters trained. All 256 fixed-length steps were nonzero.
The frozen primary was aligned **second-block K, Walsh bit 2**. Count an edge win
when its mean of four order-averaged assignment/query margins beats the reference
by more than 5e-6; the stronger test requires all four margins to improve.

| Primary comparison | Initial / subset-rank: within-fit IDs | Initial / subset-rank: out-of-fit IDs |
| --- | --- | --- |
| Edge wins vs hold | 5/8, 5/8 | 5/8, 5/8 |
| Edge wins vs first-block K | 2/8, 2/8 | 0/8, 0/8 |
| Edge wins vs second-block Q | 5/8, 5/8 | 5/8, 4/8 |
| Edge wins vs symmetric second-block K | 3/8, 4/8 | 5/8, 5/8 |
| All four margins improve vs hold | 0/8, 0/8 | 0/8, 0/8 |

Neither factor passed the preregistered mean-margin or behavior gate (>=6/8
against each reference). Every arm and hold scored 32/64 binary choices and 0/16
completely ranked edge/order quartets on each non-fit split. In fact every prompt
preferred token 2 (`on`) to 16 (`standby`).

**What the key actually did:** for each of 64 within-fit-ID check prompts and
eight fixed snapshots, the donor prefix differed only in the entity *before*
the same first status cue. Replacement happened before reading the remaining
six tokens; no future inputs were supplied to the donor. First-block cue keys
were identical and substitution changed no output (512/512). Second-block
substitution moved the answer logit gap by >5e-6 in **512/512** cases, including
both untrained holds and label-symmetric controls. For the primary trained K
the average absolute gap movement was 0.000837 / 0.000804; for symmetric K it
was larger (0.001192 / 0.000981). The signed contribution to the shared binding
margin (factual minus substituted) was +0.000160 / +0.000167, versus
+0.000246 / +0.000241 already at hold. Sensitivity to one key therefore does
not establish that learning there produced better retrieval. This does not
ablate all cue keys, or show that an entire block is unnecessary.

**Post-hoc calibration, not a success gate:** [offset readback](key_depth_offset_readback.py)
fits one vocabulary-pair bias by binary logistic likelihood on the 64 fit rows only.
For primary K this makes 34/64 and 33/64 within-fit-ID choices correct, exactly
the same as calibrated hold; both remain 32/64 on the transfer split. Even a
label-informed split-specific threshold (not deployable) could score at most
34/64 and 35/64 within-fit, and 34/64 in transfer. A shared offset cannot reorder
the members of a quartet. A next learning test should include absolute answer
supervision and test the transmitted value/readout rather than assuming deeper
keys, additional bias or larger key motion will solve assignment.

## Evidence

All 34 snapshots' full-vocabulary ordinary Rust, independently reimported ordinary
wheel, and shared-prefix Rust logits were bit-identical on 6,528 rows in this run.
The protocol permitted a bounded numeric difference for cached inference; it did
not measure speed. Direct dense reference gradients also agreed on the first fit
batch in every lane. 196 Rust causal-LM tests and 148 Python procedure tests passed.
A first unit test incorrectly expected any distinct key to cause a distinct final
f32 logit; its red result was kept. A manual before-append edit now independently
checks exactly which key/value bytes and outputs a substitution produces.

Local receipts and raw rows: `2026-11-04-native-contextual-key-learning-v1`.
The predecessor executable and this experiment's executable were both retained.
No pretrained checkpoint, GPU, external ML framework, new wheel or publication.

| Artifact | SHA-256 |
| --- | --- |
| Prior plan | `113a0efec6e84807acdad593bf265004267577dfadc1a54bb8005b4f563b0f8b` |
| Prior readback | `6cde9819797df12802bb7b3fb6919635247eb2abe6f19609a9cc249e13b016a8` |
| Native executable | `304a1a1fe84ac72bbbb77e20fd970fd260ebd8a41044017c4d72c7c03d34832b` |
| Frozen protocol | `8595426d95733205cd9ec2e67d47bcba4ef7b7b0ae0ee387a6016098be4e602c` |
| Fit | `5d147ec8513ed10ad4754c369ab5b95dc54b27ea086a0069acff40d17e28e403` |
| Serving | `a22cbfd00a545e66bd71426999e1df77797ae3926e77726c88a05732a348b8af` |
| Key substitutions | `4b128416be4b9a2a25a6cdebf21cfc8d2dada25890c88a0fd7912f3e139105b8` |
| Model-free readback | `dda80723da2c4bded81eff4e50662d9044157e81d88b0ad99fa9202898e69c42` |
| Post-hoc offset readback | `0ae73105d65dd5da25cbea3b77ebc5820b1b3a929d6e894c46ffc0402fb37a9f` |

Build the offline Rust example with `cargo build --locked --offline --release -p st-nn --example native_quartet_projection`.
Use the installed prior CPU development wheel, Python 3.12 with `-I -B`, and
`SPIRALTORCH_BACKEND=cpu`. In a fresh private parent, run the [study](contextual_key_learning.py)
as follows, using each stage's printed SHA for its successor. Use an external
240-second wall bound for each physical phase. No retry or endpoint selection.

```text
freeze PROTOCOL PRIOR_DIR PRIOR_PLAN_SHA PRIOR_READBACK_SHA DRIVER
train  PROTOCOL PLAN_SHA DRIVER
serve  PROTOCOL PLAN_SHA DRIVER TRAIN_SHA
probe  PROTOCOL PLAN_SHA DRIVER TRAIN_SHA SERVE_SHA
audit  PROTOCOL PLAN_SHA TRAIN_SHA SERVE_SHA PROBE_SHA
```
