# Conditional answer retention in Rust

Use `NativeCausalLmRankOneProjection` when the trainable parameter is a
right vector at one GPT-2 projection site. The left axis and base model
stay fixed. Each row supplies a prompt and a pair `[gold, alternative]`
of distinct vocabulary IDs.

`step_conditional_answers` reduces the gold-token cross-entropy
**conditional on the pair**. It does not by itself prevent the rest of
the vocabulary from taking probability away from both candidates.
Use a separate training roster to protect that second quantity:

```rust
use st_nn::causal_lm::{
    CausalLmConditionalAnswerConfig, CausalLmResult, CausalLmRetainedConditionalAnswerStep,
    NativeCausalLmRankOneProjection,
};

fn train(
    projection: &mut NativeCausalLmRankOneProjection,
    guard_prompts: &[Vec<usize>],
    guard_pairs: &[[usize; 2]],
    batches: &[(Vec<Vec<usize>>, Vec<[usize; 2]>)],
) -> CausalLmResult<Vec<CausalLmRetainedConditionalAnswerStep>> {
    let mut guard = projection.prepare_conditional_margin_and_mass_retention(
        guard_prompts,
        guard_pairs,
        0.01,
    )?;
    let config = CausalLmConditionalAnswerConfig {
        length: 0.04,
        max_trials: 2,
        mass_inward_cosine: Some(0.05),
    };
    batches.iter().map(|(prompts, pairs)| {
        projection.step_conditional_answers_retaining(prompts, pairs, config, &mut guard)
    }).collect()
}
```

- The roster must have at least one initially positive gold-minus-rival
  margin. Every such margin must remain strictly positive after each
  commit. Initially incorrect rows do not become permanent sign anchors.
- If prepared with mass retention, the arithmetic mean of
  `P(gold) + P(alternative)` over **all** guard rows also has a fixed
  floor. The absolute allowance is subtracted once from the starting
  mean, so it cannot compound across steps. The sign-only constructor
  skips this mass measurement.
- For a floor at or above 0.5 the comparison uses the equivalent
  ceiling on the mean probability of all other tokens. Rust computes
  this probability directly from the mass loss, rather than
  subtracting a rounded pair probability from one. In a mass check,
  `outside` is present when the floor rounds to 1 or when the rounded
  pair-mass fields alone would give a different verdict. When present,
  compare `outside.mean_outside_mass` against
  `outside.maximum_mean_outside_mass`; otherwise the pair fields
  reproduce the verdict unless `log_pair` is present. A positive floor
  below the normal f64 probability range is compared in log space;
  use `log_mean_pair_mass >= minimum_log_mean_pair_mass` in that
  receipt. If `log_outside` is present, compare
  `log_mean_outside_mass <= maximum_log_mean_outside_mass` instead.
  This handles a positive outside ceiling below the normal f64 range.
  These are logarithms of arithmetic means, not means of logarithms.
  All measurements refer to the same rows.
- `mass_inward_cosine` controls a different check: it can project the
  search direction and require non-increasing mean negative log pair
  mass on the **update batch**. The optional `log_candidate_mass_nll`
  reports the log of that mean loss when it is positive but subnormal.
  Set `mass_inward_cosine` to `None` for direct conditional descent
  under the same roster floor. For accepted trials read
  `step.committed`; inspected trials expose their roster checks in
  `retention`, including the mass and its floor when enabled.
- Keep the guard with that projection trajectory. A hold preserves
  the projection and leaves the guard usable; a commit advances its
  owner without resetting the floor. Do not carry the guard to other
  weights or use held-out evaluation prompts for it. Reuse also
  revalidates the roster against the receiving model's work limits. There is no
  guarantee on new prompts, full-vocabulary top-1, or throughput.

Each call commits at most one step. A later error does not roll back
earlier calls in the loop; record each returned step as it completes
when you need a durable per-batch trace.

## Restart without resetting the guard

Call `projection.checkpoint_with_retention(&guard)` to get bytes and a
SHA-256 pin. Save both, and keep the expected pin independently of the
checkpoint being verified. Use the original frozen decoder, including its
LoRA, geometry, and readout policy:

```rust
let plan = NativeCausalLmRankOneProjection::admit_retained_checkpoint(
    &source, &bytes, &expected_pin,
)?;
let (mut projection, mut guard) = plan.restore(guard_prompts, guard_pairs)?;
```

Admission checks the size, pin, format and both projection identities
without any model evaluation. Restoration requires the exact original
ordered guard roster, checks its work budget, and replays it at the
original factors. Saving also checks that the entire original roster,
including non-anchors, fits the current model's work budget. The signed
margins, mean mass floor, outside ceiling,
and active log-space floor must reproduce bit-for-bit. The saved endpoint
must pass that reconstructed guard before it is returned.

This projection checkpoint binds the guard and projection, not a training-data
cursor or schedule. It contains factors, summary statistics, and a roster digest,
not prompt rows or cached activations; treat it as training data rather than
as a redacted export. The v3 format does not load older research-only
checkpoints. A v3 checkpoint that predates the log-outside field is
rejected if rebuilding the roster requires that field. A different
baseline or schema is an error, not permission to start a new guard
at the saved endpoint.

For a fixed ordered schedule, use `NativeCausalLmConditionalAnswerSession`
to bind those weights and the guard to a batch cursor. It supports repeated
epochs without expanding the schedule or rebasing the guard. A session
checkpoint embeds the pinned v3 projection checkpoint and adds the schedule
digest, policy, repeat count, cursor, accepted-update count, and starting
projection. Legacy v3 and v4 session checkpoints lack the starting projection
but are still readable.
Older v3/v4 sessions may contain a retention record without the log
outside-mass ceiling at subnormal limits. Only legacy session resume
recomputes a missing ceiling, requires every recorded guard field to
reproduce and checks the saved endpoint using the recomputed guard.
New sessions and direct retention restore still require an exact baseline.
`prepare_resume` checks the outer and inner pins before model evaluation.
Resume requires the exact original pool, ordered batches, and guard roster.
`advance_to_checkpoint_new` stages at most 1,024 positions and only advances
the live session after successful no-clobber publication. If a complete file
was installed but cleanup or parent-directory sync failed, the live session
remains at the previous position. The
`ConditionalCheckpointPublicationUncertain` error carries the staged pin.
Do not overwrite the path or trust its durability. If the file remains
readable, verify it against that pin and explicitly resume from it only if
choosing that checkpoint over the prior live state.

An exhausted session can start a new pool and schedule with `next_phase`.
The child keeps the exact weights and original guard and starts its own
counters at zero. Its checkpoint pins both the starting projection and
the immediate predecessor's digest. The digest links the
checkpoints but does not prove the predecessor's training history.
An unchanged restored session emits the exact admitted bytes, including
its original header encoding. Consuming a batch replaces that checkpoint
identity, even when the candidate update was refused.

`serving_from_checkpoint` reconstructs the pinned projection without the
prompt pool, original guard roster, or training workspace. It checks the
outer and inner pins and model recipe; it does not replay the guard.
Use `prepare_resume` with the original inputs to resume training.

The explicit read-only `serving_from_legacy_checkpoint` also admits archived
repeated v2 and family v4 sessions with a v2 retention record. It checks
both pins and both projection identities against the exact source decoder.
Linked v4 phases use a different format. No checkpoint is converted, no
guard is replayed, and these archived formats cannot resume training.

The [Python binding](python/native-causal-lm.md#guarded-rank-one-training-and-restart)
calls this same implementation without Torch. There is no WASM training
binding for this projection.
