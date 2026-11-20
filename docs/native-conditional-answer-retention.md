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
  receipt. When a nearby positive allowance leaves a subnormal
  difference, use the represented residual for a normal baseline or
  when the logs of a subnormal baseline and allowance cannot be
  distinguished reliably. Otherwise keep the log-space difference.
  If `log_outside` is present, compare
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
  revalidates the rows the guard will evaluate against the receiving
  model's work limits: sign anchors only for a sign-only guard, or the
  whole roster for a mass guard. There is no guarantee on new prompts,
  full-vocabulary top-1, or throughput.

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

The checkpoint binds the guard and projection, not a training-data cursor
or schedule. It contains factors, summary statistics, and a roster digest,
not prompt rows or cached activations; treat it as training data rather than
as a redacted export. The v3 format does not load older research-only
checkpoints. A v3 checkpoint that predates the log-outside field is
rejected if rebuilding the roster requires that field. Likewise,
an old v3 checkpoint that omitted or recorded a different active log
pair floor after near cancellation is rejected on restore. A
different baseline or schema is an error, not permission to start a
new guard at the saved endpoint.

The [Python binding](python/native-causal-lm.md#guarded-rank-one-training-and-restart)
calls this same implementation without Torch. There is no WASM training
binding for this projection.
