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
  receipt. These are logarithms of arithmetic means, not means of
  logarithms. All measurements refer to the same rows.
- `mass_inward_cosine` controls a different check: it can project the
  search direction and require non-increasing mean negative log pair
  mass on the **update batch**. Set it to `None` for direct conditional
  descent under the same roster floor. For accepted trials read
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

This is a Rust-only API. It does not require Torch and does not expose
a Python or WASM training entry point.
