# Unpadded Answer QKV Updates

The [transfer study](KEY-LAG-TRANSFER.md) exposed a length/position sensitivity,
including on already-trained identity pairs. The next training experiment
needs variable-length inputs without silently changing how examples are
weighted. This change supplies that native Rust path. It does **not** establish
an improvement in language quality or repair the failed transfer gates.

## Contract

`NativeCausalLmQkvAdamW::answer_sequence_gradients(prompts, gold, max_rows)`
computes full-vocabulary CE at each prefix's final position and its QKV weight
derivative. Each nonempty prefix has **one answer and one vote**, regardless
of its length. Gold labels never become model inputs.

`step_answer_sequences` collects the whole logical batch's mean gradient and
commits one AdamW update. The implementation shares the compensated f64
sample-weighted reducer with `answer_gradients_microbatched`, and reuses the
native document-window helper to group only adjacent equal-length inputs.

- No padding, truncation, EOS insertion, cross-document concatenation or sorting.
- Mean gradients are weighted by the actual number of sequences in each group,
  including short final groups; they are not averaged as equal-sized groups.
- Admission checks every prefix, label and actual group shape before any forward
  pass. Existing per-group activation, geometry and training-work limits remain.
- At most 4,096 sequences, 1,024 groups and the existing logical QKV-work bound
  of `2^30` are admitted. `max_rows` must be nonzero; it is a ceiling, not padding.
- A failed update leaves weights, optimizer moments, clock and serving identity
  unchanged. Zero learning rate retains the existing freeze policy.
- Checkpoint format and serving ownership are unchanged. The caller must retain
  the input order, grouping limit, labels and schedule for exact continuation;
  the optimizer checkpoint alone does not record the next training batch.

This is not all-token language-model loss: the existing shifted-target geometry
accumulator weights **target tokens**, whereas this answer-only API weights
**sequences with one supervised answer**. They must not be interchanged.

Given a loaded model whose vocabulary contains these IDs:

```rust
use st_kernel_contracts::adamw::{AdamWConfig, AdamWStep};
use st_nn::causal_lm::{CausalLmResult, NativeCausalLm, NativeCausalLmQkvAdamW};

fn example(model: &NativeCausalLm) -> CausalLmResult<(Vec<u8>, String)> {
    let mut train = NativeCausalLmQkvAdamW::new(model, 0, AdamWConfig::default())?;
    let short = [1, 3, 8];
    let long = [1, 5, 9, 4, 7];
    let prefixes = [&short[..], &long[..]];
    train.step_answer_sequences(
        &prefixes, &[2, 6], 2, AdamWStep::new(0.01, 0.0)?
    )?;
    train.checkpoint()
}
```

The offline `native_key_lag_fit` example accepts
`"batch_kind": "unpadded_sequences"` in `fit` and `gradient` actions.
Omitting it preserves the legacy equal-length admission policy; variable
lengths are never enabled by an implicit fallback. The example keeps its
smaller 128-sequence / 512-position input limits. The model's own limits can
be stricter. The new API is Rust-native; no Python wheel or WASM exposure is
claimed here.

## Verification, 2026-11-10

209 native decoder tests and two example tests pass. New cases cover
per-answer versus per-prefix-token weighting, equal-length compatibility,
ordinary and lagged attention with frozen LoRA/geometry, uneven groups,
single-clock updates, zero-rate freezing, exact resume, stale prefix snapshots,
invalid late inputs, roster bounds and the unchanged geometry budget.

A separate CPU witness uses the saved synthetic two-block, width-32,
64-vocabulary model under lag 0 and lag 1. Only the first eight existing fit
questions are used, with lengths 9/11/13. It does not read transfer/held-out
questions or pretrained GPT-2 weights. All checks and tolerances were fixed
before execution; 82 native calls and 20 small optimizer updates completed
without retries.

| Check | Ordinary | Lag 1 |
| --- | ---: | ---: |
| Maximum gradient error vs old singleton mean, cap 1 | 2.98e-8 | 4.47e-8 |
| Maximum gradient error vs old singleton mean, cap 2 | 5.77e-8 | 8.99e-8 |
| Independent full-vocabulary NLL error | 3.07e-8 | 8.71e-9 |
| Old/new fixed-length gradients and two-step checkpoints | Exact | Exact |
| Four-step endpoint vs resumed 2+2, including moments | Exact | Exact |
| Old/new executable serving and ordinary/cache equality | 8/8 | 8/8 |

The tolerance for every gradient component and answer loss was absolute
`2e-6`. Resume and compatibility comparisons require exact bytes, not that
tolerance. Numerical agreement is not a speed or quality benchmark.

| Local evidence | SHA-256 |
| --- | --- |
| Native executable | `d6895e12017dfd75b4345beac239f389bc3b12262dac77dd70135793fbaaf310` |
| Frozen witness plan | `7e7399bd879a8a02263685b38794bede686b5f7de9cccf5a8a9694ac8f93e0e1` |
| Witness result | `eb91258114b0e9bd888cab7e25d3dac73a1876a5090117582e63f397e2ff0636` |

Requests, stdout/stderr, checkpoints, build/source inventories, procedure and
receipts remain in local validation folder `2026-11-10-native-ragged-qkv-v1`.
The old executable, weights and experimental originals are unchanged.
Run the repository's structural checks with:

```sh
cargo test --locked --offline -p st-nn --lib causal_lm:: -- --test-threads=1
cargo test --locked --offline -p st-nn --example native_key_lag_fit
```

Next scientific step: preregister matched length/position-varied training and
new evaluation conditions before running it. The already-inspected transfer
questions cannot be relabeled as fresh holdout after this design change.
