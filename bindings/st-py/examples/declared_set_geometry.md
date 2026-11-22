# Learn a choice without losing its probability mass

For supervised choices such as two verb forms, a model can prefer the correct
verb more strongly while assigning *both* verbs less probability. The native
CPU decoder exposes two separate mean losses on an explicit set S:

| Score | Meaning |
| --- | --- |
| `within_nll` | `-log P(target given S)`: ordering inside the declared set |
| `coverage_nll` | `-log P(S)`: probability lost to the rest of the vocabulary |
| `full_nll` | Their sum, ordinary unsmoothed target NLL |

This example assumes a loaded `NativeCausalLm`, a trainable geometry adapter,
and vocabulary IDs from that same model. No Torch or NumPy is involved.

```python
ids = [*prefix_ids, gold_id]
shape = [1, len(ids)]
supervision = dict(
    labels=[-100] * len(prefix_ids) + [gold_id],
    contrasts=[(len(prefix_ids), gold_id, [other_verb_id])],
)
next_model, step = model.step_declared_set_geometry_at_distance(
    ids, shape, target_l2=1 / 256, **supervision
)
print(step["accepted"], step["before"], step["proposed"])
model = next_model  # unchanged on a finite tie or regression
```

The step proposes a within-set update at the requested Euclidean distance in
trainable geometry coordinates. It guards both first-order loss changes, then
scores the resulting model **once** on the same batch. Both measured means must
strictly decrease. There is no retry, adaptive line search, change to frozen
base/LoRA, or guarantee for individual examples or unseen prompts. Invalid
inputs, zero gradients and infeasible directions raise an error rather than
pretending to have evaluated a candidate. Ordinary training APIs are unchanged.

For monitoring only, `model.declared_set_nll(ids, shape, **supervision)`
needs no adapter and builds no gradient. For a custom update,
`within, coverage, scores = model.geometry_declared_set_gradients(ids, shape, **supervision)`
returns two snapshot-bound handles after one decoder forward. Coverage is
differentiated directly, not by subtracting rounded full and within gradients.

Positions are flat, unshifted target positions in `[batch, time]`; a window's
first position cannot be a target. Rival IDs must be sorted, unique, and
exclude the target. The two means have the same denominator: all non-ignored
labels. An active row with no declaration (or with the entire vocabulary in
its set) contributes ordinary CE to within and exactly zero to coverage.
Strictly improving coverage requires at least one proper subset. This is a
source-tree feature; an older installed release may not expose these methods.
