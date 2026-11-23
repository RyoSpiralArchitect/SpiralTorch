# Can one train-only verb offset stand in for a uniform value read?

This is a mechanism control on the **previously scored** synthetic number
agreement prompts. No new data, model/geometry/LoRA updates, pretrained
model, gradients, GPU, release or network. Keep both teachers, all three
bases and all three orders, both templates, and both original gate scales.
No per-template, per-input, per-channel or transfer-selected fitted offset.

Target: UU, the uniform-trained gate served with a uniform read. Primary
source: UA, the same donor served with an attention read, with the previously
frozen per-cell scaling in the common-RMS experiment. Negative control: the
zero-gate origin with the same frozen backbone and LoRA. The score coordinate
is D = f64(logit[7]) - f64(logit[8]) (singular minus plural). A fitted
scalar adds to D, not to an attention weight, vocabulary distribution or
generation sampler; serving one prompt needs neither a label nor its mate.

Fit both constants separately per base, order, teacher, scale, and source,
on **all 64 training prompts together** (32 signed pairs, both templates):
- Task offset: exactly maximize the number of training pairs where both
  signs are right. A pair gives the strict interval (-D_positive, -D_negative);
  equality is not a success. Among maximizers prefer zero if optimal,
  otherwise the midpoint of the widest elementary optimal interval; break
  ties by nearest zero then smaller signed midpoint. Use a representable f64.
- Mimic offset: the least-squares constant, mean(D_UU - D_source) over all
  training prompts. This needs the training predictions, not their labels.
- Keep the unshifted source and factual UU. The origin's task offset is
  necessarily shared across orders/scales for a fixed base and teacher.

The original archive has training logits for origin and raw UU but not the
raw UA or either RMS-matched serving condition. Use the same pinned native
CPU wheel to run all four training-prefix cells, including the raw UU
positive control. Compare *all eleven* factual logits there bit-for-bit to
the saved training journal. Rehearsal: 2 groups x 4 x 8 = 64 native forwards;
formal: 18 groups x 4 x 64 = 4,608. Reuse all transfer logits, zero new
evaluation forwards. One process per partition (60/120 s respectively),
no retries; 4 GiB disk floor. Seal both partitions' fitted coefficients
and independently check the Rust fits **before** admitting transfer data.

Report both-correct counts, individual hits, zero ties, mean weaker margin,
agreement with UU's individual decisions, UU successes reproduced/missed,
new successes absent from UU, all prompt gap RMSE vs UU. Report pooled,
per base, and per order/template, retaining failures on all bases. Repeated
orders are not independent examples. Two non-deployable transfer diagnostics:
(1) among the UU wins, how many are individually separable in the source and
how many can be covered *together* by one constant across both templates per
group? (2) could a single finite f64 offset match every UU row decision in
that group? Report bounds/booleans only, never a transfer-derived scalar.
A training-fit miss is not proof of structural impossibility. Conversely
recovering all wins does not mean the other rows or full logits agree.
A scalar control matching performance is not proof that the actual
internal geometry is a scalar. No significance or fresh generalization claim.
