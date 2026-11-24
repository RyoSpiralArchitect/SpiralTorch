# Read prefix levels or adjacent innovations?

Freeze before observing either carrier on these examples. Compare only the input
to the same causal quarter-turn accumulator: levels (existing v1), or innovations
(new v3). For innovations d[0]=0 and d[t]=x[t]-x[t-1] inside each document;
S[t]=R*S[t-1]+d[t]. The feature is S[t]/(t+1), not a running variance or a
learned detector. There is no supervision, other sequence, or future row in that
feature. Identity output mix, strength 1/4, frozen zero bias and three trainable
gates at the final MLP. Frozen base and LoRA, weakest-lane objective, packed
two-prefix update and delivered residual RMS are identical between arms.

Reuse bases 557/563/569, orders 17/31/73, 2 epochs, both teachers and the
predecessor's training/transfer lexical splits, but move the neutral marker from
just before token 10 to immediately after separator 9 on both templates. Check
prompt uniqueness/disjointness, including the preceding study. Formal: 64
training and 32 transfer prefixes, 36 endpoints x 64 updates = 2,304. Two origins
per base (one per carrier). Independently reconstruct factual feature contributions,
gate VJPs, diagonal calibration, and step directions; witness first and last
delivered residuals on actual native site outputs. Record all step owners,
parameters, paired scores and endpoint bytes. Both zero-gate origins must give
identical full-vocabulary logits. Score 42 models x 64 training, seal before
42 x 32 transfer prompts. Rehearsal base 571 order 5 one epoch, eight prompts
per partition: four endpoints x four updates. Freeze manifests before either
worker, one attempt each, 120s and 300s CPU wall respectively; stop on first
failure. No real GPT-2 calls, language quality claims, GPU or publication.

For each base and teacher, compare median paired common-direction ratio on unique
training pairs at zero gate, as in the predecessor. Mechanism gate: increase
>0.05 on BOTH templates, without a decrease in median absolute common gain and
without any optimistic direction exceeding physical norm 16. Task gate: on
EVERY template and order, innovations must beat both zero-gate and level by
>0.001 in mean weaker signed margin, increase the number of pairs with both
decisions strictly correct, and not reduce individual verb decisions. Require
at least two of three bases for head-number, report nearby-number separately.
Report margin-only, both-correct counts, ties and negative gradient cosines even
when a gate fails. No adjusting the filter, limit, or schedule after viewing data.
