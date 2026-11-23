# Does removing full-vocabulary CE make an answer choice learnable?

Frozen before inspecting the new weights' outputs. This follows the
[answer-scope comparison](../answer-scope/RESULTS.md); it is not a retry of
that study. Local CPU, generated weights and symbolic prompts only. No
Python ML stack, real model, GPU, downloads, publishing or old English data.

## Matched factors

Use every previously untested base 503, 509 and 521 from the parent
name/coordinate-keyed generator (same width three, two layers, 11-token
vocabulary). Use identical numeric frozen LoRA and six starting geometry
coordinates. Only placement: second MLP output. WaveGate and matched affine.
Data and teachers as in the parent: all four number combinations on eight
training and four held-out ordered noun pairs, 32/16 prompts. The symbolic
task and split are reused, not new English prompts.

Cross 2 geometries x 2 teachers (head/attractor) x 2 answer-token loss
mixtures (0.5 and 1.0) x 2 schedules (single and balanced pair) x 3 orders
(17/31/73), on **every** base. No EOS in either training loss. Mix 0.5
weights full-vocabulary gold CE and gold CE conditional on tokens 7/8
equally; 1.0 uses only the latter. Two epochs. Within a lexical frame
pair each head-singular prompt with its complement (both numbers flipped):
each pair has opposite gold under either teacher. Sort pairs by SHA-256
of ASCII `<order>:pair:<head-singular-row-key>`; orient each pair by sorting
members by SHA-256 of `<order>:row:<row-key>`; break digest ties by row key.
Repeat the permutation. Train sequentially on that *same* order or take a
single Rust cohort step from both gradients at a common snapshot. Both
use the same native f64 mean-gradient ray: step length n/256 for n example(s).
Each cell sees 64 examples and travels 0.25 in parameter space. Pair cells
intentionally have half as many optimizer steps; this is not equal time,
endpoint displacement or number of weight updates.

Overall: 144 endpoints, 9,216 training VJPs, 6,912 geometry updates. Check
each actual f32 displacement against the normalized negative mean gradient,
relative error <=1e-4, verify individual loss dots and frozen ownership.
No floor, skipped zero gradient, resampling or retries. Read all 11 next
logits once for each of the 48 prompts at each of 6 origins and 144
endpoints (7,200 forwards). No free generation. Archive actual gradients,
steps, logits, and final geometry plus companion. One formal CPU launch,
240 s limit, retain partial evidence and stop at first error. Use disjoint
repeated-noun examples and base 541 for the separate rehearsal.

## Numerical and mechanistic checks

For each training prompt at each origin capture both covers (mixtures) and
both possible gold labels, independent of which teacher is trained (768
origin VJPs). Define d=z7-z8, p=softmax([z7,z8])[0]. The difference
g(plural)-g(singular) equals grad d for either mixture. Pure binary
gradients should be -(1-p)*grad d and p*grad d. Cross-check the pure pair
against the **mixed** difference; absolute vector-error ceiling
2e-6 * max(1, all four gradient norms), with relative errors and undefined
cosines also reported. Do not turn an undefined direction into a pass.
Recompute all origin losses from full logits (abs 3e-6).
After endpoint bytes are sealed, measure the pure pair on every held-out
row at the immutable origins (192 VJPs; no training, tuning or choice of
arms from them). Never pass those gradients to an optimizer.

Use the origin train gradients to predict displacement if gradients did
not change over training: sum each actual schedule's unit descent rays times
its step length over both epochs. On held-out prompts compare the predicted
h and S using their analytic margin derivatives with (a) the same local
derivatives applied to the actual endpoint displacement and (b) observed
margin changes. These are first-order diagnostics, not guarantees or update
rules. Undefined zero mean directions are recorded, not normalized to noise.

## Outcomes and controls

Held-out primary d is **first-token** log odds, no stop. H=mean(sign(head)*d),
A=mean(sign(attractor)*d), with singular +1. h=H(head-trained)-H(origin);
S=((H-A)(head-trained)-(H-A)(attractor-trained))/2. The primary comparison
is pure vs mixed with single-example updates. Report the same contrast for
balanced pairs separately, and the schedule interaction, not the best of
the two as primary. For each base/geometry/schedule the probability gate
requires all three orders to satisfy h(pure), S(pure), h(pure)-h(mixed)
and S(pure)-S(mixed) each >0.001 nats, and h(pure)>0 on at least 3/4 noun
pairs. Decision gate also requires at least one more correct first-token
choice versus both origin and matched mixed in each order; ties are not
wins. Need >=2/3 bases for a transfer claim per geometry and schedule.
Wave specificity requires >=2 Wave-passing bases with both paired scope
advantages exceeding affine's by >0.001 in every order for that schedule.
Report base/arm/order, training vs held-out, menu wins/ties, full-vocabulary
wins and gold NLL, within-menu NLL and menu mass. No score/threshold tuning.

Separate supervised readout control, **not a geometry arm**: train a
two-class linear probe with native CPU tensors on (1) one-hot token IDs in
all six prompt slots, (2) positionless token counts, and (3) each of the
six frozen origins' 11 next logits. Population-standardize columns using
training rows only (std floor 1e-4), append one intercept. Zero weights,
64 full-batch CE updates, lr 0.1 and L2 coefficient 0.01 including intercept.
Train each feature set under both teachers: 16 probes, 1,024 separate
linear steps. No use of held-out rows for fitting, stopping or choosing
features. Position-aware control must get >=15/16 under each teacher;
bag cannot exceed 12/16 under either (mirrored mixed-number cases are
indistinguishable). Decoder probe results diagnose this readout and budget,
not an impossibility result about the hidden states. If the geometry gates
fail, close these claims locally rather than adjusting seeds or length.
