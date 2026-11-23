# Does order-sensitive prefix transport make a shared residual learn the head?

A new local CPU study, not a rerun of the [conditional-loss comparison](../conditional-update/RESULTS.md).
Freeze before measuring any responses. No real GPT-2, English evaluation, GPU,
downloads, free generation, publication, hyperparameter search or retry of the formal study.

At the second MLP output let S[-1]=0, S[t]=R S[t-1]+x[t], u[t]=S[t]/(t+1).
Add strength * (gate * u[t] + bias) channelwise to the site's original x[t].
R is identity, a positive quarter turn on each consecutive pair of channels,
or the negative quarter turn. An odd final channel always passes unchanged.
All previous rows enter the summary with the same coefficient norm. No role
labels or token positions are supplied to the adapter. Identity is order
insensitive on *the supplied prefix activations*, not on a positional decoder.
It is not a pointwise affine layer. The two learned vectors start at zero and
all three modes must return exactly the frozen logits initially.

Use artificial weights from the parent generator, seeds **557/563/569**, the
same numerical frozen LoRA as before, one geometry placement, width three,
strength 0.25, six trainable values. Use the parent's symbolic agreement
design: 32 training prompts on eight noun-pair frames and 16 held-out prompts
on four other frames. The split is reused; these are not 16 new observations
about English. Cross 3 bases x 3 rotations x 2 teachers (head or attractor)
x 3 fixed orders (17,31,73). Pure CE conditional on token 7/8 at the final
prompt position; no EOS target. Same balanced examples, two epochs of the
parent's deterministic complementary pairs, 2/256 displacement per pair
from the *same* snapshot. All endpoints have 32 steps, 64 training VJPs,
and a prescribed path of 0.25. No clipping or skipping zero gradients.
Archive origin and final weights, individual gradients, post-step identities
and all 11 logits. Score all 48 prompts at nine origins and 54 endpoints;
also score the three frozen LoRA-only references on these same prompts to
verify the zero residual (3,024 geometry and 144 reference forwards).
Formal limit: 1 CPU attempt, 240 seconds, first error stops. Run implementation
checks on different artificial weights first. No loop to get green measurements.

Let d=z7-z8, H=mean(sign(head)*d), A=mean(sign(attractor)*d). For matched
order, h=H(head-trained)-H(origin), S=((H-A)(head-trained)-(H-A)(attractor-trained))/2.
For each base compare **positive quarter turn minus identity**. A probability
win requires h, S, h-h(identity), S-S(identity) all >0.001 nats and improved
H on >=3/4 held-out frames for every order. A decision win additionally needs
>=1 more strict correct two-token winner than both origin and identity in
every order; ties count separately. Require >=2/3 bases for transfer.
Report negative turns identically as a secondary handedness check, not a
replacement for the primary. Include training outcomes, unrestricted
vocabulary accuracy/NLL, ties, and both teachers. A failed gate is a result;
do not retune the rotation or reclassify a per-order win as transfer.
