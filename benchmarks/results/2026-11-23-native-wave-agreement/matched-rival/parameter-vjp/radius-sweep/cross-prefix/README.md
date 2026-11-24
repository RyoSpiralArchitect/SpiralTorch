# Cross-prefix fixed-step transfer, training rows only

This is a new **descriptive** analysis of already captured gradients, not a
new decoder run or a change to the frozen real radius-learning experiment.
Fix the comparison before examining its outputs: R=1 versus R=4, both factual
hosts separately, all 24 original training prefixes (six lexical quartets).
Use the gold-minus-rival parameter derivatives, gate then bias, at the same
stored site input and incoming adjoint. Radius 4 was previously selected on
these same rows; this is deliberately not a generalization result.

For a donor row i, receiver j, donor radius s and receiver radius r, compute

    T[r,s](j,i) = (1/256) * dot(g[r,j], g[s,i]) / norm(g[s,i]).

Positive means that moving a fixed Euclidean distance in the donor's
margin-ascent direction would locally favor the receiver's own correct verb
with its incoming adjoint held fixed. For binary rival CE this is the
normalized descent direction when rival probability is positive. This is a
tangent prediction, not an actual finite step, post-update logit, or
recomputed decoder derivative at R=4. Never multiply gradients from
different factual hosts.

Exclude the donor's entire lexical quartet, **not just its own row**. For
each receiver, average all 20 other-quartet donors, ten with the same gold
verb and ten with the other. Report both strata, their balanced mean, and
all six receiving-quartet means. This gives 480 ordered pairs per host,
not 480 independent examples. A one-quartet toy archive has no eligible
pairs; return no estimate, not zero or a self-comparison.

Use all four combinations. Report the diagonal change T44-T11, and split it
symmetrically into donor rotation and receiver sensitivity:

    donor    = ((T14-T11) + (T44-T41)) / 2
    receiver = ((T41-T11) + (T44-T14)) / 2

The receiver term includes its magnitude **and** direction. Separately show
the unit-to-unit alignment at R=1 and R=4 to remove both gradient magnitudes.
Call a balanced improvement consistent across these lexical quartets only
if both T44 and T44-T11 are strictly positive for every fold on both hosts.
Zero or reversal is not a win. No new radius/threshold search, p-value,
held-out inference, release decision or change to the matched learning gate.

Compute row dot products with the native CPU wide accumulator (f32 storage
and result), and check every cell against a separate double-precision
`math.fsum` calculation. Reject nonfinite or zero-gradient directions, an
unbalanced quartet, a mismatched hash or a cell with scaled error above
2.5e-7 (scale: product of the two reference norms). Repeat the complete
fold classification with the reference matrix and require agreement.
Preserve raw native matrices and ordered pair rows with a fresh output
directory; do not load pretrained weights or evaluation records.
