# Do the fitted gate coordinates or the serving value source carry the difference?

This is a post hoc mechanism check on the previous synthetic decoder study.
The 32 transfer prompts have already been scored; these are not fresh quality
data. Keep **all** bases (557/563/569), orders (17/31/73), head and nearby
teachers, templates and pairs, including the bases where uniform never won.
Use the archived frozen backbone, frozen LoRA, 3-coordinate final-MLP gate,
native CPU wheel and f32 readout. No fitting, gradients, retries, new prompts,
fresh English evaluation, pretrained model, GPU, or public upload.

Cross the saved gate coordinates (trained with attention or uniform) with
the serving source (attention or uniform), leaving the original backbone
attention in place. First apply the four literal swaps. Then use the same
four directions, with one positive scalar per cell fixed *only* from archived
training prefixes. For each base/teacher/order the target is the geometric
mean of the two factual diagonal residual RMS values. For a raw cell, compute
R from actual delivered final-row residuals on all training examples/channels,
not parameter length: f32 site output minus its original f32 input. Multiply
gate coordinates by target/R in f64 and round once to f32. Bias remains frozen
at zero. Require finite nonzero RMS and relative error <=2e-5 for the matched
cells; no clipping, second pass or per-test-example adjustment. Compare all
training-row site outputs to the native two-input geometry in both arms. RMS
matching does not match per-example energies, orientations, or logit scales.

Freeze endpoint, training source, old transfer journal and code digests before
scoring. Freeze *all* proposed serving artifacts for both partitions before
running either model. Verify ownership and the two raw diagonals' 11 logits
bit-for-bit against the archived native scores, failing on the first mismatch.
Rehearsal: one base, one order, two teachers, eight arms, eight transfer
prompts = 128 forwards. Require its no-model audit before the formal worker:
18 groups x eight arms x 32 prompts = 4,608 forwards. One native batched site
witness per group/arm per up-to-16 training rows: 16 rehearsal and 576 formal.
CPU wall limits 60 s rehearsal / 120 s formal, 4 GiB disk floor, one worker
per partition and exclusive outputs. No actual model or optimizer updates.

For each template, order, teacher and base report both-correct verb pairs
(both signed margins >0), individual correct verbs, ties and mean weaker
margin. For each scalar convention report the four outcomes in order AA,
AU, UA, UU (coordinate donor first, serving source second). For a pair that
fails AA and passes UU, AU alone means switching the read sufficed, UA alone
means switching the weights sufficed, both means either sufficed, neither
means both were needed among these cells. Record all 16 outcome patterns,
including those where both diagonals agree, both pooled and per base.
For margins also report simple effects at each fixed value of the other
factor and the interaction UU-UA-AU+AA. Repeated orders are not independent
examples. No threshold selection, significance test, or default promotion.
