# Does targeting the weaker signed lane change the verb decision?

Freeze before this study sees any model output. Use the existing three small synthetic
decoder bases (557, 563, 569) and each base's frozen numerical LoRA. Train only the
three gates of the final MLP positive temporal quarter turn, strength 1/4, zero frozen
bias and identity output mix. No real GPT-2, English evaluation, GPU, network,
publication, tuning, retries, checkpoint selection or additional training set.

Let Dp,Dn be token 7 minus token 8 on the two prefixes with supervised number
0 and 1 respectively. Compare independent binary CE, rank (softplus of minus the
half-gap), and weakest_lane = softplus(-min(Dp,-Dn)). No free loss coefficient.
For the last arm only the weaker lane receives the loss derivative. At exact
equality share it equally. Select by the signed margin, not by comparing rounded
losses. Both decisions are correct iff the weaker signed margin is strictly positive;
ties are not correct. Record each actual selected lane and both gradients.

## Frozen inputs and work

Use the same four heads h=3..6, training nearby noun a=h+1 or h-1 cyclically,
new transfer noun a=h+2; modifier c=h+2 in training, h+1 in transfer. Marker 0
comes immediately before the final token on **both** templates:
[c,h,i,9,a,j,0,10] and [h,i,9,c,a,j,0,10], i,j=1 or 2. Train and transfer use
the same positions but disjoint bare (h,a) pairs. Verify exact token lists are
distinct from the preceding six-, seven-, and eight-token studies, including the
leading-zero paired-gap study; frame families and the model bases are reused.
Rehearsal uses base 571, h=a=3, c=5/6 for train/transfer, eight prompts on
each partition. Freeze both manifests before any model call.

Group by template, frame and nonsupervised number. Orient each pair as supervised
0 then 1; sort the group keys and shuffle once per order, repeated across the
epochs and all arms. Formal: three bases, teachers head and nearby, orders
17/31/73, two epochs, 64 training inputs, 32 transfer inputs; 54 endpoints,
64 updates per endpoint, 3,456 updates, 6,912 differentiated lanes. One two-row
native decoder VJP and one sealed gate update each time (even when one seed is
zero). Fix delivered RMS = 0.25*(2/256)/sqrt(3), same calibrated diagonal
residual metric and physical parameter norm cap 16. No clipping. Use exactly
192 train-only unit-margin captures and 192 metric calibration prefixes.
Full 11-logit scoring: 3,648 training forwards on all origins and endpoints,
seal and independently audit the training journal, only then open transfer for
1,824 forwards. Rehearsal: order 5, one epoch, six endpoints, 24 updates,
8 captures/calibration prefixes, 56 train and 56 transfer forwards.
First failure stops, one worker per split, rehearsal 120s/formal 300s CPU wall caps.

Keep exact state bytes at every update. Independently verify loss, signed
seeds, source and endpoint owners, first-update sum of two single-lane gradients
(per-coordinate relative 3e-4 plus absolute 2e-7), every physical RMS (relative
1e-4), direction and descent, plus first and last steps on the actual training
site rows (absolute RMS discrepancy <=2e-6). Record all conditions and ties.

## Advancement, not retuning

The primary question is head-number transfer. Within **every template and order**
on a base, require strictly more pairs with both decisions correct under weakest_lane
than under origin, binary CE **and** rank, and mean weaker signed margin greater
than all three by more than 0.001 logit. Require >=2/3 bases. Separately report
the continuous margin-only gate with the same threshold and scope, and individual
head decisions (no worse than any reference in the same cells for a quality
claim). Repeat the full accounting for the nearby teacher as a control, not as
a substitute for head. List pair orderings, raw flips and gains/losses of both-
correct pairs, including unique pairs versus repeated orders. A null result
ends this comparison: no changing ties, objectives or thresholds on these prompts.
