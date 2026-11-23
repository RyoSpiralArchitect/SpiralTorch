# Matched delivered residual RMS: training protocol v1

Freeze before calibration or training. Synthetic CPU study, no GPT-2 or external data.
Reuse the prefix-transport formal split (32 train, 16 previously scored evaluation
prompts, balanced complementary pairs, candidate token IDs 7 and 8). No new holdout
claim. Use bases 557/563/569, orders 17/31/73, two epochs, both labelings (head,
attractor), both identity and positive quarter turn. Rehearse separately on base 571
with four train and four evaluation rows, one order and one epoch. Frozen LoRA and
decoder; only the final MLP prefix-transport gate changes, bias stays exactly +0.

Fit once per base/rotation on the final prompt row of each TRAIN prefix without
targets. With strength s, native f32 transport mean m and width w, let
q[j] = mean_i (s*m[i,j])^2. For physical gate displacement d, calibration RMS is
sqrt(sum(q[j]*d[j]^2)/w), before rounding when adding to the residual stream.
This fixed metric remains applicable because no parameter before the site changes.
Uniform follows the raw pair-mean gradient. Diagonal divides each coordinate by
q[j]+mean(q)/16. Cyclic uses q[(j+1)%w] instead, with the same damping and
spectrum. For all rules exclude exact null coordinates; reject zero descent.
Scale each step to RMS 0.25*(2/256)/sqrt(3), not to equal parameter L2.
Cap physical L2 at 16 per step; fail instead of clipping or silently skipping.
Require realized RMS relative error <= 1e-4, strict descent on the pair mean,
and an independent comparison against native residual outputs (absolute tolerance
2e-6). Complementary members get equal votes with binary conditional CE.

Each of 108 formal endpoints takes 32 real updates, 2 VJPs/update: 3,456 updates,
6,912 training VJPs. Score all 11 logits for 48 prompts at 6 shared origins and
108 endpoints. Archive parameters, gradient and step receipts, and first failure.
CPU only, one formal attempt (max 300 s); error stops the run, no retries.
No user model, GPU, release, external transfer or history rewrite.

Primary: positive turn, channel diagonal vs both uniform and cyclic. For every
order within a base require h=H(head-trained)-H(origin) and selectivity
S=((H-A)(head-trained)-(H-A)(attractor-trained))/2 > .001, gaps in h and S
against BOTH controls > .001, and improved head margin on at least 3/4 frames.
H and A are mean correctly oriented head/attractor verb margins on the reused
evaluation set. Claim decision transfer only if head choices strictly improve
over origin and both controls in every order; require >=2/3 bases for transfer.
Report identity controls, order ranges, both teachers and turn comparison;
fixed-parameter-length historical endpoints are descriptive only. No claim
about language-model quality, a new holdout or generalization to real GPT-2.
