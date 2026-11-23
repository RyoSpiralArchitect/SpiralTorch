# Paired token-gap learning at matched delivered RMS

Freeze before any new model call. Only small synthetic native CPU decoders. No real GPT-2,
GPU, network, release or reuse of old evaluation results for choosing the loss weight.
Test the existing three initializations (557, 563, 569), not three new independent models.
Freeze the base and numerical LoRA. Train only the three gate coordinates at the last MLP,
with zero fixed bias, strength 1/4, positive quarter-turn temporal transport and identity
output mixing. Do not select routes, sweep coefficients, tune a threshold or retry.

## Three equal-work objectives

Each packed pass contains two distinct, equal-length contexts. The first has target-role
number 0 (correct token 7); the second differs only at that number marker and has number 1
(token 8). The nearby role, lexical identity and all other tokens match. Let Dp,Dn be the
two token-7-minus-token-8 logits. R=(Dp-Dn)/2, C=(Dp+Dn)/2. Let C0 be the zero-gate
mean for these exact two training inputs, recorded before learning, not a transfer statistic.

- Binary control: (softplus(-Dp) + softplus(Dn))/2, ordinary independent two-token CE.
- Rank ablation: softplus(-R).
- Primary anchored rank: softplus(-R) + (C-C0)^2/2, fixed stiffness 1 in logit units.

Each uses one two-row decoder backward and one sealed geometry handle per update. No
normalization by whole vocabulary. Apply the identical train-only diagonal residual metric
and direction rule to all three: RMS = (1/4)*(2/256)/sqrt(3), physical cap 16, no clipping.
The anchor penalizes endpoint *drift since initialization*, not an exact constraint on an
individual step. Record actual f32 weights, parameter gradients, score/loss components and
rounded movement; independently check signs, derivatives, positive descent, calibrated RMS
relative error <=1e-4. At each trajectory's first update verify the paired gradient against
the two audited single-lane margin VJPs (relative 3e-4 plus absolute 2e-7 per coordinate).
Check actual first/last native residual changes against the calibration (absolute RMS 2e-6).

## Data, budget and order

At fixed vocabulary 11, token 0 is a constant leading marker. For h in 3..6 set a=h+1
or h-1 cyclically in 3..6 for training, a=h+2 for transfer. For training c=h+2; for
transfer c=h+1. Place c either before the head [0,c,h,i,9,a,j,10] or before the nearby
noun [0,h,i,9,c,a,j,10]. Cross i,j in {1,2}. Training has 64 rows, transfer 32: same
templates, length and marker positions within a template, disjoint bare head/nearby pairs.
Check both lists against the preceding six-, seven- and eight-token study inputs. No extra
vocabulary or natural-language claim. For the separate base-571 rehearsal, h=a=3, c=5 in
training and c=6 in transfer: eight rows per split, both placements, one order 5, one epoch.

For each teacher (head or nearby noun) group by template, lexical frame, and the other
number. Within each pair keep supervised number 0 first and number 1 second; sort groups
then shuffle once by each seed 17/31/73, reuse that order for both epochs and all arms.
Formal: 54 endpoints, 64 updates each = 3,456 updates and 6,912 differentiated prompt
lanes. Three zero origins capture 192 train-only unit margin VJPs and calibrate on the same
192 prefixes. Score the three origins and all 54 endpoints once on training (3,648 full-logit
forwards); seal and audit all training evidence before opening transfer. Then score these 57
states on transfer (1,824 forwards). Rehearsal: six endpoints, 24 updates, 8 unit captures,
8 calibration prefixes, 56 training and 56 transfer forwards. Check all 11 logits, owners
and endpoint reloads. One attempt per split, CPU 120s rehearsal / 300s formal, first failure
stops, no retries or favorable checkpoint selection. A consumed transfer split is never fresh again.

## Frozen decision rules

On each template and training order measure whether D0>D1 when flipping only the teacher's
number. Also report signed verb margin, first-choice accuracy (threshold zero), ties and
both directions of changes, for both teachers. The primary head-transfer criterion on a base
requires more correctly ordered head pairs than **both origin and binary** on every template
and order, and mean head half-gap > both by 0.001. Require >=2/3 bases; saturation at a fully
correct origin is recorded, not discarded. To claim raw decision improvement additionally
require more correct head decisions than both references in the same cells of >=2 qualifying
bases. Separately, call the center penalty effective on transfer only if, on both templates
and every order for both teachers, mean absolute pair-center drift is positive for rank alone,
at least 10% smaller with anchored rank, and ordering accuracy is no worse (>=2/3 bases).
This last criterion need not imply either quality criterion. Report training and transfer
separately, all nulls, and individual flips on repeated questions as repeated measures.
