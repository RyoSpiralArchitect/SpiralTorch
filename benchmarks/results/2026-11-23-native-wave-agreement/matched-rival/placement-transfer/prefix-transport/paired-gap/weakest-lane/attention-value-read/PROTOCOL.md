# Does actual last-block attention read a more useful value than position alone?

Freeze before any model outputs on these prompts. Use the same three-coordinate
gate and frozen bias at the final MLP. Change its source to the final block's
projected V. Compare (1) the actual merged attention context already used by
the decoder, (2) the causal uniform V mean, (3) V at zero-based slot 2 or the
latest visible slot if the prefix is shorter. Never replace the backbone's
ordinary attention. Add strength 1/4 times gate times source to the MLP branch.
This tests a skip connection, not an interpretation of attention as an explanation.
No trained router or labels/paired input during inference.

Same frozen small bases 557,563,569 and LoRA, train/transfer lexical split,
teachers head/nearby, orders 17,31,73, and two epochs as the preceding
study. Place neutral 0 immediately BEFORE separator 9 on both eight-token
templates, distinct from the latest study's position after 9. Slot 2 therefore
reads the head marker on one template and neutral 0 on the other: an explicit
positional control, not a label-blind hyperparameter search. No oracle position
chosen per template. Uniform distinguishes weighting from merely changing the
underlying values. Targets 7/8, native weakest-lane objective, train only gate.
Equal delivered residual RMS = .25*(2/256)/sqrt(3), diagonal calibration on train,
physical parameter norm cap 16. No clipping, selection, or retries.

Freeze both manifests. Formal: 64 train/32 transfer prompts. 3 routes x 2
teachers x 3 orders x 3 bases x 64 updates = 3456; 576 origin captures and
576 calibration prefixes. Score all 9 origins and 54 endpoints on training
(4032 full-logit forwards), seal and replay the training journal before opening
transfer (2016 forwards). Two native physical witnesses per trajectory on all
training site inputs. Rehearsal: base 571, eight train/eight transfer, order
5, one epoch, six endpoints, 24 updates, 24 captures/calibration, 72 train and
72 transfer forwards. One worker per partition, 120s/300s CPU wall respectively;
stop at first failure. Save every gradient, step, checkpoint, and raw score.

Require all zero-gate logits bit-identical across routes. Independently reconstruct
attention at the target row from Q/K/V, scale and key lag (allow f32 reduction
error); verify selected source, attribution, calibrated norms, step receipts,
first/last actual movement, restored endpoint owners, and no new KV history
in implementation tests. Report attention mass on the two number markers on
TRAIN only; attention weights alone are not evidence of decisions. Compute the
origin common-direction capacity for head and nearby pairs in each route's own
physical metric. Mechanism gate: on each template, median common-gain ratio
exceeds BOTH controls by >.05, median absolute common gain is no lower, every
pair defined and no ideal direction outside physical cap. Task gate: on every
template and order, actual attention exceeds initial and BOTH controls by >.001
in mean weaker signed margin, has strictly more both-correct pairs, and no fewer
individual correct verbs. Report the continuous-only gate separately. Require
>=2/3 bases for head, report nearby separately; retain every null/negative
control. No pretrained GPT-2, English evaluation, GPU, external publication,
or changes to these gates after seeing data.
