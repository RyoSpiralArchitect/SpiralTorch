# Can a direction change admit the same per-prompt envelope?

Freeze before implementation or new measurement. Generated ordinary decoder with
the existing nonzero LoRA and WaveGate, the three unchanged training examples
also used as anchors, two epochs, tile 11. No GPT-2, held-out text, GPU or network.
For each of LoRA, geometry and joint compare these three policies from the same
weights and schedule, no retries or fallback after an unmatched step:

- Raw sealed loss gradient (the previously measured capped-mean control).
- Remove its component along the normal estimated for the highest-KL probe anchor.
- Exactly the same operation and cost, but use that anchor's cyclic successor.

Use one uncommitted raw-gradient probe at parameter L2 1/256 per shaped step.
Measure KL from origin to probe for all anchors, break ties by first index.
For the selected prompt form u = J(origin)^T (p_probe - p_origin) for the
last-position vocabulary logits. In selected trainable parameter coordinates
use g - u (u.g)/(u.u); one f32 rounding, normalize by the existing joint
ray contract. No regularization, blend sweep or learning based on these outcomes.
Report zero/unresolved normals or retained gradient below 0.001 of the raw
norm as failures, never as successful zero steps. Check actual origin loss
gradient dotted with realized displacement is negative on accepted shaped steps.

Unchanged scalar search: mean forward KL in [0.98T,T], T=1/65536 nat, every
anchor <=2T, parameter ceiling 1/8, at most 20 measured distances. Every step
recomputes its direction. Primary: count accepted updates before first refusal
for every arm; did largest advance at the previously blocked first update
for geometry/joint, and did its matched-cost control also advance? No ranking
of final losses across unequal horizons. Check old raw pins/refusals, realized
KL independently from logits, immutable refused state, and report the cost of
probes and VJPs separately. This is neither a convergence nor a quality test.
