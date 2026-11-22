# Preserve a feasible loss step, try one angular midpoint only on a miss

Frozen before new implementation or measurement. Use exactly the generated
ordinary/WaveGate decoder, nonzero LoRA, three previous training examples also
used as the ordered KL bank, two epochs, tile 11. No real GPT-2, held-out
evaluation, GPU, publication or additional training data.

The previous capped raw-gradient control is pinned. Compare three parameter
owners (LoRA, geometry, joint) under two opt-in rescue policies: largest
KL at one raw probe, or its cyclic successor in the bank. Six primary
trajectories, stopping at the first terminal refusal, no restarts, parameter
sweeps or target changes. Raw controls are verified at each actual origin
using a separate one-step session; verifier work is counted separately.

1. Run the unchanged raw scalar search. If accepted, use **exactly** that
   step with no probe, extra candidate, anchor backward or correction.
2. Only a finite, measured bounded miss can enable the second direction.
   Keep the first refusal, reuse its anchor references and the very same sealed
   loss gradient. Probe the raw ray at L2 1/256. Select the first maximal-KL
   anchor, or its successor for the control; one extra anchor VJP at the origin.
3. Let h be the prior full secant projection of g off the selected normal.
   If its norm is unresolved (<0.001 of g), do not form a direction. Otherwise
   use (g + norm(g) * h/norm(h))/2 before normalizing the candidate step.
   The two unit directions receive equal weight even when their norms differ
   greatly. Verify actual original-gradient dot movement is negative when accepted.
4. Use the identical search on that one direction, from the untouched origin.
   Per search: mean forward KL in [0.98T,T], all three KLs <=2T, T=1/65536,
   parameter ceiling 1/8, <=20 scalar trials. Per attempt <=40 scalar trials
   plus one uncommitted probe. Invalid inputs and unexpected numerical errors
   are not rescue triggers. Never return a best-effort undersized update.

Primary endpoints: accepted raw/rescued counts before first terminal refusal;
did the originally blocked geometry or joint reach one or six updates? Did the
successor also do so? All six LoRA updates must remain raw and identical to the
prior control. Check actual KL, selected owner movements, preserved refusal
state, midpoint direction and extra work. Record in-sample losses at the
actual stopping horizon; do not rank across horizons or claim generalization.
If neither correction advances, retain the negative result rather than
retuning the mixing fraction. A rescue is evidence of feasibility, not
evidence of better language learning or of an equal-cost comparison to raw.
