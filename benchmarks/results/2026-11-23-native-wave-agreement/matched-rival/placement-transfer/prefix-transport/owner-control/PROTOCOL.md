# Gate-only versus bias-only learning on the fixed symbolic agreement task

Freeze before new observations. This is a CPU-only follow-up on artificial
weights, not another GPT-2 call or a new English evaluation. The [parent
experiment](../RESULTS.md) ablated parameters *after* joint training. Here
only one vector can move from the very first step. No fitted hyperparameters.

Use the parent's exact bases 557/563/569, frozen numerical LoRA, geometry
site, f32 readout, width three, strength 0.25, initial all-positive-zero
coordinates, 32 training / 16 reused evaluation prompts, two teachers,
three orders 17/31/73, two epochs, complementary pairs and 2/256 parameter
displacement per update. Cross (gate trainable, bias fixed) or (bias
trainable, gate fixed) with identity, positive and inverse quarter turn.
The native equal-vote ray must use only the three active coordinates.
Archive both raw member gradients, the realized parameter vector and native
receipt at every step; the inactive three parameters must remain bitwise +0.
Check the projected mean gradient, member dots, displacement and chain of
weight identities, not just a trainable-parameter count.

Before training, match all 18 masked origins to the archived frozen logits
on all 48 prompts. First-pair gradients must match the co-trained parent's
at the same origin. Across rotations, all bias-only active trajectories,
member losses and all logits must agree bitwise; the inactive gate gradient
is allowed to differ. Reference checks use original bytes, not another
forward/backward. Record full logits for 18 origins and 108 endpoints:
6,048 snapshot forwards, 3,456 updates, 6,912 training VJPs. First rehearse
with separate artificial base 571. The formal process is exclusive, first
error stops, one attempt, 240 s CPU-worker wall limit; no formal retry.

Primary is positive quarter-turn gate-only. Control A is gate-only with no
rotation; control B is positive quarter-turn bias-only. Compute d=z7-z8,
H=mean(sign(head)*d), A=mean(sign(attractor)*d), h=H(head-trained)-H(origin),
S=((H-A)(head-trained)-(H-A)(attractor-trained))/2. For each base **every
order** must have h and S >0.001 nats, improve H on at least 3/4 held-out
frames, and exceed the control's h and S by >0.001. Report passes separately
for A, B, and both. For the joint decision gate also require strictly more
correct head choices than origin and both controls in every order. Require
2/3 bases to claim transfer here. Report inverse turn as secondary, not a
substitute. Prior six-coordinate joint training is a historical comparator,
not an equal-capacity control.

Report all arms, both teachers, losses, ties and winner changes. Separate
agreement from head/attractor conflict; count switches and report changes
in conflict head choices. Decompose each complete lexical frame's margins
into offset, head, attractor and interaction contrasts and verify exact
reconstruction to numerical tolerance. These are diagnostics, not new gates.
No quality claim from margin movement alone. No GPU, real-model run,
external upload, or modification of the original evidence.
