# Choose a trained prefix route without opening transfer examples

Freeze before new model calls. Synthetic CPU only. First complete the independent rehearsal,
then make one formal attempt (120 seconds, stop on first failure; no retries). No pretrained
GPT-2, natural-language evaluation, gradient, update, GPU, network, or release. Reuse the gated
endpoints from [the matched channel study](../PROTOCOL.md) without changing their weights.
This is a discrete choice among trained directions, not a continuously learned rotation or
equal total training cost: fitting the original three alternatives cost three times one arm.

## Inputs and selector

Use bases 557/563/569, the 32 six-token training prefixes, teachers head/attractor, training
orders 17/31/73, and all three frozen endpoints in order identity, Cayley, inverse Cayley.
For each base, run all three zero-gate origins and all 18 endpoints once on each training
prefix: 2,016 full-logit forwards in total. All three origins must be bit-identical and
their verb logits must match the original native training traces. Check each reloaded
endpoint against its original owner, masks and SHA-256. The selector must only receive
training logits. No evaluation prompts, labels, logit, or endpoint scores enter selection.

Within each lexical frame, form the four assignments (head, attractor) = 00, 01, 10, 11.
Let D be logit[7] - logit[8], computed after widening each stored f32. The head minus
attractor selectivity of this quartet is (D01-D10)/2. For each candidate subtract the
origin selectivity; negate this gain for an attractor-trained endpoint. Choose the route
with the largest **minimum** gain across training frames; tie-break by mean gain, then
lowest route index. Make one choice per base/teacher/order, not per transfer template.
No threshold fitting, endpoint refitting, default promotion, or choosing a winner after
inspection. The blind choice is (base + training_order + teacher_index) mod 3.
Also freeze a secondary decision threshold per base: the mean origin D over all training
rows, labels ignored. The primary decision threshold remains zero.

## Transfer and controls

After sealing and independently verifying all choices, use 32 new eight-token prompts.
For each head h in 3..6 take cyclic offsets a=h+2, c=h+1, d=h+3 within that vocabulary.
For i,j independently 1 or 2, place both modifiers on the head [h,c,d,i,9,a,j,10]
or on the nearby noun [h,i,9,a,c,d,j,10]. Same multiset, length, final token and
answers, different positions. Check disjointness from prior six-/seven-token training
and evaluation lists and rehearsal. Same nouns, not unseen vocabulary or natural text.
Rehearse on base 571, order 5, four training rows and eight new two-modifier prompts
using h=a=3, c=5, d=6; rehearsal and formal gates are distinct.
Score the same complete bank once on these prompts: 2,016 formal forwards; rehearsal
uses 36 training and 72 transfer forwards. Zero weights are a reference; never refit.
Once scoring starts the new transfer set is consumed even if the worker stops.

Report every order/template and five policies: selected, blind, and the three fixed
routes. h = mean signed head margin gain for the head-trained endpoint. S is half
the sum of the oriented selectivity gains of the two teacher endpoints. For each base,
on both templates in all orders, require h and S > 0.001 and both greater than identity
and blind by > 0.001. Require at least two of three bases **and** macro-average selected
h and S each > every fixed policy and blind by > 0.001 across the 18 balanced cells.
This latter check avoids comparing a chosen endpoint to itself as if it were a new route.
For a decision-transfer claim also require strictly more raw correct head decisions
than origin, identity and blind in every cell of at least two margin-passing bases.
Do not promote centered decisions or sensitivity ranks to the raw decision gate.

Report raw and training-centered counts including ties and both directions of flips
for both teachers; report head-number ordering with the attractor fixed, and
attractor-number ordering with head fixed (two comparisons per frame). These rankings
cancel any constant verb bias. The matched templates, repeated training orders, bases
and controls are not independent language samples. All computations after sealing
consume saved f32 logits; do not rescore if a reporting bug is found.
