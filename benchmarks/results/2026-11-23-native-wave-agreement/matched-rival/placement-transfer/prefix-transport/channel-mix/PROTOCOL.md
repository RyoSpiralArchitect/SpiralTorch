# Fixed cross-channel mixing at matched delivered RMS, v1

Freeze before model access. Synthetic native CPU decoders, no new pretrained GPT-2
call, GPU, external transfer or publication. A dense mixing matrix changes the
subspace a three-parameter gate can deliver; it is not additional trainable capacity.

## Intervention

Keep the positive quarter-turn temporal transport, width 3, last MLP, strength
1/4, frozen numerical LoRA and frozen +0 bias. Three maps after the branch:
legacy identity; primary Q with rows (15,-6,10), (10,15,-6), (-6,10,15) divided
by 19; or its transpose. Q is orthogonal and preserves the common channel mode.
The three gates start at zero. Use diagonal, train-only residual-RMS updates in
all arms: target (1/4)*(2/256)/sqrt(3), original damping, per-step physical
cap 16, no clipping. Arithmetic, position admission, cache size and owner
identity must remain explicit. Do not choose the rotation sign from results.

## Matched learning

Use the 32 previously used synthetic training prompts, bases 557/563/569,
teachers head/attractor, orders 17/31/73, two epochs of 16 complementary
pairs. Each of 54 endpoints trains for 32 steps: 1,728 real updates, 3,456
native VJPs. Repeat the unmixed arm contemporaneously; all 18 endpoint
artifacts and its 576 steps must match the saved residual-matched control.
Compare train-only second moments to independent pre-existing site inputs and
check routed VJPs at origin. Check actual native output and measured
residual RMS at the first/last step of each path on every training prefix;
relative error on all steps <=1e-4, native output RMS absolute error <=2e-6.
One independent rehearsal, base 571, the four old training prompts, order 5,
one epoch, and only rehearsal examples. Stop on the first failure. Formal CPU
limit 300s, one training attempt, no retries; keep the first failure and records.

## New evaluation

Only after all endpoints and the training manifest are sealed, score native
11-logit rows at nine origins and 54 endpoints on 32 new seven-token prompts
(2,016 scoring forwards). No evaluation gradients. For each h in 3..6,
let a be h+2 and c be h+1 cyclically in 3..6; i,j independently range over
1,2. Templates are [h,c,i,9,a,j,10] and [h,i,9,a,c,j,10]. Same token multiset,
length and final token, modifier attached to different roles. No modifiers in
the six-token training template. These are new *prompts and compound frames*,
not unseen vocabulary; the underlying bare noun pairs have appeared in earlier
evaluation. Use h=3 only and different (rehearsal-only) a=h and c=6 on base
571. Freeze the exact lists and check disjointness against previous train/test
and the other split before loading a model. Once scored these are not unused.

For each template and order measure H (mean signed head margin), A (mean signed
attractor margin), h = H(head trained) - H(origin) and S =
((H-A)(head trained) - (H-A)(attractor trained))/2. Primary Q must exceed
.001 in h and S, beat BOTH identity and transpose in each by >.001 and improve
at least 3/4 lexical frames on each template in every order. Decision gate
also requires strictly more correct head choices than origin and both controls
on each template/order. Require >=2/3 bases for transfer. Report the inverse
arm as a control even if better, ties, both teachers, order ranges, and all
null results. This does not test natural-language understanding.
