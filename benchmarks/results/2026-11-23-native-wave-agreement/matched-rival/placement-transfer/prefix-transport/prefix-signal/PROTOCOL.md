# Where does the head-number signal reach a prefix gate?

Freeze before observing responses. This is a new read-only decoder diagnostic,
not a retry of the failed standalone property test, another optimizer sweep,
or an additional measurement on pretrained GPT-2. CPU only, no updates,
downloads, held-out evaluation, publication or retries of the formal worker.

Use the previous three width-three artificial bases 557, 563 and 569, the
same frozen numerical LoRA and the same 32 *training* prompts (eight lexical
frames with all four number combinations). One prefix-transport placement
at the final MLP, strength 1/4; positive-zero gate/bias, gate trainable,
bias fixed. Read only training prompts from the frozen split. Reuse all
three rotations; positive vs identity is primary, inverse secondary.
Rehearse on separate base 571 before the single formal 240-second worker.

Each native capture takes the unit raw margin z7-z8, *not* cross-entropy.
It returns the factual site inputs X[t], contributions M[t]=R^lag X[t]/T,
and the final-site incoming adjoint U. For the gate slope, sum M[t] times
B, where B is the actual bias derivative (strength times U rounded to f32).
Compare this decomposition to the factual geometry VJP and to the
difference of two ordinary native conditional-CE gradients with opposite
labels. At every origin compare z7 and z8 to saved original logits. Take
independent central differences of all three gate axes at +/-1/128.
Limits: summed slopes and CE difference absolute error <=2e-6 times
max(1,max abs actual); finite difference absolute error <=2e-5 plus 2%
of the axis slope. Stop at first failed check, retain the journal.

For each lexical frame, flip head number while fixing attractor, and vice
versa. Split the resulting gate-slope difference exactly and symmetrically:
(Delta M)*mean(B) + mean(M)*(Delta B), separately by source position and
axis. Report the magnitude of each route and cancellation before and after
summing source positions. The earlier activations must be bit-identical
until the first changed token. This is a local product decomposition, not
an assertion that all grammatical information is stored in a position.

For each initially wrong strict head decision only, compute the optimistic
first-order fraction of the gap covered by a norm-1/4 gate in the direction
of its *own* signed gradient. Also score that one oracle perturbation on the
same prompt (not training, not a common deployable gate). Report ties,
zero slopes and actual fractions; neither the linear approximation nor
one chosen direction is a global nonlinear bound. No sweep. Formal: 288
captures, 576 ordinary VJPs, 1,728 central-difference forwards; at most
288 additional forwards for the conditional oracle. Hash outputs and
verify all reductions without a model.
