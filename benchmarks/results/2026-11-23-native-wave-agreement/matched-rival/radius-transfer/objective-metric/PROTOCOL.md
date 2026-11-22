# Does the local channel map change the conflict between two losses?

Reuse all first-step anchors with *both* full-vocabulary and binary-choice CE
gradients on the same initial affine GPT-2 host: seeds 17, 31 and 73, three
distinct old training prompts. Use the one available artificial anchor first.
This is a retrospective mechanism check, not a new independent quality test.
Do not open the old assessment plan/logits, load a decoder, update weights, or
run the pending radius learning study. Run the archived-input kernel once, CPU,
60 seconds per data set, stop on the first error. Preserve failures.

The placement is the **last** block's attention output, the rest of that
block is row-local, and only the last prompt position has a supervised token.
For the affine residual, the stored bias gradient is therefore
`f32(strength * incoming)`, with strength exactly 1/4. Recover each incoming
coordinate by scaling by four. Reject zero, subnormal, minimum normal, or
nonfinite bias coordinates: rounding at underflow need not be invertible.
Require a finite binary32 inverse, exact round-trip, and exact native replay
of **both the gate and bias bytes**, for **both** objectives. Independently
compare the recovered binary incoming vector with minus the archived rival
probability times the independently captured unit-logit-margin incoming
vector (relative L2 <= the original capture tolerance). Do not infer a
full-vocabulary incoming vector for the WaveGate host.

Hold fixed the site input, pre-adapter gate/bias, residual strength, and
these two recovered *affine-host* incoming vectors. For each radius 1 and 4,
use native Rust to compute the four local gate/bias VJPs (affine,
saturation, projection, both) for both seeds. Reuse each arm's local forward.
The affine and saturation controls must be bit-identical across radii.
Retain the raw binary32 vectors, not just a sign or a cosine. Keep actual
archived R=1 WaveGate-host gradients separate from this intervention.

Primary descriptive question: on the previously identified driver.11 conflict,
does the complete R=1 map alone switch an adverse full-loss slope to a favorable
one with the *incoming loss directions held fixed*? If not, do not credit the
map alone. Show all three prefixes, both separate stages and R=4 regardless
of outcome. The sign is computed from the exact integer dot of the two saved
binary32 parameter gradients. Positive dot means the negative choice gradient
also descends full CE. Report cosine (invariant to overall positive scaling)
and the full-loss tangent at an ideal negative-choice displacement of length
1/256. If there is no choice direction, report it as undefined. This is not
a rounded native step or a measured finite loss. No significance or radius
selection criterion; the reserved real comparison and its wheel are unchanged.
