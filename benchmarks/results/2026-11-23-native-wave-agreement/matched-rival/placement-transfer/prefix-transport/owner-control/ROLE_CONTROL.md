# Constructed role-routing control, specified after the training outcome

This checks a different claim, not a new training cell or a way to pass a
failed transfer gate. Do not use language prompts, decoder weights, hidden
activations, learned readouts, or any previously held-out observations.

Six positions, three channels, activations all zero except channel x holding
a signed number at positions 1 and 4 (zero-based). Put every pair of signs
through the Rust standalone prefix operator at amplitudes 1 and 1/2. Both
positions use the same amplitude. Fixed gate=(1/8,1/8,0), bias=0,
strength=1/4. At the final position, lags are four and one. With no rotation
the final residual is proportional to (head+attractor,0); conflicting pairs
are identical. For a positive turn it is proportional to (head,attractor);
for an inverse turn to (head,-attractor). The factor is amplitude/192.

At fixed weights, run both final-axis cotangents: for rotated inputs, x
sensitivity with respect to the two number inputs must be (1/192,0);
y must be (0,+1/192) or (0,-1/192). For identity the corresponding pairs
are (1/192,1/192) and (0,0). Also check the two raw parameter gradients
against the formula and positive recovered signs in every rotated case.
One CPU worker, 24 forwards, 48 local VJPs, zero updates, 60 s maximum.
This construct checks that the primitive and its derivative *can* route
roles encoded this way; it says nothing about whether real decoder hidden
states encode grammatical number on a shared axis.
