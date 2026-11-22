# Where do the matched updates cancel?

This follow-up uses only the archived training steps of the matched-rival pilot.
It is retrospective: the initial observation that WaveGate ended closer to its
starting point was already known. No model is loaded, no examples are rescored,
and none of the first pilot's files are modified.

## Accounting fixed before inspecting the step breakdown

For each training order and arm, take actual f32 state differences `u[i]` in
gate-then-bias order, not raw or re-normalized gradient approximations. Check
all steps against the pinned training plan, state chain, gradient tape and
native distance receipts. Let `L = sum(norm(u[i]))`. Partition the squared
endpoint-to-path ratio exactly into:

    norm(sum(u))^2 / L^2 = (sum(norm(u[i])^2) + 2*sum(dot(u[i], u[j]), i<j)) / L^2.

Keep the diagonal and four disjoint off-diagonal bins, defined by whether the
subject number and attractor number match or flip between the two sentences.
Negative dot products mean actual updates oppose each other. Gate and bias use
the same full-path denominator, so their contributions add. For 24 examples,
the four bins contain 60, 72, 72 and 72 unordered pairs (276 in total).

The primary comparison is affine minus WaveGate in the verb-pair objective.
For each order, subtract the corresponding bins; they must sum to the
observed difference in squared endpoint-to-path ratios. Call the gap primarily
opposite-subject accounting only if the gap is positive and the sum of the two
opposite-subject bins accounts for more than half of it on **every** order.
This is a signed accounting share; other bins can offset it. Check full CE as
a control, not as another chance to pass the verb-pair comparison.

Two views help interpret the accounting:

1. Within each lexical pair, swap only the subject number, at a fixed singular
   or plural attractor. Conversely, swap only the attractor at fixed subject.
   Each cosine is between two **actual steps at their own positions** in the
   same run. Mark stronger subject reversal only if both fixed-attractor means
   are negative for WaveGate and lower than affine's in all three orders.
2. On the balanced two-by-two training design, partition total step energy
   `sum(norm(u[i])^2)` into the common, subject-number, attractor-number,
   interaction and within-cell residual components. These are orthogonal
   contrasts over the *rows*, not a claim of orthogonal parameter vectors.
   Report gate and bias parts, with no block-specific renormalization. The
   within-cell residual includes lexical and trajectory effects.

These measurements describe recorded, evolving parameter states, not gradients
at one common weight. Gold verb identity is tied to subject number. A large
subject contrast can represent a shared verb direction with alternating sign,
not necessarily learned syntax. No endpoint decomposition establishes a cause
of held-out margin differences or suggests that all cancellation is harmful.

## Reproduction

Run `python3 -ISB test_paths.py -v`. Then run `python3 -ISB analyze.py ARCHIVE
--manifest SHA256 --kind rehearsal --output /new/location/tiny.json`; use
`--kind gpt2` for the real archived paths. The manifest hash must come from the
previous run. The report records the inputs and analysis implementation hash.
Output must be new and outside ARCHIVE. Logit files and the prior per-item
score report are not opened. The dataset plan also contains evaluation metadata;
the analysis selects only its training rows.
