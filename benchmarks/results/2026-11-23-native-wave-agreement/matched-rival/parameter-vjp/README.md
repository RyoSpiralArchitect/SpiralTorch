# One factual margin, four parameter pullbacks

Use only the prior matched-rival **training** rows and the two initial adapters.
Hold the base, gate, bias, strength (0.25), log radius (0) and prompt fixed.
For each of 24 prefixes use both factual decoders (affine, full WaveGate); alternate
the order of the hosts across prefixes. No LoRA or other geometry placement.
Per host there is one factual decoder forward and one backward seeded by
+1 at the gold logit, -1 at the other verb, zero elsewhere. It is not a CE seed.
The Rust API computes affine (A), saturation only (S), projection only (P) and
both (B) parameter VJPs at the same captured attention output and upstream.
These are four local maps, not four counterfactual decoders.

Record actual f32 site input, arriving margin gradient and gate/bias gradients
for each arm. Only the final prompt position has a nonzero arriving adjoint.
Gate and bias order is fixed. Factual hosts may have different adjoints; do
not pool those as if they were identical. Their pre-adapter inputs must match
bitwise. At the first training case of each previous order, check the factual
VJP against the archived binary CE gradient divided by minus the f32 rival
probability (relative L2 error at most 0.001; these are different f32 backward
seeds). No other prior trajectory endpoint is a same-state reference.

Primary descriptive rule, fixed before the GPT-2 measurement: projection
resembles the full map if **both** (i) P attenuates the joint parameter-gradient
norm relative to A and (ii) the vector distance B-to-P is smaller than B-to-S.
Require this on at least 18 of 24 rows on **each** host; the reverse rule for
saturation is symmetric. Zero/equality is not a win. Also report block norms,
cosines to A, and the relative nonadditivity norm `||B-S-P+A||/||A||`.
This is a statement about local derivatives, not endpoint movement, learning
quality, causal recovery of prior held-out margins, or a preferred optimizer.

Preflight: numerical differentiation and independent local forward/VJPs on
artificial weights, then replay the tiny archive on the new native extension.
The real worker is once only: CPU, 300 seconds maximum, stop on first error,
no training, held-out scoring, generation, GPU, retries or publishing. A fresh
private directory contains the plan, incremental raw JSONL, bounded worker
receipt and result. The previous wheel, model, data and evidence stay untouched.
