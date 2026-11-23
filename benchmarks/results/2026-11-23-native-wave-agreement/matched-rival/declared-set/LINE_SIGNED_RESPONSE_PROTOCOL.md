# Signed response of the synthetic first-attention bias

Follow-up to [the one-sided sensitivity study](LINE_SENSITIVITY.md), not a
repeat, extension or part of the real GPT-2 pullback experiment. Use the
archived synthetic CPU extension (SHA-256
`ca72f2436471f8d754d02057bccb5a13b89383d6ccd64f9c76e88c9cc70f0d5f`)
and journal `682daccccc95e7dc01c5f02776210fea52958fd3f58ddf6ac9e3909eee677a49`.
No training, adaptive scale, new data or pretrained-model calls.

1. Reconstruct the exact generated ordinary/wave host, including the fixed
   nonzero LoRA. Read full last-token logits on [1,2], [2,4], [2,1,3],
   in that order. Using independent immutable copies, offset only the f32
   `transformer.h.0.attn/bias[0]` by -1/256, +1/256, -1/128 and +1/128.
   Pin all three model/artifact identities, verify the realized displacement
   and verify that every other scalar, the LoRA and the origin stay unchanged.
2. Keep the five sets of logits, including the origin. Across 15 full-vocab
   synthetic readouts, compute KL(origin || variant) in nats. The old journal
   stores the norm, not the sign, of the source coordinate gradient: exactly
   one of the two 1/256 variants must reproduce its three KLs (abs 3e-15 nat,
   rel 1e-10). This is a read-back check, not a direction selection rule.
3. Call the earlier concentration persistent **only** if the middle prompt is
   the unique peak and takes at least 85% of the summed KL in all four
   variants. Equal peaks, zero total and an unsuccessful gate are not wins.
   Report per-prompt KL in units of T=1/65536 nat, shares and, separately for
   each sign, the ratio of total KL on doubling the displacement. Quadratic
   behavior would give 4; no fitting, pass threshold or global bound for this
   descriptive ratio. Stop this single CPU worker on the first error.
4. Even if concentration persists, moving the geometry while dropping or
   moving the existing LoRA is **not** a matched placement comparison.
   Inspect the Rust placement contract; do not run a mismatched training arm
   or change admission as part of this no-update study.
