# Post-hoc: does an identical parameter step have identical local leverage?

This is a follow-up to the completed path partition, not another preregistered
gate. It uses only the already audited, single-target, single-rival, mix-one
training journals and the pinned path report. It does not read prediction rows.

For a gold-minus-rival logit margin `m`, the binary CE is
`log(1+exp(-m))`. Its derivative is `-q`, where `q` is the rival probability
within the pair. Rust's `RestrictedRow::blend` assigns the logit seeds
`-q` and `+q`, rounded to f32, with no other class receiving gradient in this
case. Consequently the stored CE parameter gradient gives an estimate of the
gold-margin gradient norm `norm(g)/q`. The local linearized change along the
*actual* parameter step is `-dot(g, step)/q`. Neither number is an observed
post-step logit change. Native backward arithmetic is finite precision.

Match affine and WaveGate by training case and step within each fixed order.
The first case of each order is distinguished because neither adapter has yet
been updated; subsequent cases compare different visiting states. Reject zero
f32 mass, inconsistent restricted loss, non-finite norms and invalid projection.
Do not apply this identity to full CE, smoothed/mixed objectives, or multiple
rivals. The estimate does not separate a local kernel Jacobian from the arriving
gradient, and it does not measure transfer to another example.

Run `python3 -ISB test_sensitivity.py -v` for the independent centered-difference
check on a known margin. Run `python3 -ISB sensitivity.py ARCHIVE --paths REPORT
--paths-sha256 HASH --output NEW_JSON` using the output of `analyze.py`.
The source Rust paths are `crates/st-tensor/src/classification/declared.rs` and
`crates/st-tensor/src/classification/competition.rs`.
