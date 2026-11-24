# The local saturation stage can remove a choice/full-loss conflict

On the previously identified `driver.11` training prefix, **saturation alone
changes the sign** of the full-vocabulary loss derivative along the negative
binary-choice gradient. Projection alone does not. Both incoming loss
directions here come from the *same affine-host decoder*, so a different
downstream decoder response is not needed for this particular sign change.
This is a local geometric effect, not an observed change in next-token quality.

The [protocol](PROTOCOL.md) and Rust/Python implementation were committed at
`b636e50` before computing the intervention on the archived GPT-2 inputs.
There were three eligible prefixes, all plural, one first step per previous
order. No new decoder forward, backward, update, generation, or held-out scoring.
New work: six paired local kernel calls, 48 gate/bias VJPs, CPU, one attempt.

## With both incoming adjoints fixed

Cosine between the two parameter gradients. Positive means stepping against
the choice gradient also descends full CE to first order. The final column
is the separate historical WaveGate host, with its *own* incoming gradients,
not another arm of the fixed-input intervention.

| Prefix | Affine | Saturation | Projection R=1 | Both R=1 | Projection R=4 | Both R=4 | Factual Wave R=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| driver.11 | -.0572 | +.2895 | -.0188 | +.2863 | -.0188 | +.2864 | +.2846 |
| teacher.11 | +.0490 | +.3236 | +.0379 | +.3210 | +.0379 | +.3219 | +.3208 |
| actor.10 | +.1731 | +.3223 | +.1399 | +.3227 | +.1399 | +.3237 | +.3201 |

For `driver.11` the predicted full-NLL change at an **ideal** negative-choice
step of length 1/256 is +15.73 millionths for affine, +0.076 for projection
alone at R=1, -50.66 for saturation alone, -8.15 for both at R=1, and -29.70
for both at R=4. No such counterfactual step was taken. On the three rows
changing R=1 to R=4 changed the two-gradient cosine by at most 0.0011, but
magnified the absolute full-loss tangent for the complete map by 3.47-3.65x.
That is not evidence to promote R=4 or tune a production threshold.

## Which channels changed?

Post-hoc, using the raw parameter vectors. For `driver.11`, 11 of 768
channels changed between affine and saturation; the other 757 produced
identical full and choice gate/bias bytes. All raw gate magnitudes were at
most 0.3125 (threshold 1). It was the **post-affine value**, not the raw
gate, that crossed the saturation boundary. The threshold mask matched
the changed channels exactly on every row: 11, 10 and 10, respectively
(zero on the artificial control). The nearest real value was more than
0.089 from the boundary; separate and fused multiply-add rounding agreed
on its membership.

For the driver prefix, gate coordinates accounted for 99.98% of the increase
in the gradient dot; the bias change alone did not reverse its sign. Channel
496 had the largest absolute change (36.1% of the sum of absolute changes;
post-affine value -30.84). Omitting its two coordinates from *both dot
products* makes both dots positive. Omitting any other single channel leaves
the affine-negative/saturation-positive contrast intact. This arithmetic
partition is **not** a channel ablation or a claim about what the channel
encodes. The other two prefixes had no sign conflict to begin with.

## Checks and reproduction

- All six recovered affine full/choice gradients matched both original
  parameter blocks byte for byte. Relative L2 discrepancy between recovered
  choice incoming and the independent scaled unit-margin incoming was at
  most 1.14e-6. Across all four R=1 maps the corresponding parameter-vector
  discrepancy was at most 1.49e-6. The margin anchor is a different backward
  seed, not a bitwise identity. Exact dyadic dots agreed with independent
  wide dot products; both summary files replayed without loading a model.
- 337 Rust LM/geometry tests, 21 installed Python geometry tests, 67 installed
  Python LM tests, and 8 stdlib analysis controls passed. Actual test-only
  counters measured three alternative map constructions for the pair vs six
  for two separate calls. No wall-time speedup is claimed.
- The kernel is also usable outside a decoder:
  `first_maps, second_maps = batch.wave_parameter_pullback_pair_buffer(first_f32, second_f32, config)`.
  Each result contains independent native-endian float32 gate/bias buffers
  for affine, saturation, projection and both. They are local derivatives,
  not update handles. For native decoder updates use the
  [shared sealed gradients](../SHARED_OBJECTIVES.md).

For a copy of the known archives, run the following from this directory.
`CAPTURE` is the training-only margin capture, not the pending learning plan;
`MATCHED` can be the partial training-only restore. Output paths must be fresh.

```sh
"$VENV/bin/python" -I -B study.py measure gpt2 "$CAPTURE" "$MATCHED" "$ANCHORS_JSON" "$NEW_OUT" "$EXTENSION_SHA256"
python3 -ISB study.py verify gpt2 "$CAPTURE" "$MATCHED" "$ANCHORS_JSON" "$NEW_OUT" "$EXTENSION_SHA256"
python3 -ISB channel_partition.py gpt2 "$NEW_OUT/rows.jsonl" "$NEW_OUT/support.json" --capture "$CAPTURE"
```

CPU extension SHA-256: `19e30105f47aae02e7f13014c488314a7c4b667ab9e376be2a2f94a2802df3d7`.
Raw training-input tape: `cb1b245bf9435a3b36fbe31a80b8468bc6a65d973d163f37482089dd0d88f9f8`.
Primary summary: `64428d1872f0b0eaa9d6c1e48db32b0f9d69281d3c8416bd434c44f5b8e4409f`.
Post-hoc support: `40ad65399961d3eea4a5055800955083f4c0c3aba94b62152d79c5f5579faaca`.
