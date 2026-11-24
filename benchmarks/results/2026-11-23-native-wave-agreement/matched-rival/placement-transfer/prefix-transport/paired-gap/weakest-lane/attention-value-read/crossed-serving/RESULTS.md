# Uniform wins needed the fitted gate *and* the serving read

On the already-used synthetic subject-agreement prompts, neither transplant
carried the wins by itself. The only combination that got both members of a
number pair right was **uniform-trained coordinates with a uniform read**.
This held before and after matching delivered residual RMS on training data.
All such wins were on base **563**; bases 557 and 569 had none. This is a
within-model intervention, not evidence that uniform attention generally works
better. In all cells the backbone's ordinary attention was unchanged.

| Task and scale | Attention / attention | Attention / uniform | Uniform / attention | Uniform / uniform |
| --- | ---: | ---: | ---: | ---: |
| Head, literal weights | 0 | 0 | 0 | 24 |
| Head, common RMS | 0 | 0 | 0 | 11 |
| Nearby noun, literal weights | 0 | 0 | 0 | 14 |
| Nearby noun, common RMS | 0 | 0 | 0 | 6 |

Numerator: both singular and plural verbs correct; denominator: **144 pair
observations per cell** (16 pairs x 3 bases x 3 orders). First axis is the
coordinate donor; second is the serving source. No ties. Head individual verb
counts, in table order, were 142, 139, 144, 167 of 288 raw and 140, 133, 144,
155 matched. The 24 and 11 head successes cover eight and four distinct pairs
on base 563 respectively, not 24 or 11 independent examples. The raw/matched
overlap was eight pair/order observations: 16 were lost and three newly
gained. For the nearby task the 14 raw and six matched wins were disjoint.

## What changed in the weaker of the two predictions?

At matched training RMS the mean head weaker-margin effects of changing the
serving source to uniform were +0.00481 with attention-trained coordinates
and +0.01120 with uniform-trained coordinates. The interaction was +0.00639
pooled, but +0.01903 on base 563 and below +0.00011 on each other base. The
global matched RMS is not a per-channel, per-example or per-logit match.

In a **post hoc** read of saved logits, let p and n be the singular-minus-plural
logit gaps on the two prompts. The weaker signed margin is exactly
`(p-n)/2 - abs((p+n)/2)`: verb separation minus shared offset magnitude.
Holding uniform-trained weights fixed in donor/direction (with the prespecified
per-cell magnitude matching), the UA-to-UU move on the 11 matched head wins
improved the weaker margin by +0.02948 on average: **+0.03290 from recentering,
-0.00342 from separation**. Recentering was the larger contribution in all 11.
Across *all* 144 head pairs the corresponding contributions were +0.01021 and
+0.00099. This is an accounting identity, not evidence for an independent
semantic concept or intervention on a particular channel. A useful next control
is a single prediction-offset correction estimated from training pairs only,
tested against the same gate/read combination, before adding a learned router.

## Evidence and reproduction

[Registration](PROTOCOL.md) and [all 288 template-level outcome rows](strata.tsv).
Frozen source commit `1d749d8`. The parent capture is the saved [value-read
study](../RESULTS.md). Two native CPU workers, no retries: 128 rehearsal and
4,608 formal forwards, no weight updates and no pretrained model calls.
The 1,152 formal factual rows reproduced **all 11 logits bit-for-bit**; 576
batched Rust geometry checks reproduced all delivered training site rows.
Largest relative common-RMS error: 6.40e-8 rounded up (limit 2e-5).
Four pure-Python unit checks cover physical scale, rounding/zero delivery,
signed pairs, ties, sufficiency classification and the interaction sign.
No new English data, no GPU, no public upload.

| Sealed artifact | SHA-256 |
| --- | --- |
| Both treatment plans and source pins | `fdeda563994b622f94a54338369d237bbd68a425131a5fb7d4143ec7c1bf64b0` |
| Rehearsal, all logits and sites audited | `ca4cad8b11b605c9a40217146b0458c6ebef9de1bcdfe457c35ebd64fc5ccb48` |
| Formal, all logits and sites audited | `26dc374dced142e1c69644347c1e43d0d010b4d69033d4f40805657198c32eb0` |
| Full signed-margin decomposition | `425b98de5ae6c76978fa25a90df78d754d3d0269dbc4f463f418e720aa50604f` |
| All template-level outcomes | `d91cdb3071919726b35ae9a1f98bce6382ad5b42688f8fd2044faf4fc59525b9` |

With the parent and new captures available, these commands check the recorded
results **without** importing an ML framework or running inference:

```sh
python3 -ISB audit.py "$PARENT" "$CAPTURE" rehearsal "$FROZEN_SHA" --verify
python3 -ISB audit.py "$PARENT" "$CAPTURE" formal "$FROZEN_SHA" --verify
python3 -ISB gap_readback.py "$CAPTURE/formal" "$FORMAL_SHA" "$EMPTY_OUTPUT_DIR"
```

With the two portable archives, `recover.py verify "$DELTA" "$PARENT_TAR"
"$DELTA_SHA" "$EMPTY_SCRATCH_DIR"` re-extracts source, checks Git blob identities,
and repeats both audits and the logit decomposition without a wheel or a checkout
in its scratch directory.
