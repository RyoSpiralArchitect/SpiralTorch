# Learning the number contrast: ordering is not a verb decision

[Protocol](PROTOCOL.md) | [All 216 template/order cells](orders.tsv) |
[Rust objective](../../../../../../../crates/st-nn/src/causal_lm/paired_token_gap.rs)

The [weakest-lane follow-up](weakest-lane/RESULTS.md) tested the weaker decision directly
on new synthetic inputs: the continuous head-margin gate passed, but the both-correct gate did not.

Only synthetic, two-layer CPU decoders were used. With the base and numerical LoRA frozen,
each condition trained the same three gate coordinates at the last MLP. Two contexts differed
only in the number marker of the supervised noun. All arms used the same delivered residual
RMS per step. No real GPT-2, old English evaluation, GPU, free generation or publication.

## Matched transfer

These are repeated observations: 32 transfer inputs (16 matched pairs), three existing
initializations, and three training orders. Ordering means the singular context gave token
7 a higher margin over token 8 than the plural context did. It does **not** require either
margin to cross the decision boundary. Raw correctness counts strictly positive signed
margins; ties are not counted as correct.

| Teacher and split | Measure | Origin | Independent binary CE | Rank | Rank + center anchor |
| --- | --- | ---: | ---: | ---: | ---: |
| Head, training | Correctly ordered pairs / 288 | 96 | 180 | 180 | 174 |
| Head, training | Correct verbs / 576 | 276 | 323 | 322 | 321 |
| Head, transfer | Correctly ordered pairs / 144 | 48 | 90 | 100 | 98 |
| Head, transfer | Correct verbs / 288 | 144 | 165 | 165 | 165 |
| Nearby noun, training | Correctly ordered pairs / 288 | 96 | 192 | 192 | 192 |
| Nearby noun, training | Correct verbs / 576 | 288 | 312 | 312 | 306 |
| Nearby noun, transfer | Correctly ordered pairs / 144 | 48 | 96 | 96 | 96 |
| Nearby noun, transfer | Correct verbs / 288 | 144 | 156 | 156 | 153 |

All three head objectives improved over the origin. On transfer the head verb choices were
identical **question by question** across the three trained arms, not just tied in aggregate.
All 21 net additional correct decisions came from base 563. Base 569 was already fully
correct on relative ordering with zero learned gate; neither that fact nor any of the
training orders was excluded. The preregistered head ordering and ordering-plus-raw gates
each passed **0/3 bases**, versus the required two.

Anchoring reduced mean absolute pair-center drift compared with rank alone in all 18 head
cells and all 18 nearby-noun cells. Only 6/18 in each teacher reached the declared 10%
reduction. Mean transfer drift for head: 0.02270 (rank) vs 0.02104 (anchored); for the
nearby noun: 0.03983 vs 0.03643. The separate drift-and-order criterion also passed **0/3**.

## Where the eight extra ordered pairs came from

Compared with independent CE, all eight gains from the anchored objective were on base
557, on **three** distinct head-number pairs repeated across orders. There were no offsetting
ordering losses. For those eight observations the learned half-gap `R` was positive
(0.000113 to 0.002506), but the shared center `C` was still -0.524 to -0.432 logits.
Both contexts therefore chose token 8. Rank alone gained ten repeated orderings on this
base; none of these orderings translated into two correct verb choices either.

A direct read-back from the saved logits explains the distinction. Both lanes are correct
exactly when `min(Dp, -Dn) = R - abs(C) > 0`. On base 557 none of the 48 head pairs per
arm crossed this boundary. On base 563 each arm made both decisions correctly on 21/48;
on base 569 each remained at 0/48 despite ordering every pair correctly. This suggests a
next comparison that trains the weaker signed lane, rather than anchoring a possibly
already-biased center. It does not establish that such an objective will improve transfer.

## Verification and replay

The separate base-571 rehearsal completed before the single formal attempt. Formal:
54 immutable endpoints, 3,456 packed updates (6,912 differentiated lanes), 192 training-only
unit-margin captures and 192 calibration prefixes. The 3,648 full-logit training forwards
were sealed and audited before 1,824 transfer forwards. All 54 first paired gradients were
checked against two independent single-lane VJPs, and all 108 first/last physical residual
checks passed. Maximum relative RMS error across all updates was 2.27e-6 (limit 1e-4).
The compiled development wheel passed all three binding tests. The initial Rust build failure
from nonexhaustive tensor layouts was preserved separately from model runs. No scientific
worker was retried.

With the checked-out study source and the local journal archive unpacked, these commands
use the standard library only. They do not import a decoder or fit weights:

```sh
python3 -ISB test_study.py
python3 -ISB audit.py "$CAPTURE/evidence/formal" "$CAPTURE/parents/base/formal" d97ad574b956fce4dddad67c1d535f4e9409b83592862501e15e5538615896bb
python3 -ISB report.py "$CAPTURE/evidence/formal" 7715d7c7815a589d30e80c82240b6746d53e5c401c0324fd63f57036a559e367 "$EMPTY/decisions.json" "$EMPTY/orders.tsv"
cmp orders.tsv "$EMPTY/orders.tsv"
```

| Artifact | SHA-256 |
| --- | --- |
| Rehearsal result | `269b9cea0d1c01574bf3f040952db5bf0ce016c152d5051b30cec4efc7610341` |
| Formal result | `7715d7c7815a589d30e80c82240b6746d53e5c401c0324fd63f57036a559e367` |
| Descriptive decisions, Python 3.8 and 3.12 | `f15ecd159530365318f4e4f4d922d874509cb3afd016c89928e9a8fc202e3d5a` |
| Full cell table | `c5bc109a812e2f0924669efa2df787f6a2ad317be8bc172bf9052ffe484de0dc` |
| Development wheel (not a release candidate) | `1aab9ea2fe7edff5be2c49c6c12ee741b4c8375fb7d1d636918e67f06f9b6329` |
| Local journal archive, 136 verified files including parents and wheel | `be74ce943af23d3537e8dfc891588e9032930e447bebf98edbae3673e7ba5d61` |
