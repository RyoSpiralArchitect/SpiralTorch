# A fitted scalar recovers much of the serving effect, but not its decision order

Can the prior uniform-read wins be explained by moving the singular/plural
threshold? **Often, once the trained gate is present. Not from the zero-gate
backbone on these particular wins.** All original uniform wins were on
synthetic base 563. The already scored prompts are not independent new tests.

UA: uniform-trained gate, served using attention. UU: same donor, uniform
serving read (with the two prespecified per-cell scalings in the RMS control).
Origin: no active gate, same frozen backbone and LoRA. A scalar adds to the
candidate contrast only, not to model weights. Each column below counts
pair/order observations, not distinct examples; there are 144 per head task.

| Head task and scale | Control on UA | Both right | Of the original UU wins | Other wins |
| --- | --- | ---: | ---: | ---: |
| Common RMS (UU: 11) | No shift | 0 | 0/11 | 0 |
| Common RMS | Train mean UU-minus-UA shift | 15 | **8/11** | 7 |
| Common RMS | Maximize training pairs correct | **40** | **3/11** | 37 |
| Literal weights (UU: 24) | No shift | 0 | 0/24 | 0 |
| Literal weights | Train mean UU-minus-UA shift | 30 | **21/24** | 9 |
| Literal weights | Maximize training pairs correct | 50 | **20/24** | 30 |

The common-RMS mean shift reduced UA-to-UU gap RMSE from 0.01958 to 0.00852.
The count-maximizing shift scored more pairs, but RMSE rose to 0.20977;
it is **not** the same predictor. Its 40 successes were 19 on base 563
and 21 on base 569. The task-fitted zero-gate origin also got 21 pairs right
(on base 569), **none** of the 11 UU wins. Both origin controls overlapped
zero original head wins at either scale. In the nearby-noun task the UA
mean shift recovered 3/6 common-RMS and 8/14 literal wins.

## What no constant can do

For raw source contrasts p and n, one shift can make both prompts correct
iff it lies strictly between -p and -n. On base 563 all 11 common-RMS UA
winner intervals could be covered (one scalar per order); the training-fit
misses are not proof that those three cases intrinsically need a variable
shift. On literal UA the upper bound was only 21/24 (7/8 in each order).
On the zero-gate origin **none** of the UU-winning head pairs was even
individually separable at either scale, regardless of the offset. This credits
the gate/host with a different ordering, not the specific serving read with
creating all of those separations.

Covering the winning subset is weaker than reproducing all 32 choices.
On base 563 no scalar on UA reproduced *all* UU choices in any order, raw
or matched. A post hoc witness on the matched head task at order 17:

| Prompt key | UA singular-minus-plural | UU singular-minus-plural |
| --- | ---: | ---: |
| `prehead.345.22` | +0.020812 | +0.012646 |
| `prenear.634.22` | +0.035969 | -0.002504 |

Even a monotone increasing function of UA's scalar contrast cannot give the
*smaller* source value a positive sign and the *larger* source value a
negative sign. Separate template-specific constants alone cannot reproduce
every choice either: matched head has within-template contradictions in
five of six template/order groups on base 563 (the other group is feasible).
This is a property of these predictions, **not** evidence that reversing
those particular choices improves language ability. A useful next test is
to train a gate on pair ordering with a separately measured threshold and
compare on fresh lexical combinations, against the calibrated baseline.

## Implementation and evidence

[Registration](PROTOCOL.md), [all 432 cells](strata.tsv). Public Rust API:
`st_nn::language::paired_offset::PairedTokenOffset`, with `fit_coverage`,
`match_reference`, and single-prompt `adjusted_gap`; a Cargo example is
available as `cargo run -p st-nn --example paired_token_offset`.
The strict solver treats boundary ties as misses, handles intervals with
no representable f64 interior, and prefers zero if already optimal. Native
Rust produced the frozen offsets and applied them to the held-out journal.
An exhaustive Python interval oracle and an independent sum verified **every**
training fit. Seven Rust tests and four harness tests passed; the current
default-feature crate and external consumer were type-checked with the
matching cached Cargo dependencies. No new binding wheel or full workspace
build was needed for this readout.

Rehearsal: 64 native forwards on training prefixes. Formal: 4,608 such
forwards; all 1,152 raw UU comparator rows reproduced all 11 archived logits
bit-for-bit. Both fits were sealed before any transfer logit was admitted.
Transfer: arithmetic on old outputs only. No retries, optimizer updates,
pretrained models, GPUs or publication.

| Artifact | SHA-256 |
| --- | --- |
| Frozen protocol and code | `1027a252b47f765628f94b9ac5f6f05b8aca2c14952b85a8e0f9a6adc33ec277` |
| Both pre-transfer fit seals | `dc8d25cad74316d711b4d829008cbe603a27632cf181c497280a17bf82232178` |
| Rehearsal result | `990dad3af3e7539a34fa22aaaad8a559534caaa6c913559895a9919370348acd` |
| Formal result | `4bba04cc6802282aa625cd53d4519e35bf767a9f0533cb06d900905def902658` |
| Decision-order witnesses | `7411634d97fd3635212c7cf93590788a32146c3a2a0f86a746b5efc61bab9ebc` |
| Complete table | `9856782b9c08c946b5a68897f873aa440a0f43f8a0094dcc4a1aaf44ebf0e0d1` |

For a cold replay, supply the new private delta and the two original portable
archives. This validates the Git source blobs, recompiles only the small Rust
calibrator and recomputes the fits, archived comparator logits, outcomes and
witnesses without importing any model:

```sh
python3 -ISB recover.py verify "$OFFSET_DELTA" "$VALUE_JOURNALS" \
    "$CROSS_DELTA" "$OFFSET_DELTA_SHA256" "$EMPTY_SCRATCH_DIR"
```
