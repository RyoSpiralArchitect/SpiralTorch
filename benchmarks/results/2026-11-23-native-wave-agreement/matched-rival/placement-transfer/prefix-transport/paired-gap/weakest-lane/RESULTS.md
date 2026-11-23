# The weakest verb margin improved; the pair did not become correct

[Preregistered protocol](PROTOCOL.md) | [All 216 cells](orders.tsv) |
[Preceding paired-gap experiment](../RESULTS.md)

This is a new set of synthetic 8-token prompts on the same three tiny decoder
bases, not a rerun of the leading-marker study or evidence on natural English.
Only three final-MLP gate coordinates learned; the base and numerical LoRA were frozen.
Pair the two head numbers (or two nearby-noun numbers) with everything else held fixed.
A pair is correct only when both contexts strictly prefer their own verb.

## What changed

Independent binary CE averages the two lane losses; rank operates on their relative
ordering. Rust's new `weakest_lane` minimizes their maximum binary CE, with an equal
subgradient at an exact tie. On **head transfer**, a preregistered continuous gate
passed on **2/3 bases** (557, 569): on every template and training order the mean of
the smaller signed margin exceeded the origin, binary CE and rank by >0.001 logit.
The nearby-noun control passed this gate on **1/3**. The primary both-correct-plus-margin
gate and the additional no-raw-regression gate passed on **0/3** for either teacher.

| Head number | Measure | Origin | Binary CE | Rank | Weakest lane |
| --- | --- | ---: | ---: | ---: | ---: |
| Training | Mean weaker signed margin | -0.22046 | -0.17274 | -0.19917 | -0.16693 |
| Training | Individual verb decisions / 576 | 270 | 285 | 285 | 275 |
| Training | Both decisions correct / 288 | 0 | 0 | 0 | 0 |
| Transfer | Mean weaker signed margin | -0.22276 | -0.17735 | -0.20350 | -0.17035 |
| Transfer | Correctly ordered pairs / 144 | 51 | 60 | 60 | 55 |
| Transfer | Individual verb decisions / 288 | 132 | 144 | 144 | 135 |
| Transfer | Both decisions correct / 144 | 0 | 0 | 0 | 0 |

These counts repeat **32 distinct transfer inputs** over three initializations and
three training orders, not independent linguistic examples. On the nearby teacher,
transfer also had zero both-correct pairs in all arms; individual correct was 120 at
origin and 138 / 138 / 120 after binary / rank / weakest. There were no weakest-margin
ties on either split.

## Why the loss is not a quality win

Compared with CE, all head raw-decision differences were on base 563: six gains and
15 losses across eight distinct inputs, for a net loss of nine. Nine repeated pairs
(four distinct pairs) went from one correct decision to **neither**. On 557 and 569
the weakest objective **always** selected the first (singular) lane during training
(384/384 on each base, both teachers and orders combined). Across all 1,152 weakest
updates the split was 875 first, 277 second, zero ties. Thus a transferable increase
in the weaker margin does not by itself show learning of the grammatical contrast.

A margin identity explains how these measurements coexist. Write R for the half-gap
and C for the shared center. Then the weaker signed margin is R - abs(C). On base
557, relative to binary CE, the weakest objective reduced mean absolute center by
0.02654 logit but **reduced** mean R by 0.00937, net +0.01717. Base 563 also traded
ordering for centering (+0.00435 from centering, -0.00213 from R). Base 569 had
small gains on both terms (+0.00150, +0.00011). These are descriptive differences
of already-scored endpoints, not interventions on the two terms.

A separate descriptive [training-only diagnostic](local_capacity.py) uses the exact
two origin gate gradients for each pair. In the calibrated RMS metric, 95/96 head pairs
had opposing signed gradients (negative inner product). For each pair allow its own
best direction: maximize its smaller **first-order** gain on the metric ball. Relative
to the smaller of the two single-lane optima, the median simultaneous capacities were
0.317, 0.467 and 0.135 on bases 557, 563 and 569. All 192 directions (including the
nearby teacher) respected the physical cap. This is an optimistic, per-pair local
calculation at **untrained** weights; it is neither a measured finite update nor one
gate direction for the corpus. It motivates testing whether the adapter can independently
steer the two contexts instead of tuning the weakest-loss weight.

## API and verification

On this development build the two prefixes must be distinct. Row one favors the first
token ID; row two favors the second. No sampling, decoding temperature, or optimizer
step is implicit:

```python
r = model.paired_token_gap(prefix_singular, prefix_plural, [singular_id, plural_id], "weakest_lane")
print(r["weakest_lane"])  # signed_margin; active: [first_lane, second_lane]
g = model.geometry_paired_token_gap_gradients(prefix_singular, prefix_plural, [singular_id, plural_id], "weakest_lane")
```

The gradient call requires a trainable geometry adapter. It returns a handle sealed to
the original model state. Existing binary/rank/anchored reports keep their prior fields.
This is a source API, not a new PyPI release.

Rust tests covered saturated or coincident rounded losses, true ties, finite differences,
and layout equivalence. The native CPU binary passed all 375 tests in its causal-LM
module. The CPU wheel passed five binding tests including the two
new native weakest-lane checks. Each synthetic worker ran exactly once; the rehearsal
made 24 updates. Formal: **3,456 updates**, 6,912 differentiated lanes, 192 unit-margin
captures, 192 calibration prefixes, 54 endpoints. All first-step gradient, all 108
physical first/last checks and all step checks passed (maximum relative RMS error
2.86e-6 vs limit 1e-4). Training's 3,648 full-logit forwards were sealed before the 1,824
transfer forwards. No real GPT-2, GPU, network, publication, or favorable retry.

Python 3.8 and 3.12 independently regenerated the same result, cell table and descriptive
diagnostics. The first floating-point capacity diagnostic failed an artificial exact-
opposition test; its source and output were retained separately. The corrected helper
projects on the dual metric segment in rational arithmetic before selecting a direction;
neither diagnosis ran the model. Replay after unpacking the private archive:

```sh
python3 -ISB audit.py "$CAPTURE/evidence/formal" "$CAPTURE/parents/base/formal" \
  bdf037246e0836b07f993a04679cbb43d4eb3abb6657ccec27fc23c7a630f6a4 --table "$EMPTY/orders.tsv"
cmp orders.tsv "$EMPTY/orders.tsv"
python3 -ISB readback.py "$CAPTURE/evidence/formal" \
  45ad798cb2b80627d09afc5b97889e2dd8435d4eedb5a1186fda43fc8d8c4955 "$EMPTY/decisions.json"
python3 -ISB local_capacity.py "$CAPTURE/evidence/formal" \
  bdf037246e0836b07f993a04679cbb43d4eb3abb6657ccec27fc23c7a630f6a4 "$EMPTY/capacity.json"
```

| Artifact | SHA-256 |
| --- | --- |
| Rehearsal result | `76b70f0ccd3adb2cbec514ce3e430e7e30e74afa27a3786145dc2b7bdfb48409` |
| Formal numeric result | `45ad798cb2b80627d09afc5b97889e2dd8435d4eedb5a1186fda43fc8d8c4955` |
| Complete table | `cfa38f158c26fe1e35a72fdd027f5ae45089ea2f627ac49ade7067c987256ef2` |
| Endpoint decision decomposition | `50e41c4bbfefebf114cb6788b4d853432c47700a3b2b420d92c352dce320634a` |
| Initial first-order diagnostic | `055d8c565e8f57898a7f8047d1076d2d3bdd5f84139cb3e885cd4e1bf9469cf8` |
| Development wheel | `2cc87ccddf5db19bffac446cae80e79fc0fb4723048158fa332eb8c6ba40f936` |
| Private archive (143 verified files) | `a0f48250de7d7dfa869bac02e8f2fc55ed427c3060980d434151d62153c77cf2` |
