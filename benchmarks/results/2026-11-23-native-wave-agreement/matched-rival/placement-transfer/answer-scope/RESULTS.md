# Removing stop supervision did not transfer the first-token decision

On the three fresh generated bases, **0/3 passed the probability gate and
0/3 passed the decision gate**, for both WaveGate and matched affine at the
second MLP. No Wave-specific advantage. On all 36 head-trained endpoints
(3 bases x 2 geometries x 2 objectives x 3 orders), **the same 8 of 16
held-out answers won as at their respective origins**: zero switches, zero
ties. In every case the largest measured movement in any prompt's
first-token log odds was smaller than the smallest absolute starting margin.
The split is held out from training but the symbolic task was reused.

The [frozen protocol](PROTOCOL.md) distinguishes next-token probabilities
from the probability of the subsequent terminator. This is not a rerun of
the earlier successful complete-line placement gate on its original base,
and neither result measures pretrained English language quality.

## All bases

Head-margin change `h` and the matched two-teacher interaction `S` use
**only** first-token log probabilities, even when the training loss included
the stop. Medians over training orders, in nats; gold gain is the origin
minus endpoint full-vocabulary gold answer-token NLL. Orders are not new
models. Figures below are rounded; criteria use every unrounded order.

| Base | Geometry | h: answer | h: through stop | S: answer | Gold gain: answer |
| --- | --- | ---: | ---: | ---: | ---: |
| 101 | Wave | 0.000380 | 0.000516 | -0.0000255 | -0.000171 |
| 101 | Affine | 0.001072 | 0.001067 | <1e-8 (abs) | 0.046019 |
| 211 | Wave | 0.008501 | 0.008501 | 0 | 0.100281 |
| 211 | Affine | 0.000059 | 0.000256 | 0.0000190 | 0.000043 |
| 307 | Wave | <1e-8 (abs) | <1e-8 (abs) | <1e-8 (abs) | <1e-8 (abs) |
| 307 | Affine | <1e-8 (abs) | <2e-8 (abs) | <2e-8 (abs) | <1e-7 (abs) |

The apparent best gain (211/Wave) did not distinguish the two teachers:
the answer-trained `S` was zero in all three orders, and the head-margin
gain was almost identical when training through stop. At order 31 its
training-split answer gain was 0.10066 nats with 16/32 correct before and
after; on the test split 0.10028 with 8/16 before and after. Test log odds
moved at most 0.03047 nats across its answer-trained endpoints versus a
nearest initial boundary of 0.33856 nats. Removing the stop did not turn
a loss reduction into a changed decision.

See [every matched order](per-order.tsv), including complete-line stop
contributions, candidate-conditional gains, switches and ties. The full
readback contains both training and held-out splits and endpoint distances.

## What the gradients suggest

**Exploratory readback, not an additional pass or a preregistered gate.**
Two order keys start on prompts where the two teachers demand opposite
labels. This gives two genuinely matched comparisons at *identical*
parameters and inputs on each base. The remaining order starts on an
agreeing prompt; its two gradients matched exactly. The [paired vectors
summary](opposite-labels.tsv) includes both geometries and training scopes.

On 211/Wave with answer-only training, the opposite-label gradient cosine
exceeded 0.99999999999 on both prompts. Each head-trained trajectory ended
about 0.25 from its origin after travelling 0.25. On 307/Wave, both cosines
were below -0.99999998; the path also travelled 0.25 but ended only
0.000112..0.000113 away. These are observations about two initial contrasts
and saved endpoints, not proof that every subsequent step had the same
geometry. They suggest different mechanisms behind the same null accuracy.

A useful next control is **pure binary conditional CE** versus the current
50% mixture on unselected weights. For a binary menu, pure conditional CE
makes the two opposite-gold gradients at a shared input proportional to
opposite signs of the same logit-margin derivative (if it is nonzero).
That tests whether the full-vocabulary component masks the label contrast;
it does not guarantee that the resulting direction distinguishes contexts.

## Receipts and reproduction

Protocol `1443a6b`, frozen implementation `d830663`. The formal CPU worker
completed on its first attempt: **4,608 updates, 3,744 complete-menu score
records, 234 direct full-vocabulary forwards, 72 final geometry artifacts**.
Every recorded displacement was checked against the signed normalized
gradient from canonical f32 values. Worst relative length error 3.96e-5,
worst relative direction error 5.26e-5; direct token costs differed by
at most 4.45e-16 nats. The disjoint finite-difference rehearsal is excluded.
No model download, real GPT-2 call, GPU, free generation or publication.

The [native API guide](../../../../../../bindings/st-wasm/tests/native_causal_lm.md)
shows token and through-stop supervision in Python and WASM. Rust decoder
tests: 358 passed. Installed CPU-wheel session suite: 35 passed, one skipped.
Actual Node/WASM: three objective mixtures matched direct logits, three
native parameter steps, five invalid-input checks. Browser not exercised.
One build stopped at the disk floor; a Python check was stopped before
launch. New compiler intermediates were transparently compressed before
resuming. A Node invocation rejected an obsolete CLI flag before loading
WASM; the ESM test then ran without the flag. These setup receipts are
preserved separately from the single formal model attempt.

Local evidence directory: `native-answer-scope.jsHOaB` in the SpiralTorch
validation folder. The weight files, immutable sources, receipts and journal
are bound by the plan and manifest. To verify without loading a model,
supply new output paths in an existing writable directory:

```sh
python3 -ISB test_scope.py
python3 -ISB verify.py "$EVIDENCE/formal" "$OUT/result.json"
python3 -ISB readback_table.py "$EVIDENCE/formal" "$OUT/result.json" "$OUT/pairs.tsv" "$OUT/grad.tsv"
cmp per-order.tsv "$OUT/pairs.tsv"
cmp opposite-labels.tsv "$OUT/grad.tsv"
```

| Item | SHA-256 |
| --- | --- |
| Formal plan | `dbf6bd97539b88747e79a786b2965985c0e013ff53854a1b374721af0bd86d7f` |
| Formal raw manifest | `16b3e552e4fc2f2a76eaef80591404a817665d4d63c50c9d2c66f1394057ac3a` |
| Full numeric readback | `276a30ab8f24f9788094d6a5b8049dea320c5e51980208f646a0f3c989537b84` |
| Scoped comparisons | `1028c26594b200096f0708bb6c3217418adac4ca34df12c3bb07a4f8ba736a1a` |
| Opposite-label readback | `acb5d5f520e575c2f94efeecf6c24c418c5e660795e1028ee50b3d068e9fa7fd` |
| Private archive of attempts | `a11e104fa5ad47552c6760576585d1243f86132a5635eec8d247108fbb67d856` |
| CPU development wheel | `3870a1306f56fa03415ba383a6d2a857297f329281f357d6fe5cd6cdc0c150a7` |

Follow-up: [pure two-token loss and balanced updates](../conditional-update/RESULTS.md)
tests whether reversing the local gradient when the label flips is sufficient
to change the answer selected on held-out symbolic prompts.
