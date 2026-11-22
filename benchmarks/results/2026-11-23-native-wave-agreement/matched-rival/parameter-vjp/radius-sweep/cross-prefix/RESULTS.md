# A larger radius amplifies agreement and interference at a fixed step length

The local net effect was positive for all six receiving lexical quartets on
each of the two factual hosts. At an identical parameter-step length of
1/256, R=4 gave **3.50x the balanced first-order margin change** at R=1.
This was mostly a change in how the receiving prefix responded to a step,
not a change in which way the donating prefix would step.

Each host has the same 24 **training** prefixes. Each receiver is paired
with its 20 donors from other quartets, ten sharing its correct verb and
ten requiring the opposite verb. There are 480 ordered comparisons per
host; these are not 480 independent sentences. No decoder, geometry forward
or backward, update, held-out scoring, or generation occurred in this study.
Both R=4 local derivatives use the original, fixed incoming adjoints; there
is no observation here of what a decoder actually trained at R=4 would do.

Mean tangent change in gold-minus-rival logit margin, in **millionths of a
logit per normalized step**:

| Factual host | Other-quartet donors | R=1 | R=4 | Change |
| --- | --- | ---: | ---: | ---: |
| Affine | Same verb | 32.373 | 112.855 | +80.482 |
| Affine | Opposite verb | -25.982 | -90.497 | -64.515 |
| Affine | Balanced | 3.196 | 11.179 | +7.983 |
| WaveGate | Same verb | 32.387 | 112.902 | +80.515 |
| WaveGate | Opposite verb | -25.938 | -90.343 | -64.405 |
| WaveGate | Balanced | 3.224 | 11.279 | +8.055 |

In the crossed-radius decomposition, receiver sensitivity accounts for
99.924% / 99.927% of the balanced change (affine / WaveGate); donor
rotation accounts for 0.076% / 0.073%. Unit-to-unit mean cosine shifts only
from .07327 to .07342 / .07389 to .07404. The receiver term includes
both its norm and direction. The local effect is not a 3.50x longer
parameter update, nor a 3.50x accuracy or held-out learning gain.

| Receiving quartet | Affine R=4 minus R=1 | WaveGate R=4 minus R=1 |
| --- | ---: | ---: |
| actor | +7.926 | +8.013 |
| doctor | +8.669 | +8.732 |
| driver | +8.345 | +8.414 |
| farmer | +7.544 | +7.614 |
| singer | +7.453 | +7.526 |
| teacher | +7.962 | +8.030 |

The increase is small in absolute units and accompanied by substantial
opposite-verb interference. It is a reason to run the already specified
learning comparison, not to change its threshold or make R=4 the default.

## Evidence and reproduction

The [analysis rule](README.md) and code were committed at
`595679fe9bd7f1b6d8c636f5f8f98841f2db1ce5` before running on the
real-model gradient archive. The native CPU wheel used Rust 1.99.1, f64
accumulation and f32 Gram output; no Torch or NumPy. Every cell in both
48-by-48 matrices was independently checked with a binary64 compensated
sum. The largest absolute dot error divided by the product of row norms
was 4.82e-8 or less (limit 2.5e-7). Both precision paths agreed on every
fold classification. A one-quartet toy correctly returned no estimate.

Inputs are the captured margin and subsequent radius archives identified by
their manifest hashes in `transfer.py`. With the source-built CPU wheel:

```sh
"$VENV/bin/python" -I -B transfer.py run "$CAPTURE" "$SWEEP" real "$EXT_SHA256" "$NEW_OUTPUT"
python3 -ISB transfer.py verify "$CAPTURE" "$SWEEP" real "$EXT_SHA256" "$NEW_OUTPUT"
```

The second command replays saved matrices without importing SpiralTorch.
The local wheel SHA-256 is
`1d2cf81992449146a9cc4fe15fd796c5f167d332a23c69788b15fc779baa150b`;
extension SHA-256
`349bc0091e968c85f0cd0ee81e1b38631d6d301b7ba5833cc40419c13d9085fd`;
real-gradient result SHA-256
`0aed1b8571b04a4c5462f1b0138c3855857baad2782abc1f839da7e3e6fddd5c`.
No pretrained model or evaluation corpus is required for either command.
