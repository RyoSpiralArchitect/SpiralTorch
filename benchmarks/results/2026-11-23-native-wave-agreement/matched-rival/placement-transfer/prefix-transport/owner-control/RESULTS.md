# Removing bias competition was not sufficient to change the verb

The [frozen training comparison](PROTOCOL.md) is complete. All 108 trained
endpoints selected the same held-out verb as their origins on all 16 prompts,
for either teacher. Both the primary quarter turn and secondary inverse turn
passed the probability and decision gates on **0/3 artificial bases**,
against the gate-only identity control, the bias-only control, or both.
This is a result about the tested recipe, site, length budget and reused
symbolic task, not a result about fine-tuning GPT-2.

| Base | Correct head choices before/after (of 16) | Head-margin gain from quarter-turn gate alone | Advantage in that gain over identity gate | Held-out frames improved |
| --- | ---: | ---: | ---: | ---: |
| 557 | 4 / 4 | +0.0003314 to +0.0003318 | +0.0000861 to +0.0000863 | 2/4 |
| 563 | 8 / 8 | +0.0003163 to +0.0003164 | +0.00000451 to +0.00000481 | 4/4 |
| 569 | 8 / 8 | +0.00001840 to +0.00001846 | -0.000000731 to -0.000000722 | 4/4 |

Ranges are across three orders of the same examples; values are rounded.
Base 557 increased the mean head margin but its teacher-selectivity change
S was negative in every order. Base 563 improved all four frames in every
order but did not approach the 0.001 margin or selectivity requirements.
On 569 even the small advantage over identity reversed sign. Removing the
competing bias is not a universal way of improving the rotation. The
[per-order table](orders.tsv) contains all conditions, both teachers and
splits, whole-vocabulary scores, ties, strict choices, and changes in the
offset, head, attractor and interaction components of the margins.

**How far were the wrong decisions from flipping?** A descriptive [read-back
of the same logits](wrong-conflicts.tsv) selects held-out conflicts that the
origin got strictly wrong. A fraction of one means the head-trained endpoint
reached the tie, zero means no progress, negative means it moved away.
For the primary gate, the best fraction across prompts and orders was
**0.0210%, 0.2748%, 0.0211%** on bases 557, 563 and 569. In each order,
4/6, 2/4 and 2/4 of their wrong conflict cases moved toward the correct
choice. This is observed movement, not an upper bound on what an optimal
gate or a different site could achieve.

## What was actually measured

The same installed CPU extension as the parent completed the single formal
worker in 0.99 s (240 s limit; these decoders have width three). It did 3,456
paired updates, 6,912 individual training VJPs and 6,048 full-vocabulary
snapshot forwards. The 18 origins matched archived frozen logits bitwise;
216 origin member gradients matched the parent's original joint-training
measurements. All inactive coordinates remained bitwise positive zero. The
worst relative error from the independent three-coordinate ray was 1.22e-6
(ceiling 1e-4). Rotation preserved all 1,152 matched bias-only step
signatures and all endpoint logits bitwise. No reruns, real-model calls or
GPU. The preliminary base-571 rehearsal is separate from these counts.

A separate [constructed activation check](ROLE_CONTROL.md) **did not
complete**: the runner at `637e38d` called a PyO3 list property as a method.
One native standalone forward returned, zero local VJPs or role predictions
were validated. The source now reads `batch.output` as a property; its shape
and accessor are unit tested. The failed worker and first error were retained,
with no native retry. Do not count this check as a positive native result.

The next discriminating measurement is the sensitivity of the *verb gap*
to each token position and gate axis at the frozen decoder site, before any
update. That distinguishes a weak incoming number signal from a strong one
that the downstream readout ignores. More orders of the same training
examples cannot make that distinction. The subsequent [native margin-signal
measurement](../prefix-signal/RESULTS.md) traces both factors separately
on the training prompts and tests finite, per-prompt gate perturbations.

## Replay without model calls

Use new output paths. Local validation roots: `native-prefix-owner.clB6K0`
(new) and `native-prefix-transport.83xMiQ` (parent). The raw archives, base
weights and original failed worker log remain local. The original real GPT-2
[48-call diagnostic](../../../parameter-vjp/RESULTS.md) is separate; it was
verified from its archived journal, not rerun.

```sh
python3 -ISB test_owner.py
python3 -ISB verify.py "$NEW/formal" "$PARENT/formal" "$OUT/result.json" "$OUT/orders.tsv"
python3 -ISB boundary.py "$NEW/formal" "$PARENT/formal" "$OUT/wrong-conflicts.tsv"
cmp orders.tsv "$OUT/orders.tsv"
cmp wrong-conflicts.tsv "$OUT/wrong-conflicts.tsv"
```

| SHA-256 | Artifact |
| --- | --- |
| `afe6a3c70b4abc2c2d310b61c9ef8bead9ca9004fb31a0dead7c7ec84999b14d` | Formal plan |
| `29cac1ee5b5c832d69da9e1e8ddf1b81429fbf4523cb200a45b7df4cc94d8aaa` | Formal completion manifest |
| `3f5ab707f346d465923334f13dbd0710002dc0ac7dd1f540c958003a8d6103a3` | Full formal step journal |
| `4d456228f0299c2f8ee6f34d0a0ee7b06787c214f8124edb1d3c94ce8e9b478b` | Verified result JSON |
| `fc3a15271af721f03d084a39b46b934f403ff9ce567e33df1c015a58a24a2af1` | All orders TSV |
| `88a2df1510045b678c86c712e189015a23153a7c2086e5ffde0b7b4ebef1c79f` | Wrong-conflict read-back |
| `fb46bf290f19b49a7d2c15fa4223459b2728a2a1aae0e236adab193b6e9f165c` | Private archive of journals and worker receipts |
