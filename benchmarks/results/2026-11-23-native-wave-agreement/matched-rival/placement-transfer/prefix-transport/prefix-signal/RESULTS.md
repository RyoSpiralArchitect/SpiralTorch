# Number reached the gate; a parameter step was not a matched output intervention

We measured the unit z7-z8 gradient on the [frozen design](PROTOCOL.md):
32 artificial **training** prompts per base, zero gate/bias at the last MLP,
frozen numerical LoRA, three rotations. No optimizer update, held-out scoring,
pretrained GPT-2 call, GPU run or external publication. These are numerical
observations on generated width-three decoders, not English agreement results.

## Sensitivity and attainable movement

Compare singular and plural heads at the same noun pair and attractor number.
For the positive quarter turn, the pre-adapter value at the changed number
position and its transported prefix mean differed on **all 48 matched pairs**
(16 per base). All strictly earlier site rows were bit-identical. This is
evidence of sensitivity to the input change, not of grammatical understanding.

| Base | Input-feature route larger than readout route | Median norms: feature / readout route | Initially wrong head decisions | Most of an initial gap closed by a prompt-specific gate |
| --- | ---: | ---: | ---: | ---: |
| 557 | 1/16 | 0.00807 / 0.02996 | 24/32 | 16.26% |
| 563 | 16/16 | 0.00291 / 0.000485 | 16/32 | 1.72% |
| 569 | 0/16 | 0.00000828 / 0.000149 | 16/32 | 0.0549% |

The routes refer to the *difference in gate gradients*, not the difference
in logits. For two inputs let M be the transported features and B the
strength-scaled readout adjoint. We decompose the difference in M times B
into (difference in M) times (average B) and (average M) times (difference
in B). The allocation is symmetric and sums to the observed slope difference
up to rounding. The [per-pair table](pairs.tsv) includes direct and later
positions, both roles, both route vectors and cancellation. The [cell
table](cells.tsv) includes the identity and inverse rotations. There is no
single dominant route across bases.

On each initially wrong prompt we also tested the gate vector of length 1/4
pointing in its own signed margin gradient direction. **0/56** primary
perturbations reached even a tie; **0/168** across all rotations. These are
actual finite forwards, not just a local prediction. They are *different
vectors for different prompts*, not a common learned adapter. Nor does one
direction prove no better nonlinear perturbation exists. The 56 cases are
the same across rotations and the eight lexical frames are reused.

## Why parameter distance was misleading

In a [descriptive replay](scales.tsv) on the original 32 prompts, median
gate gradient norm divided by bias gradient norm was **1.68%, 2.31%, 5.86%**
on the three positive-turn bases. On 569 the transported prefix features
had the **largest** median norm (0.0882 vs 0.0369 and 0.0475) while the
incoming token-margin adjoint had by far the **smallest** median norm
(0.00641 vs 0.999 and 0.534). A readable change in the adapter input does
not guarantee a useful change in the verb logit. Bias-only fractions in
that table are first-order predictions, not additional perturbation calls.

The next training comparison should match the RMS of the change actually
delivered to the residual stream, instead of matching parameter distance
alone. Estimate scales on training data only, keep them fixed, and include a
single scalar-gain control before attributing improvement to channel-specific
geometry. On a base with a weak arriving gradient, a different placement
is a separable hypothesis.

## Verification and reproduction

One formal worker measured **288** factual traces, **576** opposite-label
conditional-CE VJPs, **1,728** central-difference forwards and **168**
prompt-specific finite perturbations. The largest absolute discrepancy in
the reconstructed gate slope was 2.24e-9; opposite-label CE difference
1.79e-7; central difference 7.49e-6. Full training-prompt target/rival
logits reproduced the historical origin exactly. Rehearsal used separate
base 571. Both ran once, under a shared 240-second, stop-on-error CPU budget.

All 1,361 Rust library tests passed. The new Python binding also exercised
f32, f64 and f32/wide-backward, the swapped margin and early rejection. An
initial *test fixture*, not a decoder result, attempted to attach two nested
placements that artifact construction already disallows; the corrected
unsupported-site test uses a legal model. The separate failed standalone
role-control run in the previous study was **not rerun**. For this probe see
the [Python and Rust contract](../../../../../../../bindings/st-py/examples/native_prefix_margin_signal.md).

Local evidence directories in SpiralTorch validation: `native-prefix-signal.4pAwbO`
(new), `native-prefix-transport.83xMiQ` (bases) and `native-prefix-owner.clB6K0`
(masked origins). New native code changes two historically pinned files;
replay old experiments from the original commit `05ac7be`, not by relaxing
those checks. Both previous masked plans were reproduced byte-for-byte
from the old source blobs in this validation.

```sh
python3 -ISB test_signal.py
python3 -ISB verify.py "$NEW/formal" "$PARENT/formal" "$MASKED/formal" "$OUT/result.json" "$OUT/cells.tsv" "$OUT/pairs.tsv"
python3 -ISB scale_readback.py "$NEW/formal" "$PARENT/formal" "$MASKED/formal" "$OUT/scales.tsv"
cmp cells.tsv "$OUT/cells.tsv"
cmp pairs.tsv "$OUT/pairs.tsv"
cmp scales.tsv "$OUT/scales.tsv"
```

| Artifact | SHA-256 |
| --- | --- |
| Formal plan | `d95d05931e41f713bf8504e7eba339c902747a2a123bf2c195fdb5889fb1b9de` |
| Formal completion manifest | `c5815a53eb9bd596c138b2c35ac14b0cb460fcb1337326e313b1d4b77b6f1fd1` |
| All original formal rows | `40fa48fcc32477e1306fc27c16d16cad95c474359915746eac7f426881badcfe` |
| Numeric readback | `52acbc0ae9f8af467d75d4914b3fb492871c5627f01c0437da4bb124b412d83a` |
| Read-only scale control | `587841c1e58f73a640755b3f9ce43e893fd99ecb063f830e5f71e065a0abed43` |
| Development wheel | `4e3c8d52961f6e317dc91edd8d89e50bf7edd656d57e6b82514f486568be7631` |
| Journal and worker archive | `142119897cbf75990e3b932085c6e308c4ee928354d781677d8cc1a62286c440` |
