# Matching delivered residual motion helps one base, not verb-choice transfer

The [frozen protocol](PROTOCOL.md) compares uniform, channel-specific and
deliberately misassigned (cyclic) preconditioners at equal final-row residual
RMS. Real optimizer updates, **synthetic** width-three decoders and previously
scored synthetic evaluation frames. This is not a pretrained GPT-2 run.

## What actually learned

Ranges below are across three predefined orders, on the same 16 evaluation
prompts. Head gain is the change in mean signed head-verb margin under the
head teacher. Selectivity contrasts head-vs-attractor margins after training
under each of the two teachers. A gain in one is not a gain in the other.

| Frozen host | Positive-turn diagonal head gain | Teacher selectivity | Head choices correct, before and after | Primary margin gate |
| --- | ---: | ---: | ---: | --- |
| 557 | .01557 to .01693 | -.00377 to -.00344 | 4/16 | No: selectivity negative |
| 563 | .01803 to .01808 | .00827 to .00832 | 8/16 | Yes: both controls beaten in every order |
| 569 | .000526 to .000529 | .000145 to .000148 | 8/16 | No: gains too small |

On 563, the minimum order-matched advantages over uniform were +.00617 in
head gain and +.00134 in selectivity; over cyclic they were +.01234 and
+.00344. All four lexical frames improved. Positive turn also exceeded
identity diagonal in both metrics on this base. Identity diagonal met the
joint margin criterion on **0/3** bases; positive turn on **1/3**, short of
the required two. **Not a single evaluation choice changed** in either
teacher arm at any of the 108 endpoints. The decision gate passed on 0/3.

In a descriptive [readback of initially wrong head choices](wrong-gaps.tsv),
all eight wrong prompts on 563 moved toward the boundary in all orders under
positive-turn diagonal. The median fraction of the initial gap closed was
9.95%, the maximum 30.15%. These are the *same eight prompts* in three orders.
On 557 the median was 1.53% (best 3.90%); on 569, 0.94% (best 2.06%).
The [per-order table](orders.tsv) includes train and evaluation, both rotations,
both teachers' decisions, and all controls. Do not select an order post hoc.

## What was matched

3,456 equal-vote updates from 6,912 training gradients; 192 label-free
calibration prefixes (six fits), 5,472 recorded full-logit forwards, 216
initial member gradients matched against earlier raw training receipts.
Base, numerical LoRA, positive-zero bias and upstream network remained fixed.
Target RMS was 0.00112764; the worst relative error over all stored updates
was 1.16e-6, without clipping. Physical parameter L2 ranged from .1234 to
.6274, deliberately **not** constant. The native residual operation measured
before/after on every calibration input at the first and last step of every
path differed from the metric prediction by at most 1.72e-9 RMS.

The real GPT-2 [48-pass derivative experiment](../../../parameter-vjp/RESULTS.md)
is separate: it had zero updates. The gain on 563 here does not establish
an improvement on that model or on new natural-language text.

## First failure and independent closure

The first wheel build reported a duplicate PyO3 methods implementation.
The registration fix preserved the Rust operator; the subsequent CPU wheel
passed all four binding tests, and the native Rust suite passed 1,363 tests.
The separate-seed rehearsal completed 24 updates. The formal worker ran once,
finished all model calls, wrote 236 hashed files, then **exited nonzero** in
its offline verifier. No model or failed training worker was run again.

The old verifier divided each position by six *before* summing to derive a
rounded transport mean. Rust divides the sum once. On two of the 41,472
native residual coordinates this put the expected output one binary32 step
away from the recorded value. The [exact readback](verify_exact.py) sums
signed, rotated binary32 inputs as exact fractions, divides once, and rounds
ties to even. It keeps the original code hash and all original gates,
including exact output equality. **41,472/41,472 coordinates** now agree;
no tolerance changed, and the original failure receipt is still a failure.
Independent Python 3.8.20 and 3.12.6 reads gave byte-identical JSON and TSV.

Local evidence anchors (SHA-256): formal spec
`9e97a9bb137f606bc1a151d4f1868d9a95a3ee5db18b4391233b8c11bcf82976`;
completed model journal manifest
`3268ed35efa6744bda3b7a9c9cd53af8d1b0d3286a3427e56ee640a88ee0a27f`;
exact results `e981a6a92437b7ce38ad49bc49bfa0051fc806e2e08c7d0460afb8b11e9b28da`.
The private portable journal archive contains 290 byte-verified files
(`da7f5311b3be584a2e9c333022ec67cbcd2068bf05aeeae034c70d939ba48931`);
the CPU wheel SHA-256 is
`7df909ab1a45ae584d7ac48eca924f4d0e65f2e65dc87c5fca187f2ee048eb62`.
The table is descriptive evidence from one synthetic task, not a promotion
of this optimizer default. A next discriminating comparison would hold the
same delivered RMS and parameter count fixed while changing the **direction
seen by the verb readout**, and test on unused lexical frames.
