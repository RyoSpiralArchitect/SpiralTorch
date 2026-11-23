# A trainable prefix rotation reached the decoder, but bias dominated this task

The new Rust recipe has a real loss gradient and a compact cache. On this
**artificial agreement task**, it did not improve which verb was selected.
These are separate implementation and learning results. No real GPT-2, new
English test set, GPU run or publication.

The [matched two-context follow-up](paired-gap/RESULTS.md) trains the contrast itself
with three gate coordinates: some new pairs were ordered correctly without changing
the verb either context selected.

## Matched training

The [preregistered comparison](PROTOCOL.md) used three freshly generated
decoders (width three), frozen numerical LoRA and six trainable parameters
at the second MLP output. The identity-rotation control sees the same site's
ordinary prefix mean, not a model without position embeddings. Quarter turns
keep each incoming channel pair's norm before averaging, but can change
cancellation in the sum. The odd third channel is unchanged by rotation.

All 9 zero-initialized origins had bit-for-bit identical logits to their
matched frozen reference. 54 endpoints, 1,728 cohort updates, 3,456 training
VJPs, 3,024 snapshot and 144 reference forwards. All 1,728 realized f32
updates matched the independently computed normalized negative mean gradient;
worst relative vector error 1.244e-6 (ceiling 1e-4). No formal retry.

| Generated base | Head wins before/after (of 16) | Quarter turn advantage in head-margin change, over three orders (nats) | Gate share of squared step, head teacher, median |
| --- | ---: | ---: | ---: |
| 557 | 4 / 4 | +3.94e-6 to +4.77e-6 | 0.0363% |
| 563 | 8 / 8 | +5.21e-6 to +5.26e-6 | 0.9143% |
| 569 | 8 / 8 | +1.18e-7 to +1.39e-7 | 0.3567% |

Positive vs identity is the declared primary comparison. Its head-margin
advantage was positive in all 9 matched orders but far below the 0.001
criterion even at its best. **0/3** bases passed the probability or decision
gate. The secondary inverse turn also passed on **0/3**. No two-token winner
switched on any of the 16 held-out prompts at any of the 54 trained endpoints
(either teacher). These are reused artificial lexical frames and three orders
of the same training set; nine matched orders are not nine independent bases.
[All train/test orders](orders.tsv) include both teachers' accuracies,
whole-vocabulary gold NLL, restricted NLL, ties and margin effects.

## Which parameter moved the output?

In the [saved step vectors](gate-fractions.tsv), the gate's share of squared
motion was at most 9.07% on **every single update**; the rest was bias.
Parameter distance does not, by itself, isolate output effects. We therefore
[specified a separate endpoint intervention](COMPONENTS.md) after the study:
gate alone, bias alone, both, and neither, preserving the original base and
companion. This cost 2,736 synthetic forwards, zero backward calls or updates.
All unaltered origins and endpoints reproduced their original logits bitwise.

Across the 16 prompts, the bias contribution to the 7-vs-8 margin had greater
RMS than gate at **48/54** endpoints, including **27/27 head-trained** endpoints.
We average the two intervention orders and report the interaction separately.
The other six were all attractor-trained on base 569 and *both* RMS effects
were below 5.5e-8 nats. No gate-only or bias-only endpoint changed a held-out
winner. The [per-endpoint table](component-orders.tsv) keeps those exceptions,
absolute magnitudes, interactions, signed effects, and all winner switches.

**Next test:** freeze bias at zero during learning and compare the three gate
vectors at equal step length, placement and dimension, with a three-parameter
bias-only control. This removes the competing route during training rather
than deleting its result afterwards. The subsequent [masked training
comparison](owner-control/RESULTS.md) tested this on the same artificial
weights; transfer to a pretrained model has not been demonstrated here.

## Runtime and reproduction

Rust owns the schema, prefix scan, reverse scan and compact f64 state.
The cache adds two f32-sized words per channel per placement instead of a
row per token. Full sequence budgets and immutable branches still apply.
The [client guide](../../../../../../bindings/st-wasm/tests/native_causal_lm.md)
shows the artifact shape. 1,358 Rust library tests passed, including quadratic
forward/finite-difference references, independent document boundaries, overflow
admission, parameter-only gradients and learned cache restoration. The installed
native Python wheel and actual Node/WASM and headless Chrome each exercised
all three rotations on a separate small fixture. Max absolute discrepancy in
the compared gradients, loss and logits: 8.48e-9. This is not a backend
speed benchmark or evidence of accuracy on natural language.

Local raw evidence: `native-prefix-transport.83xMiQ` in the SpiralTorch
validation folder. A Python 3.12 pip refused the isolated 3.8 target before
installation; the retained compatible 3.8 pip installed the exact same local
wheel without network or dependencies. No native or WASM model experiment
was retried to get a passing result. To check the evidence *without model
execution*, use new output paths:

```sh
python3 -ISB test_study.py
python3 -ISB verify.py "$E/formal" "$OUT/results.json" "$OUT/orders.tsv"
python3 -ISB components.py verify "$E/formal" "$E/components" --json "$OUT/components.json" --table "$OUT/components.tsv"
cmp orders.tsv "$OUT/orders.tsv"
cmp component-orders.tsv "$OUT/components.tsv"
python3 -ISB blocks.py "$E/formal" | cmp - gate-fractions.tsv
```

| Artifact | SHA-256 |
| --- | --- |
| Formal plan | `c97829b0567f4cbe69d548b559b108f8b88021b4893a14932bf3e27b2622fa1c` |
| Formal numeric readback | `74437a17603c8f19ee3fae3e1802adfaa4f459126c68a8edf3a731b378ea7515` |
| Component readback | `fbf2595899b4c9bc06d995ae9a8d587608a37f644036fd084544883df2c58f43` |
| Development wheel (not published) | `de8a260963a3a88f6e50768579e4335443a3b4ecff30a40577a57dfda8abba15` |
| WASM module used in both clients | `9860bd507e0efdb36042a2eeb3cb183cef44160e5e2dae44dc11262a82179173` |
| Private archive of journals and receipts | `d2319dd051287e68ba9ae133deaeae6ffa34d1e5867e67cd7fc381e445860784` |
