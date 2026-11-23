# A correct binary gradient did not transfer the subject role

On three small **generated** GPT-style decoders, dropping the full-vocabulary
loss made the two possible labels on each input give opposite gradients.
It did not teach these six adapter parameters to choose a different verb
on any held-out prompt. These are symbolic examples, not the separate
[GPT-2 English training-prefix measurement](../../parameter-vjp/RESULTS.md).

## What changed, and what did not

The [frozen design](PROTOCOL.md) crossed WaveGate and channel-affine at the
second MLP output, the grammatical head vs intervening noun as teacher,
a half-full-vocabulary vs pure two-token loss, and single-example vs
balanced-pair updates. Each used all three example orders and generated
bases 503, 509, 521. All other weights, including LoRA, stayed fixed.
All runs saw the same 64 examples per endpoint and the prescribed total
parameter-path length of 0.25. Single and paired runs did **not** have the
same optimizer step count, endpoint, or output-distribution change.

- At all 192 training origin/prompt combinations, the pure gradients for
  opposite labels were antiparallel (largest cosine -0.999999999999996).
  The two vectors agreed with the analytic binary derivative recovered
  independently from the mixed gradients: largest absolute/relative
  residuals 3.081e-8 / 4.699e-7. This tests changing the label on the
  **same input**, not discrimination between different contexts.
- Neither geometry passed the probability or decision criteria on any of
  the three bases with either update schedule. No Wave-specific advantage
  passed. For either teacher, neither of the two answer tokens overtook the
  other on any of the 16 held-out prompts at any of the 144 endpoints.
  Head-trained menu accuracy stayed at its origin (8/16, except 7/16 for
  521/Wave). These are two-token comparisons, not whole-vocabulary argmaxes.
- The task itself was learnable: a linear probe trained on token identities
  by position scored 16/16 for each teacher; removing position gave 12/16
  for each. A probe on the 11 frozen decoder logits scored 7-12/16 for the
  head and 7-13/16 for the intervening noun. Probe capacity is not matched
  to the adapter, and this does not bound information in hidden states.

See [all paired comparisons](tables/per-order.tsv), [schedule interactions](tables/schedule-interaction.tsv),
[origin gradients](tables/origin-gradients.tsv), and [probe wins and ties](tables/probes.tsv).
All 6,912 native parameter updates (9,216 training VJPs) were checked against
the actual mean gradient. Maximum relative step error was 5.077e-5
(limit 1e-4). Separate from training: 768 training-origin VJPs, 192 held-out
origin VJPs after sealing endpoints, 7,200 snapshot forwards and 1,024
linear-probe updates. No pretrained-model call, GPU or generation.

## How local was the gradient prediction?

For base 521/Wave, the direction from starting to trained weights predicted
a negative head-margin change, but the observed change was positive. A
[separate protocol](SECANT.md), fixed after seeing this discrepancy, used
one order, both schedules, both geometries and all three bases. It measured
ten intermediate or endpoint positions on each of 12 straight lines. All
origin and endpoint logits replayed bit-for-bit; 2,016 synthetic forwards,
zero new updates. For the narrowest resolvable symmetric interval on each
line, counting unresolved cases separately:

| Local check | Consistent | Inconsistent | Unresolved |
| --- | ---: | ---: | ---: |
| Mean head-signed margin | 5 | 1 | 6 |
| Per-prompt margin vector (RMS criterion) | 8 | 0 | 4 |

For **521/Wave, single**, near the origin (half-width 1/256 of the line) the
observed half-difference was -1.751e-6 vs -1.746e-6 predicted. At +1/64 the
observed change was already positive; at the endpoint it was +3.120e-4 vs
-4.471e-4 predicted. The local derivative worked at the narrowest scale
here; extrapolation to the trained weights did not. For **521/Wave, pair**,
the first resolvable mean (half-width 1/64) was -2.180e-6 vs -3.945e-6,
outside tolerance despite equal signs. Its row-vector check passed at
1/256. That aggregate discrepancy is not resolved by the single-example
result. Neither case justifies a general guarantee about the backward pass.

[All forward curves](secants/curves.tsv) and [all central differences](secants/central.tsv)
include the small and unresolved signals. The next discriminating experiment
is a position-conditioned residual versus a positionless control with equal
trainable capacity and placement, not another tuning round for these six
global coordinates. The probe motivates that comparison; it is not its result.

## Verify without model execution

In this directory, set E to the local validation folder
`native-conditional-update.BrOFFE`, and OUT to a fresh writable directory.
The readers check the original model/adapter identities, logits, individual
steps and projections without evaluating a model. Keep the protocols and
pinned sources byte-identical.

```sh
python3 -ISB test_study.py
python3 -ISB test_secant.py
python3 -ISB analyze.py "$E/formal" "$OUT/result.json" --tables "$OUT/tables"
python3 -ISB secant.py verify "$E/formal" "$OUT/result.json" "$E/secant" --output "$OUT/curves.json"
diff -r tables "$OUT/tables"
```

| Artifact | SHA-256 |
| --- | --- |
| Formal plan | `dfd38f776ebf059c024449bb211a9cbedb8be6d14c739202c1bff50c2eec659d` |
| Numeric readback | `ec081e8987bb56af70716de5227a9355f0326e419e0b06cb935567685bd917ff` |
| Directional readback | `987a7da9e136e21c39a9aeca7dce3049b2d10442aa2a554e7cfb8dca1d71b371` |
| Private archive of first attempts | `8d685f6ed97a31dc73e04db27f8a2082978364409b88732a901f4069ba149b25` |
