# Saturation and projection on the same GPT-2 attention output

**The earlier 16% residual was a comparison of two trained models, not the
gain of the Wave operator.** Holding a fixed-Wave-trained model's own input,
gate and bias constant, enabling both stages left **54.6-55.5%** of its plain
affine residual RMS. Even with both stages switched off, this trained state's
relative residual was only **29.4-30.5%** of the affine-trained state's.
These two ratios multiply to the observed 16.3-16.7%. The state comparison still
includes differences in LoRA and hidden inputs; it cannot isolate gate training.

| Factual trained state | Local quantity | Plain | Saturation only | Projection only | Both |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed Wave | Forward residual RMS | 1 | 0.607-0.617 | 0.570-0.581 | 0.546-0.555 |
| Fixed Wave | Input-adjoint correction RMS | 1 | 1.000 | 0.893-0.899 | 0.893-0.899 |
| Affine | Forward residual RMS | 1 | 0.768 | 0.366 | 0.361 |
| Affine | Input-adjoint correction RMS | 1 | 0.996 | 0.311 | 0.424 |

Each row is normalized to its **own** plain-map energy, pooled over 96
training windows, then square-rooted. Ranges cover the three training orders.
Correction is `dx-dy`, not the full gradient. The factual incoming gradient
`dy` is shared by all four local VJPs; there are no counterfactual LM losses.

## What The Rare Coordinate Did

On fixed-Wave-trained weights, one coordinate crossed the affine saturation
threshold in **every** document, with no gate outside the limit. These 96
coordinates per checkpoint (of 1,689,600) accounted for **69.05-69.96%**
of the unsaturated, unprojected forward residual's squared energy. The
plain-map *rounded adjoint correction* on those coordinates was exactly zero.
For each of 96 windows in each of three orders, saturation-only had exactly
the same correction energy as plain, and both stages had exactly the same
correction energy as projection-only. This is equality of scalar energies, not
proof that every counterfactual vector or every upstream gradient is identical.
The receipts do not identify the coordinate, token position or label mask there.

Both stages reduced forward energy, with substantial overlap. Averaging the
two enabling orders, saturation removed 32.4-33.0 percentage points of the
plain reference energy and projection 36.8-37.3 points on fixed-Wave weights.
On affine-trained weights these contributions were instead 20.7 and 66.3
points. There is no owner-independent "percentage caused by geometry".
For affine-trained adjoints, adding saturation to the projected map actually
*increased* correction RMS from 0.311 to 0.424 of plain.

The local measurements suggest a practical next test: train the same four
equal-capacity maps from a shared origin on a new split and compare answer
likelihood and accuracy. A trained model with a different map need not see the
same hidden inputs or loss adjoints. This experiment measured neither generation
quality nor a new generalization advantage.

## Evidence

- [Frozen design](README.md), commit `c82c55b42`; [native wheel identity](build.json), source `33ab011b2`.
- One attempt, six immutable checkpoints, 576 factual forward/backward calls,
  2,304 local maps, **zero updates and zero evaluation records**. 112.32 CPU
  seconds, 116.28 seconds elapsed. No learned-radius checkpoint was opened.
- All 576 original scalar receipts matched, including owners, CE, geometry
  and parameter gradients. The reselected factual forward and input adjoint
  had max absolute difference zero for every window.
- [Pooled energies and signed contributions](receipts/result.json);
  [compressed journal pins](receipts/archive.json). Run
  `python3 -I -S -B verify.py receipts --archive` to compare all factual
  entries, recompute all signed local contrasts and verify both state/operator
  product decompositions without executing a model. `--originals DIR` also
  checks the train file and six original checkpoint hashes.
