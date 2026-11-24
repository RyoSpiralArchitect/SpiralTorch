# Fixed-radius replay of captured last-attention rows

This is a local Jacobian experiment, not a decoder run. Reuse only the 24 training
prefixes from the completed margin-pullback study, each under its original
affine and WaveGate host. Keep each host's own incoming unit-margin adjoint.
The sibling rehearsal uses four disjoint artificial prefixes. No model weights,
held-out inputs, labels for new cases, training, decoding, GPU or network are needed.

## Frozen design

- Use the original Rust f32 CPU geometry runtime and its unmodified gate/bias
  and strength (0.25). Sweep only the non-trainable scalar log-radius of the
  full WaveGate. The saturation, porosity and curvature never change.
- Evaluate nominal radii in order 1, 0.5, 2, 4, 8, 16, on all captures at each
  radius. All 48 baseline gate/bias pairs must first match the stored full-map
  VJPs byte-for-byte, including signed zero. Stop at the first error. No retry.
- For each row, concatenate gate and bias. A radius passes that row only if its
  norm is at least half the archived affine-map norm, at least twice its
  radius-one norm, and its cosine to its radius-one direction is at least 0.99.
  All three tests must hold on the **same** row. An undefined ratio/cosine fails.
  Select the smallest tested radius >= 1 with at least 18 of 24 passing rows
  **in each host** (3 of 4 for the artificial rehearsal), or select none. These
  are engineering thresholds for a future candidate, not quality targets.
- Also report both parameter blocks separately and the forward footprint:
  `||y_R-y_1|| / ||y_1-x||` for the residual site's output, not the LM's logits.
  These secondary measurements are not selection filters. Save raw f32 outputs
  and parameter VJPs; do not recompute downstream adjoints at the new radius.
- CPU only; 90 seconds per cohort, no retry. The parent archive and the old
  method are read-only. All six adapter variants are recorded, even on a null
  selection. No candidate is installed as a new model default.

## Reproduction

Use the prior private observation archive (its manifest hash is pinned in
`sweep.py`) and the interpreter with its original CPU extension. `freeze` and
`verify` use only the Python standard library. Use a fresh private output
directory; the supervisor and all result files reject overwrites.

```sh
python3 -ISB sweep.py freeze "$ARCHIVE" rehearsal "$OUT/toy.json"
python3 -ISB ../bounded.py "$OUT/toy" 90 "$NATIVE_PYTHON" -IB sweep.py measure "$ARCHIVE" "$OUT/toy.json" "$OUT/toy"
python3 -ISB sweep.py verify "$ARCHIVE" "$OUT/toy.json" "$OUT/toy"
python3 -ISB sweep.py freeze "$ARCHIVE" gpt2 "$OUT/real.json"
python3 -ISB ../bounded.py "$OUT/real" 90 "$NATIVE_PYTHON" -IB sweep.py measure "$ARCHIVE" "$OUT/real.json" "$OUT/real"
python3 -ISB sweep.py verify "$ARCHIVE" "$OUT/real.json" "$OUT/real"
```
