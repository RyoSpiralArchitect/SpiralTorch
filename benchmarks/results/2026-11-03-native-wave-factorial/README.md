# Same-input factorial: which part of WaveGate contracts the local map?

The [previous read-back](../2026-11-03-native-extraction-pullback/RESULTS.md)
compared *different trained states*. This test holds the actual hidden input,
gate, bias, strength, curvature, radius and arriving CE adjoint fixed within
each window. It does not score four different language models.

## Frozen scope

- Six update-96 snapshots: affine-trained and fixed-Wave-trained, from each of
  orders 109/113/127. Same pinned 96 training records and masks, canonical order.
  No learned-radius checkpoint, tokenizer, held-out input, decoding or update.
- One factual Rust forward/backward per snapshot/window: 576 model calls.
  On its captured attention output evaluate four local maps and their input VJPs:
  plain affine, porous saturation of gate and affine value only, radius projection
  only, both. Strength 0.25, curvature -1, saturation 1, porosity 0.05, log radius 0.
- The reselected factual arm must match the actual rounded output and input
  adjoint exactly. The cross-build check is stricter than an average:
  each ordinary observation must match its independently archived prior receipt,
  including CE, upstream correction, trainable gradients and owners. Stop at
  the first discrepancy; no retry. CPU and elapsed time limited to 30 minutes.

## Estimands

Pool the sums of squared `output-input` and `dx-dy` over all 1,689,600
coordinates per endpoint. For each of those two energies, let A, S, P and B
denote affine, saturation-only, projection-only and both. Report
`sqrt(S/A)`, `sqrt(P/A)`, `sqrt(B/A)`, and the fraction of affine energy on
coordinates crossing the *post-gate affine* saturation threshold.

For descriptive attribution of the energy change, show both enabling orders.
The saturation share relative to A is `((A-S)+(P-B))/(2*A)`; projection is
`((A-P)+(S-B))/(2*A)`. Their sum is `(A-B)/A`. Also retain the interaction
`(B-S-P+A)/A`; do not assume the stages add or that negative contributions
are errors. These are order-averaged reductions in **squared energy**, not NLL
or percentages of tokens. Zero reference energy has no relative attribution.

Compare the two trained owners separately. The same local map run on different
owners need not have the same hidden states, gates or incoming gradients.
No statistical quality gate or claim about the historical training trajectory.

## Run

```sh
"$VENV/bin/python" -I -B study.py --originals "$SAVED_TRAINING_RUN" --model "$MODEL" --wheel "$INSTRUMENT_WHEEL" --commit "$STUDY_COMMIT" --output "$NEW_DIR"
```

The pinned factual scalar archive and independent checkpoint boundary files
are sufficient to authenticate inputs relative to these local pins; they are
not signatures. Originals, logs and partial failures stay in the private run.
