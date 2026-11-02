# Endpoint residuals and loss adjoints on the training records

Read-only follow-up to the [matched affine/WaveGate study](../2026-11-03-native-wave-affine-extraction/RESULTS.md).
The parameter norms differed; they do not tell us the size of either the
effective hidden-state change or the gradient actually delivered to LoRA.

## Fixed method

Use all nine update-96 checkpoint files: ordinary affine, fixed-radius WaveGate,
learned-radius WaveGate, each in orders 109, 113 and 127. Use the same 96
**training** records in canonical order at every endpoint. Reuse the pinned
token IDs and answer masks; do not open an evaluation document, tokenize,
decode, rank variants by new held-out scores, take an optimizer step, or tune
anything. Each document gets one native mean answer-token CE and its reverse
traversal. The decoder's existing checkpoint replay is part of that evaluation.
There are 864 calls, 148 supervised tokens and 2,200 hidden rows per endpoint.
One CPU process, one attempt, 1,800 seconds total, stop at the first error.

For each endpoint, pool squared magnitudes over the actual 2,200 x 768
coordinates, then take square roots:

- Forward: `||output - input|| / ||input||`, with placement strength, LoRA
  projection and arithmetic exactly as executed.
- Backward: `||dx - dy|| / ||dy||`, where `dy` arrives at the residual's
  output and `dx` is the full adjoint passed back to the LoRA projection.
  This difference isolates the correction from the identity path.
- For gate/bias (and trainable log radius), report the RMS of **96 separate
  document gradient L2 norms**. This is not the norm of a batch-mean gradient.
- For Wave only, count pre-saturation excursions and pool local projection
  gains over rows. Report both their mean and minimum so an isolated strongly
  contracted row cannot be hidden. Zero-strength bypass has no conditioning.

Compare within each training order. There are no independent test replicates,
no decision threshold, and no claim about the historical trajectory or causal
attribution: every observed checkpoint owns its own LoRA and hidden inputs.
The original training wheel and the new, instrumented wheel are distinct
artifacts; the model/adapter bytes are the same ones saved by the original fits.

## Run

```sh
"$VENV/bin/python" -I -B test_metrics.py -v
"$VENV/bin/python" -I -B observe.py --originals "$PREVIOUS_PRIVATE_RUN" --model "$MODEL" --wheel "$INSTRUMENT_WHEEL" --output "$NEW_DIR" --commit "$FROZEN_COMMIT"
```

`metrics.py` pins the three independent checkpoint boundary files, base files
and train-only file. `observe.py` pins the instrument wheel and loaded extension,
checks all installed bytes, and blocks Python reads of any other file in the
old private run. The JSONL contains scalar receipts, no hidden vectors, logits,
parameter arrays, model weights or text. Failures retain their partial journal.
