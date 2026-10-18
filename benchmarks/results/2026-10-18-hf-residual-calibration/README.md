# Training-Only Residual Calibration of Saved HF Artifacts

This is a post-training counterfactual diagnosis, not another FT sweep. The
earlier learned-radius study improved CE relative to fixed radii, but ordinary
affine still had lower CE and much larger residuals. Does shrinking its residual
magnitude account for part of that gap without changing any learned weights?

WaveGate's radial map changes magnitude along its affine direction. Per-token
normalization would remove the radial behavior under study. Instead, calibrate
one fixed ordinary-affine strength per layer using aggregate residual RMS over
64 predetermined training windows. No new geometric kernel, Python replacement
formula, per-token normalization, optimizer step or dynamic controller is added.

## Declared Protocol

Use all three seeds (41/43/47) and final artifacts of ordinary affine plus fixed
and learned radius from 1/4. There are fifteen untouched references and twelve
calibrated ordinary controls. Each control starts from its seed's original
ordinary weights, not the previous calibrated control.

For 1,047 training windows, calibration indices are `floor(i*1047/64)` for
`i=0..63`. In forward layer order 8/9/10/11, observe the current ordinary stack
and set its layer strength to `old_strength * target_RMS / current_RMS`.
The target is the saved geometry state's per-layer relative residual RMS on
these same windows: `sqrt(sum ||output-input||^2 / sum ||input||^2)`.
Earlier layers are therefore calibrated before measuring a later layer's input.
After four updates, all calibration-layer ratios must agree within 0.1%.

Zero/nonfinite observations or out-of-domain strengths fail, rather than being
clipped, retried or searched. Then evaluate all 159 development windows. Report
their actual residual mismatch; never fit to development CE or RMS. The reserve
chapters remain unopened. No checkpoint selection or default promotion follows.

All fifteen untouched CE values must exactly reproduce their prior receipts.
Every fitted artifact is Rust-validated and reloaded for an exact first-window
logit check. The base and all learned tensors must remain unchanged; only the
ordinary control's four recipe strengths may change. New artifacts and source
IDs are retained separately, never overwriting a trained checkpoint.

## Boundaries

This does not match optimization trajectories, per-token energy, training
capacity or all evaluation distributions. Models have different learned
weights; a remaining CE gap cannot be uniquely attributed to geometric shape.
Learning-radius arms still have four extra trained scalars. The diagnostic can
motivate a future training experiment, not replace it or prove general quality.

Corpus, token arrays, native traces, fitted strengths and learned weights stay
private. Public results retain every case, CE contrast, untouched-reference
development RMS, artifact lineage and verification. Calibration RMS operands
stay private in both references and fit stages. Matched ordinary development
RMS also stays private: its ratio to the original ordinary RMS can recover the
first layer's strength even on a different split. Public mismatch reports use
only fixed absolute-relative-error bands (within 0.1%, 1%, 5%), not exact ratios.
This is a bounded publication policy, not a formal privacy guarantee. The
public projection is separate from complete hash-bound private case results.
Native conditioning includes per-site learned radii,
so public summaries use an explicit scalar allowlist instead of exporting those
reports. The runtime is the already-qualified observation wheel; no new build
or PyPI release is part of this diagnostic.

## Replay

```sh
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" -I -B benchmarks/results/2026-10-18-hf-residual-calibration/replay.py \
  --expected-plan-id "$INDEPENDENTLY_RETAINED_PLAN_ID" \
  --book-private "$PRIVATE_BOOK_ROOT" --radius-runs "$PRIVATE_RADIUS_RUNS" \
  --model-dir "$PINNED_LOCAL_MODEL" --output "$NEW_PRIVATE_OUTPUT"
```

Use `tools/owned_process.py` with a 3,600-second outer deadline and five-second
cleanup grace. The output directory must be new; all case dispositions are
retained by the shared run-manifest helper. No silent restarts or dropped cases.
