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

## Observed Results

All 27 declared cases completed without retries or optimizer updates. All 15
untouched references reproduced their earlier development CE exactly. All 48
layer/control training RMS matches passed 0.1%; development was not refitted:
11/48 matches stayed within 0.1%, and all 48 stayed within 1%.

Mean development CE across seeds 41/43/47 (lower is better):

| Saved geometry | Geometry CE | Calibrated ordinary CE | Geometry minus control |
| --- | ---: | ---: | ---: |
| Fixed radius 1 | 4.110879 | 4.111593 | -0.000714 |
| Learned radius from 1 | 4.108385 | 4.109312 | -0.000927 |
| Fixed radius 4 | 4.096002 | 4.097747 | -0.001746 |
| Learned radius from 4 | 4.090802 | 4.093193 | -0.002391 |

The original ordinary-affine mean remains **4.065289**, lower than all geometry
and calibrated-control means. Shrinking the ordinary residual moves CE toward
the saved geometry results. All twelve within-seed calibrated comparisons favor
geometry, by 0.000704 to 0.002454 CE, but those are small differences between
differently trained weights with only approximate held-out RMS matching. This
is not a causal geometry advantage, matched-FT result or deployment promotion.

The next useful training comparison should control effective residual strength
from the start rather than infer its effect only after training. The native
Torch-free execution path remains the product goal; these HF runs are an
optional interoperability/comparison path, not a replacement for that runtime.

`results.json` contains every projected case and contrast. `execution.json`
binds the frozen protocol, 15 source artifacts, installed package and window
identity. A read-only saved-state check revalidated all 27 candidates, 102 traces,
intermediate native artifact IDs and calibration-rule values without loading
the model. The actual model/logit checks belong to the original run receipts.
Private raw records, traces and artifacts were retained unchanged.

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

To recheck saved results without another model run:

```sh
"$PYTHON" -I -B benchmarks/results/2026-10-18-hf-residual-calibration/verify_saved_state.py \
  --private-runs "$PRIVATE_CALIBRATION_RUNS" --book-private "$PRIVATE_BOOK_ROOT" \
  --radius-runs "$PRIVATE_RADIUS_RUNS" \
  --expected-results-id "$INDEPENDENTLY_RETAINED_RESULTS_ID" \
  --expected-manifest-id "$INDEPENDENTLY_RETAINED_MANIFEST_ID"
```
