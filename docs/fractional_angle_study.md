# Matched Angular GL Learning

The completed [learned-gain study](fractional_gain_study.md) favored the
ordinary short filter over both GL variants, while GL full beat GL short.
That result remains unchanged. This new experiment removes the angle versus
log-order optimizer-coordinate difference using the [Rust chart](fractional_angle_chart.md).
It does not assume that fractional history will improve language quality.

The [completed nine-run comparison](../benchmarks/results/2026-10-05-fractional-angle-study/README.md)
finds lower endpoint loss for GL full on both sets in all seeds, while the
short-filter loss gap nearly disappears. Final short-parameter tolerance
checks nevertheless fail in all seeds; that result is preserved separately
from successful saved-state verification and the quality measurements.

## Fixed Comparison

| Arm | History | Shape Coordinate |
| --- | --- | --- |
| `ordinary_angle_short` | Independent Torch two-tap filter | angle |
| `history_angle_short` | Rust normalized GL, K=3 | angle |
| `history_angle_full` | Rust normalized GL, K=32 | angle |

All three have identical initial parameter names, values and registration
order: `gate`, `local_gate`, `history_angle`, `log_gain`. There are 1538
trainable parameters at F=768. Gates start at zero, angle at the fixed
approximately atan(1/2) literal, and gain at sqrt(5). Every adapter starts
as identity, with the same initial [-2,1] filter up to f32 rounding.

Rust supplies alpha=1+2*tan(angle) and the GL forward/differentials. The
ordinary control reuses independent Torch trigonometry, gain and shifts;
it observes Rust's chart only for the common domain policy and order
telemetry, not for filter coefficients or gradients. No production backend
routes GL math through Torch. Equal coordinates do not promise bitwise
trajectories or equal arithmetic costs.

`bindings/st-py/examples/hf_fractional_pride_angle.json` fixes the same local
GPT-2 snapshot, Pride/Alice hashes and partitions, first-MLP insertion,
seeds 41/43/47, 512 updates, batch 2, context 128, Adam 0.001, strength 0.1
and two CPU threads as the preceding study. Nine runs total 4608 primary
updates plus 18 continuation-only updates. Endpoints remain locked until
all runs and exact within-arm continuation checks complete. Development
cannot select settings, duration, checkpoint or reported arm.

Primary CE contrasts are GL short minus ordinary short, GL full minus
ordinary short, and GL full minus GL short. Report every seed and losing
outcome, plus angle/native-order/gain/effective-gate trajectories. Initial
losses must match exactly; initial gate-gradient norms use rtol=2e-6,
atol=2e-7. Keep failed criteria visible. Short-filter tensor parity over
eight preflight updates is not long-run parameter or optimizer parity.

## Domain Failure Policy

Every arm uses the strict native chart domain after f32 conversion:
`-atan(1/2) < angle < pi/2`. No clipping, wrapping, projection, retries with
new settings, skipped arms or failure-conditioned endpoint selection.
On a chart exit the whole study stops. The journal retains the arm,
phase, attempted step, completed-primary count and observed angle, marks
the failure terminal, and prevents resume and endpoint scoring. Nonfinite
angle observations are recorded as strings, not invalid JSON numbers.

Earlier valid checkpoints remain intact. The failed update is not made
resumable, even if an optimizer step had already executed. A failure during
either continuation check also locks endpoints despite complete primary
checkpoints. Ordinary infrastructure interruptions remain resumable with
the existing exact source/runtime/data/cursor checks. No previously frozen
study or runtime is modified by these new client-side failure semantics.

## Running

Freeze this client, `hf_fractional_gain_study.py`, `hf_fractional_lag_study.py`,
`hf_wave_gate_long_horizon.py`, `hf_wave_gate_conditioning.py`, the config,
`preflight_fractional_gain_study.py` and `summarize_wave_gate_long_horizon.py`.
Use the verified angular native runtime and existing hash-matching local
assets, with the frozen client/package on `PYTHONPATH` and offline HF flags.

First admit all three arms using actual hidden states and the previous
completed gain study as a read-only base/data/schedule reference:

```sh
python -P -B "$FROZEN_CLIENT/preflight_fractional_gain_study.py" \
  --coordinate angle --client-root "$FROZEN_CLIENT" \
  --package-root "$FROZEN_PACKAGE" \
  --config "$FROZEN_CLIENT/hf_fractional_pride_angle.json" \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --previous-study "$COMPLETED_GAIN_STUDY" \
  --output "$NEW_ADMISSION_JSON"
```

Then launch once into a new directory:

```sh
python -P -B "$FROZEN_CLIENT/hf_fractional_angle_study.py" \
  --config "$FROZEN_CLIENT/hf_fractional_pride_angle.json" \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

Repeat identical arguments with `--resume` only for a nonterminal interrupted
run. Completed resume verifies existing artifacts without retraining or
rescoring. Summarize the completed plan/results/journal with the frozen
Torch-free summary tool; save into a fresh output outside primary artifacts.

After terminal success, use `tools/verify_fractional_gain_study.py --coordinate
angle` with the frozen client/runtime inventories and derived summary to
verify saved parameters and named Adam states without retraining or scoring.
It separately compares final ordinary-short and GL-short parameter/Adam
tensors at the preflight tolerance (rtol=3e-6, atol=3e-7), reports byte and
value equality separately, and retains failed tolerance checks. Valid saved
states do not imply paired-state equivalence. This does not compare full
gradient trajectories or establish long-run bitwise parity.

These are exploratory reused books, not pristine confirmation. Three
minibatch-order seeds do not establish significance or general LLM superiority.
Full-history normalization changes short taps too, so even a positive result
would not uniquely isolate long-memory causation. No speed, browser/GPU,
unique-parameter-recovery or generation-quality claim follows. Publish
results, hashes and verification only, not weights, corpus or private logs.
