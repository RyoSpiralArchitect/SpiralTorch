# Rust Angular GL Learning Path

Source: `f8e8e6aa60f7c9a4211243e91f1f1a565d26e50b`.
API and limitations: [angular chart](../../../docs/fractional_angle_chart.md).

The preceding [gain study](../2026-10-05-fractional-gain-study/README.md)
favored an ordinary learned two-tap filter over both GL variants, but used
different shape coordinates. Rust now supplies the bounded chart
`alpha = 1 + 2*tan(angle)` and its VJP/JVP. This composes with the existing
normalized GL operator instead of reconstructing coefficients in Python.

## Real-Model Connection Check

Offline local pretrained GPT-2, frozen base, first-MLP insertion, CPU f32,
actual hidden shape [2,128,768]. Compare ordinary short and angular GL K=3
with the same four parameter names and 1538 parameters, initial filter,
gain, zero feature gates, Adam 0.001 and strength 0.1. Use the first eight
training batches of the existing fixed schedule for each seed 41/43/47.
This is a separate training-only preflight, not a rerun of the 512-step study.

All 48 auxiliary training updates and 12 continuation-only updates completed.
After each paired update, compare full gradient tensors, parameters and Adam
state with fixed rtol=3e-6 and atol=3e-7. Every tensor passed. Maximum absolute
differences over all compared elements/steps/seeds:

| Quantity | Maximum Absolute Difference |
| --- | ---: |
| Parameter gradients | 1.373701e-7 |
| Updated parameters | 9.965152e-8 |
| Adam state | 2.281740e-8 |
| Training loss, descriptive | 2.384186e-6 |

Initial parameter hashes and first losses match exactly. Later cross-arm
updates are tolerance comparisons, **not bitwise equality**. Each arm's own
save/restore continuation exactly matches its uninterrupted parameters,
Adam state and update record. Base weights/gradients remain unchanged.
No endpoint losses are computed, no weights saved, and different training
batches must not be read as a loss-reduction curve.

The ordinary short filter has a larger angle domain; all observed angles
in this preflight stay inside GL's positive-order chart. A future fixed
study must declare the domain policy and report domain exits, not clip or
drop them. Eight updates do not establish long-run parity or quality.

## Native And WASM Evidence

140 Rust tests and 680 implementation Python regressions passed, including
31 new Python angle cases. The final publication-inclusive run passed 681
tests. These cover finite differences, nonzero-gate outputs and all
parameter/input gradients versus the independent ordinary implementation,
buffer/sequence transport, joint JVP, invalid charts and interrupted tiny-HF
resume. The shared trainer rejects a post-update chart exit before writing
a new checkpoint. Old log-order states and semantics remain unchanged.

Eight compiled-WASM fixtures and strict TypeScript checks passed. The new
fixture composes native chart and GL parameter VJPs for 1000 synthetic SGD
updates: MSE goes from 0.857356564 to 7.945771e-14. JS owns the optimizer,
Rust owns the map and differentials. This verifies executable WASM learning,
not a browser/GPU run, pretrained quality or unique parameter recovery.

Native release, wasm32 release, explicit CPU/text binding check, strict
native/WASM st-frac Clippy and 34 non-timing benchmark correctness tests
passed. Existing JIT deprecation warnings remain. The initial optional-Torch
test signature failure and its repair are recorded, not a changed tolerance.
All 577 earlier study/client/runtime files were rehashed unchanged.

## Reproduction

Build the native extension and freeze this commit's package/client sources.
The client directory contains `hf_fractional_gain_study.py`,
`hf_fractional_lag_study.py`, `hf_wave_gate_long_horizon.py`,
`hf_wave_gate_conditioning.py`, `hf_fractional_pride_gain.json` and
`preflight_fractional_angle_chart.py`. Use matching existing local assets:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  PYTHONPATH="$FROZEN_PACKAGE" python -P -B \
  "$FROZEN_CLIENT/preflight_fractional_angle_chart.py" \
  --client-root "$FROZEN_CLIENT" --package-root "$FROZEN_PACKAGE" \
  --config "$FROZEN_CLIENT/hf_fractional_pride_gain.json" \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --previous-study "$COMPLETED_GAIN_STUDY" \
  --output "$NEW_PREFLIGHT_JSON"
```

For compiled-WASM checks, build `spiraltorch-wasm` for wasm32, generate Node
bindings with wasm-bindgen 0.2.104, then run
`node bindings/st-wasm/tests/fractional_angle_chart.mjs <generated-module.js>`.

`preflight.json` retains every training record and per-tensor error;
`validation.json` binds code, native/WASM artifacts, frozen runtime/client,
tests and private-log hashes. `wasm-learning.json` is synthetic evidence
only. Verify public file integrity with `shasum -a 256 -c SHA256SUMS`.
Public consistency checks do not independently replay private tensors.
No weights, corpus text, runtime packages or raw private logs are published.
