# Causal Elliptic Factorial Study: Completed

All 12 runs completed 512 updates each, followed by sealed endpoint evaluation.
Both the original process and completed-resume process exited successfully.
Protocol: [elliptic_causal_study.md](../../../docs/elliptic_causal_study.md).

**The pointwise tangent control wins.** In every seed on both evaluation sets,
elliptic features lose to their corresponding tangent control, and causal mixing
loses to the corresponding pointwise path. All four arms improve the unchanged
base model. This verifies useful adapter learning, not an advantage from geometry
or this causal mixing design.

## Results

Mean cross-entropy over seeds 41/43/47; lower is better:

| Arm | Pride tail, 120 blocks | Alice transfer, 32 blocks |
| --- | ---: | ---: |
| Unchanged base | 4.073713 | 4.011807 |
| Tangent, pointwise | **3.566816** | **3.673728** |
| Elliptic, pointwise | 3.717387 | 3.759478 |
| Tangent, causal | 3.600876 | 3.710587 |
| Elliptic, causal | 3.764982 | 3.782360 |

Causal-minus-pointwise CE is +0.034060 / +0.036859 for tangent and
+0.047595 / +0.022882 for elliptic (Pride / Alice). Every paired seed difference
is positive. Geometry-minus-tangent is +0.150571 / +0.085750 without mixing and
+0.164106 / +0.071773 with mixing.

The difference-in-differences is +0.013535 / -0.013977. The negative mean Alice
interaction is **not a win**: mixing worsens both maps for every seed, and only
one seed has a negative interaction. `summary.json` retains all five contrasts,
all per-seed values and their sample standard deviations, not a significance
claim or a count of blocks as independent experiments.

## Verification

- 6144 primary updates, plus 24 continuation-only validation updates. Every arm
  has 8450 trainable parameters; initial parameter hashes and batches are paired.
- All 12 final checkpoint hashes verify. Both projection gradients are finite
  and nonzero after the zero-readout startup step. The frozen base is unchanged.
- All 12 adapter/Adam next-update continuations match exactly. Completed resume
  does not rerun endpoints, and sealed result/journal bytes stay unchanged.
- All six rerun pointwise arms exactly reproduce their previous training records,
  development probes and per-block endpoints. These are deterministic replays,
  **not six new independent confirmations**. The unchanged baseline also matches.
- Initial preflight passes 90 tests; the parent optional-Torch import fix passes
  91. The combined client with merged lazy telemetry and an added parity case at
  the actual `[2,128,3]` orientation shape passes 97 tests with no skips/failures.
  The running experiment's frozen runtime was never replaced by those clients.
- The extended summary reproduces the preceding nonlinear study byte for byte.
  It also reproduces this study from the public compressed plan/results without
  importing Torch or evaluating the model again.

## Evidence And Reproduction

Executed revision: `03cee5fb8c65cffefaf4d60da4c5691a32dc6175`.
Study ID: `2d5860c91905b26b6b29e729b94cfa9c710bb11ade2e4125f827c778a89c3954`.
Uncompressed result SHA-256:
`d7b0ff56b3573c8e9d0a4f790aa95b043a145030afe56f3d302a13729184cd8f`.

`plan.json.gz` and `results.json.gz` preserve the full numeric plan and results
losslessly. `journal.json`, `summary.json`, `validation.json`, the original
`preflight.json` and `SHA256SUMS` distinguish planned, executed and verified state.
Raw logs, checkpoints, runtime, model and corpus text remain local.

From the repository root, using a new output path:

```sh
python -S tools/summarize_wave_gate_long_horizon.py \
  --plan benchmarks/results/2026-10-02-elliptic-causal-factorial/plan.json.gz \
  --results benchmarks/results/2026-10-02-elliptic-causal-factorial/results.json.gz \
  --journal benchmarks/results/2026-10-02-elliptic-causal-factorial/journal.json \
  --output /tmp/elliptic-causal-summary.json
```

The run uses a frozen float32 CPU GPT-2, its first MLP output, batch 2, 128-token
unpadded contexts, Adam 0.001 and residual strength 0.1. The geometric map and its
causal VJP are Rust-owned. Ordinary causal attention is a Torch control with the
same mathematical map, not bitwise matched arithmetic or matched compute.

Pride/Alice endpoints were previously inspected; this is exploratory reuse, not
pristine confirmation. No speed, significance, resident-GPU or general geometry
failure claim is made. Do not promote this causal adapter as a quality-improving
default. Preserving the local feature path and learning a contextual correction
is a next architectural hypothesis, not an explanation established by this run.
