# Learned Gain: Pretrained Connection Check

Source revision: `ab9f4ea35c9089cf57c7a11fb593555c2d57b668`.
This validates the [fixed learned-gain study](../../../docs/fractional_gain_study.md),
not its nine long-run outcomes or a quality improvement.

The existing local GPT-2 and Pride/Alice assets matched their previously
recorded hashes. The frozen base, data partition and minibatch schedule also
matched the completed history/energy factorial. No assets were downloaded.
The actual first-MLP activation shape was `[2, 128, 768]`.

## Observed

All three arms have 1538 trainable parameters, identity initialization and
matching initial history maps within fixed f32 tolerance. Each made two
training-only updates and a separate reference/replay continuation pair:
six auxiliary training updates plus six continuation-only updates in total.
Parameters and Adam states matched after continuation; the base was unchanged.
No held-out losses were computed and no model weights are included here.

| Arm | Shape After Two Updates | Gain After Two Updates |
| --- | ---: | ---: |
| Ordinary two-tap, angle coordinate | 0.4629037082195282 | 2.237732172012329 |
| GL short, alpha coordinate | 1.9985129833221436 | 2.237732172012329 |
| GL full, alpha coordinate | 1.998513102531433 | 2.237732172012329 |

Gain began at `2.2360680103302`. First shape/gain gradients were zero because
the history gate began at zero; both became nonzero on the next update.
All arms' first two training losses were identical. These losses refer to
different minibatches and must not be presented as a before/after quality
curve. Angle and alpha are different coordinates, not directly comparable
parameter estimates. Full trajectories and fixed-budget endpoint comparisons
remain the primary experiment's job.

## Validation And Reproduction

- 135 Rust tests; 604 Python regression tests, including 18 gain-study tests.
- 34 benchmark correctness tests, with no timing comparison.
- Seven real compiled-WASM fixtures and TypeScript checking passed.
- Native release, explicit CPU/text check and formatting passed.
- Strict native/WASM `st-frac` Clippy passed; binding Clippy allowed the
  preexisting argument-count/type-complexity categories, not all warnings.
- The new package's 71 files and seven client files matched current sources.
  The 423 recorded prior study/client/runtime files remained unchanged.

[preflight.json](preflight.json) contains the emitted numerical records.
[validation.json](validation.json) records hashes and verification scope.
`SHA256SUMS` covers these three publication files. Runtime packages, checkpoints,
corpus text and raw private logs remain local.

Use `tools/preflight_fractional_gain_study.py` with the arguments in the
[protocol](../../../docs/fractional_gain_study.md#run-and-resume), a matching
local model/corpus, the completed factorial study, and an isolated built
SpiralTorch package. The primary run must use its own fresh directory and
the fixed three arms, three seeds and 512-update budget. This preflight did
not select winners, checkpoints, duration or hyperparameters.
