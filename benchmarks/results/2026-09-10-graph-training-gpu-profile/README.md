# Scheduled Graph Training GPU Profiles

Rust now exposes opt-in private-device profiling from an NN `InferencePlan`.
It uses the ordinary training encoder and existing compute-pass boundaries,
without changing shader math, forward-only inference, loss/VJP guards or SGD.
The production Python/WASM profiler classes are not exposed yet; the browser
benchmark calls the same Rust implementation directly.

## Source And Scope

- Implementation and frozen native/WASM/Python products: `04a014f0`.
- Uninstrumented baseline: `df64ed06`, with identical core/NN/tensor sources to the
  preceding evidence-only HEAD `336c73db`.
- Native Apple M4/Metal; isolated Chrome 152.0.7977.83 WebGPU. Browser physical
  GPU and background contention remain UNKNOWN. No CUDA/Furnace run here.
- Eager Torch 2.12.1 MPS, fallback disabled, no equivalent finite-check/rollback
  machinery, no `torch.compile` or fastest-PyTorch claim.

## Verified Behavior

Each client ran 21 conditions: the six standard/wide shape-depth recipes at
seeds 17/29/43, plus module-compatible gain scaling for the smallest recipe.
Each condition compares 12 successive real SGD updates on three workspaces:
profiled, uninstrumented timestamp-capable, and ordinary. First three profiles
are warmups, nine retained. This is 504 profiled updates, 378 retained profiles
and 1,512 main updates including controls, not long-running application training.

All compared losses, final predictions, input VJPs, raw/effective gradients and
parameters matched exactly between these controls (maximum absolute error 0).
Pending reuse, dropped receipts, numerical rollback, explicit recovery, owning
reads after workspace drop and unchanged ordinary device features passed.
The browser also rejects an injected completion-promise failure.

Final checks: 114 backend tests, 733 NN tests, six integration tests, 30 Python
GPU tests without skips, native/WASM backend strict Clippy, CPU-only check and
formatting passed. Independent Torch CPU/MPS replay passed 24 trajectories and
5,184 comparisons; its maximum absolute error was about 1.19e-7.

## Failures Found And Fixed

- Putting every full state into one final browser DOM report crashed the page
  after all 21 GPU cases completed. Full captures now stream per case into a
  hash-bound JSONL file; the DOM contains only a short status.
- Same-command-buffer query resolution often returned final-update timestamps
  as `0/0` on native Metal. Resolution now waits for the training submission to
  complete. The wait belongs only to diagnostic readback, not ordinary `step`.
  Ambiguous zero pairs mark timings incomplete and null the affected phase
  total/span; they never mean zero update cost.
- This exposed `unimplemented!()` in pinned WGPU 0.20's browser queue completion.
  It now uses the browser's owning completion Promise; rejection drops the
  callback and disconnects the waiting channel instead of certifying success.

The adopted 504 profiles contain no missing/ambiguous or zero intervals. Earlier
failures and raw captures remain archived. Equal nonzero ticks are still allowed
as quantized zero intervals by the contract.

## Uninstrumented Regression

One fresh paired round per matrix/client: two warmups and eight retained rotated
blocks per cadence, eight SGD updates and all requested loss readbacks per block.
There are 1,800 raw intervals, 1,440 retained and 14,400 updates over reset
trajectories. Values below are geometric means of baseline/candidate case-median
ratios: greater than one means the candidate took less time.

| Matrix / Client | Immediate | Deferred |
| --- | ---: | ---: |
| Standard / Native | 1.014 | 0.973 |
| Standard / Browser | 1.000 | 0.991 |
| Wide / Native | 1.050 | 1.039 |
| Wide / Browser | 1.002 | 1.004 |

This is not a speedup or universal no-regression claim. Individual ratios range
as low as 0.805 for standard native immediate and 0.903 for wide browser deferred.
All numerical gates passed; wider eager Torch MPS remains roughly twice as fast.
Every case, retained range and loss trajectory is preserved.

Diagnostic pass times have substantial dispersion and instrumentation effects.
They are not uninstrumented kernel costs: `forward_mixed`, `dense_backward` and
`update` may each contain multiple dispatches. Compute sums exclude copies,
inter-pass gaps and query/readback work. Per-phase median shares need not sum to
one. Dense backward and adjacent pointwise fusion are follow-up candidates for
paired uninstrumented experiments, not conclusions from these timestamps alone.

## Evidence

`summary.json` contains per-shape profile summaries, regression ratios and limits.
`manifest.json` binds 70 losslessly compressed raw artifacts to both compressed
and decompressed SHA-256 hashes. Browser profile JSONL captures are retained in
full, including all three states. Redundant regression browser case streams are
represented by identical complete cases inside their browser JSON reports.
Frozen binary/module hashes and commands are in the build/check receipts.

Ordinary `Module::forward`, `ModuleTrainer`, `pure::Tensor` storage and general
autograd routing remain unchanged. The earlier one-off parallel GPU VJP failure
is still UNKNOWN; the serial passes here do not establish its cause. No push,
PR, merge or release was performed during this work.
