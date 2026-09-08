# Resident Training Transpose Experiment: Not Promoted

The candidate is numerically valid in the tested fixtures, but **is not adopted
on main**. This branch retains the attempted optimization; do not treat its
`perf` commit as a demonstrated performance improvement.

- Baseline: `00a955a701aa21821fdca34a7c16d27d712aa997` (main plus opt-in diagnostic).
- Candidate: `753119911d9b3f3d22a1c8d0092ec33f70c96401`.
- Previous main optimization, PR #2093, remains separate and unchanged.

## Change And Method

The candidate changes cooperative tile loading so transposed operands put
their contiguous source axis on neighboring x lanes. Dot-product order,
activation/finite guards, SGD, pass coalescing and readback cadence are unchanged.
The common shader is touched, so this is not isolated to the high-level trainer;
general host-tensor performance has not been newly benchmarked here.

Both clients ran the same nine fixed recipes as the preceding training benchmark:
three seeds, depths 2/8/16, N-D shapes `[2,16,32]`, `[4,16,64]`, `[4,32,128]`,
eight nonzero-rate SGD steps per interval, two warmups and eight retained blocks
for each immediate/deferred loss-readback cadence. A whole-matrix repeat was
declared before inspecting the completed first browser result. No shapes,
tiles, losses, sample counts or numerical tolerances were tuned between runs.

Native baseline/candidate/eager-PyTorch order rotates; browser pair order
alternates. Owned GPU timing jobs were serialized with no compilation during
timing. Apple M4/macOS scheduling, other applications and power state remain
uncontrolled. Browser adapter metadata is a separate probe, not exact-device
attestation. These are bounded synthetic diagnostics, not LLM quality claims.

## Results

Each entry is the range across three seeds of old/candidate median elapsed
ratios. Above 1 favors the candidate. These are not confidence intervals.

| Run | Depth | Native Immediate | Native Deferred | Browser Immediate | Browser Deferred |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | 2 | 0.75-1.07 | 1.06-3.76 | 1.00-1.09 | 0.62-1.86 |
| First | 8 | 0.83-1.02 | 0.87-1.04 | 0.99-1.01 | 1.18-1.33 |
| First | 16 | 1.01-1.13 | 0.99-1.40 | 0.97-1.01 | 0.99-1.01 |
| Repeat | 2 | 0.72-1.02 | 0.99-1.13 | 0.96-1.00 | 0.91-1.03 |
| Repeat | 8 | 1.00-1.02 | 1.02-1.34 | 1.06-1.14 | 0.88-1.34 |
| Repeat | 16 | 0.90-1.05 | 0.99-1.01 | 0.93-0.99 | 0.99-1.00 |

The large first-run native deferred gains did not persist. Browser depth-16
results are mostly flat or slightly worse. Some depth-8 browser cases improve,
but they do not justify a universal default, or a device/shape routing threshold.
The candidate still does not resolve the largest native workload's MPS gap.

## Correctness And Verification

- Backend tests: 103 passed with real GPU tests enabled; the opt-in profile test
  is separately executed, not silently counted as a normal test pass.
- New coverage includes 72 shape/tile/kernel/accumulation combinations, testing
  rectangular transposed VJPs, partial edges, input gradients and actual SGD.
- Native and browser each passed 18 VJP fixtures and three 128-step learning
  trajectories, plus finite guards, saturated GELU and all-layer rollback/recovery.
- Independent CPU/MPS PyTorch correctness replay passed for both clients.
- Both complete timing matrices passed read-only captured-state validation,
  maximum absolute error about `1.49e-8` against recorded PyTorch states.
- All 18 native recipe/run pairs have matching baseline/candidate state
  fingerprints. These are consistency receipts, not independent device attestations.
- All 720 browser interval receipts were revalidated in order.
- Whole-workspace format check, seven benchmark-admission tests and strict
  backend Clippy on native and wasm32 passed. No PR CI was requested for this
  unpromoted experiment.

## Evidence And Next Test

`summary.json`, `validation.json` and `validation-repeat.json` retain compact
results and source bindings. `local-raw-manifest.json` identifies the complete
raw captures retained in the durable local log directory. **Those large raw
captures are not bundled here**: remote readers cannot independently revalidate
the numeric arrays from the manifest alone. Reproduction commands are in
`docs/resident_nn_training_benchmarks.md` at the candidate source.

The opt-in split-pass timestamp test identifies matrix work and bias-gradient
aggregation as larger costs than SGD in that diagnostic mode. It changes Metal
scheduling and excludes host/readback cost; it is not a measurement of the
coalesced production path. Its test executable hash was not captured, so retain
the narrower compile-log/source context rather than claiming binary attestation.

Next hypothesis: specialize the stage attributes already known by the Rust
training plan (transpose orientation and fusion flags), rather than add dynamic
tile-loading selectors to the shared fast path. Inspect and benchmark that
separately, preserve this failed candidate, and do not move policy into Python
or JavaScript. No general N-D autograd or automatic `ModuleTrainer` migration is
claimed by this experiment.
