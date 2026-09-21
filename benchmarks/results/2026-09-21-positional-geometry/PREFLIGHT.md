# Excluded Preflight Attempts

No performance measurements use these attempts.

- `baseline-native-build`: fixture omitted the required `st-vision/nerf` feature;
  build failed. Its original manifest is retained. The corrected feature build
  succeeded at the unmodified mainline runtime.
- `baseline-wasm-build`: actual st-vision dragged the native Faer/spindle/
  atomic-wait chain into wasm32 and failed. The recorded dependency tree identifies
  that route. Native-only Faer scoping at 91d7f038 enabled an actual st-vision WASM
  build without changing numerical kernels or upgrading any dependency.
- `baseline-contract`: the seven added actual crate tests ran against unchanged
  runtime source: three controls passed; four layout/bounds/update contracts failed.
  Cargo stopped before the separate nerf_regression target. It was not a completed
  training-regression run. The original unformatted test is `nerf_geometry.rs` here.
- `frozen-baseline-native-build`: reuse of the workspace Cargo target by the
  standalone fixture failed with incompatible cached spiral-config type identities.
  The dependency-tree report does not justify a dependency upgrade. The stale
  executable copied after this failure is quarantined below `excluded-stale-build`,
  not admitted to a measurement cohort. The corresponding WASM build succeeded,
  but will also be rebuilt in the fixture-private target for uniform provenance.

Formal baselines and candidates use the same unchanged fixture sources, separate
from workspace verification builds. Existing caches and failure records are kept.

## Training Preflights

- The first layout/phase candidate passed 59 library tests and 8 geometry tests,
  then the previously CI-excluded training regression diverged at the legacy
  width-dependent ramp initialization. Its receipt also records a formatting
  overlap; it is diagnostic, not an admitted frozen-source verification.
- Seeded Xavier initialization stopped the divergence and passed the old test.
  The old absolute bound (loss < 0.2) also admits an entirely black image (0.18).
- `seeded-training-preflight` evaluated seeds 0, 1, 7, 13 at fixed midpoints after
  20 steps. Default seed 13 was fully inactive; this is retained as a failure.
- `positive-density-preflight` starts the density head at constant 0.1. All
  four seeds improve, but fail the exploratory combined 1%-improvement and
  better-than-black criterion after just 20 steps. No seed or learning rate
  was changed to hide this result.
- The final regression asserts measurable progress (>1e-5 absolute MSE decrease)
  over these same fixed points, not convergence or a novel-scene quality gain.
  This is a revised engineering regression criterion, NOT an independently
  held-out or preregistered scientific result. The original stronger failures
  remain published. An explicit density-head test verifies active initial
  gradients independently of that loss threshold.

- `runtime-contracts` failed to compile a new test because it treated the
  existing infallible transpose/row-sum APIs as Results. The helper was corrected.
- `runtime-contracts-corrected` ran all 1,001 core tests: 1,000 passed; the new
  independent libm recurrence comparison differed by one float64 ULP. Its
  numerical oracle now permits 4 eps absolute on [-1,1]. Cache-key identity,
  hit reuse and history-independence checks remain exact; no production
  trigonometry change was made to RoPE.

- `verification-before-lint-cleanup` passed all 1,790 core/NN/vision contract
  tests, then strict feature-enabled vision Clippy found three old iterator
  style issues. They were corrected without changing arithmetic order.
- The second verification attempt again passed all contracts; strict Clippy
  then required four fixed-width test slices to use `as_chunks`. These are
  test-only changes; no lint exemptions were introduced.
- The shared workspace target later produced incompatible cached serde_json
  identities when building the public WASM binding, after the real st-vision
  WASM check had succeeded. `wasm-build-shared-cache-failure` preserves the full
  compiler output and driver state. Public client builds use a fresh scoped
  `positional-geometry-clients-v1` target; existing caches were not deleted and
  dependency versions were not changed. Previously successful source-bound
  stages are reused only after rechecking their commit and all source hashes.
