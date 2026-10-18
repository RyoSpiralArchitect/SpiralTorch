# Bounded Independent Review

A read-only local reviewer inspected the new observation scope, native report
transport, normal-hook integration, lazy exports and fixed saved-state runner.
It did not run models, builds, tests or network operations or modify any files.
Unrelated PR #2272 remained outside the review.

## Finding And Resolution

The original native-report sink checked capacity when issued. An outer sink
could remain pending while a reentrant inner forward filled the last slot,
then append a seventeenth report. The equality-only precheck would subsequently
stop recognizing the full buffer.

The capture now enforces `>= 16` at the actual append as well as the initial
precheck and counts every discarded report. An exact 15-plus-pending-outer-plus-
inner regression verifies the cap and omissions. Independent follow-up closed
the finding; the completed source review reported no remaining actionable
issues in the bounded scope.

## Main-Owned Validation

Initial runtime tests retained two issues: test fixtures attempted to deepcopy
an intentionally non-pickleable Rust kernel, and combined CPU/dtype conversion
attempted an unsupported float64 conversion on MPS. Tests now reconstruct the
adapter and load its state; observation explicitly transfers to CPU before
float64 conversion. The original failure log remains retained.

The installed reviewed wheel passed 158 related tests, including real CPU/MPS
HF training with exact geometry/base/optimizer/scheduler and RNG equality with
observation on/off. Six model-free tests cover all 28 saved cases, external
state identities, row-weighted aggregation, incomplete/nonfinite coverage,
ordinary controls and installed-wheel identity. The runner also accepts the
actual Torch `mps:0` device spelling, verified before model measurements.

These are engineering and scoped source checks, not a security audit of all
dependencies, a speed qualification, or evidence that a geometry improves LM
quality. Actual saved-state measurement and publication are recorded separately.
