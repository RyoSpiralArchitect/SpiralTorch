# Independent review and disposition

Read-only local independent reviews were used for this increment. Reviewers
did not edit the worktree or run overlapping builds/GPU workloads. The parent
agent performed the recorded tests and native/browser executions. No formal
PR, CI run, or merge is part of this increment.

## Findings addressed

1. Large cancelling chart products erased the homogeneous overlap anchor.
   Compensated accumulation now retains the low overlap; an analytic failing
   pure regression was reproduced before correction.
2. Collapsing the compensated dot before adding the anchor lost the residual
   in `1 - 1 + epsilon^2`. The anchor now participates before collapse.
3. Summing each complex product before compensation erased its smaller real
   term. Primitive real products now enter the compensated sums separately.
4. The observation-only qualification mode retained relative failures but
   still discarded the trajectory on finite absolute/relative CE, parameter,
   derivative, or mixture-weight failures. These finite learning failures are
   now recorded, with `passed=false`; legacy entry points remain strict.
5. Subtracting the base endpoints before compensation erased a recoverable
   derivative. Each signed endpoint is now accumulated separately in Rust and
   WGSL. The failing regression was reproduced before correction, then the
   full analytic vector was checked in pure Rust, native GPU and browser.
6. A malformed/nonfinite router control could be recorded as an ordinary
   qualification failure. Reference and observed contrast preflight now
   reject missing/empty/mismatched/nonnumeric/nonfinite vectors and finite
   inputs whose f32 contrast overflows.

The final bounded follow-up found no actionable issue in fixes 5 and 6 and
confirmed that the earlier dot fixes and frozen gates were preserved by source
inspection. This is not a claim of exhaustive numerical verification.

## Regressions and limits

The shared native/browser boundary suite contains eight cases: three
tiny/saturated controls and five analytic cancellation cases, with 30
nonzero endpoint components. CPU/GPU agreement is not the sole oracle:
the cancellation cases have analytic expected derivatives. The separate
Torch reference constructs normalized complex projectors and uses autograd.

These fixes do not prove accurate arithmetic for every ill-conditioned finite
chart, and do not waive learning-trajectory or router-contrast failures.
The fixed six-case study's outcome is recorded separately in
[reviewed/results.json](reviewed/results.json). The initial failed observations
and their source hashes remain in [results.json](results.json) and
[validation.json](validation.json).

The changes are opt-in. Earlier three-metric fixture bytes and historical
contextual failures are not replaced by this study. No language-quality,
fine-tuning efficacy, or speed advantage has been established.
