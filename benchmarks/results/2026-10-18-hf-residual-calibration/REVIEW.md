# Bounded Pre-Measurement Review

A read-only local reviewer inspected plan, protocol, executor, model-free tests
and README plus the CI discovery line and required dependency references. It
did not access corpus, weights, models, builds or network resources. Unrelated
PR #2272 remained excluded.

Two publication findings were closed before measurement:

1. Target/current training RMS values would reconstruct fitted strengths.
   Public projection now removes calibration operands from both untouched
   references and intermediate/final fit stages, retaining private receipts.
2. Even development RMS ratios reveal the first ordinary layer's strength
   because its original and calibrated inputs/weights are identical. Matched
   control development RMS is therefore private too. Exact mismatch ratios
   become fixed boolean bands, preventing that direct algebraic reconstruction.

Final independent follow-up reported no remaining actionable findings in the
projection and documentation. This is a bounded publication policy, not a
formal privacy guarantee or proof against arbitrary model inference attacks.

Main-owned checks passed eight model-free protocol/projection tests, 38 existing
installed-wheel affine/artifact tests, Ruff and diff checks. No calibration or
pretrained-model measurement had been executed at this source-freeze boundary.
Actual calibration tolerance, artifact roundtrip logits and loss reproduction
remain runtime gates; a source review does not establish those outcomes.

## Saved-State Verification Follow-Up

A read-only review of the two new verifier/test files found missing stage-state
identity checks. The verifier now reconstructs each intermediate state using
only preceding layer strength updates and uses Rust-issued canonical artifact
IDs for source, intermediate and final stages. Regressions reject missing,
swapped and incorrect IDs. The focused independent follow-up found no remaining
actionable findings; it did not run tests or inspect private outcomes.

Main-owned verification subsequently passed all 27 candidates and 102 retained
traces, including native stage identities, learned-state equality, the fitted
strength rule and exact public projection. Thirteen model-free tests pass.
The original measurement and source files were unchanged by the verifier fix.

Final bounded publication review found no actionable findings: all 27 cases,
12 contrasts, displayed means/ranges, mismatch counts and in-scope hash links
agree. The reviewer checked public files only; raw trace reaggregation and
private state verification remain main-owned evidence, not independent reruns.
