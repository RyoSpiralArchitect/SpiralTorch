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
