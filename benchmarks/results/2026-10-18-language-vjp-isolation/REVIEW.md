# Review Dispositions

Independent local reviewer: read-only source inspection and analytic witnesses.
The implementing agent ran the recorded GPU tests. The reviewer did not execute
tests, change files, or certify model quality.

| Finding | Disposition |
| --- | --- |
| Unshifted centering loses dominant-key derivative | Reproduced; shifted compensated centering; shared native/browser regression |
| Wide value-gradient reduction drops a small residual | Reproduced; separate-exponent compensation; shared regression |
| Backward denominator drops tiny terms retained by forward | Reproduced, including packed denominator tail; compensated denominator and native scratch regression |
| Existing normal-output midpoint rounding ignores low/tail | Open, separate shared-helper issue; reproducer retained; no correctly-rounded or IEEE-f64 claim |
| Denominator regression could vacuously pass if both weight and tail became zero | Added an explicit finite, positive shader-weight assertion and reran the test |

Parent review also found and reproduced a regression in the first shifted-dot
repair: subtracting a large reference value before preserving its correction
lost a small value. Both components now survive products and centering; the
failing intermediate and successful four-case regression are retained.

The denominator witness is ill-conditioned as a whole-gradient comparison.
Its scratch-tail test is the primary attribution check. Remaining observed
query-gradient residuals are reported separately, not attributed to the fixed
denominator or described as mathematically unavoidable. No zeroing workaround
was applied.

The source review checked fourteen-word scratch producers/consumers, causal
reference-key selection, per-lane dimension indexing and workgroup budgets.
The kernel still has bounded extended precision. Private accumulator spills,
extra score scans and value reads require separate performance measurement.
Neither this local review nor passing tests substitute for external CI.
