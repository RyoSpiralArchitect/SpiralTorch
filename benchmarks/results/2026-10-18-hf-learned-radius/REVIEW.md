# Bounded Pre-Training Review

A read-only local reviewer inspected `plan.json`, `replay.py` and
`test_replay.py`, using the existing book helpers as reference. It reported no
actionable findings: radius-only trainability changes and the additional four
parameters, input/runtime pins, fixed-control replay gates, fourteen-case
failure retention, fresh-process resume comparisons and aggregate-only radius
reporting were consistent. Unrelated PR #2272 was excluded.

The reviewer did not run tests, preflight, models or training, and changed no
files. Main-owned checks before launch passed five model-free tests and 51
installed-wheel radius/HF Trainer tests, including CPU/MPS numerical and resume
checks. The initial Ruff lambda-style finding was replaced with an equivalent
local function before freezing the source. None of these source/fixture checks
establish a pretrained-model outcome or a quality advantage.

## Publication Review

Before pushing, main found that a site's exact maximum parameter change could
reveal its individual scalar radius update. A deterministic public projection
now removes all 56 per-site maxima across all fourteen outcomes. The original
private results, manifest, checkpoints and logs remain unchanged; their hashes
are retained. The projection does not remove cases, losses or paired contrasts.

The independent reviewer then inspected the current eight-file publication
scope: README, results, execution, validation, publication projection, saved-
state verifier, tests and this review record. It also checked the three private
receipts and plan/replay hash references. It reported no remaining actionable
findings: all fourteen outcomes and six paired contrasts were preserved, hash
links and README statistics agreed, resume claims were bounded, and no raw
text, generated text or individual-weight disclosure remained.

The reviewer did not load checkpoint weights, rerun tests/verifiers or repeat
optimizer/RNG equality checks. Those remain main-owned: all fourteen result/log
hashes, 26 checkpoints and 156 companions were rechecked; radius and parameter
aggregates were recomputed; both fresh-process resumes matched final state,
optimizer/scheduler/MPS RNG tensors and HF RNG bytes. Seven publication tests
and 29 shared model-free regressions passed. These checks do not imply quality
superiority, power-loss recovery or equivalence across untested environments.
