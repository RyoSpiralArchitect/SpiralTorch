# Attention Shift-Conservation Follow-Up

The isolated GPU regression already established a broken sum before this plan:
online exponential rescaling in VJP statistics disagrees with weights recomputed
at the final maximum. The two-pass correction passes the same conservation bound
without masking gradients, dropping Key-bias or relaxing comparison tolerances.

Before the new language run, fix this scope:

- Reuse width16/seed11, all ordinary/learned/frozen Poincare cases, all 1,024
  batches, optimizer controls and initial parameter bytes unchanged.
- Compare to the immediately preceding AdamW-repaired native debug, uninterrupted
  run and the original independent float32 Torch oracle. Both new/previous native
  runs use Rust 1.99, Apple M4/Metal and the same CLI/build profile.
- Make one fresh uninterrupted run. Do not resume old moments or retry for a
  more favorable outcome. Keep all strict failures and all three cases.
- No new prompt search, generation, test-data inspection, model promotion,
  relaxed tolerance, dropped coordinate, Python optimizer or provider call.
- Preserve private inputs, checkpoints, reports and exact executed runner/binary
  hashes. Public outputs are numerical localization and verification records.
- The driver builds the native debug example itself with Cargo/Rust 1.99 before
  launching. Its build receipt binds the reported executable to the recorded
  compiler, profile and unchanged reviewed sources; an arbitrary supplied
  executable is not accepted. This safeguard was added in pre-run review.

Pinned input SHA-256:

| Input | Hash |
| --- | --- |
| request | `b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c` |
| independent Torch | `96d37dd50825679d52687bb699ab8bcbeb4424af881d75ecabd566bcccc0090b` |
| previous repaired native report | `6bb2a87dcf87ef83ef209402eb713b8d8bc196ea6ceca560136341982b427317` |

This is a numerical follow-up, not evidence of geometry efficacy or language
quality. The new statistics have an additional score-only pass. No speed benefit
is claimed; the existing forward-only timestamp benchmark cannot measure that
backward cost. Earlier failed studies are not retroactively qualified.
