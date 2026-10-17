# AdamW Numerical Isolation And Bounded Language Follow-Up

The optimizer-only comparison preceded this note: its unchanged float32 Torch
gate failed before the complement-correction patch and passed afterward. Keep
both results. The scalar float64 trajectory is diagnostic, not a newly relaxed
gate. The existing full-language failures remain failures.

This language follow-up is specified before running the changed kernel:

- Use only `w16-seed11`, the first, smallest condition in the earlier capacity
  matrix, not a seed selected for favorable new results.
- Reuse its exact request, initial values, 1,024 batches, ordinary/learned/frozen
  Poincare cases and optimizer controls. Do not change Attention, masks, betas,
  epsilon, schedules, decoder, corpus, or tolerance.
- Run one fresh native Rust study with the changed kernel. No automatic retry,
  no resumed old moments, no new generation/probe-data search, no promotion.
- Reuse the previously recorded independent Torch report, whose request identity
  is verified by the existing comparison. Bind all inputs to committed receipts.
- Keep the full native report, checkpoint, errors and weights private. Publish
  all three cases' numerical localization and hashes, including failures.

Pinned SHA-256 identities from the earlier committed private-receipt manifest:

| Input | SHA-256 |
| --- | --- |
| request | `b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c` |
| independent Torch | `96d37dd50825679d52687bb699ab8bcbeb4424af881d75ecabd566bcccc0090b` |
| previous native final report | `6f4ca125c9d4ded06d8cf3fb06f518992efa66dc90e7f912c38ba62efb2edcf5` |

The original full study was segmented; this bounded run is uninterrupted. That
is not a new exact-resume experiment. Build/device identities are retained, and
no change in this one seed can retroactively qualify the six prior studies or
establish geometry efficacy, language quality, speed, or hardware portability.
