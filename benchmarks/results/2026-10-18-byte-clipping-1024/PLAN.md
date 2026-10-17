# Request-bound clipping: 1,024-update language study

Declared before execution. This is a one-seed mechanism, numerical agreement,
and process-recovery experiment, not a geometric quality or speed claim.

## Fixed design

- Reuse the complete private width-16, seed-11 request, including corpus,
  initialization, window order, 1,024 updates and held-out evaluation schedule.
- Batch 2, sequence 32, hidden 32, two heads, one block, geometry width 4,
  curvature -0.75. Ordinary, learned Poincare, and frozen Poincare arms.
- Preserve AdamW, all learning rates, epsilon, decay and freeze policies.
- Only wrap v9 in v10, selecting either `disabled.v1` or
  `active_global_l2.v1` with `max_norm=0.01` (rounded by Rust to float32).
  This threshold is fixed from the preceding small mechanism experiment.
- Run the installed source-candidate wheel continuously, then in a separate
  worker killed immediately after sealing update 512, then resume in a fresh
  process to 1,024. Require byte-identical final checkpoint/report, unchanged
  parent artifacts, verified lineage and restored inference/generation equality.
- Compute a fresh independent CPU PyTorch reference for each condition. Retain
  the existing strict parameter, geometry-delta and AdamW moment gates without
  relaxing tolerances or masking Key-bias gradients. Publish failures too.
- Compare all disabled-control case payloads with the pinned earlier native and
  Torch reports, separately from strict comparison; schema/request ids change.
- Serialize GPU work. Bound each native worker and CPU reference to 1,800
  seconds; bound the three-worker qualification command to 6,000 seconds.
  No automatic retries, seed search or threshold selection from these results.

## Input pins

| Artifact | SHA-256 |
| --- | --- |
| Original v9 request | `b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c` |
| Original Torch report | `96d37dd50825679d52687bb699ab8bcbeb4424af881d75ecabd566bcccc0090b` |
| Latest unclipped native report | `ddc47f8bcd053391daf265cb1a3f914bdc78bceeb8c4ea4371792772ef80f280` |
| Candidate wheel | `74fdba8c31dbca9e6200992467447e986fbfc742a681c9bb85e8fdc78c0f36d9` |
| Installed native extension | `f1539edd9305f54ce8ac25534b267cfca8e85a8d7520058e12d727440af5d305` |

The runner seals plan, source, wheel and derived-request hashes before work and
rechecks them afterwards. Private corpus, complete requests, raw reports,
weights, checkpoints and logs remain local. Publish metrics, dispositions,
artifact hashes, validation records and reproduction instructions only.

## Interpretation

A recovery pass establishes this checkpoint/optimizer/geometry path, not
power-loss durability. Lower loss on this one seed does not establish a general
Z-space benefit. The previous long study failed ten Key-bias scalar tolerances
despite near-identical losses; a new failure remains a failure even when localized
to an analytically shift-invariant component. Tiny clipped SGD previously failed
the geometry-learning qualification; this AdamW study does not requalify it.
