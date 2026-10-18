# Main Integration Follow-Up

PR #2269's reviewed head `92814e5e73e650be1b3841e63aaa87b4d0961f91`
completed its last macOS GPU check successfully. After PR #2271 merged, two
additive conflicts remained against main
`cdea76a5a17bfa0f03ea22c2ad9e0f233663db56`.

The merge preserves both parents' CI commands, both the 512-update diagnosis
limits and the packaged language/Python recovery roadmap. It removes only the
now-obsolete sentence proposing the diagnosis as future work. No training,
optimizer, numerical criteria, retained results or historical manifests change.

Validation after resolution:

- 42 diagnosis, rechecker, launch/adoption and saved-model qualification tests.
- 31 main-branch tests: compatibility runner (2), installed-store offline
  contract (14), recovery qualifier metadata (3), wide fixture generation (4),
  wide publication (3), and AdamW evidence qualification (5).
- Generated wide-rounding fixture bytes equal the retained fixture:
  `6d44dd27a675e5d2ffc19c59c9c03c49cc8a0b952917a8c5fc063675d10cef09`.
- Whitespace checks pass. Initial missing-test errors in the intentionally
  sparse review checkout were resolved by materializing only those test/helper
  paths; the primary checkout's intentional deletions were not restored.

These 73 offline tests do not replace CI for the new merge head. The earlier
green check belongs to its exact original head, not this integration commit.
