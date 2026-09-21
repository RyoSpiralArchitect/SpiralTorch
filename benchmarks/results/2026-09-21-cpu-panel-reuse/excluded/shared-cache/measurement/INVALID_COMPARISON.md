# Excluded source comparison

The baseline and candidate executable SHA-256 values in the build receipt are
identical (3a7bb6d52fe1585e60fad9b3861cce005ede128dfb33bca2f416c5c3d61133fa).
The candidate build log did not rebuild st-tensor after switching worktrees
through a shared Cargo target directory. Source identity failed even though the
numerical gates passed. These complete records are retained for audit only.
No timing from this directory supports a baseline-versus-candidate claim.

The replacement build forces the changed crate and harness to rebuild with
mtime-only touches, verifies source bytes against Git, requires matching-root
compilation logs, and rejects identical executable hashes before measurement.
