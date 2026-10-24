# One authorized hosted retained-boundary attempt

This diagnostic-only branch is not a release or a proposed main-branch change.
It reuses the registered byte_study_replay.yml workflow filename solely to
dispatch the retained caller once against this reviewed branch.

The native source is pinned to e053c314d54be06b346e30e5ea1a065d7f3fdc90.
The original request and all 25 cases stay unchanged. There is no warmup,
test-suite GPU prelude, math change, tolerance change, or automatic retry.
This is not the historical native binary or physical GPU.

Only authenticated age ciphertext is committed. Original packet SHA-256:
be7dc0c48c6b23c93964ec9319ce8ae490ea473ddefa1a715c792a2ff61e4605.
Ciphertext SHA-256:
a74a3369eb9f9dba60982cda47930db83233cbfd9baf4377675b7773e1a13915.

The input identity is a short-lived repository secret. The result identity
never leaves the operator's local private evidence folder. Public artifacts
contain only encrypted evidence and its ciphertext hash; runtime errors and
partial outputs are included in the encrypted archive, not public logs.

The workflow rejects reruns and an unexpected branch/harness commit. The
operator dispatches once and removes the input secret after decryption or
terminal failure. A separately triggered second run is not authorized.
The platform can terminate a job before sealing; the worker's 120-minute
deadline leaves 60 minutes in the 180-minute job for build and evidence capture,
but artifact recovery after platform termination is not guaranteed.

Even an exact-match result leaves release_ready=false and issue #2289 open.
PyPI publication, tag/version updates, and PR #2291 remain held.
