# CPU-only ciphertext bridge

The maintainer-approved exception grants `contents:write` only to a separate,
short CPU staging job. The final qualification job must keep `contents:read`
and `actions:read`; no personal-token fallback is allowed. This is not a repair
of the earlier failed read-only draft probe and must not rerun that probe.

`tools/byte_release_input_bridge.py` renders a dedicated, unmergeable branch's
workflow and stages one pinned, unpublished release asset. Its reader performs
GET requests only, including an authenticated unique-first-run check. The
workflow runs for at most 15 minutes; the complete fetch worker has one shared
300-second deadline and no retry. The checked-out commit, specification digest,
repository, workflow, branch, run ID and attempt 1 are all bound. Both the
runner and local coordinator must reject earlier or ambiguous submissions.

The job has no input decryption key, wheel installation, model import, training
command or publication step. The token is scoped to the fetch step and checkout
does not persist credentials. Only a successful, byte-verified ciphertext is
passed to the pinned upload action; receipts and safe error types are retained
even on failure. Signed URLs, tokens and private response bodies are not logged.
Artifact uploads are immutable, uncompressed, and retained for seven days.

The local coordinator must obtain the completed first run and exact artifact
ID, size and SHA-256 before constructing the read-only consumer pin. A passing
bridge means only that encrypted bytes were staged. The offline Actions input
verifier still requires an authenticated bounded downloader and a separately
reviewed candidate amendment before the final hosted GPU run. No frozen source,
wheel, local proof, numerical tolerance or candidate count changes here.

Use synthetic encrypted data to establish the new route before production
input. Keep the original failures and all complete evidence. Delete only owned
temporary draft assets after verified recovery, without moving or creating tags.
Neither helper authorizes execution or publication on its own.
