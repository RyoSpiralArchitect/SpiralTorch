# Read-only Actions input

`byte_release_actions_download.py` authenticates two fixed GitHub metadata GETs
for the externally pinned producer run and artifact, then downloads the exact
archive. It accepts a direct stream or one redirect to GitHub's documented
artifact endpoints: `results-receiver.actions.githubusercontent.com` or a single
Azure storage-account label under `blob.core.windows.net`. The latter is
GitHub-managed transport, not creation of a new storage service. API credentials
are never sent to either redirect endpoint. Userinfo, ports, IP literals, unsafe
paths, fragments, other domains and second redirects reject.
[GitHub artifact domains](https://docs.github.com/en/actions/reference/runners/self-hosted-runners#accessible-domains-by-function)

The reader makes five GETs for a direct stream, six for a redirected one.
Both run and artifact identities must still match after transfer, including
attempt 1, successful completion, unchanged source and non-expired artifact.
Metadata-supplied download URLs are ignored. Metadata responses stay within
64 KiB; archive and ciphertext keep the existing 129 MiB limits and exact
lengths/SHA-256. The existing bounded offline ZIP verifier performs extraction.
Partial first outputs remain exclusive. Credentials/URLs are excluded from
receipts and error text. No retry, write request, token acquisition or fallback
exists. The caller must supervise the whole operation within one deadline.

`byte_release_actions_probe.py` provides a separate 300-second owned worker and
15-minute CPU-only preflight on macOS, with contents/actions read permissions.
It only accepts small pinned inputs (at most 64 KiB), no decryption key or ML
execution, and checks clean source, expected pin digest, fixed branch, unique
submission and attempt 1. Its rendered workflow must stay on the dedicated
unmergeable probe branch. Safe receipts/logs and the original ciphertext archive
and member (including partial first outputs on failure) are retained as Actions
artifacts, never plaintext or a key. Recovery requires the download operation's
successful completion record and exact saved bytes, not merely matching files.
The completed producer is never rerun by this probe.

Unit tests and a small live probe do not qualify a candidate. The actual input
must still bind the frozen source/wheel, unchanged full local proof, reviewed
controller and external amendment before the one hosted GPU qualification.
