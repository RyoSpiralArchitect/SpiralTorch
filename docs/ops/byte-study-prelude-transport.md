# Private transport for the fixed-wheel diagnostic

This is offline preparation for the [historical-prelude pair](byte-study-prelude-pair.md).
It does not authorize another GPU attempt, upload, secret provisioning, branch
push, CI dispatch, release, tag, version change or unrelated merge. The workflow
template remains under `docs/ops`, outside GitHub's workflow registration path.
The three public placeholders deliberately fail admission until explicitly
rendered and reviewed for an approved campaign.

## Local Archive Checks

`tools/byte_study_pair_transport.py` uses only the standard library and existing
offline validators. It never imports the native extension or foreign ML packages.
The input contains exactly the eight pinned original v9 failure files under
`original/version-9/` and the recovered wheel under `wheel/`. The wheel and its
native extension are independently hash-pinned; rebuilding is not equivalent.
Archive timestamps and ownership are fixed for reproducible plaintext packaging.

```sh
python -I -S -B tools/test_byte_study_pair_transport.py -v
python -I -S -B tools/byte_study_pair_transport.py pack-input \
  --input "$PRIVATE_FAILURE_ROOT" --wheel "$RECOVERED_PRIVATE_WHEEL" \
  --output "$FRESH_PRIVATE_INPUT_ARCHIVE"
python -I -S -B tools/byte_study_pair_transport.py unpack-input \
  --input "$FRESH_PRIVATE_INPUT_ARCHIVE" --sha256 "$REVIEWED_PLAINTEXT_SHA256" \
  --output "$FRESH_PRIVATE_INPUT_DIRECTORY"
```

Use a private parent directory, mode `0700`. Files are created exclusively with
mode `0600`; existing archives or extraction directories are never overwritten.
The parser accepts only bounded plain USTAR regular files, not links, special
files, extended headers, duplicate names, traversal, missing or extra members,
unbounded expansion or hidden trailing archives. Result archives carry an exact
file-size/hash manifest and reject file changes observed during packing. This is
for a quiescent private directory, not an atomic snapshot of an actively changing
or adversarial filesystem. A failed operation may leave incomplete private files;
their presence is not success. Quarantine them and use a fresh location for any
separately permitted offline repair, never resume the directory silently.

## Approved Hosted Execution

Only after separate approval, render
`byte-study-prelude-pair.workflow.yml` with the input ciphertext SHA-256, plaintext
archive SHA-256 and a new public result recipient. The result private key must
stay local. Prove local encryption/decryption with that exact key pair before
rendering the recipient. The runner also encrypts a harmless probe with its pinned
`age` tool before any diagnosis, so a malformed recipient cannot first fail at
final result sealing. The runner's encryption probe alone does not prove local
possession of the corresponding private key.
The input identity is a separate, narrowly named repository secret;
it is written outside the evidence directory, removed on step exit and unset
before decryption/diagnostic subprocesses. Never put keys in evidence or logs.
Only encrypted input belongs on the diagnostic branch.

On that branch only, the template can replace the already-registered
`.github/workflows/byte_study_replay.yml`. Review and commit the complete rendered
branch, then dispatch its exact commit once. Do not open a PR during this bounded
experiment: ordinary pull-request CI could launch an additional GPU workload.
The inspected base revision triggers ordinary CI only for `main` pushes or PRs,
and release builds for version tags; recheck triggers before an eventual push.
`run_attempt == 1` rejects reruns of a run; it does not prevent a second distinct
manual dispatch. Keep a durable one-dispatch receipt and do not dispatch again
on timeout, ambiguous response or failure.

The template installs the recovered wheel with no index, dependencies or cache,
using Python 3.12.10 on `macos-26`. There is no Rust build or package publication.
It invokes the fixed pair caller exactly once, with 90 minutes per arm and a
210-minute job limit. No loop or retry is introduced. Record actual runner image
and native/device provenance; a matching image does not mean the same physical
GPU or driver state. The current source checkout remains pinned independently
from the diagnostic harness commit.

## Recovery Is Not Qualification

The sealing step runs after ordinary step failures, packages only the dedicated
evidence directory, and encrypts it to the local holder's public result recipient.
The only uploaded files are ciphertext and its SHA-256 list, with three-day
retention. A host loss, hard job timeout, missing encryption tool or artifact
service failure can still prevent recovery. No success should be inferred from
an empty or absent result, and such a failure does not authorize another attempt.

After downloading the exact authorized run's artifact, verify ciphertext hashes
against its saved artifact/dispatch receipt, decrypt locally, and pin the hash of
that authenticated plaintext archive before extraction:

```sh
python -I -S -B tools/byte_study_pair_transport.py unpack-result \
  --input "$AUTHENTICATED_PRIVATE_RESULT_ARCHIVE" --sha256 "$PLAINTEXT_SHA256" \
  --output "$FRESH_PRIVATE_RESULT_DIRECTORY"
```

The command checks the whole archive and every declared member. Authenticating
decryption and manifest verification prove transport integrity, not the identity
of a remote experiment by themselves: also bind the recorded run, attempt,
harness, native wheel, source and raw comparison receipts. Preserve mismatch,
prelude, timeout, supervision, provenance and packaging errors separately.
Neither transport success nor two non-reproducing arms resolves the historical
failure or permits publication. Every transport receipt has `release_ready=false`.
