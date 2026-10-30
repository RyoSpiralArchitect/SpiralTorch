# Offline Actions input verifier

`tools/byte_release_actions_input.py` validates supplied bridge-run and artifact
metadata and unpacks one already downloaded, externally pinned archive. It is
a CPU-only component, not an enabled candidate route, downloader or permission
grant. It does not replace the failed read-only draft probe's result.

The pin binds repository identity, a dedicated bridge branch, reviewed bridge
head, registered workflow ID/path, run ID and attempt 1. Supplied metadata must
show a completed successful manual run from that same repository, not a fork,
and the exact non-expired artifact ID/name/size/digest from that run. Booleans
and floats cannot alias integer identities. Metadata alone is not authenticated;
a future caller must retrieve it through the approved authenticated read path.
No URL or credential is accepted in the pin. API-provided URLs are not used.

Before ZIP parsing, the complete downloaded archive must match its fixed bytes
and SHA-256. Both outer ZIP and inner ciphertext stay within the existing 129
MiB transport cap. This is an additional container within that cap, not a limit
increase. A bounded classic single-member ZIP directory is required: no ZIP64
directory/locator overrides, multi-disk archives, archive comments, trailing
records or oversized directories. The immediately preceding ZIP64 locator is
rejected before `ZipFile` can replace otherwise bounded classic directory counts.
The sole ordinary member must be the pinned basename ending in `.age`, using
stored or deflated compression, with the exact expected length and SHA-256.
Unsafe paths, extra files, symlinks, special files and ZIP encryption reject.
Outputs are exclusive and private; failed partial output cannot be overwritten.
The original archive identity is checked again after extraction.

This verifies bytes, not cryptography. An `.age` name does not establish valid
encryption: the producer's reviewed encryption and the consumer's authenticated
age decryption remain required. It also does not certify the bridge's permissions
or prove absence of earlier fresh bridge runs. External first-submission
ownership, reviewed source and the separately approved transport amendment remain
necessary. The helper cannot admit, dispatch or promote a release.

The existing Actions API documents artifact metadata and download under
Actions read permission, but live access with the final job's token has not
been tested for this route. The proposed CPU bridge permission exception is
still a separate decision. No bridge workflow is added or modified here.
[GitHub artifact API](https://docs.github.com/en/rest/actions/artifacts)

Follow-up integration must bind the original admission and ciphertext in the
outer candidate descriptor, preserve complete local evidence and the frozen
numerical checkout, and review download/deadline/credential handling. The current
candidate verifier intentionally does not recognize this helper's pin schema.
CPU fixtures are not evidence that a producer or live GPU input path works.
