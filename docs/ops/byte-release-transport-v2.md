# Byte-release transport v2: CPU implementation boundary

This is a separate transport implementation, not an applied candidate amendment.
It neither qualifies a release nor grants authority to transfer private evidence,
dispatch a workflow, repeat a GPU test, publish a package, or carry forward a
previous local result. Record those decisions separately before live use.

The v1 tools, 96 MiB input limit, qualification workflow, numerical admission and
original failure remain unchanged. The v2 workflow is currently a renderer and
test fixture only: no deployed workflow or committed input descriptor is added
by this implementation phase.

## Two roots, one numerical contract

- The execution checkout stays clean at the admitted source commit and tree. Its
  original input verifier, offline installation and numerical runner are used.
- The controller checkout contains only allowlisted transport changes, its
  reviewed workflow and a small, non-secret input descriptor. It cannot overlay
  files in the execution checkout or substitute a numerical acceptance rule.
- Both roots are checked by commit, tree and clean status. Comparing tree entries
  does not require a controller parent commit or full Git history.
- All three v2 runtime modules must be tracked ordinary controller files before
  dispatch; a descriptor/workflow-only controller cannot reserve an attempt.
- The controller workflow must equal the complete rendering of the original
  workflow. The original runner, Python/action pins, installation flags, test
  command, environment knobs, 270-minute outer deadline and return upload step
  are preserved. Source checkout moves before admission so both roots can be
  checked before the input secrets are accessed.
- The workflow filename, fixed candidate branch and concurrency group do not
  change. A new transport version cannot create a fresh qualification namespace.

This comparison is an implementation guard, not an independent trust root. The
controller code, renderer and complete rendered workflow still require review,
and their exact revision must be approved outside that checkout.

## Identity chain

The identities are deliberately acyclic:

1. Retain the complete original input ZIP and encrypt it outside Git only after
   separate approval. Pin exact plaintext and ciphertext byte counts and hashes.
2. Create descriptor D at `.github/byte-release-input.json`. It binds candidate,
   admission digest, source commit, both object identities and one direct HTTPS
   host/path. Its schema is `spiraltorch.byte_release.transport_input.v2`.
3. Review and commit controller C, including D and the rendered workflow. D must
   not contain C's hash, an authentication token or a credential-bearing URL.
4. Record external amendment A, then pin its canonical digest in the approved
   coordinator record. A names the source and controller commits/trees, D's
   digest, the admission and complete local-evidence digests, review-record
   digest, fixed namespace, limits and the public result-encryption recipient.
   Its schema is `spiraltorch.byte_release.transport_amendment.v1`.

`validate_amendment` only checks data against an expected digest. Constructing
valid JSON does not authorize candidate application or establish that a human
review occurred. The decision to retain local 3/3 under this transport change
must be made explicitly; the original admission and result are never rewritten.
Every transport record keeps `release_ready=false`.

## Bounded input and credentials

V2 permits a plaintext ZIP up to 128 MiB and ciphertext up to 129 MiB, each with
an exact pinned length and SHA-256. It retains the v1 inner archive format and
limits: expanded total 2 GiB, one member 256 MiB, manifest 8 MiB and 16,384 payload
files. No evidence is removed to meet a smaller transport limit.

The direct-object downloader accepts only the approved HTTPS host and exact opaque path.
Credential-bearing query parameters exist only in the runtime URL secret, not
in D, A, Git, workflow inputs, command arguments or result logs. Redirects,
explicit proxies, credentials in authority fields and IP-literal hosts are
rejected, including short, octal and hexadecimal IPv4 forms. The object must
support a direct, bounded response; no provider or
account has yet been selected or validated by these CPU tests.

One owned child has a 300-second transfer deadline. Overflow, truncation, changed
bytes or HTTP failure stop the attempt; no retry or alternate endpoint is used.
Partial first files/logs are retained and cannot be overwritten. Error text
cannot echo a signed URL. Plaintext files use private permissions.

Before extraction, v2 checks the complete ZIP identity and roster. Extraction
rejects unsafe paths, links, special files, encryption, unsupported compression,
duplicate/missing/extra members, excessive expansion, wrong sizes and hashes.
The complete local chain and wheel inventory must match the frozen admission.

After the download, the URL is unset. The input private key is deleted after
decryption, including on shell exit. Only then does the controller invoke the
original source's input verifier in an isolated Python process. Its stdout,
stderr and process outcome are retained, with a 300-second deadline and no retry.
A successful process exit alone is insufficient: the exact receipt must match.
The original numerical execution step is not replaced by the controller.

The result private key stays local; only its separately pinned public recipient
is passed to the workflow. Cleanup of temporary repository secrets and remote
objects belongs to the later approved live-operation plan, not this module.

## GitHub unpublished-asset input: CPU implementation only

`tools/byte_release_github_asset.py` adds a GET-only input reader for the existing
SpiralTorch repository. It does not create a draft, upload an object, retrieve a
credential, publish a release or implement a write API. The earlier reviewed
direct-object snapshot remains separate and unchanged.

This route uses `spiraltorch.byte_release.transport_github_input.v1`, not a
direct-object URL disguised as the old schema. The other descriptor fields and
128/129 MiB limits remain the same. Its `object` record pins exactly:

- `repository`: the existing `RyoSpiralArchitect/SpiralTorch` namespace.
- `repository_id`, `release_id`, `asset_id`: positive integer identities, never
  booleans or integral floats.
- `asset_name`: an opaque encrypted `.age` filename, not a private model name.
- `release_tag`: the exact tag name carried by the unpublished draft metadata.

The externally approved amendment still binds the complete descriptor's canonical
digest. The controller must include the GitHub reader as a tracked ordinary file
in addition to the three v2 runtime files. Its rendered workflow must match the
GitHub-specific form exactly, including unchanged `contents: read` and
`actions: read` permissions and the unchanged numerical execution step.

The reader observes repository identity, unpublished draft state, the release's
asset roster and exact asset size/hash before transfer. It uses only API URLs
derived from the fixed namespace and pinned numeric IDs; it does not use a
metadata-supplied browser URL. The GitHub API can return a stream or a redirect.
[Asset API](https://docs.github.com/en/rest/releases/assets#get-a-release-asset)

In the redirect case, accept only one HTTPS hop to the exact
`release-assets.githubusercontent.com` host, using a fresh request without
Authorization or API headers. Reject userinfo, ports, fragments, unsafe paths,
alternate hosts and any second redirect. The domain is documented by GitHub for
release assets; it is not a wildcard for arbitrary GitHub-hosted content.
[Documented release domain](https://docs.github.com/en/actions/reference/runners/self-hosted-runners#accessible-domains-by-function)

Metadata is bounded to 64 KiB per response. Binary data must match the fixed
ciphertext length and hash. JSON, compression, size, identity or status errors
stop immediately. HTTP 401/403/404, throttling, expiry or server errors never
trigger another credential, a public download fallback or a retry.

After transfer, a second release observation must still show the same unpublished
draft and asset. This is an intentional state check, not a transfer retry. There
are five GET requests for a direct stream or six for one CDN hop; all run within
the same owned 300-second fetch deadline with 30-second request timeouts. A late
state failure retains the first ciphertext bytes but issues no verified result.
Two observations cannot establish that publication never occurred between them.

Only the fetch step receives the built-in workflow token as `GH_TOKEN`; it does
not use a personal token or a new token secret. The input private key remains a
separate existing transport secret. Before the frozen input verifier, the shell
unsets both `GH_TOKEN` and `GITHUB_TOKEN`; preparation rejects either if present.
The direct-object URL secret is not used by this route, and supplying it cannot
select a fallback. No token or signed CDN URL enters argv, descriptors, receipts
or error text. Metadata errors are not logged as raw response bodies.

The asset API documents a read permission, but draft visibility also depends on
access to that unpublished object. Neither this documentation nor synthetic
success proves that the actual read-only workflow token can read the eventual
draft. Do not increase permissions or substitute another token to make it work.
That access check needs a separately approved, bounded live preflight before a
candidate attempt; unavailable access means stop and revisit the transport choice.
[Draft access](https://docs.github.com/en/rest/releases/releases#list-releases)

The CPU tests cover direct/redirected streams, actual urllib handler sequencing
with an in-memory HTTPS transport, strict metadata and archive identities,
credential non-forwarding, denial/failure paths, post-transfer state changes,
workflow permissions, module presence, dispatch and return binding. They use
synthetic bytes and fake credentials only. No provider compatibility, candidate
carry-forward, independent review of this addition or release readiness is claimed.

## First attempt and return handling

Dispatch defaults to inspection: it makes GitHub metadata GET requests but no
POST or canonical claim write. It is not an offline test command. An explicitly
executed dispatch reserves the canonical claim before its single POST. An
ambiguous failure retains the claim and forbids resubmission.

Any v1/v2 dispatch, response, start, run, result, import or partial hosted output
blocks a new claim. Remote checks bind workflow ID/path, branch, controller
commit, dispatch event and attempt 1. Existing runs cannot be hidden by renaming
the workflow or using a second branch. Binding observes the first run; it never
dispatches one.

V2 has a separate result importer rather than disguising its context as v1. It
requires the terminal first run, matching A/D/context/admission and the complete
unchanged numerical evidence chain. Returned records are schema-validated and
bound by canonical hashes, not Python equality that aliases booleans and numbers.
A failed job with apparently green numerical
output, wrong context or incomplete return cannot become verified acceptance.
The raw first return is kept. A transport failure does not prove GPU execution
or a numerical defect; derived failure records state `gpu_execution=not_inferred`.

## CPU verification and limits of evidence

From this separate checkout, the offline regression entry points are:

```sh
python3.12 -I -S -B -m unittest discover -s tools -p 'test_*byte_release*.py' -v
python3.12 -I -S -B -m unittest discover -s tools -p 'test_owned_process.py' -v
```

New tests use temporary fabricated ledgers, synthetic wheel/archive bytes, local
Git fixtures and fake object/API responses. They block network/native imports
and never execute a real dispatch, encrypted transfer, GPU round or production
ledger update. A real 97 MiB synthetic ZIP exercises the old rejection/new
acceptance boundary; oversized and adversarial archives exercise the remaining
guards. Existing qualification tests remain unchanged and run alongside v2.

The tests cover input reconstruction, strict identity/size/type validation,
complete local proof, source/controller separation, exact workflow equivalence,
first-attempt ownership, credential isolation and non-green recovery. Their
success does not establish provider compatibility, encryption interoperability,
hosted execution, independent review, candidate carry-forward or release readiness.

Before candidate application, independently review the implementation and full
rendered workflow, approve the external amendment and local-result carry-forward,
select/preflight the private endpoint under a bounded live plan, and approve the
single hosted execution. Do not rebuild the wheel, alter thresholds, reset a
claim, invent candidate 2 or silently relax a frozen condition to obtain green.
