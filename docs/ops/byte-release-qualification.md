# Bounded byte-learning release qualification

Maintainer-approved direction, 2026-10-25. This changes the publication decision
rule, not the interpretation of historical evidence. Rust owns learning and
restoration; Python only selects inputs, orchestrates calls and compares saved
outputs. No numerical tolerance, kernel or diagnostic API is changed here.

The maintainer approved a hosted-only budget amendment on 2026-10-26 after
[main CI 40203512490](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/40203512490)
passed all ten jobs: its six required `StudyGpu` tests took approximately 59
minutes, making three rounds inconsistent with the original 90-minute estimate.
Hosted Metal now has four hours; local Metal remains at 90 minutes. This changes
neither the test matrix nor the two-candidate limit, and does not reset the
original decision or reinterpret old results. No candidate had been admitted
before this amendment. Timings from that engineering wheel are planning
evidence, not final-wheel qualification or a runtime guarantee.

## Two separate completion decisions

Historical investigation is resolved only by an explanatory, same-condition
failure-before/fix-after regression. Otherwise retain it as unresolved in
[issue #2289](https://github.com/RyoSpiralArchitect/SpiralTorch/issues/2289).
Do not close that issue as fixed merely because a candidate qualifies.

[PR #2294](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2294)'s
[first CI attempt](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/40189776388)
passed all ten jobs at `9300b6a6aa0b70cf8cc578d3bee37f7d5b96b082`, including
ordinary real-GPU learning/resume. It was merged as
`448b056d3312400dc753e99030fa096b39d45045`. This is the investigation cutoff,
not qualification of a future binary. Without new failing evidence, stop
historical binary searches, diagnostic API expansion and repeated reproductions.

The next release may qualify independently under the rules below. This is a
finite acceptance test for named artifacts/environments, not proof that a rare
defect cannot exist, cross-GPU bitwise determinism, or LLM quality superiority.

## Freeze the candidate before execution

Close only this release's prerequisites, including review of #2291's TLS and
optional Go-bridge repairs against current main. Its old failed GPU run stays
failed; do not merge its red revision or transfer its old audit result to a new
revision. Review the resulting source and require its ordinary CI and unchanged
security/license gates. Do not expand this release into the ecosystem roadmap.

Choose a new unused package version. Record the final clean source commit,
Cargo.lock, toolchain/features, exact three platform wheel digests, installed
native digests, qualification tooling/test hashes and requests before testing.
The release must publish those same wheel bytes, not a rebuild. Existing
0.4.32 development wheels are engineering evidence only and cannot be uploaded.
If current workflow sequencing cannot retain/test those final bytes before
publication, fix that sequencing first; do not waive artifact identity.

The two named GPU environments are local macOS Metal and hosted macOS Metal.
Both install the same admitted macOS wheel, with package/native identity checked
in each fresh process, including inference children. Require a non-CPU adapter;
no fallback. Record the actual architecture, adapter, OS/image and Python.
No GPU execution claim is made for untested platforms or an unexecuted universal2
slice. Torch, NumPy, Transformers, PEFT, Accelerate and safetensors are absent.

## Fixed acceptance matrix

Each environment runs three predeclared fresh-process rounds, sequentially.
Each round includes, without numerical diagnostic capture:

- All six existing `StudyGpu` tests in `test_nn_byte_study.py`, preserving their
  order and original `run_wgpu`/`advance_wgpu` calls. This includes the 25-case
  clipped contextual-AdamW comparison, unclipped v6/v8 SGD preparation paths,
  v4-v9 preparation chain, detached calls, freeze clocks and four audited fresh
  inference handoffs. Do not replace these with a selected-case-only comparison.
- The three existing `ByteGpu` clipping tests: query-boundary exact resume,
  active-norm SGD and active-norm AdamW moments.
- `LongResumeGpu.test_128_updates_and_two_saved_cursors`: the fixed three-arm
  plain/trainable-Poincare/frozen-Poincare AdamW study, 128 updates, split after
  updates 1 and 64. This complements, not replaces, the short contextual-query
  tests. It checks ordinary `run_wgpu`, complete `advance_wgpu`, resumed reports
  and checkpoints, finite histories, active learning and frozen optimizer clocks.

The long request is 143,920 bytes with SHA-256
`6c8feea42a494c855fd659a86b2f3b132b3d877277f3912b66af1be215f440f2`.
It uses batch 1, four bytes per window, width 4, hidden width 8, two heads,
one non-Topos residual block, two geometry columns and curvature -0.75.
This deliberately small mechanics fixture is not a long-corpus quality study.

Require exact report/checkpoint equality within each environment. Do not add
tolerances, discard fields, average runs or use captures to turn a failure green.
Offline negative controls must catch changed weights/moments, dropped histories,
empty or reordered cases and changed serialization. Existing CPU-only validation
must remain available; optional-feature skips must be explicitly expected, never
accepted as execution of a GPU test.

Every round retains the exact inputs and available outputs before comparison,
including failures. No normal-path diagnostic observer is enabled. Metadata
collection is after numerical execution, not a new GPU prelude. All descendants
are owned by the enclosing supervisor; a child success without a complete audit
is incomplete evidence.

## Hard stop rules

- Maximum two candidates for this release decision: the initial candidate and
  one reviewed implementation repair. An unchanged retry is not a repair.
- Per candidate, local Metal has one 90-minute wall-clock budget and hosted
  Metal has one four-hour budget. Each covers all three rounds, children and
  postludes, not a new budget per test or per round.
  Thus additional GPU qualification is at most eleven environment-hours across
  both candidates. Ordinary source CI and wheel builds are recorded separately;
  they do not replace these rounds or reset their budgets.
- Stop at the first unexpected numerical failure, process error, missing
  evidence, cancellation or timeout. Preserve it. Do not start remaining rounds
  or the other environment after a failure is known. Infrastructure interruption
  is incomplete, not a pass; it does not authorize an automatic restart.
- Changing the source, native bytes or numerical build settings after an attempt
  invalidates qualification of affected artifacts and consumes the repair
  candidate slot. Renaming a branch/output directory does not reset the limit.
- At the limit, stop publication and make an explicit implementation/scope
  decision. Do not quietly disable a failing route, relax equality, add retries
  or grow the test matrix. Do not expand historical diagnosis by default.

Preparation and CPU-only tests do not consume a GPU round. Freeze the complete
harness before admitting candidate 1. A bare unittest invocation without the
artifact checks, shared deadline, process ownership and audit is engineering
evidence only; it cannot count as a candidate round.

## Publication done

All named GPU rounds, the actual Linux/macOS/Windows wheel gates, native
dependency-free smoke, existing security/license checks and publication dry-run
must pass for the frozen candidate. Review the finite evidence and unresolved
limitations before release. Upload only the tested wheels under the new version;
verify PyPI digests, index visibility and clean installed-package origin/version.
Keep the historical issue open and disclose the qualified scope in release notes.

Then return the main workstream to language learning/inference development.
Reopen investigation for a new retained mismatch on the supported path, not
merely because a new model, geometry idea or optimization is available.

## Implementation status

The long-study recipe and portable checks are implemented in
`tools/byte_release_long_resume.py` and
`bindings/st-py/tests/test_nn_byte_release_qualification.py`.
Their real-GPU test requires both existing runtime opt-in and
`SPIRALTORCH_RUN_BYTE_RELEASE_QUALIFICATION=1`; it is not added to ordinary CI.
The artifact-bound worker, fixed matrix, child verification, local attempt ledger
and encrypted hosted handoff are implemented. Their actual end-to-end hosted
execution, final build receipts and candidate admission remain unverified. This
document does not claim that qualification has run.

The read-only `tools/byte_release_artifacts.py` component inventories complete
wheel bytes and checks the installed native extension, Python facade, shipped
`spiral`/`examples` files and original metadata. Install the qualification
environment with `pip --no-compile`: cached bytecode, extra package files,
symbolic links and shadowed module origins are rejected. Installer-rewritten
`RECORD` is excluded from original-metadata equality; it is not used to establish
which bytes came from the wheel. This check does not select a GPU runtime.

`tools/byte_release_schedule.py` supplies the sequential three-process ownership
and decreasing-budget primitive. A zero exit requires independent evidence
verification; failure, cancellation, timeout and incomplete evidence stop the
sequence. It has no standalone execution CLI or publication authority. Candidate
ledger admission, the frozen test worker and its child/evidence verification
are integrated by `tools/run_byte_release_qualification.py`. It runs the frozen
matrix in three fresh interpreters, checks complete parent/child wheel identity,
rejects missing evidence despite zero exits, and verifies saved exact bytes.
Existing numerical calls/assertions stay in their original tests; opt-in
`byte_release_evidence` hooks retain already available host bytes. They add no
GPU snapshot or diagnostic dispatch and do not evaluate factories in normal CI.
Each retention label requires its exact payload filename set, not just an event
count. Removing an expected/observed file from both disk and index cannot pass.
Raw inputs/results and failure records are private by default. Offline tests
exercise synthetic success, tampering, missing audits, failure and timeout;
they are not evidence of GPU execution.

## Preparing and running the frozen acceptance

After reviewing/merging the release prerequisites, choose an unused version,
merge its reviewed release PR and tag that exact source once. The tag-push
`release_wheels.yml` run builds/tests all three final wheels, signs and retains
`signed-release-payload-<tag>`, but does not attach or publish a GitHub Release
or upload to PyPI. Record that original run and artifact IDs/digests; download
and verify its exact-tag provenance before admitting those same wheel bytes.
Do not use `finalize_release=true` or a second build for this candidate.
Keep the complete build logs and a JSON receipt for each platform containing
`source_commit`, `wheel_sha256`, `toolchain` (the actual compiler identity), and
`features` (the resolved release feature list). The freeze tool hashes the whole
receipt; it cannot establish the truth of operator-authored build records, review
approval, unused PyPI versions or successful platform CI. Verify those separately.

After the complete candidate gates and evidence review pass, use
`recover_github_release.yml` with that original tag-push run ID to finalize the
unchanged signed payload. Then use the separate publish-from-release dry-run
and upload path. The legacy combined build/publication option is not an
alternative to qualification; it would build different, unqualified bytes.

Use one persistent private ledger outside the source tree for this decision.
Initialize it once, then never choose a new ledger to reset the attempt budget:

```sh
python3 -I -B tools/prepare_byte_release_qualification.py init-ledger "$LEDGER"
python3 -I -B tools/prepare_byte_release_qualification.py freeze \
  --linux-wheel "$LINUX_WHEEL" --linux-build-receipt "$LINUX_RECEIPT" \
  --macos-wheel "$MACOS_WHEEL" --macos-build-receipt "$MACOS_RECEIPT" \
  --windows-wheel "$WINDOWS_WHEEL" --windows-build-receipt "$WINDOWS_RECEIPT" \
  --output "$ADMISSION"
```

Retain the printed admission digest independently as `ADMISSION_SHA256`. Do not
hand-edit the admission. Admission schema v2 freezes both environment budgets;
the ambiguous old scalar-budget format is rejected, not silently upgraded.
The canonical decision and candidate slots remain unchanged. Candidate 2 additionally requires `--candidate 2`, the
previous `--prior-admission`, and the reviewed implementation `--repair-review`
PR URL. Admission requires a retained terminal failure of candidate 1; it rejects
unchanged retries, unfinished environments and third candidates. Review URLs
record provenance, not proof that review was approved.

Install the final macOS wheel into a fresh environment with `--no-compile`, no
foreign ML packages, and a clean checkout of the frozen commit. `PYTHON` below
is that environment's interpreter. Default invocation only checks CPU-side
identity; it neither starts GPU tests nor consumes the environment claim:

```sh
"$PYTHON" -I -B tools/run_byte_release_qualification.py \
  --admission "$ADMISSION" --admission-sha256 "$ADMISSION_SHA256" --wheel "$MACOS_WHEEL"
"$PYTHON" -I -B tools/run_byte_release_qualification.py \
  --admission "$ADMISSION" --admission-sha256 "$ADMISSION_SHA256" --wheel "$MACOS_WHEEL" \
  --ledger "$LEDGER" --environment local-metal --execute
```

One exclusive environment claim is written before launching any round. A failed
or interrupted claim is not reusable, even if its output folder was renamed or
removed. Hosted execution is sequential, using the exact same admission and
wheel, and is permitted only after all three local rounds verify. The ledger is
an operator-owned filesystem record, not a distributed lock or authenticated
remote attestation. Before hosted execution, the coordinator must transfer and
verify the canonical admission/local-result evidence, ensure no other owner is
running, and preserve/return the first hosted claim and outcome. Do not initialize
an independent hosted ledger. Use the fixed handoff below, not an untracked
manual invocation on a new runner.

All receipts keep `release_ready=false`. Even six verified GPU rounds do not
perform publication, close #2289, or substitute for the three wheel gates,
security/license checks, review and same-byte upload verification above.

## Hosted handoff and recovery

`byte_release_handoff.py` checks the saved chain from local acceptance receipt
through all three round seals to their original bytes. It does not run learning
again or reinterpret a prior failure. Its bounded ZIP rejects missing/extra
members, unsafe paths, links, changed bytes and oversized payloads. Input ZIPs
must fit 96 MiB before age encryption; larger inputs stop preparation rather
than silently dropping evidence. Keep all plaintext and identity files private.

After three local rounds have qualified, pack the same ledger and macOS wheel:

```sh
python3 -I -S -B tools/byte_release_handoff.py pack-input \
  --input "$LEDGER" --wheel "$MACOS_WHEEL" --output "$INPUT_ZIP" > "$PACKET_RECEIPT"
```

Use the pinned age 1.3.2 tool to generate separate input and result identities in
a private directory. Check an encrypt/decrypt round trip and retain the result
identity locally; never give it to GitHub. Encrypt `INPUT_ZIP` to the input
identity's public recipient. Store only that input identity in the temporary
repository secret `ST_BYTE_RELEASE_CANDIDATE_1_INPUT_IDENTITY` (or `_2_` for the
reviewed repair). Do not reuse PyPI or provider credentials for transport.

Create the dedicated controller branch
`spiralreality/byte-release-candidate-1-20261025` (or candidate 2) directly from
the frozen source commit. Add exactly one commit containing exactly one file:
`.github/byte-release-input.zip.age`. No code/workflow edits belong on this
branch, and it must not merge into main. Push it only after reviewing the frozen
source and ciphertext digest. The controller is the frozen source plus its
opaque packet; the workflow checks out the original source separately and never
rebuilds its wheel.

The dispatcher performs read-only checks unless explicitly executed:

```sh
python3 -I -S -B tools/byte_release_hosted.py dispatch \
  --ledger "$LEDGER" --controller "$CONTROLLER_CHECKOUT" \
  --packet-receipt "$PACKET_RECEIPT" --recipient "$RESULT_RECIPIENT"
```

Only after those checks pass, the reviewed coordinator adds `--execute` once.
An exclusive canonical dispatch record is saved before the POST. A timeout or
ambiguous response leaves that record consumed: observe, never resubmit. Bind
the first run with the read-only `bind-run --candidate-directory` operation.
It may be called again to observe delayed API visibility; it never submits work.

The manual `byte_release_qualification.yml` workflow rejects run attempts other
than 1, any prior/duplicate dispatch on that candidate branch, controller/source
drift and wrong input digests before accessing the input secret. A shared
non-cancelling concurrency group prevents overlapping hosted candidates. This
is a single-coordinator policy, not protection against a repository administrator
bypassing it. The native supervisor owns one four-hour hosted budget across all
three rounds; the job's 270-minute outer limit preserves the existing 30-minute
setup/sealing allowance, not extra GPU execution time. An exhausted job
or unavailable result artifact is incomplete evidence, never a retry trigger.

Only `result.zip.age` and its ciphertext checksum are uploaded. Download the
artifact from the bound run using the authenticated GitHub account, verify its
GitHub artifact identity/digest and ciphertext checksum, then decrypt locally
with the retained result identity. Do not accept an archive sent through an
unverified channel. Pin the resulting plaintext ZIP digest before import:

```sh
python3 -I -S -B tools/byte_release_hosted.py recover-result \
  --candidate-directory "$CANDIDATE_DIRECTORY" --archive "$RETURN_ZIP" \
  --sha256 "$RETURN_PLAINTEXT_SHA256"
```

Recovery checks the bound first run is terminal, retains the raw return intact,
and rechecks the acceptance chain before writing a successful canonical receipt.
A failed GitHub job cannot promote otherwise-green numerical output. Malformed
or incomplete returns produce an explicitly coordinator-derived failure record;
they do not claim whether GPU execution occurred. Existing outcomes are never
overwritten. Retire the temporary input secret after the original run is
terminal; preserve encrypted artifacts and the local result identity until
recovery is complete. No command here publishes a package or closes #2289.
