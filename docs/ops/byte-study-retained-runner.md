# Same-process retained-boundary caller

This is an explicit diagnostic caller of the Rust-owned
[retained-boundary API](byte-study-boundary-probe.md#same-process-python-observation).
It is not a training workaround, automatic retry, release gate, or explanation
of the intermittent continuous/resumed discrepancy. The caller's orchestration
has offline stub tests; execution of this dedicated caller on a GPU still needs
separate authorization and qualification.

## Fixed scope

`tools/run_byte_study_retained.py` accepts the retained original v9 failure with
request SHA-256
`6711703a49361d1d2d577912dbe86d47d4d42122273fee797fb27fd6abf8c52c`.
It keeps all 25 cases in request order, selecting case 23,
`mixture-11-3-train-query-train`, for its update-1 boundary. A different request,
case, or earlier failure version is rejected, not silently substituted.

In one fresh Python process, it calls native `advance_wgpu(1)`, restores that
same-attempt checkpoint and calls `advance_wgpu(resume=...)`, then calls
`run_wgpu_with_retained_boundary(...)` for the full two updates. No learning
math or optimizer semantics are reconstructed in Python. All three calls cover
the full request; the selected boundary is the only diagnostic readback.

This is a **new cold-process protocol**, not a replay of the historical test's
preparation chain, earlier tests, Rust prelude, driver, or native binary. It adds
host I/O between the partial/resume/full phases. Handle retention also changes
resource lifetimes. A passing attempt can therefore only be non-reproduction
under this instrumented protocol, never proof of repair or determinism.

## Offline preflight

The failure root must contain the original `version-9/` artifact directory,
including its manifest and all original files. Keep these payloads private.
Default invocation only verifies existing hashes/identity. It does not import
SpiralTorch, request a GPU, or launch a worker:

```sh
python -I -S -B tools/run_byte_study_retained.py \
  --failure-root "$PRIVATE_FAILURE_ROOT"
python -I -S -B tools/test_run_byte_study_retained.py -v
```

The tests use stub native objects, injected failures and mocked processes. They
are protocol checks, not GPU training or numerical-correctness evidence.

## Separately authorized execution

Execution requires `--execute-once`, a fresh output directory, a private wheel,
and a clean native source checkout at
`e053c314d54be06b346e30e5ea1a065d7f3fdc90`. The installed native extension must
match the selected wheel. Retain the pinned build command/log/toolchain and
wheel hash separately: matching a source checkout alone does not prove how a
wheel was built. The worker checks the required client surface before training.

```sh
python -I -B tools/run_byte_study_retained.py \
  --failure-root "$PRIVATE_FAILURE_ROOT" \
  --source-root "$PINNED_NATIVE_SOURCE" --wheel "$PRIVATE_WHEEL" \
  --output "$NEW_PRIVATE_OUTPUT" --timeout-seconds 7200 --execute-once
```

The supervisor launches one isolated worker with a finite process-group
deadline. Its exclusive worker claim prevents a second launch against the same
output, including direct use of the internal worker entry. No worker subprocess
is allowed during the actual diagnostic. Existing replay import guards prohibit
the listed foreign ML packages. No workflow dispatch, build, upload, tag or
publication occurs inside this tool. The older historical replay workflow does
not run this caller and must not be described as qualifying it.

## Evidence and interpretation

The caller saves exact `run_outcome_json` and `report_json` before consuming the
boundary once. A map/read failure is recorded separately and cannot erase a
preceding report mismatch. Reference extraction happens after the full run and
uses the same-attempt partial checkpoint, not the old historical first state.

The report and checkpoint acceptance comparisons use exact original bytes.
Decoded field differences and component comparisons only help localize them;
they never replace the exact comparison or introduce a tolerance. The selected
case's two loss points are recorded separately: an aggregate mismatch in a
different case is not evidence about the selected case's captured boundary.

`worker-result.json` precedes runtime metadata collection. The supervisor keeps
worker outcomes separate from timeout/metadata/exit status, preserves partial
files, and hashes retained evidence. The metadata collector may acquire a GPU
device, so it only runs after the first native phase has returned successfully.
Preflight or first-phase failures record `not_collected` instead; they do not
initialize another runtime merely to describe the failure. A readable checkpoint
does not establish update acceptance; even a rejected SGD update can leave
readable old values.

The worker record's `returncode` describes the comparison before the postlude,
not necessarily the eventual process exit. Missing or failed metadata makes
the process and supervisor fail without rewriting that comparison record.
The supervisor validates the saved adapter/native identity and records
`metadata_status` separately. Equal numerical results with missing provenance
produce `metadata_error`, not a successful diagnostic.

Different update-1 weights localize divergence to or before that boundary, not
to a specific kernel. Equal weights with different next-forward loss narrow
the next inquiry to inputs, rebinding/transient state and forward execution.
Neither observation alone attributes a driver bug. `release_ready` stays false,
including on exit zero. Do not retry to green or lift the publication hold.

For an explicitly scoped comparison that restores the historical same-process
prelude while reusing the exact recovered wheel, see the separate
[prelude-pair caller](byte-study-prelude-pair.md). It does not change this
single-attempt cold-process protocol or inherit its execution authorization.
