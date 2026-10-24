# Same-wheel historical-prelude comparison

This is a bounded diagnostic, not a fix, benchmark, training workaround or
release qualification. It investigates differences between the original
failure's same-process history and the later cold retained-boundary attempt.
Only the orchestration is offline-tested. Do not infer GPU correctness from
the stub tests, and do not execute without separate human authorization.

## Frozen Inputs

The caller accepts only the original v9 failure packet used by the
[retained-boundary caller](byte-study-retained-runner.md), all 25 cases in
their original order, with selected case 23 at attempted update 1.

- Native source: `e053c314d54be06b346e30e5ea1a065d7f3fdc90`, clean checkout.
- Historical Python test SHA-256:
  `c0859450ba491e7a7fd6df14b4124a90d77c153a7c443fa106523d2f2707a06b`.
- Recovered hosted wheel SHA-256:
  `4930f6a32f1774984a0bf4ab407895c957d3fd29cad4c74fd6b8553b017e0854`.
- Native extension SHA-256:
  `57488cd19ee0c30065b3489b596286d0b8cb28ee77be4ecb7228b10393fcae26`.

The wheel is the retained artifact from hosted run `40076826426`, not the
original failing binary. Do not rebuild or substitute a similarly versioned
wheel. Its internal version is not publication authorization. Keep the original
build log, source pin and recovered artifact hash chain alongside this campaign.
The installed native extension and all wheel-owned Python source files must
match that wheel. The environment must not contain the foreign ML packages
listed by the existing replay guard.

## Two Arms

One campaign launches two separate Python processes, ordinary first and retained
second. Each executes the five historical `StudyGpu` predecessors once, including
the original concurrent-run test, followed by the preparation-chain test once.
Versions 4 through 8 and the v8 fresh-inference helper remain unchanged.
Portable tests that originally followed the failure are not a prelude and are
not run. Prior Rust test executables and other Python test-file processes are
not recreated; their environment/driver effects are not controlled.

The test file is hash-pinned and imported from the native source checkout.
Only its final v9 first/resumed/full assignments are replaced in memory, and
a request-identity check is inserted before v9 training. The test file on disk
is never edited. An AST round-trip test checks that reverting those three
assignments and the inserted check recovers the original method exactly.

The common v9 wrappers preserve exact request, partial/resumed reports and
checkpoint bytes. They retain a decoded same-attempt partial checkpoint and
add host file I/O between the three native calls. This is a common observation
change, not a claim to reproduce the original interpreter instruction stream.

1. The ordinary arm calls the original native `run_wgpu()` for the full run.
   It never requests a retained boundary.
2. The retained arm calls `run_wgpu_with_retained_boundary(...)` once, keeping
   the selected update-1 handles until the complete study returns. Exact full
   report/outcome bytes are saved before the one explicit boundary read.
   Its reference is that arm's partial checkpoint, never a historical or
   cross-arm checkpoint.

All learning, optimizer state and model execution remain Rust-owned. No
Python reconstruction of gradients or optimizer arithmetic is introduced.

## Offline Checks

Default invocation validates hashes and emits a manifest only. It does not
import SpiralTorch, create a device, build a wheel, launch a worker, or dispatch
a workflow. The source checkout and original packet must already exist.

```sh
python -I -S -B tools/run_byte_study_prelude_pair.py \
  --failure-root "$PRIVATE_FAILURE_ROOT" \
  --source-root "$PINNED_NATIVE_SOURCE" --wheel "$RECOVERED_PRIVATE_WHEEL"
python -I -S -B tools/test_byte_study_prelude_pair.py -v
```

For separately approved execution, use the private, foreign-ML-free environment
containing that exact wheel, commit the qualified harness, and add a **new**
`--output "$PRIVATE_CAMPAIGN_OUTPUT" --execute-pair`. Each worker is limited to
5,400 seconds (90 minutes); a smaller `--timeout-seconds` is allowed.
Use a hosted-job cap of 210 minutes to leave time for packaging and cleanup.
Actual duration is unknown. The caller does not install packages, transfer
files, provision secrets, build, push, dispatch CI, or publish anything.

The output root must not exist. Each worker claims its arm exclusively before
native imports. There is no resume, sweep or retry option. If the ordinary
target exposes a qualified exact mismatch, the already-declared retained arm
may still run once. A prelude failure, incomplete audit, timeout, device error,
boundary read error or metadata failure stops the campaign. It does not skip
the failing phase or launch a replacement attempt.

## Evidence And Limits

The historical fresh-inference subprocesses are guarded and recorded, rather
than blocked or silently omitted. A complete passing arm expects four such
children (two each at v8 and v9); an arm stopped at a recorded target mismatch
expects two. Child identities, native hashes, parent receipts and import/spawn
audits are rechecked from their saved files. No unlisted child is admitted.

The supervisor recomputes the target verdict from saved raw reports and checks
checkpoint identity, case order and training history. It verifies the retained
reference against the selected case in that same attempt's first checkpoint.
Checkpoint CE is serialized directly as f32, while report CE is converted
through a JSON Value's f64 representation. Only this cross-format history
check compares exact f32 bits (including signed zero); report values must
themselves represent an exact f32 value. A one-f32-ULP change still fails.
Full-versus-resumed reports and retained-versus-reference checkpoint bytes
are always compared exactly. No tolerance or reconstruction replaces that
primary comparison. A discrepancy
elsewhere among the 25 cases is not attributed to the selected boundary.

Target evidence precedes the worker result, which precedes runtime metadata.
A later observation or metadata error cannot erase earlier raw files. An
earlier qualified arm mismatch remains in `comparison_outcome` even if the
second arm times out. A verified target comparison is also preserved separately
if runtime metadata fails or hangs before arm qualification. A returned native
segment in an exception traceback proves native completion even when the first
test fails its assertion. Otherwise a started test with no completion evidence
is recorded as unknown, not as proof that no native work happened; it does not
trigger a new device solely for metadata.
Incomplete worker/provenance records cannot qualify an
arm merely because its process returned zero.

Shared runner/driver state, arm order, host recording, compilation of distinct
native entry points, extended lifetimes and intermittency remain confounds.
Two passing arms are non-reproduction under this protocol. Failure in just one
arm is not proof that retention or preparation caused it. A failing captured
W1 can localize the next investigation; equal W1 does not prove all transient
state or forward inputs equal.

Keep plaintext model/checkpoint/log evidence private and transport it encrypted
if a hosted runner is separately approved. Only admitted summaries and hashes
belong in public reporting. `release_ready` remains false for every outcome;
PyPI, tags, version changes and unrelated merges remain separate decisions.
