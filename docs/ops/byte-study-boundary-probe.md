# First-update boundary probe

This is a separate, instrumented experiment for
[issue #2289](https://github.com/RyoSpiralArchitect/SpiralTorch/issues/2289),
not release qualification or a replacement for the
[unmodified replay protocol](byte-study-replay.md).
PyPI publication remains held until the cause and repair are validated.

## Question

The complete hosted v6 (SGD) and v9 (AdamW) failures agree on their first
recorded loss and disagree on their second. That second loss is evaluated
**before** backward and the second optimizer update. Matching first losses
does not establish matching first gradients or parameter updates.

The next boundary to inspect is therefore the uninterrupted model immediately
after update 1: its weights, AdamW moments when present, and update clocks.
No backward-stage or driver explanation follows from a final checkpoint alone.
The immutable original failure reports and checkpoints remain the reference.

## Instrument

`ByteCorpusStudy::run()` and `advance()` do not install an observer.
`run_with_update_observer()` explicitly invokes a callback after every submitted
update and before the next scheduled drain or forward. It passes a case name,
one-based attempted update, and an immutable model. Submission is not acceptance;
the existing receipt checks are unchanged. A callback error aborts that run.

The native `resident_byte_boundary_probe` example:

- Validates the exact request and request-bound, all-case update-1 checkpoint
  before requesting a GPU. Inputs must be regular files of at most 64 MiB.
- Runs every case in the request's original order. The default `copied` mode
  submits a single `checkpoint_snapshot()` after update 1 of the selected case.
- The optional `retained` mode calls `retain_checkpoint()` instead. It only
  clones immutable parameter and optimizer handles at update 1. Diagnostic
  GPU copies are not submitted until the observed study has returned.
- Neither mode maps the captured boundary until the observed study has returned.
  Its original scheduled drains/evaluations still occur. Mode and limitations
  are recorded in both the invocation and summary.
- Saves `run-outcome.json` **before** attempting the boundary readback. A
  failed study or diagnostic read does not replace the other result.
- Saves both canonical model checkpoints, compares them exactly, and records
  their hashes, the executable hash, adapter, original inputs and run metadata.
  Output must be a new directory. Mismatch or execution error exits nonzero.

The default mode adds GPU copies and retains staging buffers between updates.
Retained-handle mode avoids those additional submissions, but still changes host
work and GPU resource lifetimes; it is not an unperturbed control. Neither mode
changes the ordinary `run()` or checkpoint path. Later parameter/AdamW updates
allocate new buffers, so delayed copies still read the retained revision, even
after the model is dropped. Neither retention nor copying proves acceptance.
Even an equal boundary and passing report do **not** prove the
uninstrumented path is fixed. The full original fixture is synthetic; no
language-quality or throughput claim is made.

## Bounded use

Retain the first original failure before building or running the probe. Declare
the selected case, source/patch, toolchain, build flags, execution deadline and
one attempt in the private run receipt. Do not retry unchanged runs until green.
Declare the capture mode too; do not switch modes in response to a failure and
discard the previous outcome. The hosted supervisor retains the `copied` default.
Record the build/source identity separately; an executable hash alone does not
attest the Git checkout. Use a process-owning supervisor for a finite deadline;
the example itself does not impose a whole-process timeout.

```sh
cargo build --locked -p st-nn --no-default-features --features wgpu \
  --example resident_byte_boundary_probe

target/debug/examples/resident_byte_boundary_probe \
  /absolute/private/original/request.json \
  /absolute/private/original/first-checkpoint.json \
  mixture-11-3-train-query-train \
  /absolute/private/new-boundary-run
```

To predeclare the retained-handle variant, append `retained` after the new output
directory. Omitting the argument or passing `copied` preserves the existing
capture path. The retained checkpoint's explicit API is
`model.retain_checkpoint()?.snapshot()?.read()?` on native, or `read_async().await`
on WASM; keep the handles until the chosen observation window has ended before
calling `snapshot()`.

The case name must exist in that request. For the preserved v6 failure, use its
`headwise-11-1-0-frozen` case instead. These are separate declared conditions,
not a search for a passing case. Preserve the process exit status and logs,
including a timeout or partially written output directory.

Compare `run-outcome.json`'s `report.cases` with both the original failing full
report and original resumed report. Compare corresponding cases, not adapter
strings across hosts. Do not remove fields or relax numeric equality:

- If the observed run still fails and its first boundary differs, disagreement
  already exists in the captured first-update model state.
- If it still fails but that boundary agrees, the captured parameters, moments
  and clocks do not explain the subsequent discrepancy by themselves. Transient
  forward/backward state is not included in that checkpoint.
- If it does not fail, record `no mismatch observed under instrumentation`,
  not cause found or repair validated.

Host/native-binary changes are additional confounders. Keep original payloads
private; publish only summaries, hashes and reproduction instructions. This
probe creates no release, tag, wheel upload or publication authorization.

## Same-process Python observation

The separate native probe cannot inherit the original Python process history.
For a separately declared instrumented comparison, Python can now call the
same Rust runner without starting another process:

```python
observed = study.run_wgpu_with_retained_boundary(first.checkpoint, case_name)
# Persist this before readback; a failed diagnostic must not erase the run.
with (new_output_dir / "run-outcome.json").open("x") as stream:
    stream.write(observed.run_outcome_json)
boundary = observed.read_boundary()  # one-shot, including on read failure
with (new_output_dir / "observed-boundary.json").open("x") as stream:
    stream.write(boundary.to_json())
```

`first` is an existing all-case update-1 segment for this exact request.
Preflight checks its complete ownership/layout/history contract, the selected
case name, and at least two requested updates before acquiring a Python GPU
runtime. It does **not** initialize the observed run from that reference.
The observed run executes all cases in their original order from their original
initialization. Only the chosen case's first-update handles are retained.
Their copies and mapping start only at the explicit `read_boundary()` call,
after the whole run has returned, potentially after its model owners are dropped.

Rust owns this contract through `validate_retained_boundary_reference()` and
`run_with_retained_boundary()`. The returned Rust `ByteCorpusBoundaryObservation`
separates `run_outcome` from optional `boundary` handles, usable by native and
WASM callers too. WASM callers use the existing async readback after `snapshot()`;
no synchronous browser wait or new browser binding is introduced here.
Python's `ByteCorpusBoundaryObservation` has no public constructor, keeps its
`run_outcome_json` immutable, and consumes `read_boundary()` once, even on failure.

The outcome schema `spiraltorch.byte_retained_boundary.v1` distinguishes the
completed study report from an execution error and records whether a boundary
was captured. Save either outcome. A successful diagnostic read does not turn an
errored study into an accepted run; a read error leaves the original outcome
available. Callers must preserve the read error separately and compare the
completed report against the reference run, not infer a repair from read success.
In particular, rejected SGD preserves the old valid parameter bits while
advancing the attempted revision. That diagnostic checkpoint can be readable
even though the run's update receipt was rejected; it is not acceptance evidence.

This is **still instrumented**: buffer lifetimes and host work change. A new
binary also differs from any previously failing extension. The ordinary
`run_wgpu()`/`advance_wgpu()`, original prepared-study tests and hosted workflow
remain unchanged. A same-process diagnostic must be selected in advance, never
substituted for a failed original run or used as retry-until-green. It is not a
release gate and does not lift the publication hold.

## Hosted post-failure capture

The ordinary macOS GPU CI retains its original Python test commands and failure
status. Only after that step fails **and** the prepared-failure artifact upload
succeeds does `tools/run_byte_study_boundary.py` select a diagnostic.
It checks the original manifest's file hashes, request identity, update-1 cursor,
report schemas and case order. Eligibility is limited to the existing v4-v9,
two-update fixtures with at most 32 cases.

Selection is deterministic: the lowest retained version, then the first
canonical-JSON case difference in its original request order. This distinguishes
booleans, integer/float spellings and signed zero; it does not use Python's loose
numeric equality. Missing prepared evidence or no canonical case difference
causes no build or GPU attempt.
Malformed or mismatched evidence fails closed. It never hunts another case or
version for a passing result.

There is at most one build and one probe process, each with a 900-second owned
process-group deadline; the enclosing step allows 35 minutes including evidence
handling. The executable must come from Cargo's JSON artifact record for this
example, within the declared target directory, not a possibly stale hard-coded
binary. Terminal records precede post-processing. Completion identity, original
input bytes and boundary hashes are checked; a nonzero probe exit stays nonzero.
The original failed CI step remains failed even if this diagnostic succeeds.

This is the same hosted job but a **separate, instrumented process and binary**,
not capture of the already-finished original process. Its new outcome cannot
retroactively explain that process. `release_ready` is always false.
The separate one-day `synthetic-byte-study-boundary-diagnostic` artifact retains
the toy-fixture inputs, observations, source/compiler identity, hashes and logs.
It contains no executable, wheel, credentials or external corpus/model weights.
Download it into private validation storage before expiry; commit only bounded
summaries, hashes and procedures, as with the original failure artifact.

## Local instrument check, 2026-10-20

One fresh-process, 600-second-bounded attempt per preserved condition ran on
Apple M4 / Metal, macOS 26.4.1. This used the new diagnostic executable, built
without debug info or incremental compilation from base
`933a6ac735d29bd066488150eab281644edbe00f` plus this observer patch.
The local compiler reports Rust 1.97.0, not the original hosted toolchain.
Both observed studies completed and read their retained boundaries; both CLI
processes exited **1** because the exact cross-host checkpoint comparison differed.

| Condition | Observed second loss | Original failed full loss | Original resumed loss |
| --- | ---: | ---: | ---: |
| v9 contextual AdamW | 5.32478666305542 | 5.4575114250183105 | 5.324787139892578 |
| v6 frozen headwise SGD | 5.320909023284912 | 5.373128890991211 | 5.320909023284912 |

The maximum absolute first-boundary JSON numeric differences were
`3.05517155e-7` (v9) and approximately `1.0e-8` (v6).
These are measurements, not acceptance tolerances. The original large excursion
was not observed here; the hosted failure is **not reproduced, explained or
repaired** by these local, instrumented, different-binary runs.

Local validation passed 85 existing study tests with runtime tests enabled,
three observer tests (two GPU execution tests and one compile-contract test),
four CLI evidence tests, and the `wasm32-unknown-unknown` compile check.
An independent read-only review caught a pre-execution completion claim;
invocation now records intent, and only the returned outcome records completion.
The corrected patch had no remaining review findings. Browser execution of the
observer and hosted execution of this diagnostic remain untested.

Original inputs, outputs and logs remain private. Verification SHA-256 values:

- Predeclared local protocol: `0c6883b1ff6bff61356a0289d377470b40257da9eb7bfcc5d6b36c38e934126c`.
- Two-condition result: `a10d442e41fd3dcd31a7d88e98fafcad729133a60e9e6aab9d8b585d410071d8`.
- Native executable: `733ed61b6c936b14252ebefc0f4af6c0ab61605d84557934b0ba1fcb3a36391b`.
