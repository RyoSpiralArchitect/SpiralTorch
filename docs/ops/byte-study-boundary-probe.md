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
- Runs every case in the request's original order, retaining a single
  `checkpoint_snapshot()` after update 1 of the selected case.
- Does not map that snapshot until the entire observed study has returned.
  Its original scheduled drains/evaluations still occur.
- Saves `run-outcome.json` **before** attempting the boundary readback. A
  failed study or diagnostic read does not replace the other result.
- Saves both canonical model checkpoints, compares them exactly, and records
  their hashes, the executable hash, adapter, original inputs and run metadata.
  Output must be a new directory. Mismatch or execution error exits nonzero.

GPU copies and retained staging buffers alter submissions, timing and memory
lifetimes. Even an equal boundary and passing report do **not** prove the
uninstrumented path is fixed. The full original fixture is synthetic; no
language-quality or throughput claim is made.

## Bounded use

Retain the first original failure before building or running the probe. Declare
the selected case, source/patch, toolchain, build flags, execution deadline and
one attempt in the private run receipt. Do not retry unchanged runs until green.
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
