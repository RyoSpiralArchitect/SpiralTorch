# Durable Byte-LM Segments

**Source-only addition after 0.4.31; not a claim of a new PyPI release.**

`spiraltorch.run_byte_corpus_segments` is a synchronous, importable execution
primitive for the [Rust-owned paired study](byte-corpus-studies.md). It retains
ordinary weights **and** geometric coordinates, mixture/query routers, frozen
case policy, data cursor, scalar histories and any AdamW state carried by the
study checkpoint. Python does not rebuild those semantics or alter their rates.

This closes the gap between a validated in-memory checkpoint and an explicitly
recoverable local run. It is not a new scheduler, HF executor, registry, automatic
retry service or model promotion policy. Existing HF lineage/promotion/executor
APIs remain separate: a native byte-LM checkpoint is not an HF adapter.

## Run And Recover

Use a pinned request from the [corpus protocol](../byte_corpus_learning.md).
This example assumes it declares 12 accepted updates:

```python
import spiraltorch as st

st.run_byte_corpus_segments("request.json", "run-a", [0, 5, 12])
```

The output directory must not already exist; its parent must exist. Stops are
1..32 explicitly increasing absolute cursors ending at the study's total. The
zero cursor is optional and saves the initial state. Request/checkpoint and stop
validation happen before the first device request. GPU execution requires a
WGPU source wheel; CPU-only builds do not silently execute a different trainer.

If the process stops after the step-5 segment has been sealed:

```python
saved = st.load_byte_corpus_segment("run-a", 5)  # CPU-only; Rust validates all cases.
print(saved.receipt_sha256)

st.run_byte_corpus_segments(
    "request.json", "run-b", [12],
    resume_from="run-a", resume_after=5,
    expected_resume_sha256=saved.receipt_sha256,
)
```

Recovery always uses a new directory, the **exact original request bytes**, and
an explicit saved cursor. It never chooses the newest file, steals an active
run, erases a failure, retunes rates or resumes an unsealed checkpoint. The
parent claim/receipt/checkpoint hashes are recorded in the child claim. All new
cursors must advance beyond the selected checkpoint. Loading a final checkpoint
for inference is valid; running further than the declared study is not.

`expected_resume_sha256` optionally pins a previously retained receipt hash.
Reading that hash from the same store immediately before use prevents later
substitution, but is **not** independent authentication of its author or history.
The source can remain live: recovery explicitly forks the immutable selected
segment rather than claiming to take over its process. Callers own scheduling
and GPU resource coordination.

The installed CLI exposes the same primitive, without a repository checkout:

```bash
spiral-byte-corpus-run request.json run-a --stop 0 --stop 5 --stop 12
spiral-byte-corpus-run request.json run-b --stop 12 \
  --resume-from run-a --resume-after 5
```

The historical `tools/run_byte_corpus_segments.py` is now only a compatibility
wrapper around the installed package. It requires this source candidate.

## Saved Models Remain Useful

`ByteCorpusSavedSegment` contains the Rust study and checkpoint plus the opaque
report string and content digests. No device is opened when loading. The saved
owner feeds the same model-selection and serving lifecycle as any other study
checkpoint:

```python
artifact = saved.study.case_artifact(
    saved.checkpoint, "my-explicit-case-name", policy="fixed_window_frozen.v1",
)
# Or select among actual saved evaluation-boundary checkpoints using
# study.select_checkpoint(...), then apply a separate held-out serving gate.
```

See [model artifacts](byte-model-artifacts.md),
[saved-checkpoint selection](../byte_corpus_checkpoint_selection.md) and
[serving gates](byte-serving-gates.md). A successful resume does not establish
geometric quality benefit, held-outness, or deployment readiness.

## Storage Contract

- The store currently supports trusted local POSIX filesystems with hard links
  and directory `fsync`. Windows, network/object stores and hostile concurrent
  writers are outside this contract. Parent directories must be trusted.
- The exclusive directory claim is mode `0700`; files are created mode `0600`.
  Report/checkpoint bytes are flushed before their receipt is published via an
  atomic no-replace hard link. The directory is flushed after publication.
- `segment-N.json` is the seal for `report-N.json` and `checkpoint-N.json`. Only
  that segment's complete claim, request, receipt and payload are needed to
  load it. Other segments, terminal summaries and older ancestors are not
  required. Ancestor digests are references, not an authenticated chain.
- `completed.json` means all declared segments reached the original total;
  `failed.json` is best-effort diagnostic context. Abrupt death may produce
  neither. A terminal-summary write failure can leave a valid final segment;
  inspect the explicit seal instead of automatically rerunning completed work.
- Interrupted writes can leave `.pending-*` or unsealed payload files. They are
  never adopted. Keep the source unchanged; recover into a new run. A malformed
  or corrupt selected seal fails closed, without silently falling back.
- Loading uses bounded regular-file reads, rejects symlinks at the store/file
  entry, verifies hashes and calls Rust's full checkpoint validator. JSON
  payloads are bounded to 64 MiB each; metadata to 64 KiB. Exact opaque model
  bytes are never passed through a Python float/JSON round trip for storage.
- Legacy `segment_launch.v1` directories are not automatically upgraded. Their
  exact checkpoints can still be independently validated through the lower-level
  Rust API; this store does not manufacture new historical execution receipts.

The filesystem must honor these operations. File/directory `fsync` calls do not
by themselves prove power-loss resilience on every disk or platform. Requests,
weights and failure messages may be private: publish selected results/hashes,
not an entire run directory.

## Exact Resume Qualification

On 2026-10-19 (JST), [the macOS GPU job for source `09e98b0b3`](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/39141071472/job/117715511190)
failed the exact contextual-mixture clipped-AdamW resume comparison. It was a
numerical mismatch, not a timeout. The original failure did not retain full
comparison payloads, so it does not isolate restoration from two independently
executed training prefixes. No numerical fix or tolerance relaxation is claimed.

The same source's default-feature local wheel, installed without Torch,
Transformers, PEFT, NumPy, Accelerate or safetensors, passed the 25-case comparison
on Apple M4/Metal. One initial reproduction and eight predeclared fresh-process
reproductions all had byte-identical full/resumed reports and checkpoints. The
eight repetitions also shared these SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Request | `39bb6c540db50f8d9545076a91408de4b1c9b674e1e2947a427dd2ca4ee0b1de` |
| Full/resumed report | `6dc103dbbb9191ff8b693e33a773a61ca13444b9e1cb9a0408ed84f7fd7548e6` |
| Full/resumed checkpoint | `f1a5e92d562f2f95ac9ecf0ed59f472f2926c820a26d8e96432e4ee069c37670` |
| Local development wheel (not publishable) | `e5c6f75b5a028ed6dcb768cfa84eee55e9ebed25f6969d05a2b8929db9cede9d` |

These local passes do **not** resolve the CI failure or qualify a release.
After installing a source wheel, the isolated test can be reproduced with:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
SPIRALTORCH_RESUME_STUDY_DIAGNOSTICS=/trusted/new-synthetic-failure-directory \
python -I -B bindings/st-py/tests/test_nn_byte_study.py \
  StudyGpu.test_clipped_contextual_mixtures_preserve_all_frozen_clocks -v
```

The diagnostic directory is created only on mismatch and must not already
exist. It retains exact request bytes, full/first/resumed reports and checkpoints,
hashes, and bounded differences. CI uploads these **synthetic test** payloads on
failure; private training runs are not automatically published. Strict report
and checkpoint equality remains required. The next investigation must use the
first differing histories, parameter values and optimizer states, rather than
rerunning until green or treating local success as a correctness repair.

When retention is enabled, a mismatch also records a hash-bound `runtime.json`
with the active process-default GPU's name/backend/type, OS/Python versions,
and the installed native library's basename and SHA-256. Collection runs only
after a mismatch, without tensor dispatch/readback; passing comparisons do not
change their execution cadence. No environment, host name or absolute library
path is collected. Context is bounded to 16 KiB; a metadata-collection failure
retains its exception type, not its message, alongside the original comparison
payloads. This improves failure localization, not numerical correctness.

On 2026-10-19, [PR #2288's GPU job](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/39187825231/job/117853890234)
passed that clipped-AdamW comparison but failed the preparation-chain
uninterrupted/resumed report comparison for v6 headwise and v8 query-mixture
SGD. Logged values differed substantially, not merely by serialization or an
acceptable rounding tolerance. This separate comparison had not retained its
full payloads; its failure log is preserved and no root cause is asserted.
The retained full job log has SHA-256
`e83c019f7f4e4a9057848b463b545f96a9cc773e483a9d68cba6ccee9928c1bf`.

`SPIRALTORCH_PREPARED_STUDY_DIAGNOSTICS` now retains these mismatches separately
by request version, including exact request/full/first/resumed reports and the
available first/resumed checkpoints. The original `run_wgpu()` path is preserved:
no full-run checkpoint is fabricated or obtained by substituting `advance_wgpu`.
Boundary serialization and runtime-identity collection occur only on failure.
This capture change does not repair either numerical discrepancy.

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 SPIRALTORCH_STRICT_GPU=1 \
SPIRALTORCH_PREPARED_STUDY_DIAGNOSTICS=/trusted/new-prepared-failure-directory \
python -I -B bindings/st-py/tests/test_nn_byte_study.py \
  StudyGpu.test_preparation_chain_uses_one_shared_runner -v
```

Independent shader review also found an extreme-logit portability defect:
`value - maximum` could overflow before `exp` for finite opposite-sign values.
[WGSL permits indeterminate runtime results after such overflow](https://www.w3.org/TR/WGSL/#floating-point-evaluation).
The mixture shader now avoids that subtraction when the signs differ and one
magnitude is at least 128: the exponential is already below the smallest f32
subnormal. Same-sign differences and all representable nonzero exponential
masses follow the original path. Tests include amplified router gradients so
a small nonzero mass cannot be silently discarded. This is a separate fix,
not evidence explaining the ordinary-logit v6/v8 study failures.
