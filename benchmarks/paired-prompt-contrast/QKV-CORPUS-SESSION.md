# Data-Bound QKV Corpus Sessions

Status: Rust implementation and Python source-build API verified on CPU. No
pretrained-model study, GPU execution, cache deletion, push or publication.

## Ownership And Schedule

`NativeCausalLmQkvTrainer` snapshots a QKV optimizer and owns a fixed corpus
schedule. It reuses `CausalDocumentWindows::covering_targets`, `Corpus` content
identity/label validation and the previously tested ragged QKV update. Shared
`Corpus::window_rows` borrows individual window slices rather than copying the
whole corpus. No optimizer math or window policy is reconstructed in Python.

Each window is bounded by the declared context, stays within one document,
overlaps its neighbor by one token and retains a short tail. All shifted targets
are covered once per epoch before label masking. All-masked windows are omitted;
empty/single-ID documents contribute no window but remain in the data identity.
All IDs and labels, including ignored ones, are validated and hashed. This is
window-local attention with reset positions, not full-document context or a
training KV cache. No shuffle, padding, inferred EOS or automatic LR schedule.

Active windows are chunked into logical update batches and adjacent equal-width
microbatches. The complete plan is admitted before training, including sparse
output budgets and optimizer-clock capacity. Corpus transport is bounded by
16,777,216 IDs plus shared document/window limits; QKV group/work limits remain.
Learning rate is positive so each successful update advances the AdamW clock
exactly once. Warm-started optimizer steps are stored as a separate offset.

The binary session checkpoint has a bounded JSON header and existing binary
QKV optimizer payload. It binds configuration, corpus/plan identity, label
presence, update cursor, warm-start offset and optimizer identity. Unknown
header fields, bad schema, broken hashes and inconsistent clocks are rejected.
Restore first verifies optimizer/model ownership, then reconstructs the window
plan from separately supplied corpus content and checks its identity and cursor.
No serialized window ranges are trusted. Hashes are integrity pins, not
authentication. Frozen model/adapters and corpus bytes are not embedded.

Python exposes the trainer, typed stubs, one-step reports, progress/config,
checkpoint/restore and a detached `optimizer_snapshot()` for existing inference
APIs. Mutating a returned snapshot or the original optimizer does not mutate
session state. Checkpoint file storage remains caller-owned.

## Executed Verification

Four new native tests cover manual-versus-session updates with frozen LoRA and
geometry, a warm-started optimizer, skipped prompt-only windows and retained
short tails. They verify every restart boundary across two epochs, exact
optimizer/checkpoint/report equality, and idempotent completion. Additional
cases reject changed ignored data, labels, order and readout; malformed/resealed
headers; target-free or over-budget plans; and a plan exhausting the clock.
An overflowing AdamW candidate verifies that failure changes neither cursor
nor optimizer checkpoint.

Three new Python tests exercise the same public corpus path, dense/no-label
updates, exact continuation, inference snapshots and ownership isolation. They
also verify bad pins stop before corpus access, oversized data stops before ID
extraction, strict settings, changed data rejection and failed-update rollback.

| Check | Result | Wall time |
| --- | --- | --- |
| Offline native decoder tests | 230 passed; no failures/ignored | 19.37 s; bodies 2.11 s |
| CPU binding check | Passed | 17.32 s |
| One offline CPU release extension build | Passed | 115.65 s |
| Isolated Python execution | 77 passed; no failures/errors/skips | 2.04 s; bodies 0.315 s |

All four invocations passed initially and kept their source inventories unchanged.
The runner enforced a 4 GiB free-space floor, two build jobs, disabled incremental
output, a 600-second build cap and 180/120-second check/runtime caps. No guard
fired. A private 83-file source overlay used the new extension without installing
a wheel; the prior saved extension was preserved byte-for-byte. Torch, NumPy,
Transformers, Python safetensors and Accelerate imports were neither attempted
nor loaded. The checks use synthetic fixtures and establish no language-quality,
throughput, GPU or released-artifact claim.

## Evidence

Private validation directory: `2026-11-10-qkv-corpus-session-v1`. Only documentation
was added after execution. Rust formatting and whitespace checks were clean,
apart from the repository's existing nightly-only rustfmt-option warnings.

| Artifact | SHA-256 |
| --- | --- |
| Tested source inventory | `6070e7d567e0037432b0bebe77a73a6c98aeeec961e210f56608ec7e8812286a` |
| Native log | `125ae3cee2750dd9c8be7a14eaaba194e60a3a428208b5b6e243eaabd26161b2` |
| Binding check log | `ea6edfa7facffe2bb2495d04b4935389612655f93276361fd5bd3aa7eaac2a38` |
| Extension build log | `153f7882993e8e3dedc13da42ac9bc0e40a5ade965245dc82689f1bc330bec76` |
| Extension | `8623287a131965bce611a3990034cb0c4af94f7029fd4ffb0c0804e29bbddbd4` |
| Package manifest | `c995cccd37bc0374ec6a7553c256cfb15f278dc91cbb07a6bd5e3231636e06b4` |
| Python runtime log | `eaef348375e309f6513e2e4a115fe9aa47106c6a5ebebe9aa2a79f30013c7cb6` |
