# Independent Review And Repairs

## Core API

Reviewer `01a13379-ca9b-79d3-aa29-197fa239a4fe` inspected the scoped Rust
API/harness changes against `fa6113be`. No actionable introduced bug was found.
The reviewer checked that only the query-coordinate input adjoint is omitted,
query parameter and metric VJPs remain, Full stays default, missing routers
reject, and owner/revision/guard/atomic-update paths are retained. Checkpoint
policy reapplication is documented rather than silently persisted.

This was a source-only review, not independent execution of the GPU workload
or raw numerical evidence. The subsequently added native stopped-CE checkpoint
test is covered by the parent's runtime test, not this initial review.

## Evidence Review

Reviewer `01a13385-c5d3-78e1-801c-a540b4cf1d1a` separately inspected the paired
harness, browser page, independent diagnostic, scalar publisher and its tests.
The reviewer ran seven existing stdlib tests plus three in-memory mutation
checks; no GPU/browser builds, raw-array replay or file edits were performed.

**P2, fixed:** the harness initially executed Full, Stop and Full again before
reading either earlier VJP. If a later backward overwrote an earlier buffer,
the repeat comparison could falsely pass. This was a validation blind spot,
not evidence of aliasing in the current core.

The repair captures embedding and all parameter arrays immediately after each
backward, before another one can touch buffers. It also compares the retained
Full after Stop and retained Stop after repeat against independent snapshots.
The native API test uses the same strengthened observation order. No library
numerical kernel, fixture or gate threshold changed.

The same reviewer performed a bounded source-only follow-up on those two files
and confirmed the specific P2 was fixed, with no new concrete issue found.
The parent then rebuilt and reran native/browser evidence and related tests;
the final results and hashes are in `validation.json`. Initial raw evidence
and the initial weak-fixture test failure remain local, not overwritten.

## Remaining Boundary

The paired control is not a replacement for projective-v1 qualification, which
remains 4/6 in each runtime. No tolerance relaxation, numerical optimization,
large corpus sweep, language-quality advantage, browser stopped-mode training
resume, CUDA result, PR, CI run or merge is claimed by this increment.
