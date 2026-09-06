# Shared-Memory Rank Tile Sort

## Execution Evidence

Baseline `141952ac` and candidate `9c027aa2` ran in clean, separate Furnace
worktrees on the same named RTX 5090 (WGPU Vulkan versus PyTorch CUDA). Order:
baseline A1, candidate B1, candidate B2, preserved baseline executable A2.
Every run passed 54 canonical rank comparisons, source/build binding,
execution-image stability, and preflight/postflight foreign-compute-PID gates.
The request hash is identical across all four runs:
`0e80062e7078ed0ad2ffd7e36845288b98a543125decd05d8f693da3eb1caa79`.

The candidate keeps padded tiles up to 1024 in 8 KiB of workgroup memory and
retains storage-memory sorting above that boundary. Finite filtering, total
float ordering, source-index ties, and snapshot semantics are unchanged.

## Bounded Timing Result

Units below are microseconds per operation. Each cell is the median of nine
seed/requested-tile medians; each interval enqueues 16 rank operations plus a
completion fence. Requested tiles can clamp to the same effective geometry.

| Kind | Columns | A1 | B1 | B2 | A2 | PyTorch A2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TopK | 256 | 59.16 | 50.36 | 50.33 | 58.98 | 7.16 |
| TopK | 2048 | 79.68 | 59.95 | 59.83 | 79.72 | 12.91 |
| MidK | 256 | 50.38 | 36.00 | 36.13 | 50.19 | 13.62 |
| MidK | 2048 | 106.09 | 92.06 | 91.77 | 106.09 | 15.00 |
| BottomK | 256 | 59.02 | 50.38 | 50.39 | 59.06 | 7.20 |
| BottomK | 2048 | 79.55 | 59.77 | 59.63 | 79.65 | 14.17 |

Group medians fell about 13-29% from A1 to B1, with the effect retained in B2
and the preserved baseline returning to its earlier resident timing in A2.
All 54 matched cases improve when comparing the mean of their two per-run
resident medians. These are API-throughput observations, not GPU-event timing,
statistical confidence intervals, or a universal speedup. The process gate is
not an exclusive reservation; cross-framework measurements remain separate
blocks. **PyTorch is still faster.**

Resident host-to-host medians stayed around 0.76-0.80 ms and drifted between
runs. There is no demonstrated end-to-end host-call improvement. Do not
attribute upload/readback/driver overhead to shader execution time.

## Additional Validation

- Real Chrome WebGPU: 42 cases and 934 assertions, including the 1024/1025
  boundary, partial final tiles, repeated dispatch and snapshots surviving free.
  The browser reports `BrowserWebGpu` but does not expose its physical GPU name.
- Native WGPU rank tests: 8 passed locally and on Furnace, including 81
  shape/tile/k/kind boundary combinations and an exactly 8 KiB device limit.
- Rebuilt Python wheel: 70 passed, one optional skip, including 12 new boundary
  cases through `spiraltorch.WgpuRank`.
- Pinned rustfmt and native/wasm all-target strict backend clippy passed.

Browser evidence binds the generated assets and harness by hash, not by the
native source/build attestation. API and reproduction boundaries are in the
[resident rank guide](../../../docs/performance/resident_rank.md).

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| resident-guarded-baseline-141952ac.json (A1) | 16b5b25ebc35a37c66fbb9ea34f1e6dc0c00b2867b99ebce835f55e1691b5857 |
| resident-shared-sort-9c027aa2.json (B1) | 8b53aadc7f375818b4fd0ee0ddb4d1e630f8a7830a4368628a52deb08dc017d7 |
| resident-shared-sort-repeat-9c027aa2.json (B2) | 07d480948061186bc301b7b73f872b67b8314284c41cea0c01219586124dc2ba |
| resident-guarded-baseline-repeat-141952ac.json (A2) | 3038f2a20a2cd207947085d7a7d0a682b53c9196654a468534853bc8d7b26612 |
| spiraltorch-shared-sort-browser.json | 490fa11ab1b8d3eabeca8f60d893d21a34cf7b83988febbf30eaaae00a365f1e |
