# Rank Entry (TopK/MidK/BottomK)

- `plan_rank(RankKind, rows, cols, k, caps)` plans and returns a `RankPlan` with unified `Choice`.
- Backends implement `RankKExecutor` (mutable) and call `execute_rank(...)`.
- All heuristics (SpiralK/Redis/SoftLogic/Generated) are encapsulated in `unison_heuristics`.

CPU reference execution is available via `backend::cpu_exec::CpuExecutor` and
`ops::rank_cpu` (TopK/MidK/BottomK). `RankKind::MidK` selects the centered
window of width `k` from the ascending value order. Threshold-based compaction
remains available via `ops::compaction` (with `ops::midk` kept only as a
compatibility alias).

WGPU threshold/mask compaction is now exposed through
`backend::wgpu_rt::dispatch_compaction_{1ce,2ce}_buffers(...)`. The current
shader contract is `mask -> counts + packed values + packed indices`; only the
first `counts[row]` entries are valid per row, and `kind` is currently a
reserved hint rather than a shader-side branch.

For `execute_rank(...)`, use `backend::wgpu_exec::WgpuBufferExecutor`. The WGPU
exact-selection path currently wires only `TopK`; exact `MidK`/`BottomK` return
an explicit unsupported error instead of silently falling back to threshold
compaction.

This becomes the **single standard entry** for Rank‑K across WGPU/CUDA/HIP/CPU.
