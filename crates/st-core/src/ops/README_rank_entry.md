# Rank Entry (TopK/MidK/BottomK)

- `plan_rank(RankKind, rows, cols, k, caps)` plans exact TopK/MidK/BottomK and returns a `RankPlan` with an exact-rank `Choice`.
- `ops::compaction::plan_compaction(rows, cols, caps)` plans threshold/mask compaction separately via `CompactionPlan`.
- Backends implement `RankKExecutor` (mutable) and call `execute_rank(...)`.
- All heuristics (SpiralK/Redis/SoftLogic/Generated) are encapsulated in `unison_heuristics`.

CPU reference execution is available via `backend::cpu_exec::CpuExecutor` and
`ops::rank_cpu` (TopK/MidK/BottomK). `RankKind::MidK` selects the centered
window of width `k` from the ascending value order. Threshold-based compaction
is planned and executed separately through `ops::compaction` (with `ops::midk`
kept only as a compatibility alias).

WGPU threshold/mask compaction is now exposed through
`backend::wgpu_rt::dispatch_compaction(...)` (plan-aware) or the lower-level
`dispatch_compaction_{1ce,2ce}_buffers(...)` entry points. The current shader
contract is `mask -> counts + packed values + packed indices`; only the first
`counts[row]` entries are valid per row, and `kind` is currently a reserved
hint rather than a shader-side branch. `ST_COMPACTION_APPLY=sg2` enables the
alternate subgroup-prefix apply kernel for A/B testing; `fallback` forces the
serial apply path.

For `execute_rank(...)`, use `backend::wgpu_exec::WgpuBufferExecutor`. The WGPU
exact-selection path currently wires only `TopK`; exact `MidK`/`BottomK` return
an explicit unsupported error instead of silently falling back to threshold
compaction.

CUDA/HIP exact-selection executors are currently fail-fast stubs as well: they
surface an explicit error for `TopK`/`MidK`/`BottomK` until real kernel
dispatch is wired, rather than returning success without writing outputs.

This keeps exact Rank‑K and threshold compaction on distinct planner surfaces.
