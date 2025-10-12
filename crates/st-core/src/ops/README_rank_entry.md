# Rank Entry (TopK/MidK/BottomK)

- `plan_rank(RankKind, rows, cols, k, caps)` plans and returns a `RankPlan` with unified `Choice`.
- Backends implement `RankKExecutor` and call `execute_rank(...)`.
- All heuristics (SpiralK/Redis/SoftLogic/Generated) are encapsulated in `unison_heuristics`.

This becomes the **single standard entry** for Rank‑K across WGPU/CUDA/HIP/CPU.
