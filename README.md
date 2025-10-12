# SpiralTorch v1.7.5 Overlay

**What’s new (build-on v1.7.4):**
1. **Backend executors** (WGPU / CUDA / HIP) implementing `RankKExecutor` for **TopK / MidK / BottomK** with
   instant switching by `plan.choice.{mk, mkd, tile, ctile}`.
2. **Self‑Rewrite**: default **`win_threshold=0.60`** (gradual tightening). Override via env.
3. **WASM Tuner → Generated table**: format extended to carry **`mkd` / `ctile`**; WGPU heuristics `Choice` extended.
4. **MidK/BottomK**: 1CE / 2CE auto selection rule wired (uses `use_2ce` and `ctile : cols` ratio).

## Quick use
```bash
unzip -o spiraltorch-overlay-v1_7_5.zip
export ST_NO_TRACEBACK=1
cargo build -p st-core --features wgpu,logic,kdsl,kv-redis --release
```

Rust (standard planning + backend exec):
```rust
use crate::backend::device_caps::DeviceCaps;
use crate::ops::rank_entry::{RankKind, plan_rank, execute_rank};
use crate::backend::wgpu_exec::WgpuExecutor; // or cuda_exec::CudaExecutor / hip_exec::HipExecutor

let caps = DeviceCaps::wgpu(32, true, 256);
let plan = plan_rank(RankKind::TopK, rows, cols, k, caps);
let exec = WgpuExecutor::default();
execute_rank(&exec, &plan)?;
```

Self‑Rewrite stricter threshold:
```bash
# defaults to 0.60; override
export ST_REWRITE_WIN_THRESHOLD=0.60
```

Tuner → Generated RS:
- `tools/tuner/gen_generated_rs.py` now writes fields: `mkd`, `ctile`, `two_ce_hint` per (rows, cols, k).
- At runtime, WGPU heuristics merges generated + SpiralK under unified chooser.
