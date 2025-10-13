
# 🌀🕯️SpiralTorch🕯️🌀

**SpiralK + SoftLogic + (optional) WASM tuner** collaborate to pick the fastest **merge kind** and **tile width** for your hardware—then **Self-Rewrite** locks the win back into your heuristics.  
**WGPU** is the default path; **HIP/CUDA** absorb the **same unified choices**. Python wheels target **3.11–3.14**.

> **Why it’s different**
> - **Two-layer consensus:** SpiralK (runtime rules) + WASM table (offline measurements)  
> - **Unified heuristics:** One `Choice { mk, mkd, tile, ctile, … }` across WGPU / HIP / CUDA  
> - **1-CE Subgroup Top-K (WGPU):** candidates → final in a single compute pass  
> - **MidK/BottomK compaction:** 1-CE / 2-CE, tile-aware, same API  
> - **Ameba Hypergrad:** unrolled / implicit (Neumann / CG) hyper-gradients

---

## What you get

- **Rank-K family** (TopK / MidK / BottomK) with a **single entrypoint**  
  Backends implement a `RankKExecutor`, decisions are made once via **unison heuristics**, everyone uses the same plan.
- **SpiralK DSL** (K×Lisp-inspired)  
  Hard assigns (`mk:`, `tile:`) and soft rules (`soft(mk, …)`, `soft(tile, …)`) that blend with measurements.
- **SoftLogic (finite-domain solver)**  
  Explores a tiny discrete space (merge kinds, tiles) and scores candidates with your soft rules.
- **Optional WASM tuner table**  
  Autogenerates a simple piecewise `choose(rows, cols, k, sg)` for your device; the runtime gently prefers measured defaults.
- **Self-Rewrite**  
  A/B outcomes (Wilson CI) append `soft(...)` into `~/.spiraltorch/heur.kdsl` when the advantage is statistically significant.
  
---

### Features (opt-in)

- `wgpu` / `wgpu-rt`: WebGPU backends + runtime wiring
- `mps`: macOS Metal (MPS)
- `cuda`: CUDA (NVRTC/PTX loader expected)
- `hip`: ROCm HIP (stub-safe)
- **`hip-real`**: ROCm HIP + RCCL “real” path (requires ROCm toolchain & linker; gated on top of `hip`)
- **`kv-redis`**: enable Redis-backed consensus (soft hints); absent = **safe no-op**
- `logic` / `kdsl`: SoftLogic solver / SpiralK DSL

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/RyoSpiralArchitect/SpiralTorch.git
cd SpiralTorch
```

### 2) Build from source (Rust)

**CPU (default; no GPU deps)**
```bash
cargo build -p st-core --release
```

**WGPU (WebGPU; Windows/Linux/macOS)**
```bash
cargo build -p st-core --features wgpu --release
```

**MPS (macOS GPU)**
```bash
cargo build -p st-core --features mps --release
```

**CUDA (optional; needs NVRTC/Toolkit)**
```bash
cargo build -p st-core --features cuda --release
```

**HIP / ROCm (optional; real backend is feature-gated)**
```bash
export HIPCC=/opt/rocm/bin/hipcc
export ROCM_PATH=/opt/rocm
cargo build -p st-core --features hip,st-backend-hip/hip-real --release
```

### 3) Python wheels (optional)
```bash
pip install maturin==1.*
# CPU + WGPU wheel
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu
# Add other backends via features (mps / cuda / hip)
```

---

## Minimal API

**Rust (TopK via unified entry)**
```rust
use st_core::backend::device_caps::DeviceCaps;
use st_core::ops::rank_entry::{RankKind, plan_rank, execute_rank};

// describe device
let caps = DeviceCaps::wgpu(32, true, 256); // lane, subgroups, max_wg
// plan once (decisions: mk/mkd/tile/ctile/use_2ce)
let plan = plan_rank(RankKind::TopK, rows, cols, k, caps);

// choose a backend executor (WGPU/CUDA/HIP); CPU fallback exists
use st_core::backend::wgpu_exec::WgpuExecutor;
let exec = WgpuExecutor::default();

// launch
execute_rank(&exec, &plan)?;
```

`DeviceCaps` now ships backend-specific constructors (`wgpu`, `cuda`, `hip`, `cpu`) and
builder-style setters (`with_subgroup`, `with_max_workgroup`, `with_shared_mem`) so you
can describe GPUs with realistic limits while still feeding the unified heuristic
chooser a compact struct.  The helpers also expose higher level tuning hints such as
`recommended_workgroup`, `recommended_tiles`, and `preferred_k_loop` so backends can
query consistent defaults without duplicating the heuristic math.  Pair them with the
extended `prefers_two_stage(rows, cols, k)` signature when you want to peek at whether
the planner will promote the 2-pass compaction path for huge matrices.
chooser a compact struct.  It also exposes derived helpers such as
`recommended_workgroup`, `recommended_sweep_tile`, and `recommended_compaction_tile`
so you can introspect the policy or plug device-aware hints into custom tooling.
chooser a compact struct. The chooser normalizes the plans produced by the DSL, the
generated tables, and the fallback rules, aligning workgroup sizes to hardware warp
widths, honouring shared-memory budgets, and scoring each candidate before execution.
When the reported shared memory is too small for the shared-heap paths or two-stage
compaction, the planner now automatically falls back to bitonic variants so that the
plan always honours device limits.

**Python**
```python
import numpy as np, spiraltorch as st

x = np.random.randn(8, 65536).astype(np.float32)
vals, idx = st.topk2d(x, k=1024, device="auto")   # "wgpu > cuda > mps > cpu"
```

---

## Heuristics (SpiralK) — optional & powerful

SpiralK is a tiny runtime DSL for device-aware choices. Flip it on, then shape the policy per device.

```bash
export SPIRAL_HEUR_SOFT=1
export SPIRAL_HEUR_K='
  # mk: 0=bitonic, 1=shared, 2=warp (subgroup path on WGPU)
  mk:   sel(sg && (k<=128), 2, sel(k<=2048, 1, 0));
  # mkd: sub-strategy (auto/heap/kway/bitonic/warp_heap/warp_bitonic)
  mkd:  sel(mk==2,4, sel(mk==1,1,3));
  # TopK sweeping tile
  tile: sel(log2(c)>15.0, 2048,
        sel(log2(c)>13.0, 1024,
        sel(log2(c)>12.0,  512, 256)));
  # Mid/Bottom compaction tile
  ctile: sel(tile>=1024, tile/2, tile);

  # Soft hints (gently bias the solver)
  soft(mk,   2, 0.25, sg && (k<=128));
  soft(mk,   1, 0.20, (k>128)&&(k<=2048));
  soft(tile, 2048, 0.20, log2(c)>15.0);
  soft(tile, 1024, 0.15, (log2(c)>13.0)&&(log2(c)<=15.0));
'
```

**How the final choice is made (two-layer consensus)**

- **A** = SoftLogic best (your DSL soft + optional Redis soft)
- **B** = DSL **hard** assignment (if you set `mk:`/`tile:` explicitly, B wins)
- **C** = **Generated table** (tuner output)

Default policy: if **B** exists use it; else score **A** and **C** with backend-aware
occupancy/tile metrics derived from `DeviceCaps`, then add a small prior to **C**
(`SPIRAL_HEUR_GEN_WEIGHT`, default `0.10`).
If the adopted choice wins locally (Wilson CI lower bound > 0.5 with min trials), **Self-Rewrite** appends matching `soft(...)` to `~/.spiraltorch/heur.kdsl`.

---

## Regenerating the WASM table (optional)

The repo includes a tiny generator that converts tuner JSON to a Rust table:
```bash
python3 tools/tuner/gen_generated_rs.py tools/tuner/tuner_results.json \
  > crates/st-core/src/backend/wgpu_heuristics_generated.rs
```

**Example JSON**
```json
[
  {"rows": 1024, "cols_min": 4096,  "cols_max": 8191,   "k_max": 128,  "sg": true,  "mk": 2, "tile": 512},
  {"rows": 1024, "cols_min": 8192,  "cols_max": 65535,  "k_max": 2048, "sg": true,  "mk": 1, "tile": 1024},
  {"rows": 1024, "cols_min": 65536, "cols_max": 262143, "k_max": 4096, "sg": true,  "mk": 1, "tile": 2048},
  {"rows": 1024, "cols_min": 4096,  "cols_max": 65535,  "k_max": 2048, "sg": false, "mk": 1, "tile": 1024},
  {"rows": 1024, "cols_min": 65536, "cols_max": 262143, "k_max": 4096, "sg": false, "mk": 0, "tile": 2048}
]
```

---

## Ameba Hypergrad (unrolled / implicit)

Rust utilities for hyper-parameter gradients (continuous relaxation):
- **Unrolled**: expand T updates and backprop
- **Implicit**: Neumann or **CG** to solve `(I − J) v ≈ g` efficiently

> See `crates/st-core/src/autograd/hypergrad*.rs`.  
> Python glue is kept minimal; wheels can expose helpers.

---

## Safety & fallbacks

- Builds **CPU-only** by default (no GPU toolchains required).
- WGPU / CUDA / HIP are **feature-gated** and degrade safely.
- Heuristic chooser always returns a **safe** `Choice` (fills mk/tile from table or conservative defaults).

---

## Contributing

Issues & PRs welcome—especially:
- Backend kernels (WGPU subgroup variants, HIP/CUDA heap/k-way merges)
- Tuner recipes & generated tables
- New SpiralK sugar (e.g., `penalty_if(...)`, device-aware bands)

Run tests/benches on your device and share logs (latency / shapes / adapter caps).  
**AGPL-3.0-or-later** keeps it open and remix-able.

---

## Social preview

Upload a social preview PNG via **Repo → Settings → Social preview** (1200×630).  
Suggested caption: **“SpiralTorch — WGPU-first, Self-Tuning GPU Top-K (Rank-K)”**.

---

### Troubleshooting

- **No Redis?**  
  Build without `kv-redis` or leave `REDIS_URL` unset. The consensus chooser
  skips network calls and falls back to SpiralK / Generated-table safely.

- **ROCm not installed but `hip` enabled?**  
  Use `--features hip` only (stub path). The **real** path needs `hip-real`
  and a working ROCm + RCCL toolchain.

- **Wheels red?**  
  First build CPU+WGPU only: `maturin build -m bindings/st-py/Cargo.toml --release --features wgpu`
  to decouple GPU toolchain issues.

---

## License

**AGPL-3.0-or-later**. See `LICENSE`.
