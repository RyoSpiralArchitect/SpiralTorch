<p align="left">
  <a href="https://github.com/RyoSpiralArchitect/SpiralTorch/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/RyoSpiralArchitect/SpiralTorch/release_wheels.yml?label=release%20wheels&logo=github"></a>
  <a href="#license"><img alt="AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-blue"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.11–3.14-3776AB?logo=python"></a>
</p>
# SpiralTorch v1.7.2 — Self-Tuning GPU Top-K. WGPU-first. Zero-regret fallbacks.

**SpiralK + SoftLogic + WASM Tuner** pick the fastest **merge kind** and **tile width** for your hardware—then **Self-Rewrite** locks wins into your heuristics.  
WGPU is the default, HIP/CUDA absorb the same unified choices. Python wheels target 3.11–3.14.

**Why it’s different**
- **Two-layer consensus:** SpiralK (runtime rules) + WASM table (offline measurements)
- **Unified heuristics:** same `Choice{ mk, tile, … }` across WGPU / HIP / CUDA
- **1-CE Subgroup Top-K:** candidates → final in a single pass (WGPU)
- **MidK compaction kernels:** 1-CE / 2-CE paths, tile-aware
- **Ameba Hypergrad:** unrolled & implicit (Neumann / CG) utilities
  
---

## What’s inside

1. **SpiralK DSL extensions**  
   - Hard assigns: `mk:` (0=bitonic / 1=shared / 2=warp), `tile:` (256/512/1024/2048)  
   - Soft constraints: `soft(mk, …)`, `soft(tile, …)`
   - Example “portable defaults” (feel free to edit):
     ```bash
     export SPIRAL_HEUR_SOFT=1
     export SPIRAL_HEUR_K='
       # mk: 0=bitonic, 1=shared, 2=warp
       mk: sel(sg && (k<=128), 2, sel(k<=2048, 1, 0));
       tile: sel(log2(c)>15.0, 2048, sel(log2(c)>13.0, 1024, sel(log2(c)>12.0, 512, 256)));
       soft(mk, 2, 0.25, sg && (k<=128));
       soft(mk, 1, 0.20, (k>128)&&(k<=2048));
       soft(tile, 2048, 0.20, log2(c)>15.0);
       soft(tile, 1024, 0.15, (log2(c)>13.0)&&(log2(c)<=15.0));
     '
     ```

2. **SoftLogic (finite-domain solver) extensions**  
   - Search space: `mk ∈ {0,1,2}`, `tile ∈ {256,512,1024,2048}` with `tile ≤ cols`  
   - Scoring highlights (before DSL/Redis soft rules are added):
     - Small-K & subgroups → prefer `mk=warp`  
     - Mid-K → prefer `mk=shared`  
     - Huge pools → fall back to `mk=bitonic`  
     - `tile` preference follows `log2(cols)` bands

3. **Unified heuristic chooser (Unison)**  
   - Merges **DSL soft** + **Redis soft** into SoftLogic (candidate **A**)  
   - Combines with **WASM table** (candidate **C**) via “two-layer consensus” (details below)  
   - Returns a single `Choice{ mk, tile, ... }` used by **all** backends

4. **Generated table sample (WASM Tuner)**  
   - Ships with a simple piecewise `wgpu_heuristics_generated.rs`  
   - Re-generate from browser/Node JSON via `tools/tuner/gen_generated_rs.py`
---

**Requirements**
- Rust 1.74+ (stable)
- Python 3.11–3.14 (for wheels via maturin)
- Optional GPU SDKs:
  - WGPU: no extra SDKs (WebGPU drivers)
  - macOS MPS: Xcode command line tools
  - CUDA: CUDA toolkit + NVRTC
  - HIP/ROCm: ROCm toolchain (`hipcc`) when enabling `st-backend-hip/hip-real`

---

## ✅ Quick Start

### 1) Clone
```bash
git clone https://github.com/RyoSpiralArchitect/SpiralTorch.git
cd SpiralTorch
```

### 2) Build from source (Rust)

**CPU (default, no GPU deps)**
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

**CUDA (optional; NVRTC/CUDA toolkit required)**
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
# CPU + WGPU
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu
```

### 4) Minimal examples

**Rust**
```rust
use st_core::ops::topk_midk_gpu::{topk2d, Device};
let (vals, idx) = topk2d(x.view(), 1024, Device::Auto)?;
```

**Python**
```python
import numpy as np, spiraltorch as st
x = np.random.randn(8, 65536).astype(np.float32)
vals, idx = st.topk2d(x, k=1024, device="auto")
```

---

## Heuristics (SpiralK) – optional

```bash
export SPIRAL_HEUR_SOFT=1
export SPIRAL_HEUR_K='
  mk: sel(sg && (k<=128), 2, sel(k<=2048, 1, 0));
  tile: sel(log2(c)>15.0, 2048, sel(log2(c)>13.0, 1024, sel(log2(c)>12.0, 512, 256)));
  soft(mk, 2, 0.25, sg && (k<=128));
  soft(mk, 1, 0.20, (k>128)&&(k<=2048));
  soft(tile, 2048, 0.20, log2(c)>15.0);
  soft(tile, 1024, 0.15, (log2(c)>13.0)&&(log2(c)<=15.0));
'
```
---

## Two-Layer Consensus (how the final choice is made)

- **A** = Best score from **SoftLogic** (includes DSL soft + optional Redis soft)
- **B** = **DSL hard** assignment (if you set `mk:` / `tile:` explicitly)
- **C** = **Generated table** (WASM Tuner output)

**Default policy:**
1. If **B** (DSL hard) exists, **B** wins (missing fields are filled from C or A).
2. Else, compare **A vs C** by SoftLogic score and favor **C** with a small prior:
   - `SPIRAL_HEUR_GEN_WEIGHT` (default **0.10**) is added to C to gently bias toward measured defaults.
3. If the adopted choice **wins locally with significance** (Wilson CI lower bound > 0.5 and min trials),  
   **Self-Rewrite** appends a matching `soft(...)` into `heur.kdsl` to “stick” to it.

---

## Regenerating the WASM table

Feed your tuner JSON to the generator:

```bash
python3 tools/tuner/gen_generated_rs.py tools/tuner/tuner_results.json \
  > crates/st-core/src/backend/wgpu_heuristics_generated.rs
```

**Example `tools/tuner/tuner_results.json`:**
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

## Notes

> When enabled, the project may append `soft(...)` rules to `~/.spiraltorch/heur.kdsl` **on your machine only**.  
> No network calls are performed unless you explicitly use Redis (`REDIS_URL`) or HIP distributed features.
- The final `mk/tile` are **consumed by the TopK implementation** in each backend.  
  For WGPU: 1CE/2CE and tiling are already wired; you only plug the **choice** in.
- If you don’t ship a generated table, the chooser remains **safe**:  
  it falls back to **SpiralK / SoftLogic / Redis** (in that order).
- Redis “soft hints” and Self-Rewrite remain compatible—use them to converge choices over time.
