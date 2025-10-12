# SpiralTorch v1.7.2 Overlay

**TL;DR**  
This overlay teaches SpiralTorch to pick **TopK merge kind (`mk`)** and **tile width (`tile_cols`)** using a **two-layer consensus**:
1) **SpiralK DSL** (runtime, hand-authored rules + soft constraints), and  
2) **WASM Tuner generated table** (offline measurements).  
The final choice is exposed as a **unified heuristic** consumed by **WGPU / HIP / CUDA** paths.

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

## Quick Start

1) **Apply overlay**
```bash
unzip -o spiraltorch-overlay-v1_7_2.zip
```

2) **(Optional) Control `mk` / `tile` via SpiralK**  
See the env block above, then:
```bash
cargo build -p st-core --features wgpu,logic,kdsl,kv-redis --release
```

3) **(Optional) Overwrite with a WASM Tuner table**
```bash
python3 tools/tuner/gen_generated_rs.py tools/tuner/tuner_results.json \
  > crates/st-core/src/backend/wgpu_heuristics_generated.rs
```

4) **Build**
```bash
# WGPU example
cargo build -p st-core --features wgpu,logic,kdsl,kv-redis --release
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

- The final `mk/tile` are **consumed by the TopK implementation** in each backend.  
  For WGPU: 1CE/2CE and tiling are already wired; you only plug the **choice** in.
- If you don’t ship a generated table, the chooser remains **safe**:  
  it falls back to **SpiralK / SoftLogic / Redis** (in that order).
- Redis “soft hints” and Self-Rewrite remain compatible—use them to converge choices over time.
