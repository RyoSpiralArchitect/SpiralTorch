# SpiralTorch

WGPU-first, Rust-native, Torch-like tensor & autograd runtime.  
**SpiralK** (tiny DSL) drives device heuristics. Self-rewrite & Redis make them evolve.

---

## Install (Rust)
```bash
cargo build -p st-core --release
cargo test  -p st-core
```

## Device selection
```bash
export ST_DEVICE_AUTO_PRIORITIES="wgpu,cpu"
```

## Heuristics (SpiralK)
- Top/Bottom: `SPIRAL_HEUR_K`（Lab override: `SPIRAL_HEUR_K_LAB`）
- MidK: `SPIRAL_HEUR_MIDK`
- Fallback file: `~/.spiraltorch/heur.kdsl`（env が無ければ読む）

Examples:
```bash
export SPIRAL_HEUR_K='u2:(c>32768)||(k>128); ch:sel(c>16384,8192,0)'
export SPIRAL_HEUR_MIDK='wg:sel(c>32768,256,128); tile:sel(c>65536,4096,2048)'
```

## WASM tuner → Generated tables
1. Open `wasm-tuner/` in a browser and export `tuner.json`
2. Generate tables:
```bash
cargo run --manifest-path tools/Cargo.toml --bin gen_heuristics_table -- tuner.json   crates/st-core/src/backend/wgpu_heuristics_generated.rs   crates/st-core/src/backend/generated/midk_heuristics_generated.rs
```
3. Rebuild. Priority: **SpiralK (env/file) > Redis > Generated table > Fallback**

## Self‑Rewrite (opt‑in)
```bash
export ST_SELF_REWRITE=1
export ST_SR_MIN_SAMPLES=5
export ST_SR_IMPROVE_PCT=0.05
export ST_SR_COOLDOWN_SEC=3600
export ST_SR_MAX_LINES=2000
export ST_SR_EXPIRY_DAYS=30
```
Runtime will log observations → synthesize and append `soft(...)` rules to `~/.spiraltorch/heur.kdsl`.

## Redis resonance (opt‑in)
```bash
export REDIS_URL=redis://127.0.0.1/
export ST_KV_BROADCAST=1
export ST_KV_TTL_SEC=600
```

## API (sketch)
```rust
use st_core::backend::{wgpu_heuristics, midk_heuristics};
let ch = wgpu_heuristics::choose(1024, 65536, 1024, true);
let mk = midk_heuristics::choose_midk(1024, 65536, 32768, true);
println!("{:?} {:?}", ch, mk);
```

License: AGPL-3.0-or-later


---

## ROCm / HIP (experimental, feature‑gated)

- Enable HIP backend at compile time (skeleton): `--features hip`.
- Default build provides **stubs** (compile anywhere). To activate real HIP/RCCL bindings,
  build a custom `st-backend-hip` with `hip-real` and link against your ROCm toolchain.
- Distributed heuristics (Unison) can still **publish/ingest** consensus via Redis regardless of HIP.

**Three‑stage distributed TopK (outline)**  
1) local TopK (per GPU) → 2) inter‑node K‑way merge (tree/ring) → 3) finalize/allgather.  
This repo ships a **CPU fallback merger** (`distributed/topk_dist.rs`) as control path.

> Note: exact ROCm versions/APIs vary across environments. Treat HIP feature as opt‑in.
