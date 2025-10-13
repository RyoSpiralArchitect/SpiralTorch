//! Three-layer heuristics for fractional & FFT:
//! 1) SpiralK DSL (env: SPIRAL_HEUR_FRAC)
//! 2) WASM Tuner generated (wgpu_heuristics_frac_generated.rs)
//! 3) Self-Repair (online fallback) — catches misses/OOD and proposes safe settings

#[derive(Clone, Copy, Debug, Default)]
pub struct ChoiceFrac {
    pub use_2ce: bool,
    pub wg: u32,
    pub tile_cols: u32,
    pub radix: u32, // 2 or 4 for FFT
}

pub fn choose_for_frac(rows:u32, cols:u32, k:u32, subgroup:bool, ops_hint:&str) -> ChoiceFrac {
    // 1) SpiralK (env string) — simplified parser: wg, tile, radix, use_2ce
    if let Ok(src) = std::env::var("SPIRAL_HEUR_FRAC") {
        if !src.is_empty() {
            // naive parse
            let use_2ce = src.contains("u2:1") || (cols>32768);
            let wg = if src.contains("wg:256") || subgroup { 256 } else { 128 };
            let tile = if src.contains("tile:8192") || (cols>16384) { 8192 } else { 4096 };
            let radix = if src.contains("radix:4") && (cols%4==0) { 4 } else { 2 };
            return ChoiceFrac{ use_2ce, wg, tile_cols: tile, radix };
        }
    }
    // 2) Generated table (WASM tuner)
    if let Some(c) = crate::backend::wgpu_heuristics_frac_generated::choose_generated(rows as usize, cols as usize, k as usize, subgroup) {
        return c;
    }
    // 3) Self-Repair — safe fallback tuned by quick OOD guards
    //    Heuristic: if subgroup → wg=256; pick radix 4 if divisible by 4; tile by size
    let use_2ce = cols > 32768 || k >= 64;
    let wg = if subgroup { 256 } else { 128 };
    let radix = if cols%4==0 { 4 } else { 2 };
    let tile = if cols > 32768 { 8192 } else if cols>16384 { 4096 } else { 2048 };
    ChoiceFrac{ use_2ce, wg, tile_cols: tile, radix }
}
