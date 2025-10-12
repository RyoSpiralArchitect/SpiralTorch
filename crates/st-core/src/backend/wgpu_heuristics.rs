//! WGPU TopK heuristics: SpiralK(DSL) + SoftLogic + KV + WASM table + Fallback.
use crate::backend::wgpu_heuristics_generated as gen;
#[cfg(feature="logic")]
use st_logic::{solve_soft, Ctx as LCtx, SolveCfg as LCfg, SoftRule};
use super::kdsl_bridge;

/// Unified choice used by TopK backends.
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
}

/// Main entry: decide heuristics with ability stack.
pub fn choose(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    // SpiralK env (hard overrides + soft rules)
    let (hard_from_dsl, soft_rules): (Option<Choice>, Vec<SoftRule>) =
    {
        let (h, s) = kdsl_bridge::parse_env_dsl(rows, cols, k, subgroup);
        (h, s)
    };

    // 0) SoftLogic (with DSL-provided soft rules)
    #[cfg(feature="logic")]
    {
        let use_soft = std::env::var("SPIRAL_HEUR_SOFT").ok().map(|v| v=="1").unwrap_or(true);
        if use_soft {
            let ctx = LCtx { rows, cols, k, sg: subgroup };
            let (c, score) = solve_soft(ctx, LCfg{ noise: 0.02, seed: 0x5p1ra1_u64 }, &soft_rules);
            if score > 0.1 {
                return Some(Choice{ use_2ce: c.use_2ce, wg: c.wg, kl: c.kl, ch: c.ch });
            }
        }
    }

    // 1) SpiralK hard overrides
    if let Some(c) = hard_from_dsl {
        return Some(c);
    }

    // 2) KV (Redis)
    if let Some(c) = kdsl_bridge::choose_from_kv(rows, cols, k, subgroup) {
        return Some(c);
    }

    // 3) WASM-generated table (k-means nearest table)
    if let Some(c) = gen::choose(rows as usize, cols as usize, k as usize, subgroup) {
        return Some(Choice{ use_2ce:c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch });
    }

    // 4) Conservative fallback
    Some(fallback(rows, cols, k, subgroup))
}

/// Conservative fallback used when no other source is available.
fn fallback(_rows:u32, cols:u32, k:u32, subgroup:bool) -> Choice {
    let wg = if subgroup { 256 } else { 128 };
    let kl = if k >= 64 { 32 } else if k >= 16 { 16 } else { 8 };
    let ch = if cols > 16_384 { 8_192 } else { 0 };
    let use_2ce = cols > 32_768 || k > 128;
    Choice { use_2ce, wg, kl, ch }
}

// WASM Tuner / build.rs generated table (stubbed if missing)
include!("wgpu_heuristics_generated.rs");
