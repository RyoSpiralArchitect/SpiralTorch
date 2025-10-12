// Unified heuristics (WGPU-first) with two-layer consensus (SpiralK + Generated table).
use super::wgpu_heuristics_generated as gen;

#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg:  u32,
    pub kl:  u32,
    pub ch:  u32,
    pub mk:  u32,  // 0=bitonic,1=shared,2=warp
    pub tile:u32,  // tile_cols (256..2048)
}

#[cfg(feature="logic")]
use st_logic::{solve_soft, Ctx as LCtx, SolveCfg as LCfg, SoftRule, Field, Value};
use super::kdsl_bridge;
#[cfg(feature="kv-redis")]
use crate::ability::unison_mediator;

fn fallback(rows:u32, cols:u32, k:u32, subgroup:bool)->Choice{
    let wg = if subgroup {256} else {128};
    let kl = if k>=64 {32} else if k>=16 {16} else {8};
    let ch = if cols>16_384 {8192} else {0};
    let mk = if subgroup && k<=128 {2} else if k<=2048 {1} else {0};
    let tile = if cols>65_536 {2048} else if cols>8192 {1024} else if cols>4096 {512} else {256};
    Choice{ use_2ce: cols>32_768 || k>128, wg, kl, ch, mk, tile }
}

#[allow(unused)]
pub fn choose(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    // 1) collect Soft rules from DSL + Redis (low-weight)
    let (dsl_hard_opt, mut soft_rules): (Option<Choice>, Vec<SoftRule>) = {
        let (hard, soft) = kdsl_bridge::parse_env_dsl(rows, cols, k, subgroup);
        let mut sr = soft;
        #[cfg(feature="kv-redis")]
        { sr.extend(unison_mediator::soft_from_redis(rows, cols, k, subgroup)); }
        (hard.map(|h| Choice{ use_2ce:h.use_2ce, wg:h.wg, kl:h.kl, ch:h.ch, mk:h.mk, tile:h.tile }), sr)
    };

    // 2) SoftLogic solve (candidate A)
    #[cfg(feature="logic")]
    let cand_a = {
        let ctx = LCtx{rows, cols, k, sg:subgroup};
        let (c, _score) = solve_soft(ctx, LCfg{ noise:0.02, seed:0x5p1ral }, &soft_rules);
        Choice{ use_2ce:c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch, mk:c.mk, tile:c.tile }
    };
    #[cfg(not(feature="logic"))]
    let cand_a = fallback(rows, cols, k, subgroup);

    // 3) Generated table (candidate C)
    let cand_c = gen::choose(rows as usize, cols as usize, k as usize, subgroup)
        .unwrap_or_else(|| fallback(rows, cols, k, subgroup));

    // 4) DSL hard (candidate B)
    let cand_b = if let Some(h) = dsl_hard_opt {
        // fill missing fields from cand_c preferentially, else cand_a
        Choice{
            use_2ce: h.use_2ce,
            wg:  if h.wg  !=0 {h.wg}  else {cand_c.wg},
            kl:  if h.kl  !=0 {h.kl}  else {cand_c.kl},
            ch:  if h.ch  !=0 {h.ch}  else {cand_c.ch},
            mk:  if h.mk  !=0 {h.mk}  else {cand_c.mk},
            tile:if h.tile!=0 {h.tile}else {cand_c.tile},
        }
    } else { cand_a };

    // 5) Consensus: pick better of {A,B} vs C using SoftLogic score + slight generated bias
    #[cfg(feature="logic")]
    let decide = |x:Choice, y:Choice| -> Choice {
        let ctx = LCtx{rows, cols, k, sg:subgroup};
        // reuse Soft scoring by passing soft rules (DSL+Redis)
        let s_x = st_logic::solve_soft(ctx, LCfg{noise:0.0,seed:1}, &soft_rules).1; // hack: we only need scoring; reuse path
        // emulate scoring of y by swapping candidate (cheap trick: add a temporary Soft that pins y; or use fallback scoring)
        // We'll approximate: if mk/tile align with rules, bump; else rely on generated bias.
        let gen_bias: f32 = std::env::var("SPIRAL_HEUR_GEN_WEIGHT").ok().and_then(|v|v.parse().ok()).unwrap_or(0.10);
        // simple heuristic: prefer y (generated) if wg/mk/tile differ and subgroup suggests warp/shared boundaries
        let prefer_y = (subgroup && (y.mk==2) && (k<=128)) || ((!subgroup) && (y.mk==1) && (k<=2048));
        if prefer_y { y } else { x }
    };
    #[cfg(not(feature="logic"))]
    let decide = |x:Choice, _y:Choice| -> Choice { x };

    let final_choice = decide(cand_b, cand_c);
    Some(final_choice)
}

// WASM Tuner generated piecewise map (mk,tile); default impl here is a placeholder.
// The real pipeline can overwrite by regenerating this file from tuner JSON.
include!("wgpu_heuristics_generated.rs");
