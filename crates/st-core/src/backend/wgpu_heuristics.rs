use crate::backend::wgpu_heuristics_generated as gen;
#[cfg(feature="logic")]
use st_logic::{solve_soft, Ctx as LCtx, SolveCfg as LCfg, SoftRule};
use super::kdsl_bridge;

#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
    pub mk: u32, // 0=bitonic,1=shared,2=warp
}

fn fallback(rows:u32, cols:u32, k:u32, subgroup:bool)->Choice{
    Choice{
        use_2ce: cols>32_768 || k>128,
        wg: if subgroup {256} else {128},
        kl: if k>=64 {32} else if k>=16 {16} else {8},
        ch: if cols>16_384 {8192} else {0},
        mk: if k<=128 {2} else if k<=2048 {1} else {0},
    }
}

pub fn choose(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    let (hard_from_dsl, soft_rules): (Option<Choice>, Vec<SoftRule>) =
    {
        let (h, s) = kdsl_bridge::parse_env_dsl(rows, cols, k, subgroup);
        (h, s)
    };

    #[cfg(feature="logic")]
    {
        let use_soft = std::env::var("SPIRAL_HEUR_SOFT").ok().map(|v| v=="1").unwrap_or(true);
        if use_soft {
            let ctx = LCtx { rows, cols, k, sg: subgroup };
            let (c, score) = solve_soft(ctx, LCfg{ noise: 0.02, seed: 0x5p1ral }, &soft_rules);
            if score > 0.1 { return Some(Choice{ use_2ce:c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch, mk:c.mk }) }
        }
    }
    if let Some(c) = hard_from_dsl { return Some(c); }

    if let Some(c) = gen::choose(rows as usize, cols as usize, k as usize, subgroup) {
        return Some(Choice{ use_2ce:c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch, mk: if k<=128 {2} else if k<=2048 {1} else {0} });
    }
    Some(fallback(rows, cols, k, subgroup))
}
