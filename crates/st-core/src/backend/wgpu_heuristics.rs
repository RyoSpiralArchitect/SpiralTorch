// wgpu_heuristics.rs (v1.8.5): add algo_topk and honor KDSL algo
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
    pub algo_topk: u8, // 0=auto, 1=heap, 2=bitonic
}

fn fallback(_rows:u32, cols:u32, k:u32, subgroup:bool) -> Choice {
    let use_2ce = cols > 32_768 || k > 128;
    let wg = if subgroup {256} else {128};
    let kl = if k>=64 {32} else if k>=16 {16} else {8};
    let ch = if cols>16_384 {8192} else {0};
    Choice{ use_2ce, wg, kl, ch, algo_topk: 0 }
}

pub fn choose(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    let (hard_from_dsl, soft_rules, algo_dsl) = kdsl_bridge::parse_env_dsl_plus(rows, cols, k, subgroup);

    #[cfg(feature="logic")]
    {
        let use_soft = std::env::var("SPIRAL_HEUR_SOFT").ok().map(|v| v=="1").unwrap_or(true);
        if use_soft {
            let ctx = LCtx { rows, cols, k, sg: subgroup };
            let (c, score) = solve_soft(ctx, LCfg{ noise: 0.02, seed: 0x5p1ral }, &soft_rules);
            if score > 0.1 {
                return Some(Choice{ use_2ce:c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch, algo_topk: algo_dsl });
            }
        }
    }
    if let Some(mut c) = hard_from_dsl {
        c.algo_topk = algo_dsl;
        return Some(c);
    }
    if let Some(mut c) = kdsl_bridge::choose_from_kv(rows, cols, k, subgroup) {
        c.algo_topk = algo_dsl;
        return Some(c);
    }
    if let Some(mut c) = gen::choose(rows, cols, k, subgroup) {
        c.algo_topk = algo_dsl;
        return Some(c);
    }
    Some(fallback(rows, cols, k, subgroup))
}
