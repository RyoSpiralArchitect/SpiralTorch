use super::kdsl_bridge;
#[cfg(feature="logic")]
use st_logic::{solve_soft, Ctx as LCtx, SolveCfg as LCfg, SoftRule};
#[derive(Debug, Clone, Copy)]
pub struct Choice { pub use_2ce: bool, pub wg: u32, pub kl: u32, pub ch: u32 }

fn fallback(_rows:u32,_cols:u32,_k:u32,subgroup:bool)->Choice{
    Choice{ use_2ce:false, wg: if subgroup{256}else{128}, kl:8, ch:0 }
}

pub fn choose(rows:u32, cols:u32, k:u32, subgroup:bool)->Choice{
    let (hard_dsl, soft_rules) = kdsl_bridge::parse_env_dsl(rows, cols, k, subgroup);

    #[cfg(feature="logic")]
    {
        let ctx = LCtx{ rows, cols, k, sg:subgroup };
        let (c, score) = solve_soft(ctx, LCfg{ noise: 0.02, seed: 0x5p1ral }, &soft_rules);
        if score > 0.1 {
            return Choice{ use_2ce:c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch };
        }
    }

    if let Some(h) = hard_dsl { return h; }

    if let Some(kv) = kdsl_bridge::choose_from_kv(rows, cols, k, subgroup) { return kv; }

    if let Some(gen) = choose_generated(rows, cols, k, subgroup) { return gen; }

    fallback(rows, cols, k, subgroup)
}

include!("wgpu_heuristics_generated.rs");
