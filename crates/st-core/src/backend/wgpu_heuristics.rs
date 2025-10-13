// crates/st-core/src/backend/wgpu_heuristics.rs  (v1.8.7)
use crate::backend::wgpu_heuristics_generated as gen;
#[cfg(feature="logic")]
use st_logic::{solve_soft, Ctx as LCtx, SolveCfg as LCfg, SoftRule};
use super::kdsl_bridge;

// Unified Choice consulted by all RankK executors
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
    pub algo_topk: u8,    // 0=auto, 1=heap, 2=bitonic
    pub ctile: u32,       // 0=auto or tile size (cols per tile)
    pub mode_midk: u8,    // 0=auto, 1=1CE, 2=2CE
    pub mode_bottomk: u8, // 0=auto, 1=1CE, 2=2CE
}

fn fallback(_rows:u32, cols:u32, k:u32, subgroup:bool) -> Choice {
    let use_2ce = cols > 32_768 || k > 128;
    let wg = if subgroup {256} else {128};
    let kl = if k>=64 {32} else if k>=16 {16} else {8};
    let ch = if cols>16_384 {8192} else {0};
    Choice{ use_2ce, wg, kl, ch, algo_topk: 0, ctile: 0, mode_midk: 0, mode_bottomk: 0 }
}

// Backward-compatible entry (kindless). Use choose_topk/choose_midk/choose_bottomk when possible.
pub fn choose(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "generic")
}

pub fn choose_topk(rows:u32, cols:u32, k:u32, subgroup:bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "topk")
}
pub fn choose_midk(rows:u32, cols:u32, k:u32, subgroup:bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "midk")
}
pub fn choose_bottomk(rows:u32, cols:u32, k:u32, subgroup:bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "bottomk")
}

pub fn choose_kind(rows: u32, cols: u32, k: u32, subgroup: bool, kind:&'static str) -> Option<Choice> {
    let (hard_from_dsl, soft_rules, dsl_overrides) = kdsl_bridge::parse_env_dsl_plus_kind(rows, cols, k, subgroup, kind);

    #[cfg(feature="logic")]
    {
        let use_soft = std::env::var("SPIRAL_HEUR_SOFT").ok().map(|v| v=="1").unwrap_or(true);
        if use_soft {
            let ctx = LCtx { rows, cols, k, sg: subgroup };
            let (c, score) = solve_soft(ctx, LCfg{ noise: 0.02, seed: 0x5p1ral }, &soft_rules);
            if score > 0.1 {
                let mut out = Choice{ use_2ce:c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch, ..fallback(rows,cols,k,subgroup) };
                out.algo_topk   = dsl_overrides.algo_topk;
                out.ctile       = dsl_overrides.ctile;
                out.mode_midk   = dsl_overrides.mode_midk;
                out.mode_bottomk= dsl_overrides.mode_bottomk;
                return Some(out);
            }
        }
    }

    if let Some(mut c) = hard_from_dsl {
        // overlay DSL overrides
        overlay_dsl(&mut c, &dsl_overrides);
        return Some(c);
    }
    if let Some(mut c) = kdsl_bridge::choose_from_kv(rows, cols, k, subgroup) {
        overlay_dsl(&mut c, &dsl_overrides);
        return Some(c);
    }
    if let Some(mut c) = gen::choose(rows as usize, cols as usize, k as usize, subgroup) {
        overlay_dsl(&mut c, &dsl_overrides);
        return Some(c);
    }
    Some(fallback(rows, cols, k, subgroup))
}

#[derive(Default, Clone, Copy)]
pub struct DslOverrides {
    pub algo_topk: u8,
    pub ctile: u32,
    pub mode_midk: u8,
    pub mode_bottomk: u8,
}

fn overlay_dsl(c:&mut Choice, o:&DslOverrides){
    if o.algo_topk!=0  { c.algo_topk = o.algo_topk; }
    if o.ctile!=0      { c.ctile = o.ctile; }
    if o.mode_midk!=0  { c.mode_midk = o.mode_midk; }
    if o.mode_bottomk!=0 { c.mode_bottomk = o.mode_bottomk; }
}

// WASM Tuner or build.rs generated table (kept as-is, now allowed to fill algo_topk/ctile/modes)
include!("wgpu_heuristics_generated.rs");
