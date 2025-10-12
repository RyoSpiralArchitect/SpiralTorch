//! Backend-aware unified chooser (TopK/MidK/BottomK).
//! Two-layer consensus (SpiralK + Generated) + backend injections + DSL/Redis soft.

use super::device_caps::{DeviceCaps, BackendKind};
use super::wgpu_heuristics; // WGPU-only generated table

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankKind { TopK, MidK, BottomK }

#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg:  u32,
    pub kl:  u32,
    pub ch:  u32,
    pub mk:  u32,   // 0=bitonic,1=shared,2=warp
    pub mkd: u32,   // 0=auto,1=heap,2=kway,3=bitonic,4=warp_heap,5=warp_bitonic
    pub tile:u32,   // TopK sweep tile
    pub ctile:u32,  // MidK/BottomK compaction tile
}

#[cfg(feature="logic")]
use st_logic::{solve_soft, Ctx as LCtx, SolveCfg as LCfg, SoftRule, Field, Value};

#[cfg(feature="kv-redis")]
use crate::ability::unison_mediator;

use super::kdsl_bridge;

fn fallback(rows:u32, cols:u32, k:u32, caps:&DeviceCaps, kind:RankKind) -> Choice {
    let wg = if caps.subgroup {256} else {128};
    let kl = if k>=64 {32} else if k>=16 {16} else {8};
    let ch = if cols>16_384 {8192} else {0};
    let tile = if cols>131_072 {4096} else if cols>65_536 {2048}
               else if cols>8_192 {1024} else if cols>4_096 {512} else {256};
    let ctile = if cols>65_536 {1024} else if cols>8_192 {512} else {256};
    let mk = match caps.backend {
        BackendKind::Wgpu => if caps.subgroup && k<=128 {2} else if k<=2048 {1} else {0},
        BackendKind::Cuda => if k <= caps.lane_width*4 {2} else if k<=4096 {1} else {0},
        BackendKind::Hip  => if k <= caps.lane_width*2 {2} else if k<=4096 {1} else {0},
        BackendKind::Cpu  => 0,
    };
    let mkd = match mk {
        2 => if k<=128 {4} else {5}, // warp: smallK→warp_heap, else warp_bitonic
        1 => if k<=1024 {1} else {2},// shared: heap then k-way
        _ => 3,
    };
    let use_2ce = (cols>32_768) || (k>128);
    Choice{ use_2ce, wg, kl, ch, mk, mkd, tile, ctile }
}

fn gen_weight_for_backend(bk: BackendKind) -> f32 {
    let key = match bk {
        BackendKind::Wgpu => "SPIRAL_HEUR_GEN_WEIGHT_WGPU",
        BackendKind::Cuda => "SPIRAL_HEUR_GEN_WEIGHT_CUDA",
        BackendKind::Hip  => "SPIRAL_HEUR_GEN_WEIGHT_HIP",
        BackendKind::Cpu  => "SPIRAL_HEUR_GEN_WEIGHT_CPU",
    };
    if let Ok(v) = std::env::var(key) {
        if let Ok(x) = v.parse::<f32>() { return x; }
    }
    if let Ok(v) = std::env::var("SPIRAL_HEUR_GEN_WEIGHT") {
        if let Ok(x) = v.parse::<f32>() { return x; }
    }
    0.10
}

#[cfg(feature="logic")]
fn backend_soft_injections(rows:u32, cols:u32, k:u32, caps:&DeviceCaps, kind:RankKind) -> Vec<SoftRule> {
    let mut v = Vec::<SoftRule>::new();
    // mk by backend
    match caps.backend {
        BackendKind::Wgpu => {
            if caps.subgroup && k<=128 { v.push(SoftRule{ field: Field::Mk, value: Value::U(2), weight: 0.30 }); }
            else if k<=2048 { v.push(SoftRule{ field: Field::Mk, value: Value::U(1), weight: 0.20 }); }
        }
        BackendKind::Cuda => {
            if k <= caps.lane_width*4 { v.push(SoftRule{ field: Field::Mk, value: Value::U(2), weight: 0.28 }); }
            else if k <= 4096 { v.push(SoftRule{ field: Field::Mk, value: Value::U(1), weight: 0.18 }); }
        }
        BackendKind::Hip => {
            if k <= caps.lane_width*2 { v.push(SoftRule{ field: Field::Mk, value: Value::U(2), weight: 0.22 }); }
            else if k <= 4096 { v.push(SoftRule{ field: Field::Mk, value: Value::U(1), weight: 0.22 }); }
        }
        BackendKind::Cpu => {}
    }
    // mkd (sub-strategy) rules
    if k<=128 { v.push(SoftRule{ field: Field::Mkd, value: Value::U(4), weight: 0.18 }); }
    else if k<=1024 { v.push(SoftRule{ field: Field::Mkd, value: Value::U(1), weight: 0.12 }); }
    else { v.push(SoftRule{ field: Field::Mkd, value: Value::U(2), weight: 0.10 }); }
    // tiles
    let candidates = [256u32,512,1024,2048,4096,8192];
    let base = (caps.lane_width.max(16) * 16) as u32;
    let mut best = 256u32; let mut bestd = u32::MAX;
    for &t in &candidates {
        let d = base.abs_diff(t);
        if d < bestd { bestd = d; best = t; }
    }
    v.push(SoftRule{ field: Field::Tile, value: Value::U(best), weight: 0.12 });
    // compaction tile for MidK/BottomK
    if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        let ct = if cols>65_536 {1024} else if cols>8_192 {512} else {256};
        v.push(SoftRule{ field: Field::Ctile, value: Value::U(ct), weight: 0.10 });
    }
    v
}

/// Unified chooser (Rank‑K). Prefer this as the single entry for heuristics.
pub fn choose_unified_rank(rows:u32, cols:u32, k:u32, caps:DeviceCaps, kind:RankKind) -> Choice {
    // 1) Gather DSL hard + soft
    let (dsl_hard, mut soft_rules) = {
        let (hard_opt, soft_dsl) = kdsl_bridge::parse_env_dsl(rows, cols, k, caps.subgroup);
        let hard_mapped = hard_opt.map(|h| Choice{
            use_2ce:h.use_2ce, wg:h.wg, kl:h.kl, ch:h.ch,
            mk:h.mk, mkd:h.mkd.unwrap_or(0), tile:h.tile.unwrap_or(0), ctile:h.ctile.unwrap_or(0)
        });
        #[cfg(feature="logic")]
        let mut sr = soft_dsl;
        #[cfg(feature="kv-redis")]
        { sr.extend(unison_mediator::soft_from_redis(rows, cols, k, caps.subgroup)); }
        #[cfg(feature="logic")]
        { sr.extend(backend_soft_injections(rows, cols, k, &caps, kind)); }
        (hard_mapped, sr)
    };

    // 2) Candidate A: SoftLogic solve
    #[cfg(feature="logic")]
    let cand_a = {
        let ctx = LCtx{ rows, cols, k, sg: caps.subgroup };
        let (c, _score) = solve_soft(ctx, LCfg{ noise:0.02, seed:0x5p1ral }, &soft_rules);
        Choice{ use_2ce:c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch, mk:c.mk, mkd:c.mkd, tile:c.tile, ctile:c.ctile }
    };
    #[cfg(not(feature="logic"))]
    let cand_a = fallback(rows, cols, k, &caps, kind);

    // 3) Candidate C: Generated table (WGPU only) → fill mk/tile; mkd/ctile derive fallback
    let cand_c = match caps.backend {
        BackendKind::Wgpu => {
            let base = wgpu_heuristics::choose(rows, cols, k, caps.subgroup)
                .unwrap_or_else(|| super::wgpu_heuristics::Choice{
                    use_2ce:false, wg:if caps.subgroup{256}else{128},
                    kl:if k>=64{32}else if k>=16{16}else{8}, ch:if cols>16_384{8192}else{0},
                    mk: if caps.subgroup && k<=128 {2} else if k<=2048 {1} else {0},
                    tile: if cols>65_536 {2048} else if cols>8_192 {1024} else if cols>4_096 {512} else {256},
                });
            let mut c = fallback(rows, cols, k, &caps, kind);
            c.use_2ce = base.use_2ce; c.wg=base.wg; c.kl=base.kl; c.ch=base.ch; c.mk=base.mk; c.tile=base.tile;
            c
        }
        _ => fallback(rows, cols, k, &caps, kind)
    };

    // 4) Candidate B: DSL hard override (fill with C then A as needed)
    let cand_b = if let Some(h) = dsl_hard {
        let mut b = cand_c;
        if h.use_2ce { b.use_2ce = h.use_2ce; }
        if h.wg  !=0 { b.wg  = h.wg; }
        if h.kl  !=0 { b.kl  = h.kl; }
        if h.ch  !=0 { b.ch  = h.ch; }
        if h.mk  !=0 { b.mk  = h.mk; }
        if h.mkd !=0 { b.mkd = h.mkd; }
        if h.tile!=0 { b.tile= h.tile; }
        if h.ctile!=0{ b.ctile=h.ctile; }
        b
    } else { cand_a };

    // 5) Consensus: slight bias to generated (per backend)
    let gen_bias = gen_weight_for_backend(caps.backend);
    let pick = |x:Choice, y:Choice| -> Choice {
        let prefer_y =
            (caps.backend == BackendKind::Wgpu && caps.subgroup && y.mk==2 && k<=128) ||
            (caps.backend == BackendKind::Cuda && y.mk>=1) ||
            (caps.backend == BackendKind::Hip  && y.mk>=1);
        if prefer_y && gen_bias>0.0 { y } else { x }
    };
    pick(cand_b, cand_c)
}

/// Backward compat alias for TopK
pub fn choose_unified(rows:u32, cols:u32, k:u32, caps:DeviceCaps) -> Choice {
    choose_unified_rank(rows, cols, k, caps, RankKind::TopK)
}
