//! Backend-aware unified chooser (TopK/MidK/BottomK).
//! Two-layer consensus (SpiralK + Generated) + backend injections + DSL/Redis soft.

use super::device_caps::{BackendKind, DeviceCaps};
use super::wgpu_heuristics; // WGPU-only generated table

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankKind {
    TopK,
    MidK,
    BottomK,
}

#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
    pub mk: u32,    // 0=bitonic,1=shared,2=warp
    pub mkd: u32,   // 0=auto,1=heap,2=kway,3=bitonic,4=warp_heap,5=warp_bitonic
    pub tile: u32,  // TopK sweep tile
    pub ctile: u32, // MidK/BottomK compaction tile
}

#[cfg(feature = "logic")]
use st_logic::{solve_soft, Ctx as LCtx, Field, SoftRule, SolveCfg as LCfg, Value};

#[cfg(feature = "kv-redis")]
use crate::ability::unison_mediator;

use super::kdsl_bridge;

fn fallback(_rows: u32, cols: u32, k: u32, caps: &DeviceCaps, _kind: RankKind) -> Choice {
    let default_wg = if caps.subgroup { 256 } else { 128 };
    let wg = caps.align_workgroup(default_wg.max(64));
    let kl = if k >= 64 {
        32
    } else if k >= 16 {
        16
    } else {
        8
    };
    let ch = if cols > 16_384 { 8192 } else { 0 };
    let tile = caps.preferred_tile(cols, 0);
    let ctile = caps.preferred_compaction_tile(cols, 0);
    let mk = caps.preferred_merge_kind(k);
    let mkd = caps.preferred_substrategy(mk, k);
    let use_2ce = caps.prefers_two_stage(cols, k);
    Choice {
        use_2ce,
        wg,
        kl,
        ch,
        mk,
        mkd,
        tile,
        ctile,
    }
}

fn refine_choice(
    mut choice: Choice,
    fallback: Choice,
    caps: &DeviceCaps,
    cols: u32,
    k: u32,
    kind: RankKind,
) -> Choice {
    if choice.wg == 0 {
        choice.wg = fallback.wg;
    }
    choice.wg = caps.align_workgroup(choice.wg);

    if choice.kl == 0 {
        choice.kl = fallback.kl;
    }
    if choice.ch == 0 {
        choice.ch = fallback.ch;
    }
    if choice.mk == 0 {
        choice.mk = caps.preferred_merge_kind(k);
    }
    if choice.mkd == 0 {
        choice.mkd = caps.preferred_substrategy(choice.mk, k);
    }
    if choice.tile == 0 {
        choice.tile = fallback.tile;
    }
    choice.tile = caps.preferred_tile(cols, choice.tile);

    if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        if choice.ctile == 0 {
            choice.ctile = fallback.ctile;
        }
        choice.ctile = caps.preferred_compaction_tile(cols, choice.ctile);
    } else if choice.ctile == 0 {
        choice.ctile = fallback.ctile;
    }

    if !choice.use_2ce && fallback.use_2ce && caps.prefers_two_stage(cols, k) {
        choice.use_2ce = true;
    }

    choice
}

fn score_choice(choice: &Choice, caps: &DeviceCaps, cols: u32, k: u32, kind: RankKind) -> f32 {
    let occ = caps.occupancy_score(choice.wg).min(1.0);
    let mk_pref = if choice.mk == caps.preferred_merge_kind(k) {
        1.0
    } else {
        0.55
    };
    let mkd_pref = if choice.mkd == caps.preferred_substrategy(choice.mk, k) {
        1.0
    } else {
        0.7
    };
    let tile_target = caps.preferred_tile(cols, 0);
    let tile_score =
        1.0 - (tile_target.abs_diff(choice.tile) as f32 / tile_target.max(1) as f32).min(1.0);
    let ctile_score = if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        let ct_target = caps.preferred_compaction_tile(cols, 0);
        1.0 - (ct_target.abs_diff(choice.ctile) as f32 / ct_target.max(1) as f32).min(1.0)
    } else {
        1.0
    };
    let two_stage = if caps.prefers_two_stage(cols, k) == choice.use_2ce {
        1.0
    } else {
        0.75
    };

    (occ * 0.35)
        + (mk_pref * 0.2)
        + (mkd_pref * 0.15)
        + (tile_score * 0.15)
        + (ctile_score * 0.08)
        + (two_stage * 0.07)
}

fn gen_weight_for_backend(bk: BackendKind) -> f32 {
    let key = match bk {
        BackendKind::Wgpu => "SPIRAL_HEUR_GEN_WEIGHT_WGPU",
        BackendKind::Cuda => "SPIRAL_HEUR_GEN_WEIGHT_CUDA",
        BackendKind::Hip => "SPIRAL_HEUR_GEN_WEIGHT_HIP",
        BackendKind::Cpu => "SPIRAL_HEUR_GEN_WEIGHT_CPU",
    };
    if let Ok(v) = std::env::var(key) {
        if let Ok(x) = v.parse::<f32>() {
            return x;
        }
    }
    if let Ok(v) = std::env::var("SPIRAL_HEUR_GEN_WEIGHT") {
        if let Ok(x) = v.parse::<f32>() {
            return x;
        }
    }
    0.10
}

#[cfg(feature = "logic")]
fn backend_soft_injections(
    rows: u32,
    cols: u32,
    k: u32,
    caps: &DeviceCaps,
    kind: RankKind,
) -> Vec<SoftRule> {
    let mut v = Vec::<SoftRule>::new();
    // mk by backend
    match caps.backend {
        BackendKind::Wgpu => {
            if caps.subgroup && k <= 128 {
                v.push(SoftRule {
                    field: Field::Mk,
                    value: Value::U(2),
                    weight: 0.30,
                });
            } else if k <= 2048 {
                v.push(SoftRule {
                    field: Field::Mk,
                    value: Value::U(1),
                    weight: 0.20,
                });
            }
        }
        BackendKind::Cuda => {
            if k <= caps.lane_width * 4 {
                v.push(SoftRule {
                    field: Field::Mk,
                    value: Value::U(2),
                    weight: 0.28,
                });
            } else if k <= 4096 {
                v.push(SoftRule {
                    field: Field::Mk,
                    value: Value::U(1),
                    weight: 0.18,
                });
            }
        }
        BackendKind::Hip => {
            if k <= caps.lane_width * 2 {
                v.push(SoftRule {
                    field: Field::Mk,
                    value: Value::U(2),
                    weight: 0.22,
                });
            } else if k <= 4096 {
                v.push(SoftRule {
                    field: Field::Mk,
                    value: Value::U(1),
                    weight: 0.22,
                });
            }
        }
        BackendKind::Cpu => {}
    }
    // mkd (sub-strategy) rules
    if k <= 128 {
        v.push(SoftRule {
            field: Field::Mkd,
            value: Value::U(4),
            weight: 0.18,
        });
    } else if k <= 1024 {
        v.push(SoftRule {
            field: Field::Mkd,
            value: Value::U(1),
            weight: 0.12,
        });
    } else {
        v.push(SoftRule {
            field: Field::Mkd,
            value: Value::U(2),
            weight: 0.10,
        });
    }
    // tiles
    let candidates = [256u32, 512, 1024, 2048, 4096, 8192];
    let base = (caps.lane_width.max(16) * 16) as u32;
    let mut best = 256u32;
    let mut bestd = u32::MAX;
    for &t in &candidates {
        let d = base.abs_diff(t);
        if d < bestd {
            bestd = d;
            best = t;
        }
    }
    v.push(SoftRule {
        field: Field::Tile,
        value: Value::U(best),
        weight: 0.12,
    });
    // compaction tile for MidK/BottomK
    if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        let ct = if cols > 65_536 {
            1024
        } else if cols > 8_192 {
            512
        } else {
            256
        };
        v.push(SoftRule {
            field: Field::Ctile,
            value: Value::U(ct),
            weight: 0.10,
        });
    }
    v
}

/// Unified chooser (Rank‑K). Prefer this as the single entry for heuristics.
pub fn choose_unified_rank(
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
    kind: RankKind,
) -> Choice {
    let fallback_base = fallback(rows, cols, k, &caps, kind);
    // 1) Gather DSL hard + soft
    #[cfg_attr(not(feature = "logic"), allow(unused_variables))]
    let (dsl_hard, soft_rules_logic) = {
        #[allow(unused_variables)]
        let (hard_opt, soft_dsl) = kdsl_bridge::parse_env_dsl(rows, cols, k, caps.subgroup);
        let hard_mapped = hard_opt.map(|h| Choice {
            use_2ce: h.use_2ce,
            wg: h.wg,
            kl: h.kl,
            ch: h.ch,
            mk: 0,
            mkd: 0,
            tile: 0,
            ctile: h.ctile,
        });
        #[cfg(feature = "logic")]
        let mut sr = soft_dsl;
        #[cfg(not(feature = "logic"))]
        #[allow(unused_mut)]
        let mut sr: Vec<kdsl_bridge::SoftRule> = Vec::new();
        #[cfg(feature = "kv-redis")]
        {
            sr.extend(unison_mediator::soft_from_redis(
                rows,
                cols,
                k,
                caps.subgroup,
            ));
        }
        #[cfg(feature = "logic")]
        {
            sr.extend(backend_soft_injections(rows, cols, k, &caps, kind));
        }
        (hard_mapped, sr)
    };

    #[cfg(feature = "logic")]
    let mut soft_rules = soft_rules_logic;
    #[cfg(not(feature = "logic"))]
    let _soft_rules = soft_rules_logic;

    // 2) Candidate A: SoftLogic solve
    #[cfg(feature = "logic")]
    let cand_a = {
        let ctx = LCtx {
            rows,
            cols,
            k,
            sg: caps.subgroup,
        };
        let (c, _score) = solve_soft(
            ctx,
            LCfg {
                noise: 0.02,
                seed: 0x5p1ral,
            },
            &soft_rules,
        );
        Choice {
            use_2ce: c.use_2ce,
            wg: c.wg,
            kl: c.kl,
            ch: c.ch,
            mk: c.mk,
            mkd: c.mkd,
            tile: c.tile,
            ctile: c.ctile,
        }
    };
    #[cfg(not(feature = "logic"))]
    let cand_a = fallback_base;

    // 3) Candidate C: Generated table (WGPU only) → fill mk/tile; mkd/ctile derive fallback
    let cand_c = match caps.backend {
        BackendKind::Wgpu => {
            let base =
                wgpu_heuristics::choose(rows as usize, cols as usize, k as usize, caps.subgroup)
                    .unwrap_or_else(|| super::wgpu_heuristics::Choice {
                        use_2ce: false,
                        wg: if caps.subgroup { 256 } else { 128 },
                        kl: if k >= 64 {
                            32
                        } else if k >= 16 {
                            16
                        } else {
                            8
                        },
                        ch: if cols > 16_384 { 8192 } else { 0 },
                        algo_topk: 0,
                        ctile: 0,
                        mode_midk: 0,
                        mode_bottomk: 0,
                    });
            let mut c = fallback_base;
            c.use_2ce = base.use_2ce;
            c.wg = base.wg;
            c.kl = base.kl;
            c.ch = base.ch;
            c
        }
        _ => fallback_base,
    };

    // 4) Candidate B: DSL hard override (fill with C then A as needed)
    let cand_b = if let Some(h) = dsl_hard {
        let mut b = cand_c;
        if h.use_2ce {
            b.use_2ce = h.use_2ce;
        }
        if h.wg != 0 {
            b.wg = h.wg;
        }
        if h.kl != 0 {
            b.kl = h.kl;
        }
        if h.ch != 0 {
            b.ch = h.ch;
        }
        if h.mk != 0 {
            b.mk = h.mk;
        }
        if h.mkd != 0 {
            b.mkd = h.mkd;
        }
        if h.tile != 0 {
            b.tile = h.tile;
        }
        if h.ctile != 0 {
            b.ctile = h.ctile;
        }
        b
    } else {
        cand_a
    };

    // 5) Consensus: slight bias to generated (per backend)
    let cand_a = refine_choice(cand_a, fallback_base, &caps, cols, k, kind);
    let cand_b = refine_choice(cand_b, fallback_base, &caps, cols, k, kind);
    let cand_c = refine_choice(cand_c, fallback_base, &caps, cols, k, kind);

    let score_a = score_choice(&cand_a, &caps, cols, k, kind);
    let score_b = score_choice(&cand_b, &caps, cols, k, kind);
    let mut score_c = score_choice(&cand_c, &caps, cols, k, kind);
    let gen_bias = gen_weight_for_backend(caps.backend);
    if gen_bias > 0.0 {
        score_c += gen_bias;
    }
    if caps.backend == BackendKind::Wgpu && caps.subgroup && cand_c.mk == 2 && k <= 128 {
        score_c += 0.08;
    }
    if caps.backend == BackendKind::Cuda && cand_c.mk >= 1 {
        score_c += 0.05;
    }
    if caps.backend == BackendKind::Hip && cand_c.mk >= 1 {
        score_c += 0.04;
    }

    let mut best = (cand_b, score_b);
    for (choice, score) in [(cand_a, score_a), (cand_c, score_c)] {
        if score > best.1 + 0.02 {
            best = (choice, score);
        }
    }
    best.0
}

/// Backward compat alias for TopK
pub fn choose_unified(rows: u32, cols: u32, k: u32, caps: DeviceCaps) -> Choice {
    choose_unified_rank(rows, cols, k, caps, RankKind::TopK)
}
