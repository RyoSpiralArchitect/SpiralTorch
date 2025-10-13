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

fn fallback(rows: u32, cols: u32, k: u32, caps: &DeviceCaps, kind: RankKind) -> Choice {
    let wg = caps.recommended_workgroup(rows);
    let kl = caps.preferred_k_loop(k);
    let ch = caps.preferred_channel(cols);
    let (tile, mut ctile) = caps.recommended_tiles(cols);

    // BottomK relies heavily on compaction; bias toward smaller ctile to match
    // the streaming nature of the kernel.
    if matches!(kind, RankKind::BottomK) {
        ctile = ctile.min(tile / 2).max(128);
        let align = caps.lane_width.max(1);
        if align > 1 {
            let remainder = ctile % align;
            if remainder != 0 {
                ctile += align - remainder;
            }
        }
        ctile = ctile.min(tile);
    }

    let mk = caps.preferred_merge_kind(k);
    let mkd = caps.preferred_substrategy(mk, k);
    let use_2ce = caps.prefers_two_stage(rows, cols, k);
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
    // wg / kl / tile preferences from the hardware descriptor
    let preferred_wg = caps.recommended_workgroup(rows);
    v.push(SoftRule {
        field: Field::Wg,
        value: Value::U(preferred_wg),
        weight: 0.16,
    });
    let preferred_kl = caps.preferred_k_loop(k);
    v.push(SoftRule {
        field: Field::Kl,
        value: Value::U(preferred_kl),
        weight: 0.14,
    });
    let (preferred_tile, preferred_ctile) = caps.recommended_tiles(cols);
    v.push(SoftRule {
        field: Field::Tile,
        value: Value::U(preferred_tile),
        weight: 0.12,
    });
    // compaction tile for MidK/BottomK
    if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        v.push(SoftRule {
            field: Field::Ctile,
            value: Value::U(preferred_ctile),
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
    // 1) Gather DSL hard + soft
    let (dsl_hard, mut soft_rules) = {
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
    let cand_a = fallback(rows, cols, k, &caps, kind);

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
            let mut c = fallback(rows, cols, k, &caps, kind);
            c.use_2ce = base.use_2ce;
            c.wg = base.wg;
            c.kl = base.kl;
            c.ch = base.ch;
            c
        }
        _ => fallback(rows, cols, k, &caps, kind),
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
    let gen_bias = gen_weight_for_backend(caps.backend);
    let pick = |x: Choice, y: Choice| -> Choice {
        let prefer_y =
            (caps.backend == BackendKind::Wgpu && caps.subgroup && y.mk == 2 && k <= 128)
                || (caps.backend == BackendKind::Cuda && y.mk >= 1)
                || (caps.backend == BackendKind::Hip && y.mk >= 1);
        if prefer_y && gen_bias > 0.0 {
            y
        } else {
            x
        }
    };
    pick(cand_b, cand_c)
}

/// Backward compat alias for TopK
pub fn choose_unified(rows: u32, cols: u32, k: u32, caps: DeviceCaps) -> Choice {
    choose_unified_rank(rows, cols, k, caps, RankKind::TopK)
}
