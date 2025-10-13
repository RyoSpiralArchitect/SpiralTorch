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
fn fallback(_rows: u32, cols: u32, k: u32, caps: &DeviceCaps, _kind: RankKind) -> Choice {
    let wg = caps.recommended_workgroup();
    let kl = caps.recommended_kl(k);
    let ch = caps.recommended_channel_stride(cols);
    let tile = caps.recommended_sweep_tile(cols);
    let ctile = caps.recommended_compaction_tile(cols);
fn default_wg(caps: &DeviceCaps) -> u32 {
    if caps.subgroup {
        256
    } else {
        128
    }
}

fn default_tile(cols: u32) -> u32 {
    if cols > 131_072 {
        4096
    } else if cols > 65_536 {
        2048
    } else if cols > 8_192 {
        1024
    } else if cols > 4_096 {
        512
    } else {
        256
    }
}

fn default_compaction_tile(cols: u32) -> u32 {
    if cols > 65_536 {
        1024
    } else if cols > 8_192 {
        512
    } else {
        256
    }
}

fn fallback(_rows: u32, cols: u32, k: u32, caps: &DeviceCaps, _kind: RankKind) -> Choice {
    let default_wg = if caps.subgroup { 256 } else { 128 };
    let wg = caps.max_workgroup.min(default_wg).max(64);
    let kl = DeviceCaps::default_lane_quota(k);
    let ch = if cols > 16_384 { 8192 } else { 0 };
    let tile = default_tile(cols);
    let ctile = default_compaction_tile(cols);
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

fn closeness(actual: u32, target: u32) -> f32 {
    if actual == 0 || target == 0 {
        return 0.0;
    }
    let diff = actual.abs_diff(target) as f32;
    let denom = target.max(1) as f32;
    (1.0 - (diff / denom).min(1.0)).max(0.0)
}

fn score_choice(
    choice: &Choice,
    caps: &DeviceCaps,
    rows: u32,
    cols: u32,
    k: u32,
    kind: RankKind,
) -> f32 {
    let mut score = 0.0f32;

    let expected_two_stage = caps.prefers_two_stage(cols, k);
    if choice.use_2ce == expected_two_stage {
        score += 0.35;
    } else {
        score -= 0.15;
    }

    let wg_target = caps.recommended_workgroup();
    score += closeness(choice.wg, wg_target) * 0.25;
    score += caps.occupancy_hint(choice.wg, None) * 0.10;

    let kl_target = caps.recommended_kl(k);
    score += closeness(choice.kl, kl_target) * 0.15;

    let mk_target = caps.preferred_merge_kind(k);
    if choice.mk == mk_target {
        score += 0.25;
    } else {
        score -= 0.10;
    }
    let mkd_target = caps.preferred_substrategy(choice.mk, k);
    if choice.mkd == mkd_target {
        score += 0.12;
    }

    let tile_target = caps.recommended_sweep_tile(cols);
    score += closeness(choice.tile, tile_target) * 0.18;

    if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        let ct_target = caps.recommended_compaction_tile(cols);
        score += closeness(choice.ctile, ct_target) * 0.12;
    }

    if rows > 0 && choice.ch != 0 {
        let ch_target = caps.recommended_channel_stride(cols);
        score += closeness(choice.ch, ch_target) * 0.08;
    }

    score
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
    }
    // wg / kl / tile hints follow DeviceCaps helpers.
    v.push(SoftRule {
        field: Field::Wg,
        value: Value::U(caps.recommended_workgroup()),
        weight: 0.12,
    });
    v.push(SoftRule {
        field: Field::Kl,
        value: Value::U(caps.recommended_kl(k)),
        weight: 0.10,
    });
    v.push(SoftRule {
        field: Field::Tile,
        value: Value::U(caps.recommended_sweep_tile(cols)),
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
        v.push(SoftRule {
            field: Field::Ctile,
            value: Value::U(preferred_ctile),
            value: Value::U(caps.recommended_compaction_tile(cols)),
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

fn clamp_workgroup(wg: u32, caps: &DeviceCaps) -> u32 {
    let min = if caps.subgroup {
        caps.lane_width.max(32)
    } else {
        32
    };
    let max = caps.max_workgroup.max(min);
    wg.clamp(min, max)
}

fn align_workgroup(wg: u32, caps: &DeviceCaps) -> u32 {
    if !caps.subgroup {
        return wg;
    }
    let lane = caps.lane_width.max(1);
    let max = caps.max_workgroup.max(lane);
    let wg = wg.clamp(lane, max);
    let rem = wg % lane;
    if rem == 0 {
        return wg;
    }
    let down = (wg - rem).clamp(lane, max);
    let up = (wg + (lane - rem)).clamp(lane, max);
    if wg - down <= up - wg {
        down
    } else {
        up
    }
}

fn uses_shared_memory(choice: &Choice) -> bool {
    choice.mk == 1 || matches!(choice.mkd, 1 | 2)
}

fn estimate_shared_mem_bytes(choice: &Choice, caps: &DeviceCaps) -> u64 {
    if !uses_shared_memory(choice) {
        return 0;
    }
    let lane = caps.lane_width.max(1);
    let wg = choice.wg.max(lane);
    let groups = (wg + lane - 1) / lane;
    let lanes_total = groups * lane;
    let kl = choice.kl.max(1);
    let per_lane = kl.saturating_mul(8);
    (lanes_total as u64) * (per_lane as u64)
}

fn enforce_shared_memory(mut choice: Choice, caps: &DeviceCaps) -> Choice {
    if let Some(limit) = caps.shared_mem_per_workgroup {
        if uses_shared_memory(&choice) {
            let mut wg = align_workgroup(choice.wg, caps);
            let mut kl = choice.kl.max(8);
            let lane = caps.lane_width.max(1);
            let limit = limit as u64;
            loop {
                let mut candidate = choice;
                candidate.wg = wg;
                candidate.kl = kl;
                let need = estimate_shared_mem_bytes(&candidate, caps);
                if need <= limit {
                    choice = candidate;
                    break;
                }
                if wg > lane {
                    let next = wg.saturating_sub(lane);
                    if next >= lane {
                        wg = align_workgroup(next, caps);
                        continue;
                    }
                }
                if kl > 8 {
                    kl = (kl / 2).max(8);
                    continue;
                }
                choice.mk = 0;
                choice.mkd = 3;
                break;
            }
        }
    }
    choice
}

fn canonicalize_merge(mut choice: Choice, caps: &DeviceCaps, k: u32) -> Choice {
    if choice.mk > 2 {
        choice.mk = caps.preferred_merge_kind(k);
    }
    if choice.mk == 2 && !caps.subgroup {
        choice.mk = caps.preferred_merge_kind(k);
    }
    if choice.mk == 1 && matches!(caps.backend, BackendKind::Cpu) {
        choice.mk = caps.preferred_merge_kind(k);
    }
    choice.mkd = match choice.mk {
        2 => {
            if matches!(choice.mkd, 4 | 5) {
                choice.mkd
            } else {
                caps.preferred_substrategy(2, k)
            }
        }
        1 => {
            if matches!(choice.mkd, 1 | 2) {
                choice.mkd
            } else {
                caps.preferred_substrategy(1, k)
            }
        }
        _ => 3,
    };
    choice
}

fn finalize_choice(
    cols: u32,
    k: u32,
    caps: &DeviceCaps,
    kind: RankKind,
    mut choice: Choice,
) -> Choice {
    if choice.wg == 0 {
        choice.wg = default_wg(caps);
    }
    choice.wg = clamp_workgroup(choice.wg, caps);
    choice.wg = align_workgroup(choice.wg, caps);
    if choice.kl == 0 {
        choice.kl = DeviceCaps::default_lane_quota(k);
    }
    if choice.kl < 8 {
        choice.kl = 8;
    }
    choice = canonicalize_merge(choice, caps, k);
    choice = enforce_shared_memory(choice, caps);
    if !caps.prefers_two_stage(cols, k) {
        choice.use_2ce = false;
    }
    if choice.tile == 0 {
        choice.tile = default_tile(cols);
    }
    let max_tile = cols.max(256).min(8192);
    choice.tile = choice.tile.clamp(256, max_tile);
    if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        if choice.ctile == 0 {
            choice.ctile = default_compaction_tile(cols);
        }
        let max_ctile = cols.max(256).min(2048);
        choice.ctile = choice.ctile.clamp(256, max_ctile);
    } else {
        // For TopK we only use ctile as a hint. Keep sane bounds if provided.
        if choice.ctile != 0 {
            let max_ctile = cols.max(256).min(4096);
            choice.ctile = choice.ctile.clamp(256, max_ctile);
        }
    }
    if caps.backend == BackendKind::Cpu {
        choice.tile = choice.tile.min(1024);
        if matches!(kind, RankKind::MidK | RankKind::BottomK) {
            choice.ctile = choice.ctile.min(512);
        }
    }
    choice
}

fn score_candidate(
    choice: &Choice,
    baseline: &Choice,
    cols: u32,
    k: u32,
    caps: &DeviceCaps,
    kind: RankKind,
) -> f32 {
    let mut score = 0.0f32;
    let two_stage_pref = caps.prefers_two_stage(cols, k);
    if choice.use_2ce == two_stage_pref {
        score += 0.4;
    } else if choice.use_2ce && !two_stage_pref {
        score -= 0.5;
    } else {
        score -= 0.2;
    }
    let expected_mk = caps.preferred_merge_kind(k);
    if choice.mk == expected_mk {
        score += 0.6;
    } else if (choice.mk as i32 - expected_mk as i32).abs() <= 1 {
        score += 0.2;
    } else {
        score -= 0.3;
    }
    let expected_mkd = caps.preferred_substrategy(choice.mk, k);
    if choice.mkd == expected_mkd {
        score += 0.3;
    } else {
        score -= 0.1;
    }
    if caps.subgroup && choice.wg % caps.lane_width == 0 {
        score += 0.2;
    }
    let wg_diff = (choice.wg as i64 - baseline.wg as i64).abs() as f32;
    let wg_norm = caps.max_workgroup.max(1) as f32;
    score -= (wg_diff / wg_norm).min(1.0) * 0.3;
    let tile_diff = (choice.tile as i64 - baseline.tile as i64).abs() as f32;
    score -= (tile_diff / 8192.0).min(1.0) * 0.2;
    if matches!(kind, RankKind::MidK | RankKind::BottomK) {
        let ctile_diff = (choice.ctile as i64 - baseline.ctile as i64).abs() as f32;
        score -= (ctile_diff / 2048.0).min(1.0) * 0.1;
    }
    if choice.mk == 2 && !caps.subgroup {
        score -= 0.5;
    }
    score
}

/// Unified chooser (Rank‑K). Prefer this as the single entry for heuristics.
pub fn choose_unified_rank(
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
    kind: RankKind,
) -> Choice {
    let baseline = fallback(rows, cols, k, &caps, kind);
    // 1) Gather DSL hard + soft
    #[allow(unused_mut)]
    let (dsl_hard, mut soft_rules) = {
        let (hard_opt, soft_dsl) = kdsl_bridge::parse_env_dsl(rows, cols, k, caps.subgroup);
        #[cfg(not(feature = "logic"))]
        let _ = &soft_dsl;
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
        #[allow(unused_mut)]
        let mut sr: Vec<kdsl_bridge::SoftRule> = Vec::new();
        let sr: Vec<kdsl_bridge::SoftRule> = Vec::new();
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
    #[cfg(not(feature = "logic"))]
    let _ = &soft_rules;

    #[cfg(not(feature = "logic"))]
    let _ = &soft_rules;

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
    let cand_a = baseline;

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
            let mut c = baseline;
            c.use_2ce = base.use_2ce;
            c.wg = base.wg;
            c.kl = base.kl;
            c.ch = base.ch;
            c
        }
        _ => fallback(rows, cols, k, &caps, kind),
        _ => baseline,
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

    // 5) Consensus: score both candidates & bias generated table via env weight.
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
    let score_b = score_choice(&cand_b, &caps, rows, cols, k, kind);
    let mut score_c = score_choice(&cand_c, &caps, rows, cols, k, kind);
    score_c += gen_bias;
    if score_c > score_b {
        cand_c
    } else {
        cand_b
    }
    let baseline_final = finalize_choice(cols, k, &caps, kind, baseline);
    let cand_a_final = finalize_choice(cols, k, &caps, kind, cand_a);
    let cand_b_final = finalize_choice(cols, k, &caps, kind, cand_b);
    let cand_c_final = finalize_choice(cols, k, &caps, kind, cand_c);

    let gen_bias = gen_weight_for_backend(caps.backend).max(0.0);
    let mut best_choice = baseline_final;
    let mut best_score = f32::MIN;
    let mut consider = |label: &'static str, choice: Choice, weight: f32| {
        let mut score = weight + score_candidate(&choice, &baseline_final, cols, k, &caps, kind);
        if label == "dsl" {
            score += 0.3;
        }
        if score > best_score {
            best_score = score;
            best_choice = choice;
        }
    };
    consider(
        if dsl_hard.is_some() { "dsl" } else { "soft" },
        cand_b_final,
        if dsl_hard.is_some() { 1.2 } else { 0.9 },
    );
    consider("generated", cand_c_final, gen_bias + 0.5);
    consider(
        if cfg!(feature = "logic") {
            "logic"
        } else {
            "fallback"
        },
        cand_a_final,
        if cfg!(feature = "logic") { 0.8 } else { 0.6 },
    );
    consider("baseline", baseline_final, 0.4);
    best_choice
}

/// Backward compat alias for TopK
pub fn choose_unified(rows: u32, cols: u32, k: u32, caps: DeviceCaps) -> Choice {
    choose_unified_rank(rows, cols, k, caps, RankKind::TopK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_prefers_backend_guided_choice() {
        let caps = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        let good = super::fallback(4096, 65_536, 512, &caps, RankKind::TopK);
        let mut bad = good;
        bad.wg = (bad.wg / 2).max(32);
        bad.mk = 0;
        bad.mkd = 3;
        bad.tile = bad.tile / 2;
        let good_score = super::score_choice(&good, &caps, 4096, 65_536, 512, RankKind::TopK);
        let bad_score = super::score_choice(&bad, &caps, 4096, 65_536, 512, RankKind::TopK);
        assert!(good_score > bad_score);
    fn finalize_respects_shared_memory_limits() {
        let caps = DeviceCaps::cuda(32, 1024, Some(24 * 1024));
        let choice = Choice {
            use_2ce: true,
            wg: 512,
            kl: 32,
            ch: 0,
            mk: 1,
            mkd: 1,
            tile: 4096,
            ctile: 1024,
        };
        let refined = finalize_choice(120_000, 256, &caps, RankKind::TopK, choice);
        assert!(!refined.use_2ce);
        assert!(refined.wg <= caps.max_workgroup);
        if uses_shared_memory(&refined) {
            assert!(
                estimate_shared_mem_bytes(&refined, &caps)
                    <= caps.shared_mem_per_workgroup.unwrap() as u64
            );
        }
    }

    #[test]
    fn finalize_canonicalizes_merge_strategy() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let choice = Choice {
            use_2ce: false,
            wg: 300,
            kl: 32,
            ch: 0,
            mk: 2,
            mkd: 1,
            tile: 512,
            ctile: 0,
        };
        let refined = finalize_choice(20_000, 64, &caps, RankKind::TopK, choice);
        assert_eq!(refined.wg % caps.lane_width, 0);
        assert_eq!(refined.mkd, 4);
    }

    #[test]
    fn cpu_backend_dials_back_gpu_biases() {
        let caps = DeviceCaps::cpu();
        let choice = Choice {
            use_2ce: true,
            wg: 1024,
            kl: 4,
            ch: 0,
            mk: 1,
            mkd: 1,
            tile: 8192,
            ctile: 2048,
        };
        let refined = finalize_choice(10_000, 32, &caps, RankKind::TopK, choice);
        assert_eq!(refined.mk, 0);
        assert!(refined.tile <= 1024);
        assert!(!refined.use_2ce);
    }

    #[test]
    fn unified_rank_disables_two_stage_if_budget_missing() {
        let caps = DeviceCaps::cuda(32, 1024, Some(24 * 1024));
        let out = choose_unified_rank(4096, 100_000, 256, caps, RankKind::TopK);
        assert!(!out.use_2ce);
    }
}
