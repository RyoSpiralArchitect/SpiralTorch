//! Backend aware consensus chooser for TopK/MidK/BottomK kernels.
//!
//! The module blends fallback heuristics, generated tables, and optional DSL/soft
//! rules before settling on a candidate that respects device limits.  The code
//! used to be a maze of partially duplicated logic; this rewrite folds the
//! scoring into small helpers so we can plug additional learned signals later on
//! without breaking basic builds.

use super::device_caps::{BackendKind, DeviceCaps};
use super::wgpu_heuristics;

#[cfg(feature = "logic")]
use st_logic::{solve_soft, Ctx as LCtx, SoftRule, SolveCfg as LCfg};

#[cfg(feature = "kv-redis")]
use crate::ability::unison_mediator;
#[cfg(feature = "logic")]
use crate::backend::kdsl_bridge;
#[cfg(feature = "logic")]
use crate::heur::soft_learning::{SoftContext, SoftRuleLearner};

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
    pub mk: u32,
    pub mkd: u32,
    pub tile: u32,
    pub ctile: u32,
}

impl Choice {
    fn new() -> Self {
        Self {
            use_2ce: false,
            wg: 128,
            kl: 8,
            ch: 0,
            mk: 0,
            mkd: 3,
            tile: 256,
            ctile: 256,
        }
    }
}

fn fallback(rows: u32, cols: u32, k: u32, caps: &DeviceCaps, kind: RankKind) -> Choice {
    let mut choice = Choice::new();
    choice.wg = caps.recommended_workgroup_for_rows(rows);
    choice.kl = caps.preferred_k_loop(k);
    choice.ch = caps.preferred_channel(cols);
    let (tile, mut ctile) = caps.recommended_tiles(cols);
    choice.tile = tile;

    if matches!(kind, RankKind::BottomK) {
        ctile = ctile.min(tile / 2).max(128);
    }
    choice.ctile = ctile.min(tile);

    choice.mk = caps.preferred_merge_kind(k);
    choice.mkd = caps.preferred_substrategy(choice.mk, k);
    choice.use_2ce = caps.prefers_two_stage(rows, cols, k);
    choice
}

fn closeness(actual: u32, target: u32) -> f32 {
    if actual == 0 || target == 0 {
        return 0.0;
    }
    let diff = actual.abs_diff(target) as f32;
    let denom = target.max(1) as f32;
    (1.0 - (diff / denom).min(1.0)).max(0.0)
}

fn score_choice(choice: &Choice, caps: &DeviceCaps, rows: u32, cols: u32, k: u32) -> f32 {
    let mut score = 0.0f32;

    let expected_two_stage = caps.prefers_two_stage(rows, cols, k);
    if choice.use_2ce == expected_two_stage {
        score += 0.35;
    } else {
        score -= 0.15;
    }

    let wg_target = caps.recommended_workgroup();
    score += closeness(choice.wg, wg_target) * 0.25;

    let occ = caps.occupancy_hint(choice.wg, None);
    score += occ * 0.10;

    let kl_target = caps.recommended_kl(k);
    score += closeness(choice.kl, kl_target) * 0.15;

    let mk_target = caps.preferred_merge_kind(k);
    if choice.mk == mk_target {
        score += 0.20;
    } else {
        score -= 0.10;
    }

    let mkd_target = caps.preferred_substrategy(choice.mk, k);
    if choice.mkd == mkd_target {
        score += 0.10;
    }

    let tile_target = caps.recommended_sweep_tile(cols);
    score += closeness(choice.tile, tile_target) * 0.18;

    let ct_target = caps.recommended_compaction_tile(cols);
    score += closeness(choice.ctile, ct_target) * 0.12;

    if choice.ch != 0 {
        let ch_target = caps.recommended_channel_stride(cols);
        score += closeness(choice.ch, ch_target) * 0.08;
    }

    score
}

fn gen_weight_for_backend(kind: BackendKind) -> f32 {
    let key = match kind {
        BackendKind::Wgpu => "SPIRAL_HEUR_GEN_WEIGHT_WGPU",
        BackendKind::Cuda => "SPIRAL_HEUR_GEN_WEIGHT_CUDA",
        BackendKind::Hip => "SPIRAL_HEUR_GEN_WEIGHT_HIP",
        BackendKind::Cpu => "SPIRAL_HEUR_GEN_WEIGHT_CPU",
    };
    if let Ok(v) = std::env::var(key) {
        if let Ok(parsed) = v.parse::<f32>() {
            return parsed;
        }
    }
    if let Ok(v) = std::env::var("SPIRAL_HEUR_GEN_WEIGHT") {
        if let Ok(parsed) = v.parse::<f32>() {
            return parsed;
        }
    }
    0.1
}

fn finalize_choice(choice: Choice, rows: u32, cols: u32, k: u32, caps: &DeviceCaps) -> Choice {
    let mut out = choice;
    let lanes = caps.lane_width.max(1);
    out.wg = out.wg.min(caps.max_workgroup).max(lanes);
    if out.wg % lanes != 0 {
        out.wg = ((out.wg + lanes - 1) / lanes) * lanes;
        out.wg = out.wg.min(caps.max_workgroup).max(lanes);
    }
    out.kl = out.kl.max(4);
    out.tile = out.tile.max(256);
    out.ctile = out.ctile.min(out.tile).max(128);

    if matches!(caps.backend, BackendKind::Cpu) {
        out.use_2ce = false;
        out.mk = 0;
        out.mkd = 3;
        out.ch = 0;
    }

    if !caps.prefers_two_stage(rows, cols, k) {
        out.use_2ce = false;
    }
    out
}

#[cfg(feature = "logic")]
fn solve_soft_rules(
    ctx: &LCtx,
    caps: &DeviceCaps,
    rows: u32,
    cols: u32,
    k: u32,
    soft_rules: &[SoftRule],
) -> Option<Choice> {
    if soft_rules.is_empty() {
        return None;
    }
    let use_soft = std::env::var("SPIRAL_HEUR_SOFT")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(true);
    if !use_soft {
        return None;
    }
    let (candidate, score) = solve_soft(
        ctx.clone(),
        LCfg {
            noise: 0.02,
            seed: 0x5p1ral,
        },
        soft_rules,
    );
    if score <= 0.0 {
        return None;
    }
    Some(Choice {
        use_2ce: candidate.use_2ce,
        wg: candidate.wg,
        kl: candidate.kl,
        ch: candidate.ch,
        mk: candidate.mk,
        mkd: candidate.mkd,
        tile: candidate.tile,
        ctile: candidate.ctile,
    })
}

#[cfg(feature = "logic")]
#[cfg(feature = "logic")]
fn apply_hard_overrides(base: Choice, hard: &wgpu_heuristics::Choice) -> Choice {
    let mut out = base;
    out.use_2ce = hard.use_2ce;
    if hard.wg != 0 {
        out.wg = hard.wg;
    }
    if hard.kl != 0 {
        out.kl = hard.kl;
    }
    if hard.ch != 0 {
        out.ch = hard.ch;
    }
    if hard.ctile != 0 {
        out.ctile = hard.ctile;
    }
    out
}

fn from_generated(
    choice: wgpu_heuristics::Choice,
    fallback: Choice,
    caps: &DeviceCaps,
    rows: u32,
    cols: u32,
    k: u32,
) -> Choice {
    let mut out = fallback;
    out.use_2ce = choice.use_2ce;
    out.wg = choice.wg;
    out.kl = choice.kl;
    out.ch = choice.ch;
    out.tile = choice.tile_cols.max(256);
    out.ctile = choice.ctile.max(128);

    // WGPU tables mostly target merge=warp.
    if caps.backend == BackendKind::Wgpu {
        out.mk = 2;
        out.mkd = 4;
    }

    finalize_choice(out, rows, cols, k, caps)
}

pub fn choose_unified_rank(
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
    kind: RankKind,
) -> Choice {
    let baseline = fallback(rows, cols, k, &caps, kind);
    let mut candidates: Vec<(&'static str, Choice, f32)> = Vec::new();
    candidates.push(("baseline", baseline, 0.0));

    #[cfg(feature = "logic")]
    let mut learner = SoftRuleLearner::maybe_load();

    #[cfg(feature = "logic")]
    let mut soft_rules: Vec<SoftRule> = Vec::new();

    #[cfg(feature = "logic")]
    let mut ctx = LCtx {
        rows,
        cols,
        k,
        sg: caps.subgroup,
    };

    #[cfg(feature = "logic")]
    {
        let kind_str = match kind {
            RankKind::TopK => "topk",
            RankKind::MidK => "midk",
            RankKind::BottomK => "bottomk",
        };
        let (hard, soft, _ov) =
            kdsl_bridge::parse_env_dsl_plus_kind(rows, cols, k, caps.subgroup, kind_str);
        soft_rules.extend(soft);
        if let Some(h) = hard {
            let mapped = apply_hard_overrides(baseline, &h);
            candidates.push(("dsl-hard", mapped, 0.2));
        }
    }

    #[cfg(feature = "kv-redis")]
    {
        let redis_soft = unison_mediator::soft_from_redis(rows, cols, k, caps.subgroup);
        #[cfg(feature = "logic")]
        soft_rules.extend(redis_soft);
    }

    if caps.backend == BackendKind::Wgpu {
        let gen_choice = match kind {
            RankKind::TopK => wgpu_heuristics::choose_topk(rows, cols, k, caps.subgroup),
            RankKind::MidK => wgpu_heuristics::choose_midk(rows, cols, k, caps.subgroup),
            RankKind::BottomK => wgpu_heuristics::choose_bottomk(rows, cols, k, caps.subgroup),
        };
        if let Some(gen) = gen_choice {
            let mapped = from_generated(gen, baseline, &caps, rows, cols, k);
            candidates.push(("generated", mapped, gen_weight_for_backend(caps.backend)));
        }
    }

    #[cfg(feature = "logic")]
    if let Some(soft_choice) = solve_soft_rules(&ctx, &caps, rows, cols, k, &soft_rules) {
        let mut bonus = 0.0;
        if let Some(ref mut learner) = learner {
            let sc = SoftContext::new(rows, cols, k, caps.backend, caps.subgroup);
            bonus = learner.learned_bonus(&sc, &soft_choice);
        }
        candidates.push(("soft", soft_choice, 0.15 + bonus));
    }

    if candidates.is_empty() {
        return baseline;
    }

    let mut scored: Vec<(&'static str, Choice, f32)> = Vec::new();
    for (label, cand, bias) in candidates {
        let candidate = finalize_choice(cand, rows, cols, k, &caps);
        let base = score_choice(&candidate, &caps, rows, cols, k) + bias;
        if std::env::var("SPIRAL_HEUR_TRACE").ok().as_deref() == Some("1") {
            eprintln!("[heur] {label} base={base:.3}");
        }
        scored.push((label, candidate, base));
    }

    #[cfg(feature = "logic")]
    if let Some(ref mut learner) = learner {
        let ctx = SoftContext::new(rows, cols, k, caps.backend, caps.subgroup);
        let (best_idx, final_scores) = learner.rank(&ctx, &scored);
        if std::env::var("SPIRAL_HEUR_TRACE").ok().as_deref() == Some("1") {
            for (i, (label, _, _)) in scored.iter().enumerate() {
                if let Some(score) = final_scores.get(i) {
                    eprintln!("[heur] {label} learned={score:.3}");
                }
            }
        }
        return scored[best_idx].1;
    }

    let mut best = scored[0].1;
    let mut best_score = scored[0].2;
    for (_, choice, score) in &scored {
        if *score > best_score {
            best_score = *score;
            best = *choice;
        }
    }
    best
}

/// Backward compat alias for TopK.
pub fn choose_unified(rows: u32, cols: u32, k: u32, caps: DeviceCaps) -> Choice {
    choose_unified_rank(rows, cols, k, caps, RankKind::TopK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_prefers_two_stage_on_column_pressure() {
        let caps = DeviceCaps::cuda(32, 1_024, Some(64 * 1_024));
        let out = fallback(1_024, 80_000, 256, &caps, RankKind::TopK);
        assert!(out.use_2ce);
    }

    #[test]
    fn finalize_clamps_to_device_limits() {
        let caps = DeviceCaps::cuda(32, 1_024, Some(64 * 1_024));
        let raw = Choice {
            use_2ce: true,
            wg: 4_096,
            kl: 2,
            ch: 0,
            mk: 2,
            mkd: 4,
            tile: 128,
            ctile: 8_192,
        };
        let refined = finalize_choice(raw, 4_096, 65_536, 128, &caps);
        assert!(refined.wg <= caps.max_workgroup);
        assert!(refined.kl >= 4);
        assert!(refined.ctile <= refined.tile);
    }

    #[test]
    fn scoring_prefers_merge_matches() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let good = fallback(4_096, 65_536, 256, &caps, RankKind::TopK);
        let mut bad = good;
        bad.mk = (bad.mk + 1) % 3;
        let good_score = score_choice(&good, &caps, 4_096, 65_536, 256);
        let bad_score = score_choice(&bad, &caps, 4_096, 65_536, 256);
        assert!(good_score > bad_score);
    }

    #[test]
    fn choose_unified_respects_baseline_when_alone() {
        let caps = DeviceCaps::cpu();
        let out = choose_unified_rank(1_024, 4_096, 64, caps, RankKind::TopK);
        assert_eq!(out.wg, 128);
        assert_eq!(out.tile, 256);
    }
}
