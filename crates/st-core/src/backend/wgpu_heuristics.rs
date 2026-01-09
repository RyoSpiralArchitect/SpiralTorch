// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::consensus;
use super::device_caps::DeviceCaps;
use super::kdsl_bridge;
#[cfg(feature = "logic-learn")]
use super::soft_logic::learn;
#[cfg(feature = "logic")]
use super::soft_logic::SoftRule;
use super::spiralk_fft::SpiralKFftPlan;
use crate::backend::wgpu_heuristics_generated as gen;
use crate::ecosystem::{
    EcosystemRegistry, HeuristicChoiceSummary, HeuristicDecision, HeuristicSource, MetricSample,
};
#[cfg(feature = "logic-learn")]
use std::sync::Mutex;
use std::time::SystemTime;

#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub kl: u32,
    pub ch: u32,
    pub algo_topk: u8,    // 0=auto, 1=heap, 2=bitonic, 3=kway
    pub ctile: u32,       // 0=auto
    pub mode_midk: u8,    // 0=auto, 1=1CE, 2=2CE
    pub mode_bottomk: u8, // 0=auto, 1=1CE, 2=2CE
    pub tile_cols: u32,   // column tiles for ND FFT/fractional kernels
    pub radix: u32,       // preferred FFT radix
    pub segments: u32,    // ND segment count for GPU kernels
}

#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_USE2CE: &str = "use_2ce";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_WG: &str = "wg";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_KL: &str = "kl";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_CH: &str = "ch";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_ALGO: &str = "algo_topk";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_CTILE: &str = "ctile";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_MODE_MIDK: &str = "mode_midk";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_MODE_BOTTOMK: &str = "mode_bottomk";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_TILE_COLS: &str = "tile_cols";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_RADIX: &str = "radix";
#[cfg_attr(not(feature = "logic"), allow(dead_code))]
pub(crate) const SOFT_NAME_SEGMENTS: &str = "segments";

#[cfg(feature = "logic")]
#[derive(Default)]
struct NumericVote {
    sum: f32,
    weight: f32,
}

#[cfg(feature = "logic")]
impl NumericVote {
    fn push(&mut self, value: f32, weight: f32) {
        if weight > 0.0 {
            self.sum += value * weight;
            self.weight += weight;
        }
    }

    fn resolve(&self) -> Option<f32> {
        (self.weight > 0.0).then(|| self.sum / self.weight)
    }

    fn influence(&self) -> f32 {
        self.weight
    }
}

#[cfg(feature = "logic")]
#[derive(Default)]
struct BoolVote {
    positive: f32,
    negative: f32,
}

#[cfg(feature = "logic")]
impl BoolVote {
    fn push(&mut self, prefer_true: bool, weight: f32) {
        if weight <= 0.0 {
            return;
        }
        if prefer_true {
            self.positive += weight;
        } else {
            self.negative += weight;
        }
    }

    fn decide(&self) -> Option<bool> {
        let total = self.positive + self.negative;
        if total <= 0.0 {
            None
        } else {
            Some(self.positive >= self.negative)
        }
    }

    fn influence(&self) -> f32 {
        self.positive + self.negative
    }
}

fn fallback(rows: u32, cols: u32, k: u32, subgroup: bool) -> Choice {
    let max_wg = if subgroup { 256 } else { 128 };
    let caps = DeviceCaps::wgpu(32, subgroup, max_wg);
    let use_2ce = caps.prefers_two_stage_with_rows(rows, cols, k);
    let wg = caps.recommended_workgroup(rows);
    let kl = caps.recommended_kl(k);
    let ch = caps.recommended_channel_stride(cols);
    let ctile = caps.recommended_compaction_tile_default(cols);
    Choice {
        use_2ce,
        wg,
        kl,
        ch,
        algo_topk: 0,
        ctile,
        mode_midk: 0,
        mode_bottomk: 0,
        tile_cols: cols.max(1).div_ceil(1024) * 1024,
        radix: if k.is_power_of_two() { 4 } else { 2 },
        segments: if cols > 131_072 {
            4
        } else if cols > 32_768 {
            2
        } else {
            1
        },
    }
}

pub fn choose_topk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "topk")
}
pub fn choose_midk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "midk")
}
pub fn choose_bottomk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    choose_kind(rows, cols, k, subgroup, "bottomk")
}

#[derive(Default, Clone, Copy)]
pub struct DslOverrides {
    pub algo_topk: u8,
    pub ctile: u32,
    pub mode_midk: u8,
    pub mode_bottomk: u8,
    pub tile_cols: u32,
    pub radix: u32,
    pub segments: u32,
}
fn overlay(c: &mut Choice, o: &DslOverrides) {
    if o.algo_topk != 0 {
        c.algo_topk = o.algo_topk;
    }
    if o.ctile != 0 {
        c.ctile = o.ctile;
    }
    if o.mode_midk != 0 {
        c.mode_midk = o.mode_midk;
    }
    if o.mode_bottomk != 0 {
        c.mode_bottomk = o.mode_bottomk;
    }
    if o.tile_cols != 0 {
        c.tile_cols = o.tile_cols;
    }
    if o.radix != 0 {
        c.radix = o.radix;
    }
    if o.segments != 0 {
        c.segments = o.segments;
    }
}

fn describe_topk_algo(algo: u8) -> &'static str {
    match algo {
        1 => "heap",
        2 => "bitonic",
        3 => "kway",
        _ => "auto",
    }
}

fn describe_midbottom_mode(mode: u8) -> &'static str {
    match mode {
        1 => "1ce",
        2 => "2ce",
        _ => "auto",
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "Heuristic finalizer mirrors existing call signature for staged rollout"
)]
fn finalize_choice(
    kind: &'static str,
    rows: u32,
    cols: u32,
    k: u32,
    mut choice: Choice,
    source: HeuristicSource,
    score_hint: Option<f32>,
    overrides: &DslOverrides,
) -> Option<Choice> {
    overlay(&mut choice, overrides);
    let algo_hint = match kind {
        "topk" => Some(format!("algo={}", describe_topk_algo(choice.algo_topk))),
        "midk" => Some(format!(
            "mode={}",
            describe_midbottom_mode(choice.mode_midk)
        )),
        "bottomk" => Some(format!(
            "mode={}",
            describe_midbottom_mode(choice.mode_bottomk)
        )),
        _ => None,
    };
    let summary = HeuristicChoiceSummary::new(
        choice.use_2ce,
        choice.wg,
        choice.kl,
        choice.ch,
        algo_hint,
        choice.ctile,
        choice.tile_cols,
        choice.radix,
        choice.segments,
    );
    let decision = HeuristicDecision {
        subsystem: "wgpu".to_string(),
        kind: kind.to_string(),
        rows,
        cols,
        k,
        choice: summary,
        score_hint,
        source,
        issued_at: SystemTime::now(),
    };
    let registry = EcosystemRegistry::global();
    let tag_sample = |sample: MetricSample| -> MetricSample {
        sample
            .with_tag("subsystem", "wgpu")
            .with_tag("kind", kind)
            .with_tag("source", source.as_str())
    };

    registry.record_metric(tag_sample(
        MetricSample::new("heuristic.rows", rows as f64).with_unit("rows"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new("heuristic.cols", cols as f64).with_unit("cols"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new("heuristic.k", k as f64).with_unit("items"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new(
            "heuristic.use_two_stage",
            if decision.choice.use_two_stage {
                1.0
            } else {
                0.0
            },
        )
        .with_unit("flag"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new("heuristic.workgroup", decision.choice.workgroup as f64)
            .with_unit("threads"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new("heuristic.lanes", decision.choice.lanes as f64).with_unit("lanes"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new(
            "heuristic.channel_stride",
            decision.choice.channel_stride as f64,
        )
        .with_unit("stride"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new(
            "heuristic.compaction_tile",
            decision.choice.compaction_tile as f64,
        )
        .with_unit("tile"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new(
            "heuristic.fft_tile_cols",
            decision.choice.fft_tile_cols as f64,
        )
        .with_unit("cols"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new("heuristic.fft_radix", decision.choice.fft_radix as f64)
            .with_unit("radix"),
    ));
    registry.record_metric(tag_sample(
        MetricSample::new(
            "heuristic.fft_segments",
            decision.choice.fft_segments as f64,
        )
        .with_unit("segments"),
    ));
    if let Some(score) = score_hint {
        registry.record_metric(tag_sample(
            MetricSample::new("heuristic.score_hint", score as f64).with_unit("score"),
        ));
    }

    registry.record_heuristic(decision);
    Some(choice)
}

#[cfg(feature = "logic")]
fn synthesize_soft_choice(base: Choice, rules: &[SoftRule]) -> Option<(Choice, f32)> {
    if rules.is_empty() {
        return None;
    }

    let mut choice = base;
    let mut influence = 0.0f32;

    let mut use2_vote = BoolVote::default();
    let mut wg_vote = NumericVote::default();
    let mut kl_vote = NumericVote::default();
    let mut ch_vote = NumericVote::default();
    let mut algo_vote = NumericVote::default();
    let mut ctile_vote = NumericVote::default();
    let mut midk_vote = NumericVote::default();
    let mut bottomk_vote = NumericVote::default();
    let mut tile_cols_vote = NumericVote::default();
    let mut radix_vote = NumericVote::default();
    let mut segments_vote = NumericVote::default();

    for rule in rules {
        let mut weight = rule.weight.max(0.0);
        let mut score = rule.score;
        #[cfg(feature = "logic-learn")]
        {
            let (w, s) = adjust_soft_rule(rule.name, weight, score);
            weight = w;
            score = s;
        }
        if weight <= 0.0 {
            continue;
        }
        match rule.name {
            SOFT_NAME_USE2CE => use2_vote.push(score >= 0.0, weight),
            SOFT_NAME_WG => wg_vote.push(score, weight),
            SOFT_NAME_KL => kl_vote.push(score, weight),
            SOFT_NAME_CH => ch_vote.push(score, weight),
            SOFT_NAME_ALGO => algo_vote.push(score, weight),
            SOFT_NAME_CTILE => ctile_vote.push(score, weight),
            SOFT_NAME_MODE_MIDK => midk_vote.push(score, weight),
            SOFT_NAME_MODE_BOTTOMK => bottomk_vote.push(score, weight),
            SOFT_NAME_TILE_COLS => tile_cols_vote.push(score, weight),
            SOFT_NAME_RADIX => radix_vote.push(score, weight),
            SOFT_NAME_SEGMENTS => segments_vote.push(score, weight),
            _ => {}
        }
    }

    if let Some(decision) = use2_vote.decide() {
        if decision != choice.use_2ce {
            influence += use2_vote.influence();
            choice.use_2ce = decision;
        }
    }

    influence += apply_numeric_vote_u32(&mut choice.wg, &wg_vote, 1, u32::MAX);
    influence += apply_numeric_vote_u32(&mut choice.kl, &kl_vote, 1, u32::MAX);
    influence += apply_numeric_vote_u32(&mut choice.ch, &ch_vote, 1, u32::MAX);
    influence += apply_numeric_vote_u8(&mut choice.algo_topk, &algo_vote, 0, u8::MAX);
    influence += apply_numeric_vote_u32(&mut choice.ctile, &ctile_vote, 1, u32::MAX);
    influence += apply_numeric_vote_u8(&mut choice.mode_midk, &midk_vote, 0, u8::MAX);
    influence += apply_numeric_vote_u8(&mut choice.mode_bottomk, &bottomk_vote, 0, u8::MAX);
    influence += apply_numeric_vote_u32(&mut choice.tile_cols, &tile_cols_vote, 1, u32::MAX);
    influence += apply_numeric_vote_u32(&mut choice.radix, &radix_vote, 1, u32::MAX);
    influence += apply_numeric_vote_u32(&mut choice.segments, &segments_vote, 1, u32::MAX);

    if influence <= 0.0 {
        return None;
    }

    let score = (influence / (1.0 + influence)).clamp(0.0, 1.0);
    Some((choice, score))
}

#[cfg(feature = "logic")]
fn apply_numeric_vote_u32(field: &mut u32, vote: &NumericVote, min: u32, max: u32) -> f32 {
    if let Some(value) = vote.resolve() {
        let clamped = value.clamp(min as f32, max as f32).round() as u32;
        if clamped != *field {
            *field = clamped;
            return vote.influence();
        }
    }
    0.0
}

#[cfg(feature = "logic")]
fn apply_numeric_vote_u8(field: &mut u8, vote: &NumericVote, min: u8, max: u8) -> f32 {
    if let Some(value) = vote.resolve() {
        let clamped = value.clamp(min as f32, max as f32).round() as u8;
        if clamped != *field {
            *field = clamped;
            return vote.influence();
        }
    }
    0.0
}

#[cfg(feature = "logic-learn")]
fn adjust_soft_rule(name: &str, weight: f32, score: f32) -> (f32, f32) {
    let mut adj_weight = weight;
    let mut adj_score = score;
    if let Ok(weights) = soft_weights_store().lock() {
        let blend = bandit_blend_factor();
        let bandit = learn::weight_from_bandit(&*weights, name);
        let baseline = 0.5f32;
        let target = blend * bandit + (1.0 - blend) * baseline;
        if weight > 0.0 {
            let scale = (target / baseline.max(1e-6)).clamp(0.25, 4.0);
            adj_weight = weight * scale;
        }
        if let Some(bias) = weights.base_coef.get(name) {
            let bias = bias.clamp(-4.0, 4.0);
            adj_score = score + bias;
        }
    }
    (adj_weight, adj_score)
}

#[cfg(feature = "logic-learn")]
fn bandit_blend_factor() -> f32 {
    static BLEND: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *BLEND.get_or_init(|| {
        std::env::var("SPIRAL_SOFT_BANDIT_BLEND")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(0.5)
    })
}

#[cfg(feature = "logic-learn")]
fn soft_weights_store() -> &'static Mutex<learn::SoftWeights> {
    static STORE: std::sync::OnceLock<Mutex<learn::SoftWeights>> = std::sync::OnceLock::new();
    STORE.get_or_init(|| Mutex::new(learn::load()))
}

pub fn choose_kind(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    kind: &'static str,
) -> Option<Choice> {
    let (hard_dsl, soft_dsl, ov) =
        kdsl_bridge::parse_env_dsl_plus_kind(rows, cols, k, subgroup, kind);
    let soft_kv = consensus::kv_consensus_soft_rules(rows, cols, k, subgroup, kind);
    let base = fallback(rows, cols, k, subgroup);
    #[cfg(not(feature = "logic"))]
    let _ = (&soft_dsl, &soft_kv, &base);

    #[cfg(feature = "logic")]
    {
        let mut all = soft_dsl.clone();
        all.extend(soft_kv.clone());
        let use_soft = std::env::var("SPIRAL_HEUR_SOFT")
            .ok()
            .map(|v| v == "1")
            .unwrap_or(true);
        if use_soft {
            if let Some((out, score)) = synthesize_soft_choice(base, &all) {
                if score > 0.1 {
                    return finalize_choice(
                        kind,
                        rows,
                        cols,
                        k,
                        out,
                        HeuristicSource::SoftLogic,
                        Some(score),
                        &ov,
                    );
                }
            }
        }
    }
    if let Some(c) = hard_dsl {
        return finalize_choice(kind, rows, cols, k, c, HeuristicSource::HardDsl, None, &ov);
    }
    if let Some(c) = kdsl_bridge::choose_from_kv(rows, cols, k, subgroup) {
        return finalize_choice(kind, rows, cols, k, c, HeuristicSource::KeyValue, None, &ov);
    }
    if let Some(c) = gen::choose(rows as usize, cols as usize, k as usize, subgroup) {
        return finalize_choice(
            kind,
            rows,
            cols,
            k,
            c,
            HeuristicSource::Generated,
            None,
            &ov,
        );
    }
    let fallback_choice = base;
    finalize_choice(
        kind,
        rows,
        cols,
        k,
        fallback_choice,
        HeuristicSource::Fallback,
        None,
        &ov,
    )
}

/// Construct an FFT plan from the same heuristics used for TopK and emit the
/// auto-generated WGSL shader.  Returns `None` if the heuristic pipeline could
/// not find a suitable `Choice`.
pub fn auto_fft_wgsl(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<String> {
    let plan = auto_fft_plan(rows, cols, k, subgroup)?;
    Some(plan.emit_wgsl())
}

/// Produce the SpiralK hint snippet associated with the automatically emitted
/// WGSL kernel.
pub fn auto_fft_spiralk(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<String> {
    let plan = auto_fft_plan(rows, cols, k, subgroup)?;
    Some(plan.emit_spiralk_hint())
}

/// Internal helper that assembles the [`SpiralKFftPlan`] from the heuristic
/// `Choice`.
fn auto_fft_plan(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<SpiralKFftPlan> {
    let choice = choose_topk(rows, cols, k, subgroup)?;
    Some(SpiralKFftPlan::from_choice(&choice, subgroup))
}

include!("wgpu_heuristics_generated.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_fft_kernel_and_hint() {
        let wgsl = auto_fft_wgsl(512, 4096, 128, true).expect("kernel expected");
        assert!(wgsl.contains("@workgroup_size"));
        let hint = auto_fft_spiralk(512, 4096, 128, true).unwrap();
        assert!(hint.contains("tile_cols"));
    }
}
