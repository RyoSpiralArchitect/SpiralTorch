// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-amg/src/backend/wgpu_heuristics_amg.rs
//! AMG heuristics overlay: SpiralK/generated table → SoftRule blend → base score.
//! Environment variables:
//!   SPIRAL_SOFT_MODE = {Sum|Normalize|Softmax|Prob}
//!   SPIRAL_BEAM_K = <usize>
//!   SPIRAL_SOFT_BANDIT_BLEND = <0..1>  (mixing weight for bandit feedback)
use crate::profile::{DensityClass, ProblemProfile};

use st_logic::{SoftMode, SolveCfg};

use st_tensor::fractional::gl_coeffs;

fn emphasize(hit: bool, on_hit: f32, on_miss: f32) -> f32 {
    if hit {
        on_hit
    } else {
        on_miss
    }
}

fn fractional_energy(alpha: f32) -> f32 {
    let coeffs = gl_coeffs(alpha, 32);
    let l1 = coeffs.iter().map(|c| c.abs()).sum::<f32>().max(1e-6);
    l1.ln()
}

#[cfg(feature = "learn_store")]
use st_logic::learn::{load, weight_from_bandit};

// Use your existing Choice/initial_choice/base_score/soft_rules_from_spiralk implementations.
#[derive(Clone, Debug)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg: u32,
    pub tile_cols: u32,
    pub jacobi_passes: u32,
    pub score: f32,
}

impl Choice {
    /// Provide a short human-readable rationale that can be surfaced in logs or debug UI.
    pub fn explain(&self, profile: &ProblemProfile) -> String {
        let mut reasons = Vec::new();
        reasons.push(format!(
            "density={:.4} ({:?})",
            profile.density(),
            profile.density_class()
        ));
        reasons.push(format!("wg {}→{}", profile.lane_hint(), self.wg));
        reasons.push(format!("tile {}→{}", profile.tile_hint(), self.tile_cols));
        reasons.push(format!("jacobi {}", self.jacobi_passes));
        if self.use_2ce {
            reasons.push("2CE enabled".to_string());
        }
        format!(
            "wgpu-amg choice: wg={} tile={} jacobi={} score={:.3} [{}]",
            self.wg,
            self.tile_cols,
            self.jacobi_passes,
            self.score,
            reasons.join(", ")
        )
    }
}

fn initial_choice(profile: &ProblemProfile) -> Choice {
    let wg = profile.lane_hint();
    let tile_cols = profile.tile_hint();
    let mut jacobi_passes = profile.jacobi_hint() as i32;
    let diag_ratio = profile.diag_ratio();
    if diag_ratio < 0.45 {
        jacobi_passes -= 1;
    } else if diag_ratio > 0.9 && profile.mean_row_nnz() > 48.0 {
        jacobi_passes += 1;
    }
    jacobi_passes = jacobi_passes.clamp(0, 5);

    let density_class = profile.density_class();
    let mut use_2ce = matches!(density_class, DensityClass::UltraSparse | DensityClass::Sparse)
        || profile.cols() >= 16_384;
    if diag_ratio > 0.88 && !matches!(density_class, DensityClass::UltraSparse) {
        use_2ce = false;
    }

    Choice {
        use_2ce,
        wg,
        tile_cols,
        jacobi_passes: jacobi_passes as u32,
        score: 0.0,
    }
}

// Placeholder base score (replace with project-specific version).
fn base_score_amg(c: &Choice, alpha: f32) -> f32 {
    let mut s = 0.0f32;
    if c.use_2ce {
        s += 0.12;
    } else {
        s -= 0.07;
    }

    let wg = c.wg.max(64) as f32;
    let wg_focus = (wg / 512.0).min(1.0);
    s += wg.log2() * 0.09 + wg_focus * 0.11;

    let tile = c.tile_cols.max(1024) as f32;
    let tile_gain = (tile / 8192.0).min(1.2);
    s += (tile.log2() / 14.0).min(0.22) + tile_gain * 0.05;

    let jacobi_penalty = match c.jacobi_passes {
        0 => -0.04,
        1 => 0.0,
        2 => -0.02,
        3 => -0.08,
        v => (v as f32 - 2.0) * -0.09,
    };
    s += jacobi_penalty;

    s += fractional_energy(alpha) * 0.06;
    s
}

// Placeholder SoftRule source (replace with project-specific SpiralK wiring).
fn soft_rules_from_spiralk(profile: &ProblemProfile) -> Vec<st_logic::SoftRule> {
    use st_logic::SoftRule;
    const WG128: &str = "wg=128";
    const WG256: &str = "wg=256";
    const WG512: &str = "wg=512";
    const SOFT_2CE: &str = "use-2ce";
    const TILE4K: &str = "tile=4k";
    const TILE8K: &str = "tile=8k";
    const TILE16K: &str = "tile=16k";
    const JACOBI1: &str = "jacobi=1";
    const JACOBI2: &str = "jacobi=2";
    const JACOBI3: &str = "jacobi=3";
    const JACOBI0: &str = "jacobi=0";

    let density = profile.density();
    let subgroup = profile.subgroup();
    let tile_hint = profile.tile_hint();
    let wg_hint = profile.lane_hint();
    let jacobi_hint = profile.jacobi_hint();

    let mut rules = Vec::with_capacity(11);

    let wg_bias = if subgroup { 0.35 } else { 0.55 };
    rules.push(SoftRule {
        name: WG128,
        weight: emphasize(wg_hint == 128, 0.57, 0.40 + (1.0 - density) * 0.2),
        score: 0.85,
    });
    rules.push(SoftRule {
        name: WG256,
        weight: emphasize(
            wg_hint == 256,
            0.55 + wg_bias,
            0.50 + (profile.bandwidth_hint() as f32 / 16_384.0).min(0.08),
        ),
        score: 1.35,
    });
    rules.push(SoftRule {
        name: WG512,
        weight: emphasize(
            wg_hint == 512,
            0.65 + (profile.bandwidth_hint() as f32 / 16_384.0).min(0.1),
            0.35 + (profile.bandwidth_hint() as f32 / 12_000.0).min(0.25),
        ),
        score: 1.10,
    });

    let diag_ratio = profile.diag_ratio();
    let two_ce_weight = if density < 0.18 {
        0.78 + (1.0 - diag_ratio).max(0.0) * 0.25
    } else if diag_ratio < 0.55 {
        0.60
    } else {
        (0.48 * (1.0 - (diag_ratio - 0.55).max(0.0) * 0.6)).clamp(0.18, 0.48)
    };
    rules.push(SoftRule {
        name: SOFT_2CE,
        weight: two_ce_weight,
        score: if density < 0.30 { 1.35 } else { 1.05 },
    });

    let tile_base = match tile_hint {
        2_048 => (0.74, 0.28),
        4_096 => (0.80, 0.30 + (profile.bandwidth_hint() as f32 / 8192.0) * 0.05),
        8_192 => (0.82, 0.32 + (profile.bandwidth_hint() as f32 / 10_000.0) * 0.05),
        _ => (0.86, 0.34 + (profile.bandwidth_hint() as f32 / 12_000.0) * 0.04),
    };
    rules.push(SoftRule {
        name: TILE4K,
        weight: emphasize(tile_hint == 4_096, tile_base.0, tile_base.1),
        score: 0.88,
    });
    rules.push(SoftRule {
        name: TILE8K,
        weight: emphasize(tile_hint == 8_192, tile_base.0, tile_base.1 + 0.02),
        score: 0.97,
    });
    rules.push(SoftRule {
        name: TILE16K,
        weight: emphasize(tile_hint == 16_384, tile_base.0 + 0.04, tile_base.1 + 0.04),
        score: 0.92,
    });

    rules.push(SoftRule {
        name: JACOBI0,
        weight: emphasize(jacobi_hint == 0, 0.65, 0.25),
        score: 0.55,
    });
    rules.push(SoftRule {
        name: JACOBI1,
        weight: emphasize(jacobi_hint == 1, 0.85, 0.50),
        score: 1.15,
    });
    rules.push(SoftRule {
        name: JACOBI2,
        weight: emphasize(jacobi_hint == 2, 0.70, 0.35),
        score: 0.95,
    });
    rules.push(SoftRule {
        name: JACOBI3,
        weight: emphasize(jacobi_hint >= 3, 0.55, 0.20),
        score: 0.75,
    });

    rules
}

fn instantiate_soft_rules(
    profile: &ProblemProfile,
    c: &Choice,
    base: &[st_logic::SoftRule],
) -> Vec<st_logic::SoftRule> {
    let mut out = Vec::with_capacity(base.len());
    let diag_ratio = profile.diag_ratio();
    let mean_row = profile.mean_row_nnz();
    for rule in base {
        let mut weight = rule.weight;
        let mut score = rule.score;
        match rule.name {
            "wg=128" => {
                weight *= alignment(c.wg, 128);
                score *= focus_gain(c.wg, 128);
            }
            "wg=256" => {
                weight *= alignment(c.wg, 256);
                score *= focus_gain(c.wg, 256);
            }
            "wg=512" => {
                weight *= alignment(c.wg, 512);
                score *= focus_gain(c.wg, 512);
            }
            "use-2ce" => {
                if c.use_2ce {
                    score *= 1.4;
                } else {
                    score *= -0.6;
                    weight *= 0.35;
                }
            }
            "tile=4k" => {
                weight *= alignment(c.tile_cols, 4_096);
                score *= focus_gain(c.tile_cols, 4_096);
            }
            "tile=8k" => {
                weight *= alignment(c.tile_cols, 8_192);
                score *= focus_gain(c.tile_cols, 8_192);
            }
            "tile=16k" => {
                weight *= alignment(c.tile_cols, 16_384);
                score *= focus_gain(c.tile_cols, 16_384);
            }
            "jacobi=0" => {
                if c.jacobi_passes == 0 {
                    score *= 1.2 + (0.5 - diag_ratio).max(0.0) * 0.4;
                    weight *= 1.1 + (0.5 - diag_ratio).max(0.0) * 0.3;
                } else {
                    weight *= 0.5;
                    score *= 0.3;
                }
            }
            "jacobi=1" => {
                if c.jacobi_passes == 1 {
                    score *= 1.3 + (0.6 - diag_ratio).max(0.0) * 0.2;
                } else {
                    weight *= 0.4;
                    score *= 0.4;
                }
            }
            "jacobi=2" => {
                if c.jacobi_passes == 2 {
                    score *= 1.1 + (diag_ratio - 0.6).max(0.0) * 0.2;
                    if mean_row > 40.0 {
                        weight *= 1.1;
                    }
                } else {
                    weight *= 0.3;
                    score *= 0.2;
                }
            }
            "jacobi=3" => {
                if c.jacobi_passes >= 3 {
                    score *= 1.05 + (diag_ratio - 0.7).max(0.0) * 0.3;
                    weight *= 1.05 + (diag_ratio - 0.7).max(0.0) * 0.2;
                } else {
                    weight *= 0.2;
                    score *= 0.2;
                }
            }
            _ => {}
        }
        out.push(st_logic::SoftRule {
            name: rule.name,
            weight,
            score,
        });
    }
    out
}

fn alignment(actual: u32, target: u32) -> f32 {
    let diff = (actual as i32 - target as i32).abs() as f32;
    let scale = target.max(1) as f32;
    (-(diff / scale).powi(2) * 4.0).exp().max(0.05)
}

fn focus_gain(actual: u32, target: u32) -> f32 {
    let diff = (actual as i32 - target as i32).abs() as f32;
    1.0 - (diff / target.max(1) as f32).min(0.9)
}

fn score_choice(
    profile: &ProblemProfile,
    c: &Choice,
    base_soft: &[st_logic::SoftRule],
    mode: SoftMode,
    alpha: f32,
) -> f32 {
    let dyn_rules = instantiate_soft_rules(profile, c, base_soft);
    let soft = st_logic::apply_softmode(&dyn_rules, mode);
    base_score_amg(c, alpha) + soft
}

fn neighbors_amg(c: &Choice) -> Vec<Choice> {
    let mut v = Vec::new();
    for &wg in &[128u32, 256, 512] {
        let mut n = c.clone();
        n.wg = wg;
        v.push(n);
    }
    for &tc in &[4096u32, 8192, 16384] {
        let mut n = c.clone();
        n.tile_cols = tc;
        v.push(n);
    }
    for &jp in &[1u32, 2] {
        let mut n = c.clone();
        n.jacobi_passes = jp;
        v.push(n);
    }
    {
        let mut n = c.clone();
        n.use_2ce = !c.use_2ce;
        v.push(n);
    }
    v
}

pub fn choose(rows: usize, cols: usize, nnz: usize, subgroup: bool) -> Choice {
    let profile = ProblemProfile::new(rows, cols, nnz, subgroup);
    choose_with_profile(&profile)
}

pub fn choose_with_profile(profile: &ProblemProfile) -> Choice {
    let soft_mode = match std::env::var("SPIRAL_SOFT_MODE").as_deref() {
        Ok("Normalize") => SoftMode::Normalize,
        Ok("Softmax") => SoftMode::Softmax,
        Ok("Prob") => SoftMode::Prob,
        _ => SoftMode::Sum,
    };
    let beam_k = std::env::var("SPIRAL_BEAM_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let cfg = SolveCfg {
        noise: 0.02,
        seed: 0x5u64,
        beam: beam_k,
        soft_mode: soft_mode,
    };

    #[allow(unused_mut)]
    let mut soft = soft_rules_from_spiralk(profile);

    // Blend in bandit weights when the feature is enabled.
    #[cfg(feature = "learn_store")]
    {
        let sw = load();
        let lambda = std::env::var("SPIRAL_SOFT_BANDIT_BLEND")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.35);
        for r in &mut soft {
            let bw = weight_from_bandit(&sw, r.name);
            r.weight = (1.0 - lambda) * r.weight + lambda * bw;
        }
    }

    // Single pass scoring; enable beam search via beam_select when desired.
    let alpha = profile.fractional_alpha();
    let mut choice = initial_choice(profile);
    let score = score_choice(profile, &choice, &soft, cfg.soft_mode, alpha);
    choice.score = score;

    if let Some(k) = cfg.beam {
        let max_depth = 3usize;
        choice = st_logic::beam_select(
            choice,
            |c| neighbors_amg(c),
            |c| score_choice(profile, c, &soft, cfg.soft_mode, alpha),
            k,
            max_depth,
        );
    }
    choice
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explain_includes_density_bucket() {
        std::env::remove_var("SPIRAL_SOFT_MODE");
        std::env::remove_var("SPIRAL_BEAM_K");
        let profile = ProblemProfile::new(4096, 4096, 2_000_000, false);
        let choice = choose_with_profile(&profile);
        let summary = choice.explain(&profile);
        assert!(summary.contains("density="));
        assert!(summary.contains("wg=512") || summary.contains("wg=256"));
    }

    #[test]
    fn sparse_profile_prefers_two_ce() {
        std::env::remove_var("SPIRAL_SOFT_MODE");
        std::env::remove_var("SPIRAL_BEAM_K");
        let profile = ProblemProfile::new(4096, 1024, 10_000, false);
        let choice = choose_with_profile(&profile);
        assert!(choice.use_2ce);
    }

    #[test]
    fn builder_integration_prefers_wide_bandwidth() {
        use crate::profile::ProfileBuilder;

        let mut builder = ProfileBuilder::new(4096, 4096, false);
        for _ in 0..4096 {
            builder.observe_row(48, true, Some(9000));
        }
        let profile = builder.build();
        let choice = choose_with_profile(&profile);
        assert_eq!(choice.wg, 512);
        assert!(choice.tile_cols >= 8_192);
    }
}
