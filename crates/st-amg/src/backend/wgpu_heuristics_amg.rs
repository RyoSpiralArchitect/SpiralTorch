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
use st_logic::{SoftMode, SolveCfg};

use st_tensor::fractional::gl_coeffs;

fn estimated_density(rows: usize, cols: usize, nnz: usize) -> f32 {
    let volume = (rows.max(1) * cols.max(1)) as f32;
    (nnz as f32 / volume).clamp(1e-6, 1.0)
}

fn lane_pref(subgroup: bool, rows: usize, cols: usize) -> u32 {
    if subgroup {
        if cols >= 8192 { 512 } else { 256 }
    } else if rows <= 16 {
        128
    } else if cols <= 4096 {
        256
    } else {
        512
    }
}

fn tile_pref(cols: usize, subgroup: bool) -> u32 {
    match (cols, subgroup) {
        (..=4096, true) => 4_096,
        (..=4096, false) => 2_048,
        (4097..=16384, true) => 8_192,
        (4097..=16384, false) => 4_096,
        (16385..=65536, _) => 8_192,
        _ => 16_384,
    }
}

fn jacobi_pref(density: f32) -> u32 {
    if density < 0.05 {
        0
    } else if density < 0.12 {
        1
    } else if density < 0.30 {
        2
    } else {
        3
    }
}

fn emphasize(hit: bool, on_hit: f32, on_miss: f32) -> f32 {
    if hit { on_hit } else { on_miss }
}

fn fractional_alpha(rows: usize, cols: usize, nnz: usize) -> f32 {
    let density = estimated_density(rows, cols, nnz);
    let curvature = (rows.max(cols) as f32).log2().max(1.0);
    let alpha = (density.sqrt() * 0.65 + 0.25) * (1.0 - 0.05 / curvature);
    alpha.clamp(0.15, 0.95)
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

fn initial_choice(rows: usize, cols: usize, nnz: usize, subgroup: bool) -> Choice {
    let density = estimated_density(rows, cols, nnz);
    let wg = lane_pref(subgroup, rows, cols);
    let tile_cols = tile_pref(cols, subgroup);
    let jacobi_passes = jacobi_pref(density);
    let use_2ce = density < 0.25 || cols >= 16_384;

    Choice { use_2ce, wg, tile_cols, jacobi_passes, score: 0.0 }
}

// Placeholder base score (replace with project-specific version).
fn base_score_amg(c: &Choice, alpha: f32) -> f32 {
    let mut s = 0.0f32;
    if c.use_2ce { s += 0.12; } else { s -= 0.07; }

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
fn soft_rules_from_spiralk(_rows: usize, _cols: usize, _nnz: usize, _sg: bool) -> Vec<st_logic::SoftRule> {
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

    let density = estimated_density(rows, cols, nnz);
    let tile_hint = tile_pref(cols, subgroup);
    let wg_hint = lane_pref(subgroup, rows, cols);
    let jacobi_hint = jacobi_pref(density);

    let mut rules = Vec::with_capacity(11);

    let wg_bias = if subgroup { 0.35 } else { 0.55 };
    rules.push(SoftRule { name: WG128, weight: emphasize(wg_hint == 128, 0.57, 0.45), score: 0.85 });
    rules.push(SoftRule { name: WG256, weight: emphasize(wg_hint == 256, 0.55 + wg_bias, 0.55), score: 1.35 });
    rules.push(SoftRule { name: WG512, weight: emphasize(wg_hint == 512, 0.65, 0.40), score: 1.10 });

    let two_ce_weight = if density < 0.18 { 0.78 } else { 0.48 };
    rules.push(SoftRule { name: SOFT_2CE, weight: two_ce_weight, score: if density < 0.30 { 1.35 } else { 1.05 } });

    let tile_base = match tile_hint {
        2_048 => (0.74, 0.28),
        4_096 => (0.80, 0.30),
        8_192 => (0.82, 0.32),
        _ => (0.86, 0.34),
    };
    rules.push(SoftRule { name: TILE4K, weight: emphasize(tile_hint == 4_096, tile_base.0, tile_base.1), score: 0.88 });
    rules.push(SoftRule { name: TILE8K, weight: emphasize(tile_hint == 8_192, tile_base.0, tile_base.1 + 0.02), score: 0.97 });
    rules.push(SoftRule { name: TILE16K, weight: emphasize(tile_hint == 16_384, tile_base.0 + 0.04, tile_base.1 + 0.04), score: 0.92 });

    rules.push(SoftRule { name: JACOBI0, weight: emphasize(jacobi_hint == 0, 0.65, 0.25), score: 0.55 });
    rules.push(SoftRule { name: JACOBI1, weight: emphasize(jacobi_hint == 1, 0.85, 0.50), score: 1.15 });
    rules.push(SoftRule { name: JACOBI2, weight: emphasize(jacobi_hint == 2, 0.70, 0.35), score: 0.95 });
    rules.push(SoftRule { name: JACOBI3, weight: emphasize(jacobi_hint >= 3, 0.55, 0.20), score: 0.75 });

    rules
}

fn instantiate_soft_rules(c: &Choice, base: &[st_logic::SoftRule]) -> Vec<st_logic::SoftRule> {
    let mut out = Vec::with_capacity(base.len());
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
                if c.use_2ce { score *= 1.4; } else { score *= -0.6; weight *= 0.35; }
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
                if c.jacobi_passes == 0 { score *= 1.2; weight *= 1.1; }
                else { weight *= 0.5; score *= 0.3; }
            }
            "jacobi=1" => {
                if c.jacobi_passes == 1 { score *= 1.3; } else { weight *= 0.4; score *= 0.4; }
            }
            "jacobi=2" => {
                if c.jacobi_passes == 2 { score *= 1.1; } else { weight *= 0.3; score *= 0.2; }
            }
            "jacobi=3" => {
                if c.jacobi_passes >= 3 { score *= 1.05; weight *= 1.05; } else { weight *= 0.2; score *= 0.2; }
            }
            _ => {}
        }
        out.push(st_logic::SoftRule { name: rule.name, weight, score });
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

fn score_choice(c: &Choice, base_soft: &[st_logic::SoftRule], mode: SoftMode, alpha: f32) -> f32 {
    let dyn_rules = instantiate_soft_rules(c, base_soft);
    let soft = st_logic::apply_softmode(&dyn_rules, mode);
    base_score_amg(c, alpha) + soft
}

fn neighbors_amg(c: &Choice) -> Vec<Choice> {
    let mut v = Vec::new();
    for &wg in &[128u32, 256, 512] {
        let mut n = c.clone(); n.wg = wg; v.push(n);
    }
    for &tc in &[4096u32, 8192, 16384] {
        let mut n = c.clone(); n.tile_cols = tc; v.push(n);
    }
    for &jp in &[1u32, 2] {
        let mut n = c.clone(); n.jacobi_passes = jp; v.push(n);
    }
    {
        let mut n = c.clone(); n.use_2ce = !c.use_2ce; v.push(n);
    }
    v
}

pub fn choose(rows: usize, cols: usize, nnz: usize, subgroup: bool) -> Choice {
    let soft_mode = match std::env::var("SPIRAL_SOFT_MODE").as_deref() {
        Ok("Normalize") => SoftMode::Normalize,
        Ok("Softmax")   => SoftMode::Softmax,
        Ok("Prob")      => SoftMode::Prob,
        _               => SoftMode::Sum,
    };
    let beam_k = std::env::var("SPIRAL_BEAM_K").ok().and_then(|s| s.parse::<usize>().ok());
    let cfg = SolveCfg { noise: 0.02, seed: 0x5u64, beam: beam_k, soft_mode: soft_mode };

    let mut soft = soft_rules_from_spiralk(rows, cols, nnz, subgroup);
    
    // Blend in bandit weights when the feature is enabled.
    #[cfg(feature = "learn_store")]
    {
        let sw = load();
        let lambda = std::env::var("SPIRAL_SOFT_BANDIT_BLEND")
            .ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.35);
        for r in &mut soft {
            let bw = weight_from_bandit(&sw, r.name);
            r.weight = (1.0 - lambda) * r.weight + lambda * bw;
        }
    }

    // Single pass scoring; enable beam search via beam_select when desired.
    let alpha = fractional_alpha(rows, cols, nnz);
    let mut choice = initial_choice(rows, cols, nnz, subgroup);
    let score = score_choice(&choice, &soft, cfg.soft_mode, alpha);
    choice.score = score;

    if let Some(k) = cfg.beam {
        let max_depth = 3usize;
        choice = st_logic::beam_select(choice, |c| neighbors_amg(c),
            |c| score_choice(c, &soft, cfg.soft_mode, alpha),
            k, max_depth);
    }
    choice
}
