// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-logic/src/lib.rs
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub enum SoftMode {
    Sum,
    Normalize,
    Softmax,
    Prob,
}

#[derive(Clone, Debug)]
pub struct SolveCfg {
    pub noise: f32,
    pub seed: u64,
    pub beam: Option<usize>,
    pub soft_mode: SoftMode,
}

#[derive(Clone, Debug)]
pub struct Ctx {
    pub rows: usize,
    pub cols: usize,
    pub k: usize,
    pub sg: bool,
}

#[derive(Clone, Debug)]
pub struct SoftRule {
    pub name: &'static str,
    pub weight: f32,
    pub score: f32,
}

#[inline]
fn safe_sigmoid(x: f32) -> f32 {
    if x > 20.0 {
        1.0
    } else if x < -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Collapse a collection of SoftRule entries into a scalar according to the mode.
pub fn apply_softmode(rules: &[SoftRule], mode: SoftMode) -> f32 {
    if rules.is_empty() {
        return 0.0;
    }
    match mode {
        SoftMode::Sum => rules.iter().map(|r| r.weight * r.score).sum(),
        SoftMode::Normalize => {
            let sumw = rules.iter().map(|r| r.weight.abs()).sum::<f32>().max(1e-6);
            rules.iter().map(|r| (r.weight / sumw) * r.score).sum()
        }
        SoftMode::Softmax => {
            let m = rules
                .iter()
                .map(|r| r.score)
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = rules.iter().map(|r| (r.score - m).exp()).collect();
            let z = exps.iter().sum::<f32>().max(1e-6);
            rules
                .iter()
                .zip(exps.iter())
                .map(|(r, e)| r.weight * (e / z))
                .sum()
        }
        SoftMode::Prob => {
            // p_any = 1 - Π(1 - p_i) with scores interpreted as logits.
            let mut p_all_not = 1.0f32;
            for r in rules {
                let p = safe_sigmoid(r.score);
                let pw = (p * r.weight.clamp(0.0, 1.0)).clamp(0.0, 1.0);
                p_all_not *= 1.0 - pw;
            }
            let p_any = 1.0 - p_all_not;
            let eps = 1e-6;
            let q = p_any.clamp(eps, 1.0 - eps);
            (q / (1.0 - q)).ln()
        }
    }
}

/// Generic beam search helper.
pub fn beam_select<C: Clone>(
    seed: C,
    mut expand: impl FnMut(&C) -> Vec<C>,
    mut score_fn: impl FnMut(&C) -> f32,
    beam_k: usize,
    max_depth: usize,
) -> C {
    #[derive(Clone)]
    struct Node<C> {
        cand: C,
        score: f32,
    }
    let mut beam: Vec<Node<C>> = vec![Node {
        cand: seed,
        score: f32::NEG_INFINITY,
    }];

    for _depth in 0..max_depth {
        let mut pool: Vec<Node<C>> = Vec::new();
        for n in &beam {
            let nbrs = expand(&n.cand);
            for c in nbrs {
                let s = score_fn(&c);
                pool.push(Node { cand: c, score: s });
            }
        }
        if pool.is_empty() {
            break;
        }
        pool.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        pool.truncate(beam_k.max(1));
        beam = pool;
    }
    beam.into_iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .unwrap()
        .cand
}

// learn_store is optional behind the feature gate.
#[cfg(feature = "learn_store")]
pub mod learn;

pub mod nerf_trainer;
pub mod quantum_reality;
