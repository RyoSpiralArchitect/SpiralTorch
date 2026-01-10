// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::desire::DesireWeights;
use super::geometry::{RepressionField, SemanticBridge, SymbolGeometry};

pub(crate) fn schrodinger_boost(
    geometry: &SymbolGeometry,
    repression: &RepressionField,
    bridge: &SemanticBridge,
    weights: &DesireWeights,
    concept_expectation: &[f32],
    lookahead: usize,
) -> Vec<f32> {
    let vocab = geometry.vocab_size();
    let mut boosts = vec![0.0f32; vocab];
    if lookahead == 0 {
        return boosts;
    }

    let mut v_next = vec![1.0f32; vocab];
    let eps = 1e-6f32;
    let mut g_cache = Vec::with_capacity(vocab);
    for token in 0..vocab {
        g_cache.push(bridge.expectation(token, concept_expectation));
    }

    for _ in 0..lookahead {
        let mut v_curr = vec![eps; vocab];
        for (i, v_curr_i) in v_curr.iter_mut().enumerate() {
            let mut sum = eps;
            for &(j, log_syn) in geometry.syn_row(i) {
                let log_par = geometry.log_par(i, j);
                let repression = repression.value(j);
                let potential = weights.alpha * log_syn + weights.beta * log_par
                    - weights.lambda * repression
                    + weights.gamma * g_cache[j];
                let baseline = log_syn;
                let weight = (baseline + potential).exp();
                sum += weight * v_next[j];
            }
            *v_curr_i = sum;
        }
        v_next = v_curr;
    }

    let max_val = v_next
        .iter()
        .copied()
        .fold(f32::MIN, |acc, v| if v > acc { v } else { acc })
        .max(eps);
    for (idx, value) in v_next.iter().enumerate() {
        let mut log_val = value.max(eps).ln();
        log_val -= max_val.ln();
        boosts[idx] = log_val;
    }
    boosts
}
