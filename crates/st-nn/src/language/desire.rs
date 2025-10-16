// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::geometry::{ConceptHint, RepressionField, SemanticBridge, SymbolGeometry};
use super::schrodinger::schrodinger_boost;
use super::temperature::{entropy, TemperatureController};
use crate::PureResult;
use st_tensor::pure::TensorError;

#[derive(Clone, Debug)]
pub struct DesireWeights {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub lambda: f32,
}

impl DesireWeights {
    pub fn new(alpha: f32, beta: f32, gamma: f32, lambda: f32) -> Self {
        Self {
            alpha: alpha.max(0.0),
            beta: beta.max(0.0),
            gamma: gamma.max(0.0),
            lambda: lambda.max(0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DesireSolution {
    pub indices: Vec<usize>,
    pub probabilities: Vec<f32>,
    pub logit_offsets: Vec<f32>,
    pub temperature: f32,
    pub entropy: f32,
}

pub struct DesireLagrangian {
    geometry: SymbolGeometry,
    repression: RepressionField,
    semantics: SemanticBridge,
    controller: TemperatureController,
    lookahead: usize,
    top_k: Option<usize>,
    epsilon: f32,
}

impl DesireLagrangian {
    pub fn new(
        geometry: SymbolGeometry,
        repression: RepressionField,
        semantics: SemanticBridge,
        controller: TemperatureController,
    ) -> PureResult<Self> {
        if geometry.vocab_size() != repression.len() {
            return Err(TensorError::InvalidValue {
                label: "repression field must match vocabulary",
            });
        }
        if semantics.vocab_size() != geometry.vocab_size() {
            return Err(TensorError::InvalidValue {
                label: "semantic bridge must align with vocabulary",
            });
        }
        Ok(Self {
            geometry,
            repression,
            semantics,
            controller,
            lookahead: 0,
            top_k: None,
            epsilon: 1e-6,
        })
    }

    pub fn with_lookahead(mut self, iterations: usize) -> Self {
        self.lookahead = iterations;
        self
    }

    pub fn with_top_k(mut self, k: Option<usize>) -> Self {
        self.top_k = k;
        self
    }

    pub fn step(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
        weights: &DesireWeights,
    ) -> PureResult<DesireSolution> {
        if lm_logits.len() != self.geometry.vocab_size() {
            return Err(TensorError::InvalidValue {
                label: "LM logits must match vocabulary",
            });
        }
        let concepts = concept_hint.as_distribution(&self.semantics);
        let active = select_indices(lm_logits, self.top_k);
        if active.is_empty() {
            return Err(TensorError::InvalidValue {
                label: "no candidate tokens available for desire step",
            });
        }
        let mut log_base = Vec::with_capacity(active.len());
        for &idx in &active {
            log_base.push(lm_logits[idx]);
        }
        let base_max = log_base.iter().copied().fold(f32::MIN, f32::max);
        let mut log_soft = Vec::with_capacity(active.len());
        let mut base_sum = 0.0f32;
        for &logit in &log_base {
            let value = (logit - base_max).exp();
            log_soft.push(value);
            base_sum += value;
        }
        let mut log_q = Vec::with_capacity(active.len());
        for value in &log_soft {
            log_q.push((value / base_sum.max(self.epsilon)).max(self.epsilon).ln());
        }

        let schrodinger = schrodinger_boost(
            &self.geometry,
            &self.repression,
            &self.semantics,
            weights,
            &concepts,
            self.lookahead,
        );

        let mut scores = Vec::with_capacity(active.len());
        let mut offsets = Vec::with_capacity(active.len());
        for (pos, &token) in active.iter().enumerate() {
            let log_syn = self.geometry.log_syn(previous_token, token);
            let log_par = self.geometry.log_par(previous_token, token);
            let repression = self.repression.value(token);
            let g = self.semantics.expectation(token, &concepts);
            let adjustment = weights.alpha * log_syn + weights.beta * log_par
                - weights.lambda * repression
                + weights.gamma * g
                + schrodinger[token];
            let score = log_q[pos] / self.controller.value() + adjustment;
            scores.push(score);
            offsets.push(adjustment);
        }
        stabilise(&mut scores);
        let distribution = softmax(&scores);
        let entropy = entropy(&distribution);
        let temperature = self.controller.update(&distribution);
        Ok(DesireSolution {
            indices: active,
            probabilities: distribution,
            logit_offsets: offsets,
            temperature,
            entropy,
        })
    }
}

fn select_indices(logits: &[f32], top_k: Option<usize>) -> Vec<usize> {
    match top_k {
        None => (0..logits.len()).collect(),
        Some(k) => {
            if k == 0 {
                return Vec::new();
            }
            let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            pairs.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            pairs.truncate(k.min(pairs.len()));
            pairs.into_iter().map(|(idx, _)| idx).collect()
        }
    }
}

fn stabilise(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let max = values.iter().copied().fold(f32::MIN, f32::max);
    for value in values.iter_mut() {
        *value -= max;
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let mut sum = 0.0f32;
    let mut exps = Vec::with_capacity(logits.len());
    for &logit in logits {
        let val = (logit - max).exp();
        exps.push(val);
        sum += val;
    }
    exps.into_iter().map(|v| v / sum.max(1e-6)).collect()
}

#[cfg(test)]
mod tests {
    use super::super::geometry::{
        ConceptHint, RepressionField, SemanticBridge, SparseKernel, SymbolGeometry,
    };
    use super::*;
    use std::collections::HashSet;

    fn build_geometry() -> SymbolGeometry {
        let syn = SparseKernel::from_rows(
            vec![vec![(0, 0.6), (1, 0.4)], vec![(0, 0.5), (1, 0.5)]],
            1e-6,
        )
        .unwrap();
        let par = SparseKernel::from_rows(
            vec![vec![(0, 0.7), (1, 0.3)], vec![(0, 0.2), (1, 0.8)]],
            1e-6,
        )
        .unwrap();
        SymbolGeometry::new(syn, par).unwrap()
    }

    fn build_semantics() -> SemanticBridge {
        let log_pi = vec![
            vec![(0, (0.7f32).ln()), (1, (0.3f32).ln())],
            vec![(0, (0.4f32).ln()), (1, (0.6f32).ln())],
        ];
        let row = vec![1.0, 1.0];
        let col = vec![1.0, 1.0];
        let anchors = HashSet::new();
        let concept_kernel =
            SparseKernel::from_rows(vec![vec![(0, 1.0)], vec![(1, 1.0)]], 1e-6).unwrap();
        SemanticBridge::new(log_pi, row, col, anchors, 1e-6, concept_kernel).unwrap()
    }

    #[test]
    fn desire_step_produces_distribution() {
        let geometry = build_geometry();
        let repression = RepressionField::new(vec![0.1, 0.3]).unwrap();
        let semantics = build_semantics();
        let controller = TemperatureController::new(1.0, 0.7, 0.5, 0.5, 2.0);
        let mut lagrangian = DesireLagrangian::new(geometry, repression, semantics, controller)
            .unwrap()
            .with_top_k(Some(2));
        let logits = vec![2.0, 0.5];
        let weights = DesireWeights::new(0.2, 0.1, 0.3, 0.05);
        let result = lagrangian
            .step(
                &logits,
                0,
                &ConceptHint::Distribution(vec![0.6, 0.4]),
                &weights,
            )
            .unwrap();
        assert_eq!(result.indices.len(), 2);
        let sum: f32 = result.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(result.temperature >= 0.5);
        assert!(result.entropy > 0.0);
    }
}
