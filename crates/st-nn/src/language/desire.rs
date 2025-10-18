// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::geometry::{ConceptHint, RepressionField, SemanticBridge, SymbolGeometry};
use super::maxwell::NarrativeHint;
use super::schrodinger::schrodinger_boost;
use super::temperature::{entropy, TemperatureController};
use crate::PureResult;
use serde::{Deserialize, Serialize};
use st_tensor::{DesireGradientInterpretation, GradientSummary, TensorError};

const REPORT_SIZE: usize = 8;
const BIAS_UPDATE_INJECTION: f32 = 0.05;
const BIAS_UPDATE_INTEGRATION: f32 = 0.02;
const PHASE_EPS: f32 = 1e-4;
const EPSILON_BASE: f32 = 1e-6;

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DesirePhase {
    Observation,
    Injection,
    Integration,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DesireAvoidanceReport {
    pub tokens: Vec<usize>,
    pub scores: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct DesireSchedule {
    shape: ScheduleShape,
}

#[derive(Clone, Debug)]
enum ScheduleShape {
    Constant { value: f32 },
    Warmup { start: f32, end: f32, steps: u64 },
}

impl DesireSchedule {
    fn new(shape: ScheduleShape) -> Self {
        Self { shape }
    }

    fn value(&self, step: u64) -> f32 {
        match self.shape {
            ScheduleShape::Constant { value } => value.max(0.0),
            ScheduleShape::Warmup { start, end, steps } => {
                if steps == 0 {
                    return end.max(0.0);
                }
                let progress = (step as f32 / steps as f32).clamp(0.0, 1.0);
                let value = start + (end - start) * progress;
                value.max(0.0)
            }
        }
    }

    fn horizon(&self) -> u64 {
        match self.shape {
            ScheduleShape::Constant { .. } => 0,
            ScheduleShape::Warmup { steps, .. } => steps,
        }
    }
}

pub fn constant(value: f32) -> DesireSchedule {
    DesireSchedule::new(ScheduleShape::Constant { value })
}

pub fn warmup(start: f32, end: f32, steps: u64) -> DesireSchedule {
    DesireSchedule::new(ScheduleShape::Warmup { start, end, steps })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DesireSolution {
    pub indices: Vec<usize>,
    pub probabilities: Vec<f32>,
    pub logit_offsets: Vec<f32>,
    pub temperature: f32,
    pub entropy: f32,
    pub weights: DesireWeights,
    pub phase: DesirePhase,
    pub avoidance: Option<DesireAvoidanceReport>,
    pub hypergrad_penalty: f32,
    pub narrative: Option<NarrativeHint>,
}

pub struct DesireLagrangian {
    geometry: SymbolGeometry,
    repression: RepressionField,
    semantics: SemanticBridge,
    controller: TemperatureController,
    lookahead: usize,
    top_k: Option<usize>,
    epsilon: f32,
    alpha_schedule: DesireSchedule,
    beta_schedule: DesireSchedule,
    gamma_schedule: DesireSchedule,
    lambda_schedule: DesireSchedule,
    observation_override: Option<u64>,
    integration_override: Option<u64>,
    step_index: u64,
    phase: DesirePhase,
    avoidance_accumulator: Vec<f32>,
    desire_bias: Vec<f32>,
    gradient_interpretation: DesireGradientInterpretation,
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
        let vocab = geometry.vocab_size();
        Ok(Self {
            geometry,
            repression,
            semantics,
            controller,
            lookahead: 0,
            top_k: None,
            epsilon: EPSILON_BASE,
            alpha_schedule: constant(0.0),
            beta_schedule: constant(0.0),
            gamma_schedule: constant(0.0),
            lambda_schedule: constant(0.0),
            observation_override: None,
            integration_override: None,
            step_index: 0,
            phase: DesirePhase::Observation,
            avoidance_accumulator: vec![0.0; vocab],
            desire_bias: vec![0.0; vocab],
            gradient_interpretation: DesireGradientInterpretation::default(),
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

    pub fn with_alpha_schedule(mut self, schedule: DesireSchedule) -> Self {
        self.alpha_schedule = schedule;
        self
    }

    pub fn with_beta_schedule(mut self, schedule: DesireSchedule) -> Self {
        self.beta_schedule = schedule;
        self
    }

    pub fn with_gamma_schedule(mut self, schedule: DesireSchedule) -> Self {
        self.gamma_schedule = schedule;
        self
    }

    pub fn with_lambda_schedule(mut self, schedule: DesireSchedule) -> Self {
        self.lambda_schedule = schedule;
        self
    }

    pub fn with_observation_horizon(mut self, horizon: Option<u64>) -> Self {
        self.observation_override = horizon;
        self
    }

    pub fn with_integration_horizon(mut self, horizon: Option<u64>) -> Self {
        self.integration_override = horizon;
        self
    }

    pub fn vocab_size(&self) -> usize {
        self.geometry.vocab_size()
    }

    pub fn avoidance_heat(&self) -> &[f32] {
        &self.avoidance_accumulator
    }

    pub fn desire_bias(&self) -> &[f32] {
        &self.desire_bias
    }

    /// Latest gradient interpretation driving Desire feedback.
    pub fn gradient_interpretation(&self) -> DesireGradientInterpretation {
        self.gradient_interpretation
    }

    /// Update the interpretation using a precomputed feedback structure.
    pub fn interpret_gradients(&mut self, interpretation: DesireGradientInterpretation) {
        self.gradient_interpretation = interpretation;
        let damping = interpretation.damping().max(0.1);
        self.epsilon = (self.epsilon * 0.9) + 0.1 * (EPSILON_BASE * damping);
    }

    /// Convert gradient summaries into the interpretation layer and feed them
    /// back into the Desire loop.
    pub fn interpret_gradient_summaries(&mut self, hyper: GradientSummary, real: GradientSummary) {
        let interpretation = DesireGradientInterpretation::from_summaries(hyper, real);
        self.interpret_gradients(interpretation);
    }

    pub fn phase(&self) -> DesirePhase {
        self.phase
    }

    pub fn narrative_hint(&self) -> Option<&NarrativeHint> {
        self.active_narrative.as_ref()
    }

    pub fn set_narrative_hint(&mut self, hint: NarrativeHint) {
        self.active_narrative = Some(hint);
    }

    pub fn set_narrative_hint_opt(&mut self, hint: Option<NarrativeHint>) {
        self.active_narrative = hint;
    }

    pub fn clear_narrative_hint(&mut self) {
        self.active_narrative = None;
    }

    pub fn step_with_scheduler(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
    ) -> PureResult<DesireSolution> {
        let weights = self.scheduled_weights();
        self.step_internal(lm_logits, previous_token, concept_hint, weights)
    }

    pub fn step(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
        weights: &DesireWeights,
    ) -> PureResult<DesireSolution> {
        let weights = weights.clone();
        self.step_internal(lm_logits, previous_token, concept_hint, weights)
    }

    fn step_internal(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
        weights: DesireWeights,
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
            &weights,
            &concepts,
            self.lookahead,
        );

        let phase = self.classify_phase(&weights);
        if phase != self.phase {
            self.on_phase_transition(phase);
            self.phase = phase;
        }

        let mut scores = Vec::with_capacity(active.len());
        let mut offsets = Vec::with_capacity(active.len());
        for (pos, &token) in active.iter().enumerate() {
            let log_syn = self.geometry.log_syn(previous_token, token);
            let log_par = self.geometry.log_par(previous_token, token);
            let repression = self.repression.value(token);
            let g = self.semantics.expectation(token, &concepts);
            let bias = self.desire_bias.get(token).copied().unwrap_or(0.0);
            let phase_bias = match phase {
                DesirePhase::Observation => 0.0,
                DesirePhase::Injection => weights.alpha * bias,
                DesirePhase::Integration => (weights.alpha + weights.gamma) * bias,
            };
            let adjustment = weights.alpha * log_syn + weights.beta * log_par
                - weights.lambda * repression
                + weights.gamma * g
                + schrodinger[token]
                + phase_bias;
            let score = log_q[pos] / self.controller.value() + adjustment;
            scores.push(score);
            offsets.push(adjustment);
        }
        stabilise(&mut scores);
        let distribution = softmax(&scores);
        let entropy = entropy(&distribution);
        let gradient_gain = match phase {
            DesirePhase::Observation => 1.25,
            DesirePhase::Injection => 1.0,
            DesirePhase::Integration => 0.7,
        };
        if let Some(pulse) = hub::get_last_realgrad() {
            let summary = pulse.gradient_summary();
            self.controller.observe_grad(summary.norm, summary.sparsity);
        }
        let temperature = self
            .controller
            .update_with_gradient(&distribution, gradient_gain);
        self.update_tracking(phase, &active, &distribution);
        let hypergrad_penalty = self.hypergrad_penalty(phase, &active, &offsets, &distribution);
        let avoidance = self.build_report(phase);
        self.step_index = self.step_index.saturating_add(1);
        Ok(DesireSolution {
            indices: active,
            probabilities: distribution,
            logit_offsets: offsets,
            temperature,
            entropy,
            weights,
            phase,
            avoidance,
            hypergrad_penalty,
            narrative: self.active_narrative.clone(),
        })
    }

    fn scheduled_weights(&self) -> DesireWeights {
        let alpha = self.alpha_schedule.value(self.step_index);
        let beta = self.beta_schedule.value(self.step_index);
        let gamma = self.gamma_schedule.value(self.step_index);
        let lambda = self.lambda_schedule.value(self.step_index);
        DesireWeights::new(alpha, beta, gamma, lambda)
    }

    fn classify_phase(&self, weights: &DesireWeights) -> DesirePhase {
        let step = self.step_index;
        let observation_limit = self.observation_override.unwrap_or_else(|| {
            let alpha = self.alpha_schedule.horizon();
            if alpha == 0 {
                128
            } else {
                alpha
            }
        });
        let integration_limit = self
            .integration_override
            .unwrap_or_else(|| {
                let horizon = self
                    .alpha_schedule
                    .horizon()
                    .max(self.beta_schedule.horizon())
                    .max(self.gamma_schedule.horizon());
                if horizon == 0 {
                    observation_limit.max(256)
                } else {
                    horizon
                }
            })
            .max(observation_limit);
        let scheduled = if step < observation_limit {
            DesirePhase::Observation
        } else if step < integration_limit {
            DesirePhase::Injection
        } else {
            DesirePhase::Integration
        };
        match scheduled {
            DesirePhase::Observation => {
                if weights.alpha > PHASE_EPS || weights.beta > PHASE_EPS {
                    DesirePhase::Injection
                } else {
                    DesirePhase::Observation
                }
            }
            DesirePhase::Injection => {
                if step >= integration_limit && weights.gamma > PHASE_EPS {
                    DesirePhase::Integration
                } else {
                    DesirePhase::Injection
                }
            }
            DesirePhase::Integration => DesirePhase::Integration,
        }
    }

    fn on_phase_transition(&mut self, phase: DesirePhase) {
        if matches!(phase, DesirePhase::Injection | DesirePhase::Integration) {
            if !self.avoidance_accumulator.is_empty() {
                self.desire_bias = normalise(&self.avoidance_accumulator);
            }
        }
    }

    fn update_tracking(&mut self, phase: DesirePhase, active: &[usize], distribution: &[f32]) {
        match phase {
            DesirePhase::Observation => {
                let gain = self.gradient_interpretation.observation_gain();
                for (&token, &prob) in active.iter().zip(distribution) {
                    let avoid = (1.0 - prob).max(0.0) * gain;
                    if let Some(value) = self.avoidance_accumulator.get_mut(token) {
                        *value += avoid;
                    }
                }
            }
            DesirePhase::Injection => {
                self.update_bias(active, distribution, BIAS_UPDATE_INJECTION);
            }
            DesirePhase::Integration => {
                self.update_bias(active, distribution, BIAS_UPDATE_INTEGRATION);
            }
        }
    }

    fn update_bias(&mut self, active: &[usize], distribution: &[f32], rate: f32) {
        if self.desire_bias.is_empty() {
            return;
        }
        let rate = (rate * self.gradient_interpretation.bias_mix()).clamp(0.0, 1.0);
        for (&token, &prob) in active.iter().zip(distribution) {
            if let Some(value) = self.desire_bias.get_mut(token) {
                let target = (1.0 - prob).max(0.0);
                let current = *value;
                *value = current * (1.0 - rate) + rate * target;
            }
        }
        self.desire_bias = normalise(&self.desire_bias);
    }

    fn hypergrad_penalty(
        &self,
        phase: DesirePhase,
        active: &[usize],
        offsets: &[f32],
        distribution: &[f32],
    ) -> f32 {
        if !matches!(phase, DesirePhase::Integration) {
            return 0.0;
        }
        let barycenter: f32 = distribution
            .iter()
            .zip(offsets.iter())
            .map(|(p, o)| p * o)
            .sum();
        if self.desire_bias.is_empty() {
            return 0.0;
        }
        let mut bias_center = 0.0f32;
        for (&token, &offset) in active.iter().zip(offsets) {
            let weight = self.desire_bias.get(token).copied().unwrap_or(0.0);
            bias_center += weight * offset;
        }
        let penalty = (bias_center - barycenter).abs();
        penalty * self.gradient_interpretation.penalty_gain()
    }

    fn build_report(&self, phase: DesirePhase) -> Option<DesireAvoidanceReport> {
        let source = match phase {
            DesirePhase::Observation => &self.avoidance_accumulator,
            _ => &self.desire_bias,
        };
        if source.is_empty() {
            return None;
        }
        let mut pairs: Vec<(usize, f32)> = source
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| {
                if value.is_finite() && *value > PHASE_EPS {
                    Some((idx, *value))
                } else {
                    None
                }
            })
            .collect();
        if pairs.is_empty() {
            return None;
        }
        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(REPORT_SIZE.min(pairs.len()));
        let (tokens, scores): (Vec<usize>, Vec<f32>) = pairs.into_iter().unzip();
        Some(DesireAvoidanceReport { tokens, scores })
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

fn normalise(values: &[f32]) -> Vec<f32> {
    let mut buffer: Vec<f32> = values.iter().map(|v| v.max(0.0)).collect();
    let sum: f32 = buffer.iter().sum();
    if sum <= f32::EPSILON {
        return vec![0.0; values.len()];
    }
    for value in &mut buffer {
        *value /= sum;
    }
    buffer
}

#[cfg(test)]
mod tests {
    use super::super::geometry::{
        ConceptHint, RepressionField, SemanticBridge, SparseKernel, SymbolGeometry,
    };
    use super::super::maxwell::NarrativeHint;
    use super::*;
    use st_tensor::{DesireGradientInterpretation, GradientSummary};
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

    fn build_lagrangian() -> DesireLagrangian {
        let geometry = build_geometry();
        let repression = RepressionField::new(vec![0.1, 0.2]).unwrap();
        let semantics = build_semantics();
        let controller = TemperatureController::new(1.0, 0.7, 0.5, 0.5, 2.0);
        DesireLagrangian::new(geometry, repression, semantics, controller).unwrap()
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
        assert_eq!(result.weights.alpha, weights.alpha);
        assert_eq!(result.phase, DesirePhase::Injection);
        assert!(result.hypergrad_penalty >= 0.0);
        assert!(result.narrative.is_none());
    }

    #[test]
    fn scheduler_advances_phases() {
        let geometry = build_geometry();
        let repression = RepressionField::new(vec![0.05, 0.15]).unwrap();
        let semantics = build_semantics();
        let controller = TemperatureController::new(1.0, 0.8, 0.4, 0.4, 1.8);
        let mut lagrangian = DesireLagrangian::new(geometry, repression, semantics, controller)
            .unwrap()
            .with_alpha_schedule(warmup(0.0, 0.1, 2))
            .with_beta_schedule(warmup(0.0, 0.05, 4))
            .with_gamma_schedule(constant(0.02))
            .with_observation_horizon(Some(2))
            .with_integration_horizon(Some(4));
        let logits = vec![2.0, 1.0];
        let concept = ConceptHint::Distribution(vec![0.5, 0.5]);

        let mut phases = Vec::new();
        for step in 0..6 {
            let result = lagrangian
                .step_with_scheduler(&logits, step % 2, &concept)
                .unwrap();
            phases.push(result.phase);
            match result.phase {
                DesirePhase::Observation => {
                    assert!(result.weights.alpha <= 0.05 + 1e-6);
                    assert!(result.avoidance.is_some());
                }
                DesirePhase::Injection => {
                    assert!(result.weights.alpha >= 0.049);
                    assert!(result.weights.beta <= 0.05 + 1e-6);
                }
                DesirePhase::Integration => {
                    assert!(result.hypergrad_penalty >= 0.0);
                }
            }
            assert!(result.narrative.is_none());
        }
        assert!(phases.contains(&DesirePhase::Observation));
        assert!(phases.contains(&DesirePhase::Injection));
        assert!(phases.contains(&DesirePhase::Integration));
    }

    #[test]
    fn interpretation_modulates_penalty_and_bias() {
        let active = vec![0, 1];
        let distribution = vec![0.6, 0.4];
        let offsets = vec![0.15, -0.2];

        let mut baseline = build_lagrangian();
        baseline.desire_bias = vec![0.6, 0.4];
        let base_penalty =
            baseline.hypergrad_penalty(DesirePhase::Integration, &active, &offsets, &distribution);

        let mut amplified = build_lagrangian();
        amplified.desire_bias = vec![0.6, 0.4];
        let imbalance = DesireGradientInterpretation::from_summaries(
            GradientSummary::from_slice(&[1.0, -0.5, 0.75]),
            GradientSummary::from_slice(&[0.05, -0.02, 0.01]),
        );
        amplified.interpret_gradients(imbalance);
        let boosted =
            amplified.hypergrad_penalty(DesirePhase::Integration, &active, &offsets, &distribution);
        assert!(boosted >= base_penalty);

        let stable_interp = DesireGradientInterpretation::from_summaries(
            GradientSummary::from_slice(&[0.3, -0.2, 0.1]),
            GradientSummary::from_slice(&[0.3, -0.2, 0.1]),
        );
        let mut stable = build_lagrangian();
        stable.desire_bias = vec![0.9, 0.1];
        stable.interpret_gradients(stable_interp);
        stable.update_bias(&active, &distribution, 0.1);
        let stable_bias = stable.desire_bias.clone();

        let mut cautious = build_lagrangian();
        cautious.desire_bias = vec![0.9, 0.1];
        cautious.interpret_gradients(imbalance);
        cautious.update_bias(&active, &distribution, 0.1);
        assert!(stable_bias[0] < cautious.desire_bias[0]);
    }

    #[test]
    fn interpret_gradient_summaries_updates_state() {
        let mut lagrangian = build_lagrangian();
        let hyper = GradientSummary::from_slice(&[0.2, -0.1, 0.05]);
        let real = GradientSummary::from_slice(&[0.15, -0.05, 0.02]);
        lagrangian.interpret_gradient_summaries(hyper, real);
        let interpretation = lagrangian.gradient_interpretation();
        assert!((interpretation.hyper_pressure() - hyper.mean_abs()).abs() < 1e-6);
        assert!(interpretation.penalty_gain() >= 1.0);
    }
}
