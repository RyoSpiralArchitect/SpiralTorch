// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::desire::{
    DesireAvoidanceReport, DesireLagrangian, DesirePhase, DesireSolution, DesireWeights,
};
use super::geometry::ConceptHint;
use crate::PureResult;
use st_core::config::self_rewrite::SelfRewriteCfg;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const OBSERVATION_MIX: f32 = 0.35;
const INTEGRATION_MIX: f32 = 0.65;
const EPSILON: f32 = 1e-6;
const HISTORY_LIMIT: usize = 256;
const TRIGGER_EXPORT: usize = 12;

#[derive(Clone, Debug)]
pub struct DesireAutomatedStep {
    pub solution: DesireSolution,
    pub trigger: Option<DesireRewriteTrigger>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DesireRewriteTrigger {
    pub report: DesireAvoidanceReport,
    pub mean_penalty: f32,
    pub mean_entropy: f32,
    pub temperature: f32,
    pub samples: usize,
}

pub struct DesireAutomation {
    desire: DesireLagrangian,
    cfg: SelfRewriteCfg,
    penalties: VecDeque<f32>,
    entropies: VecDeque<f32>,
    last_trigger: Option<Instant>,
    vector: Vec<f32>,
    trigger_count: u64,
}

impl DesireAutomation {
    pub fn new(desire: DesireLagrangian, cfg: SelfRewriteCfg) -> Self {
        let vocab = desire.vocab_size();
        Self {
            desire,
            cfg,
            penalties: VecDeque::new(),
            entropies: VecDeque::new(),
            last_trigger: None,
            vector: vec![0.0; vocab],
            trigger_count: 0,
        }
    }

    pub fn config(&self) -> &SelfRewriteCfg {
        &self.cfg
    }

    pub fn desire_vector(&self) -> &[f32] {
        &self.vector
    }

    pub fn trigger_count(&self) -> u64 {
        self.trigger_count
    }

    pub fn mean_penalty(&self) -> f32 {
        if self.penalties.is_empty() {
            0.0
        } else {
            self.penalties.iter().sum::<f32>() / self.penalties.len() as f32
        }
    }

    pub fn mean_entropy(&self) -> f32 {
        if self.entropies.is_empty() {
            0.0
        } else {
            self.entropies.iter().sum::<f32>() / self.entropies.len() as f32
        }
    }

    pub fn penalty_samples(&self) -> usize {
        self.penalties.len()
    }

    pub fn last_trigger(&self) -> Option<Instant> {
        self.last_trigger
    }

    pub fn report(&self, limit: usize) -> DesireAvoidanceReport {
        let (tokens, scores) = self.top_desires(limit.max(1));
        DesireAvoidanceReport { tokens, scores }
    }

    pub fn into_inner(self) -> DesireLagrangian {
        self.desire
    }

    pub fn step(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
        now: Instant,
    ) -> PureResult<DesireAutomatedStep> {
        let solution = self
            .desire
            .step_with_scheduler(lm_logits, previous_token, concept_hint)?;
        Ok(self.drive(solution, now))
    }

    pub fn step_with_weights(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
        weights: &DesireWeights,
        now: Instant,
    ) -> PureResult<DesireAutomatedStep> {
        let solution = self
            .desire
            .step(lm_logits, previous_token, concept_hint, weights)?;
        Ok(self.drive(solution, now))
    }

    fn drive(&mut self, solution: DesireSolution, now: Instant) -> DesireAutomatedStep {
        self.assimilate(solution.phase);
        let trigger = self.evaluate(now, &solution);
        DesireAutomatedStep { solution, trigger }
    }

    fn assimilate(&mut self, phase: DesirePhase) {
        match phase {
            DesirePhase::Observation => {
                let snapshot = self.desire.avoidance_heat().to_vec();
                self.mix(snapshot.as_slice(), OBSERVATION_MIX);
            }
            DesirePhase::Injection | DesirePhase::Integration => {
                let snapshot = self.desire.desire_bias().to_vec();
                self.mix(snapshot.as_slice(), INTEGRATION_MIX);
            }
        }
    }

    fn mix(&mut self, source: &[f32], mix: f32) {
        if source.is_empty() {
            return;
        }
        if self.vector.len() != source.len() {
            self.vector.resize(source.len(), 0.0);
        }
        let total: f32 = source.iter().copied().sum();
        if total <= EPSILON {
            for value in &mut self.vector {
                *value *= 1.0 - mix;
            }
            return;
        }
        let inv = 1.0 / total;
        for (dest, &value) in self.vector.iter_mut().zip(source.iter()) {
            let target = (value * inv).max(0.0);
            *dest = *dest * (1.0 - mix) + mix * target;
        }
        Self::normalise(&mut self.vector);
    }

    fn evaluate(
        &mut self,
        now: Instant,
        solution: &DesireSolution,
    ) -> Option<DesireRewriteTrigger> {
        if !matches!(solution.phase, DesirePhase::Integration) {
            return None;
        }
        self.record_metrics(solution.hypergrad_penalty, solution.entropy);
        if !self.ready_for_trigger(now) {
            return None;
        }
        let trigger = self.build_trigger(solution.temperature);
        self.last_trigger = Some(now);
        self.trigger_count = self.trigger_count.saturating_add(1);
        self.penalties.clear();
        self.entropies.clear();
        Some(trigger)
    }

    fn record_metrics(&mut self, penalty: f32, entropy: f32) {
        if penalty.is_finite() {
            self.penalties.push_back(penalty.max(0.0));
            let limit = self.cfg.min_samples.max(HISTORY_LIMIT);
            while self.penalties.len() > limit {
                self.penalties.pop_front();
            }
        }
        if entropy.is_finite() {
            self.entropies.push_back(entropy.max(0.0));
            let limit = self.cfg.min_samples.max(HISTORY_LIMIT);
            while self.entropies.len() > limit {
                self.entropies.pop_front();
            }
        }
    }

    fn ready_for_trigger(&self, now: Instant) -> bool {
        let required = self.cfg.min_samples.max(1);
        if self.penalties.len() < required {
            return false;
        }
        if self.mean_penalty() < self.cfg.score_thresh {
            return false;
        }
        let cooldown = Duration::from_secs(self.cfg.cooldown_sec);
        match self.last_trigger {
            Some(last) => last
                .checked_add(cooldown)
                .map(|limit| now >= limit)
                .unwrap_or(false),
            None => true,
        }
    }

    fn build_trigger(&self, temperature: f32) -> DesireRewriteTrigger {
        DesireRewriteTrigger {
            report: self.report(TRIGGER_EXPORT),
            mean_penalty: self.mean_penalty(),
            mean_entropy: self.mean_entropy(),
            temperature,
            samples: self.penalties.len(),
        }
    }

    fn top_desires(&self, limit: usize) -> (Vec<usize>, Vec<f32>) {
        let mut pairs: Vec<(usize, f32)> = self
            .vector
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, score)| score.is_finite() && *score > EPSILON)
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        pairs.truncate(limit.min(pairs.len()));
        let (tokens, scores): (Vec<usize>, Vec<f32>) = pairs.into_iter().unzip();
        (tokens, scores)
    }

    fn normalise(values: &mut [f32]) {
        let sum: f32 = values.iter().copied().sum();
        if sum <= EPSILON {
            return;
        }
        let inv = 1.0 / sum;
        for value in values.iter_mut() {
            *value = value.max(0.0) * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::geometry::{
        ConceptHint, RepressionField, SemanticBridge, SparseKernel, SymbolGeometry,
    };
    use super::super::temperature::TemperatureController;
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

    fn build_lagrangian() -> DesireLagrangian {
        let geometry = build_geometry();
        let repression = RepressionField::new(vec![0.05, 0.15]).unwrap();
        let semantics = build_semantics();
        let controller = TemperatureController::new(1.0, 0.8, 0.4, 0.4, 1.8);
        DesireLagrangian::new(geometry, repression, semantics, controller)
            .unwrap()
            .with_top_k(Some(2))
            .with_alpha_schedule(super::super::desire::warmup(0.0, 0.1, 1))
            .with_beta_schedule(super::super::desire::warmup(0.0, 0.05, 2))
            .with_gamma_schedule(super::super::desire::constant(0.02))
            .with_lambda_schedule(super::super::desire::constant(0.0))
            .with_observation_horizon(Some(1))
            .with_integration_horizon(Some(3))
    }

    #[test]
    fn automation_reports_bias() {
        let lagrangian = build_lagrangian();
        let cfg = SelfRewriteCfg {
            score_thresh: 0.0,
            min_samples: 2,
            cooldown_sec: 0,
        };
        let mut automation = DesireAutomation::new(lagrangian, cfg);
        let logits = vec![2.0, 0.5];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let now = Instant::now();
        let step = automation
            .step(&logits, 0, &concept, now)
            .expect("automation step");
        assert_eq!(step.solution.phase, DesirePhase::Observation);
        let report = automation.report(4);
        assert!(report.tokens.len() <= 4);
        let sum: f32 = report.scores.iter().sum();
        assert!(sum >= 0.0);
    }

    #[test]
    fn automation_triggers_after_min_samples() {
        let lagrangian = build_lagrangian();
        let cfg = SelfRewriteCfg {
            score_thresh: 0.0,
            min_samples: 2,
            cooldown_sec: 0,
        };
        let mut automation = DesireAutomation::new(lagrangian, cfg);
        let logits = vec![2.0, 0.5];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let base = Instant::now();
        let mut triggers = 0;
        for step in 0..6 {
            let now = base + Duration::from_secs(step as u64 + 1);
            let DesireAutomatedStep { trigger, .. } = automation
                .step(&logits, step % 2, &concept, now)
                .expect("automation step");
            if let Some(event) = trigger {
                triggers += 1;
                assert!(event.samples >= cfg.min_samples);
                assert!(event.mean_penalty >= cfg.score_thresh);
            }
        }
        assert!(triggers >= 1);
    }

    #[test]
    fn automation_respects_cooldown() {
        let lagrangian = build_lagrangian();
        let cfg = SelfRewriteCfg {
            score_thresh: 0.0,
            min_samples: 2,
            cooldown_sec: 5,
        };
        let mut automation = DesireAutomation::new(lagrangian, cfg);
        let logits = vec![2.0, 0.5];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let base = Instant::now();
        let mut events = Vec::new();
        for offset in [1, 2, 3, 8, 9, 11] {
            let now = base + Duration::from_secs(offset);
            let DesireAutomatedStep { trigger, .. } = automation
                .step(&logits, offset as usize % 2, &concept, now)
                .expect("automation step");
            if let Some(event) = trigger {
                events.push((offset, event));
            }
        }
        assert!(events.len() >= 1);
        if events.len() > 1 {
            let (first_time, _) = events[0];
            let (second_time, _) = events[1];
            assert!(second_time as i64 - first_time as i64 >= cfg.cooldown_sec as i64);
        }
    }
}
