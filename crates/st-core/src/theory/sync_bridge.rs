// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Rust-native translation of the "Sync Theorems" appendix for Z-space control.
//!
//! The appendix presents a heuristic bridge between the structural Bures gap
//! and an observation-side `e`-process.  This module implements those formulas in
//! pure Rust so they can participate in SpiralTorch pipelines without routing
//! through PyTorch.  The design mirrors the original document:
//!
//! - [`SyncConfig`] encodes the parameters (`\alpha`, `\tau_\mathrm{B}`,
//!   `\bar\varepsilon`, `\underline{\cos^2\phi}`) together with an optional
//!   slew-rate lag.
//! - [`SyncTheoremTrainer`] keeps track of the running log `e`-process,
//!   generates gate indicators, and returns I×K labels at each step.
//! - [`SyncTheoremTrainer::run`] recreates the hitting-time viewpoint from the
//!   document, returning the log trajectory and the first observation-gate
//!   iteration if one exists.
//! - State is maintained per sample, so downstream callers can seed the log
//!   `e`-process with [`SyncTheoremTrainer::set_log_e`] and retrieve the
//!   anytime confidence curve for their own labelling policies.
//! - [`SyncTheoremTrainer::aggregate_family`] folds pairwise runs into a
//!   multi-universe signal with configurable structural policies, matching the
//!   Mandela synchrony discussion from Theorem 5.
//! - [`SyncConfig::with_azuma_c`] exposes the Azuma--Hoeffding tail bound from
//!   Theorem 2 so callers can budget additional delay mass, while
//!   [`SyncStep::effective_delta_b_sq`] surfaces the misalignment-adjusted
//!   Bures gaps used in Proposition 3.
//!
//! The implementation uses standard Rust containers (`Vec`) so it works in
//! `no_std`-averse crates, yet remains interoperable with SpiralTorch tensors by
//! accepting borrowed slices.

use std::borrow::Cow;

use thiserror::Error;

/// I×K decision labels mirroring the appendix nomenclature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IKLabel {
    /// Observation-side gate open while the structural gate is confirmed.
    Critical,
    /// Neither gate has fired and the e-process confidence exceeds `1-\alpha`.
    Safe,
    /// Any other combination of structural/observation evidence.
    Abstain,
}

/// Configuration for the synchronisation dynamics.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Anytime confidence level controlling the e-process threshold.
    pub alpha: f32,
    /// Structural Bures gap threshold `\tau_\mathrm{B}`.
    pub tau_b: f32,
    /// Upper bound on the likelihood-ratio corrections `\bar\varepsilon`.
    pub epsilon_max: f32,
    /// Lower bound on `\cos^2\phi` accounting for axis misalignment.
    pub cos_phi_min: f32,
    /// Optional number of steps spent honouring a slew-rate constraint.
    pub slew_steps: usize,
    /// Optional Azuma--Hoeffding scale `c` describing bounded log-e increments.
    azuma_c: Option<f32>,
}

impl SyncConfig {
    /// Construct a new configuration after validating the parameters.
    pub fn new(
        alpha: f32,
        tau_b: f32,
        epsilon_max: f32,
        cos_phi_min: f32,
        slew_steps: usize,
    ) -> Result<Self, SyncError> {
        let config = Self {
            alpha,
            tau_b,
            epsilon_max,
            cos_phi_min,
            slew_steps,
            azuma_c: None,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), SyncError> {
        if !(0.0..1.0).contains(&self.alpha) {
            return Err(SyncError::InvalidAlpha(self.alpha));
        }
        if !(self.tau_b.is_finite() && self.tau_b > 0.0) {
            return Err(SyncError::InvalidTau(self.tau_b));
        }
        if self.epsilon_max < 0.0 || !self.epsilon_max.is_finite() {
            return Err(SyncError::InvalidEpsilon(self.epsilon_max));
        }
        if self.cos_phi_min <= 0.0 || self.cos_phi_min > 1.0 || !self.cos_phi_min.is_finite() {
            return Err(SyncError::InvalidCos(self.cos_phi_min));
        }
        Ok(())
    }

    /// Return the Azuma--Hoeffding coefficient when configured.
    pub fn azuma_c(&self) -> Option<f32> {
        self.azuma_c
    }

    /// Attach an Azuma--Hoeffding coefficient describing deviation control.
    pub fn with_azuma_c(mut self, azuma_c: f32) -> Result<Self, SyncError> {
        if !(azuma_c.is_finite() && azuma_c > 0.0) {
            return Err(SyncError::InvalidAzuma(azuma_c));
        }
        self.azuma_c = Some(azuma_c);
        Ok(self)
    }

    /// Log threshold `\log(1/\alpha)` from Proposition 4.
    pub fn threshold(&self) -> f32 {
        (1.0 / self.alpha).ln()
    }

    fn hitting_time_denominator(&self) -> Result<f32, SyncError> {
        let denom = 2.0 * self.tau_b * self.cos_phi_min - self.epsilon_max;
        if denom <= 0.0 {
            Err(SyncError::NonPositiveDrift(denom))
        } else {
            Ok(denom)
        }
    }

    /// Analytic upper bound on the observation hitting time (Theorem 2).
    pub fn hitting_time_bound(&self) -> Result<f32, SyncError> {
        Ok(self.threshold() / self.hitting_time_denominator()?)
    }

    /// Azuma--Hoeffding tail bound for observing an additional delay `\delta`.
    pub fn hitting_time_tail(&self, delta: f32) -> Option<f32> {
        let c = self.azuma_c?;
        if delta <= 0.0 {
            Some(1.0)
        } else {
            Some((-c * delta * delta).exp())
        }
    }
}

/// Errors that can occur while driving the synchronisation bridge.
#[derive(Debug, Error, PartialEq)]
pub enum SyncError {
    /// `alpha` must lie strictly between 0 and 1.
    #[error("alpha must be in (0, 1), received {0}")]
    InvalidAlpha(f32),
    /// `tau_b` must be positive and finite.
    #[error("tau_b must be positive, received {0}")]
    InvalidTau(f32),
    /// `epsilon_max` must be non-negative.
    #[error("epsilon_max must be non-negative, received {0}")]
    InvalidEpsilon(f32),
    /// `cos_phi_min` must reside in `(0, 1]`.
    #[error("cos_phi_min must lie in (0, 1], received {0}")]
    InvalidCos(f32),
    /// The batch dimensions of the arguments disagree.
    #[error("length mismatch for {field}: expected {expected}, got {actual}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The drift implied by `(tau_b, cos_phi_min, epsilon_max)` is non-positive.
    #[error("expected positive observation drift, denominator = {0}")]
    NonPositiveDrift(f32),
    /// Family aggregation cannot proceed without at least one pair entry.
    #[error("family aggregation requires at least one pair contribution")]
    EmptyFamily,
    /// Azuma--Hoeffding scale must be positive and finite.
    #[error("azuma coefficient must be positive, received {0}")]
    InvalidAzuma(f32),
}

/// Aggregated output for a single synchronisation step.
#[derive(Debug, Clone)]
pub struct SyncStep {
    /// Updated log `e`-process for each element in the batch.
    pub log_e: Vec<f32>,
    /// Structural gate indicator `\mathbf{1}{\Delta_{\mathrm{B}}^2 \ge \tau_{\mathrm{B}}}`.
    pub structure_gate: Vec<bool>,
    /// Observation gate indicator `\mathbf{1}{\log E_k \ge \log(1/\alpha)}`.
    pub observation_gate: Vec<bool>,
    /// Assigned I×K labels.
    pub labels: Vec<IKLabel>,
    /// Linearised hitting-time bound.
    pub hitting_time_bound: f32,
    /// Expected log-`e` increment from Theorem 1.
    pub increment: Vec<f32>,
    /// Anytime confidence values `1 - 1/E_k` used for the SAFE label.
    pub confidence: Vec<f32>,
    /// Effective Bures gaps after incorporating axis misalignment.
    pub effective_delta_b_sq: Vec<f32>,
}

/// Policy describing how per-pair structural gates should be folded into a
/// family-wide signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FamilyStructurePolicy {
    /// Open the family gate as soon as any pair has crossed its structural
    /// threshold.  Mirrors a Bonferroni-style union bound.
    Any,
    /// Require all pairs to have crossed their structural thresholds before the
    /// family gate opens.
    All,
    /// Require a strict majority of pairs to confirm the structural event.
    Majority,
}

impl FamilyStructurePolicy {
    fn fold(self, gates: &[bool]) -> bool {
        match self {
            Self::Any => gates.iter().any(|gate| *gate),
            Self::All => gates.iter().all(|gate| *gate),
            Self::Majority => {
                let positives = gates.iter().filter(|gate| **gate).count();
                positives * 2 > gates.len()
            }
        }
    }
}

/// Aggregated view over a multi-universe synchronisation pass.
#[derive(Debug, Clone)]
pub struct FamilyAggregation {
    /// Family-averaged log `e`-process across the pair contributions.
    pub log_e_family: f32,
    /// Anytime confidence implied by the family log `e`.
    pub confidence: f32,
    /// Structural gate indicator after applying [`FamilyStructurePolicy`].
    pub structure_gate: bool,
    /// Observation gate indicator using the family log `e` and global
    /// threshold.
    pub observation_gate: bool,
    /// Aggregated I×K label for the family.
    pub label: IKLabel,
    /// Number of pair contributions participating in the aggregation.
    pub pair_count: usize,
    /// Index of the pair with the largest log `e` contribution, if any.
    pub dominant_pair: Option<usize>,
    /// Maximal pair log `e` value, if any.
    pub dominant_log_e: Option<f32>,
}

/// State machine that fuses structural and observational evidence.
#[derive(Debug, Clone)]
pub struct SyncTheoremTrainer {
    config: SyncConfig,
    log_e: Vec<f32>,
    log_e_mean: f32,
    iteration: usize,
}

impl SyncTheoremTrainer {
    /// Create a trainer with zeroed state.
    pub fn new(config: SyncConfig) -> Result<Self, SyncError> {
        config.validate()?;
        Ok(Self {
            config,
            log_e: Vec::new(),
            log_e_mean: 0.0,
            iteration: 0,
        })
    }

    /// Borrow the configuration used by the trainer.
    pub fn config(&self) -> &SyncConfig {
        &self.config
    }

    /// Reset the internal log `e`-process and iteration counter.
    pub fn reset(&mut self) {
        self.log_e.clear();
        self.log_e_mean = 0.0;
        self.iteration = 0;
    }

    /// Replace the internal log `e`-state.
    pub fn set_log_e(&mut self, log_e: &[f32]) {
        self.log_e = log_e.to_vec();
        self.log_e_mean = if log_e.is_empty() {
            0.0
        } else {
            log_e.iter().sum::<f32>() / log_e.len() as f32
        };
    }

    /// Borrow the internal log `e`-state.
    pub fn log_e(&self) -> &[f32] {
        &self.log_e
    }

    /// Access the observation threshold `\log(1/\alpha)`.
    pub fn threshold(&self) -> f32 {
        self.config.threshold()
    }

    /// Compute the drift predicted by Theorem 1 for each batch element.
    pub fn expected_increment(
        &self,
        delta_b_sq: &[f32],
        epsilon: &[f32],
        cos_sq_phi: Option<&[f32]>,
    ) -> Result<Vec<f32>, SyncError> {
        Ok(self
            .expected_increment_profile(delta_b_sq, epsilon, cos_sq_phi)?
            .0)
    }

    /// Jointly compute the effective Bures gaps and the log-e drift lower bound.
    pub fn expected_increment_profile(
        &self,
        delta_b_sq: &[f32],
        epsilon: &[f32],
        cos_sq_phi: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>), SyncError> {
        let batch = delta_b_sq.len();
        self.ensure_length("epsilon", epsilon.len(), batch)?;
        let (effective_delta, increment): (Vec<f32>, Vec<f32>) = if let Some(cos) = cos_sq_phi {
            self.ensure_length("cos_sq_phi", cos.len(), batch)?;
            delta_b_sq
                .iter()
                .zip(cos.iter())
                .zip(epsilon.iter())
                .map(|((delta, cos), eps)| {
                    let effective = delta * cos;
                    let drift = 2.0 * effective - eps;
                    (effective, drift)
                })
                .unzip()
        } else {
            delta_b_sq
                .iter()
                .zip(epsilon.iter())
                .map(|(delta, eps)| {
                    let effective = delta * self.config.cos_phi_min;
                    let drift = 2.0 * effective - eps;
                    (effective, drift)
                })
                .unzip()
        };
        Ok((increment, effective_delta))
    }

    fn ensure_length(
        &self,
        field: &'static str,
        actual: usize,
        expected: usize,
    ) -> Result<(), SyncError> {
        if actual != expected {
            Err(SyncError::LengthMismatch {
                field,
                expected,
                actual,
            })
        } else {
            Ok(())
        }
    }

    fn default_or<'a>(
        &'a self,
        field: &'static str,
        provided: Option<&'a [f32]>,
        batch: usize,
        fallback: f32,
    ) -> Result<Cow<'a, [f32]>, SyncError> {
        if let Some(values) = provided {
            self.ensure_length(field, values.len(), batch)?;
            Ok(Cow::Borrowed(values))
        } else {
            Ok(Cow::Owned(vec![fallback; batch]))
        }
    }

    /// Perform a single synchronisation step over a batch of structural gaps.
    pub fn step(
        &mut self,
        delta_b_sq: &[f32],
        epsilon: Option<&[f32]>,
        cos_sq_phi: Option<&[f32]>,
    ) -> Result<SyncStep, SyncError> {
        let batch = delta_b_sq.len();
        if batch == 0 {
            return Ok(SyncStep {
                log_e: Vec::new(),
                structure_gate: Vec::new(),
                observation_gate: Vec::new(),
                labels: Vec::new(),
                hitting_time_bound: self.config.hitting_time_bound()?,
                increment: Vec::new(),
                confidence: Vec::new(),
                effective_delta_b_sq: Vec::new(),
            });
        }

        let epsilon = self.default_or("epsilon", epsilon, batch, self.config.epsilon_max)?;
        let cos_sq_phi =
            self.default_or("cos_sq_phi", cos_sq_phi, batch, self.config.cos_phi_min)?;

        let (increment, effective_delta_b_sq) = self.expected_increment_profile(
            delta_b_sq,
            epsilon.as_ref(),
            Some(cos_sq_phi.as_ref()),
        )?;
        if self.log_e.len() != batch {
            self.log_e = vec![0.0; batch];
        }

        for (value, inc) in self.log_e.iter_mut().zip(increment.iter()) {
            *value += *inc;
        }
        let log_e = self.log_e.clone();

        self.iteration += 1;

        let mut structure_gate: Vec<bool> = delta_b_sq
            .iter()
            .map(|delta| *delta >= self.config.tau_b)
            .collect();
        if self.config.slew_steps > 0 && self.iteration <= self.config.slew_steps {
            structure_gate.iter_mut().for_each(|gate| *gate = false);
        }

        let threshold = self.threshold();
        let observation_gate: Vec<bool> = log_e.iter().map(|value| *value >= threshold).collect();
        let confidence: Vec<f32> = log_e.iter().map(|value| 1.0 - (-value).exp()).collect();

        let mut labels = Vec::with_capacity(batch);
        for idx in 0..batch {
            let structure_open = structure_gate[idx];
            let observation_open = observation_gate[idx];
            if structure_open && observation_open {
                labels.push(IKLabel::Critical);
            } else if !structure_open
                && !observation_open
                && confidence[idx] >= 1.0 - self.config.alpha
            {
                labels.push(IKLabel::Safe);
            } else {
                labels.push(IKLabel::Abstain);
            }
        }

        self.log_e_mean = self.log_e.iter().sum::<f32>() / batch as f32;

        Ok(SyncStep {
            log_e,
            structure_gate,
            observation_gate,
            labels,
            hitting_time_bound: self.config.hitting_time_bound()?,
            increment,
            confidence,
            effective_delta_b_sq,
        })
    }

    /// Run sequential steps, returning the log trajectory and the first
    /// observation-gate iteration if one exists.
    pub fn run(
        &mut self,
        delta_b_sq: &[f32],
        epsilon: Option<&[f32]>,
        cos_sq_phi: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<usize>), SyncError> {
        let steps = delta_b_sq.len();
        if let Some(eps) = epsilon {
            self.ensure_length("epsilon", eps.len(), steps)?;
        }
        if let Some(cos) = cos_sq_phi {
            self.ensure_length("cos_sq_phi", cos.len(), steps)?;
        }

        let mut log_trajectory = Vec::with_capacity(steps);
        let mut gate_iter = None;

        for idx in 0..steps {
            let step = self.step(
                &delta_b_sq[idx..=idx],
                epsilon.map(|eps| &eps[idx..=idx]),
                cos_sq_phi.map(|cos| &cos[idx..=idx]),
            )?;

            if let Some(value) = step.log_e.first() {
                log_trajectory.push(*value);
                if gate_iter.is_none() && step.observation_gate.first().copied().unwrap_or(false) {
                    gate_iter = Some(idx);
                }
            }
        }

        Ok((log_trajectory, gate_iter))
    }

    /// Aggregate the per-pair outputs from [`SyncStep`] into a family-level
    /// signal, matching the multi-universe synchrony recipe from Theorem 5.
    pub fn aggregate_family(
        &self,
        step: &SyncStep,
        policy: FamilyStructurePolicy,
    ) -> Result<FamilyAggregation, SyncError> {
        if step.log_e.is_empty() {
            return Err(SyncError::EmptyFamily);
        }

        let pair_count = step.log_e.len();
        let log_e_family = step.log_e.iter().sum::<f32>() / pair_count as f32;
        let confidence = 1.0 - (-log_e_family).exp();

        let structure_gate = policy.fold(&step.structure_gate);
        let observation_gate = log_e_family >= self.threshold();

        let (dominant_pair, dominant_log_e) = step
            .log_e
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, value)| (Some(idx), Some(value)))
            .unwrap_or((None, None));

        let label = if structure_gate && observation_gate {
            IKLabel::Critical
        } else if !structure_gate && !observation_gate && confidence >= 1.0 - self.config.alpha {
            IKLabel::Safe
        } else {
            IKLabel::Abstain
        };

        Ok(FamilyAggregation {
            log_e_family,
            confidence,
            structure_gate,
            observation_gate,
            label,
            pair_count,
            dominant_pair,
            dominant_log_e,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(SyncConfig::new(0.05, 0.1, 0.02, 0.9, 2).is_ok());
        assert_eq!(
            SyncConfig::new(-0.1, 0.1, 0.02, 0.9, 0).unwrap_err(),
            SyncError::InvalidAlpha(-0.1)
        );
        assert_eq!(
            SyncConfig::new(0.5, -0.1, 0.0, 0.9, 0).unwrap_err(),
            SyncError::InvalidTau(-0.1)
        );
        assert_eq!(
            SyncConfig::new(0.5, 0.1, -0.02, 0.9, 0).unwrap_err(),
            SyncError::InvalidEpsilon(-0.02)
        );
        assert_eq!(
            SyncConfig::new(0.5, 0.1, 0.0, 1.5, 0).unwrap_err(),
            SyncError::InvalidCos(1.5)
        );
    }

    #[test]
    fn expected_increment_matches_formula() {
        let config = SyncConfig::new(0.1, 0.2, 0.05, 0.8, 0).unwrap();
        let trainer = SyncTheoremTrainer::new(config.clone()).unwrap();
        let delta = [0.3, 0.25];
        let epsilon = [0.02, 0.01];
        let cos = [0.9, 0.95];
        let increment = trainer
            .expected_increment(&delta, &epsilon, Some(&cos))
            .unwrap();
        assert!((increment[0] - (2.0 * 0.3 * 0.9 - 0.02)).abs() < 1e-6);
        assert!((increment[1] - (2.0 * 0.25 * 0.95 - 0.01)).abs() < 1e-6);

        let default_increment = trainer.expected_increment(&delta, &epsilon, None).unwrap();
        assert!((default_increment[0] - (2.0 * 0.3 * config.cos_phi_min - 0.02)).abs() < 1e-6);

        let (increment_profile, effective_gap) = trainer
            .expected_increment_profile(&delta, &epsilon, Some(&cos))
            .unwrap();
        assert_eq!(increment_profile, increment);
        assert!((effective_gap[0] - 0.27).abs() < 1e-6);
        assert!((effective_gap[1] - 0.2375).abs() < 1e-6);
    }

    #[test]
    fn step_updates_gates_and_labels() {
        let config = SyncConfig::new(0.1, 0.12, 0.01, 0.95, 0).unwrap();
        let mut trainer = SyncTheoremTrainer::new(config).unwrap();
        let step = trainer
            .step(&[0.15, 0.05], Some(&[0.01, 0.01]), Some(&[0.9, 0.9]))
            .unwrap();
        assert_eq!(step.structure_gate, vec![true, false]);
        assert_eq!(step.observation_gate, vec![false, false]);
        assert_eq!(step.labels, vec![IKLabel::Abstain, IKLabel::Abstain]);
        assert_eq!(trainer.log_e().len(), 2);
        assert_eq!(step.confidence.len(), 2);
        assert_eq!(step.effective_delta_b_sq.len(), 2);
    }

    #[test]
    fn run_returns_gate_iteration() {
        let config = SyncConfig::new(0.1, 0.1, 0.01, 1.0, 0).unwrap();
        let mut trainer = SyncTheoremTrainer::new(config).unwrap();
        let delta = vec![0.2; 10];
        let epsilon = vec![0.01; 10];
        let (trajectory, gate_iter) = trainer.run(&delta, Some(&epsilon), None).unwrap();
        assert_eq!(trajectory.len(), 10);
        assert!(gate_iter.is_some());
        assert!(trajectory[gate_iter.unwrap()] >= trainer.threshold());
    }

    #[test]
    fn state_accumulates_across_steps() {
        let config = SyncConfig::new(0.1, 0.2, 0.01, 0.8, 0).unwrap();
        let mut trainer = SyncTheoremTrainer::new(config).unwrap();
        let first = trainer.step(&[0.25], Some(&[0.01]), Some(&[0.9])).unwrap();
        let second = trainer.step(&[0.25], Some(&[0.01]), Some(&[0.9])).unwrap();
        assert!(second.log_e[0] > first.log_e[0]);
        assert_eq!(trainer.log_e()[0], second.log_e[0]);
    }

    #[test]
    fn set_log_e_reinitialises_state() {
        let config = SyncConfig::new(0.1, 0.2, 0.01, 0.8, 0).unwrap();
        let mut trainer = SyncTheoremTrainer::new(config).unwrap();
        trainer.set_log_e(&[1.0, -0.5]);
        assert_eq!(trainer.log_e(), &[1.0, -0.5]);
        let step = trainer
            .step(&[0.2, 0.2], Some(&[0.0, 0.0]), Some(&[1.0, 1.0]))
            .unwrap();
        assert_eq!(step.log_e.len(), 2);
    }

    #[test]
    fn hitting_time_bound_requires_positive_drift() {
        let mut config = SyncConfig::new(0.1, 0.05, 0.2, 0.5, 0).unwrap();
        config.epsilon_max = 1.0;
        assert!(matches!(
            config.hitting_time_bound(),
            Err(SyncError::NonPositiveDrift(_))
        ));
    }

    #[test]
    fn azuma_tail_controls_deviation() {
        let config = SyncConfig::new(0.1, 0.2, 0.01, 0.9, 0)
            .unwrap()
            .with_azuma_c(0.5)
            .unwrap();
        assert!((config.hitting_time_tail(0.0).unwrap() - 1.0).abs() < 1e-6);
        let tail = config.hitting_time_tail(3.0).unwrap();
        let reference = (-0.5f32 * 9.0).exp();
        assert!((tail - reference).abs() < 1e-6);
        let trainer = SyncTheoremTrainer::new(config).unwrap();
        assert!(trainer.config().hitting_time_tail(1.0).is_some());
        assert!(SyncConfig::new(0.1, 0.2, 0.01, 0.9, 0)
            .unwrap()
            .with_azuma_c(-0.5)
            .is_err());
    }

    #[test]
    fn family_aggregation_policies() {
        let config = SyncConfig::new(0.1, 0.2, 0.0, 1.0, 0).unwrap();
        let mut trainer = SyncTheoremTrainer::new(config).unwrap();
        let step = trainer
            .step(
                &[0.25, 0.1, 0.15],
                Some(&[0.0, 0.0, 0.0]),
                Some(&[1.0, 1.0, 1.0]),
            )
            .unwrap();

        let any = trainer
            .aggregate_family(&step, FamilyStructurePolicy::Any)
            .unwrap();
        assert!(any.structure_gate);
        assert_eq!(any.pair_count, 3);
        assert_eq!(any.dominant_pair, Some(0));

        let all = trainer
            .aggregate_family(&step, FamilyStructurePolicy::All)
            .unwrap();
        assert!(!all.structure_gate);

        let majority = trainer
            .aggregate_family(&step, FamilyStructurePolicy::Majority)
            .unwrap();
        assert!(!majority.structure_gate);
    }

    #[test]
    fn family_aggregation_respects_thresholds() {
        let config = SyncConfig::new(0.2, 0.05, 0.0, 1.0, 0).unwrap();
        let mut trainer = SyncTheoremTrainer::new(config).unwrap();
        for _ in 0..5 {
            trainer
                .step(
                    &[0.5, 0.5, 0.5],
                    Some(&[0.0, 0.0, 0.0]),
                    Some(&[1.0, 1.0, 1.0]),
                )
                .unwrap();
        }

        let step = trainer
            .step(
                &[0.5, 0.5, 0.5],
                Some(&[0.0, 0.0, 0.0]),
                Some(&[1.0, 1.0, 1.0]),
            )
            .unwrap();

        let aggregation = trainer
            .aggregate_family(&step, FamilyStructurePolicy::All)
            .unwrap();
        assert!(aggregation.structure_gate);
        assert!(aggregation.observation_gate);
        assert_eq!(aggregation.label, IKLabel::Critical);
        assert!(aggregation.confidence >= 1.0 - trainer.config().alpha);
    }
}
