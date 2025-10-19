// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

use bitflags::bitflags;
use core::fmt;
use std::collections::HashMap;
use std::str::FromStr;

use crate::theory::spiral_dynamics::{HopfRegime, PsiSpiralMetrics};

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PsiComponent: u32 {
        const LOSS         = 1 << 0;
        const GRAD_NORM    = 1 << 1;
        const UPDATE_RATIO = 1 << 2;
        const ACT_DRIFT    = 1 << 3;
        const ATTN_ENTROPY = 1 << 4;
        const BAND_ENERGY  = 1 << 5;
    }
}

impl PsiComponent {
    /// Returns the default component mask used when configuration is omitted.
    pub fn defaults() -> Self {
        PsiComponent::LOSS
            | PsiComponent::GRAD_NORM
            | PsiComponent::UPDATE_RATIO
            | PsiComponent::ACT_DRIFT
    }

    fn label(self) -> &'static str {
        match self {
            PsiComponent::LOSS => "loss",
            PsiComponent::GRAD_NORM => "grad_norm",
            PsiComponent::UPDATE_RATIO => "update_ratio",
            PsiComponent::ACT_DRIFT => "act_drift",
            PsiComponent::ATTN_ENTROPY => "attn_entropy",
            PsiComponent::BAND_ENERGY => "band_energy",
            _ => "unknown",
        }
    }

    fn from_token(token: &str) -> Result<Self, String> {
        match token.trim().to_ascii_lowercase().as_str() {
            "loss" => Ok(PsiComponent::LOSS),
            "grad" | "grad_norm" | "gradnorm" => Ok(PsiComponent::GRAD_NORM),
            "update" | "update_ratio" | "ratio" => Ok(PsiComponent::UPDATE_RATIO),
            "act" | "act_drift" | "drift" => Ok(PsiComponent::ACT_DRIFT),
            "attn" | "attn_entropy" | "attention" | "entropy" => Ok(PsiComponent::ATTN_ENTROPY),
            "band" | "band_energy" | "energy" => Ok(PsiComponent::BAND_ENERGY),
            other => Err(format!(
                "unknown psi component '{}': expected one of Loss, Grad, Update, Act, Attn, Band",
                other
            )),
        }
    }

    /// Parses a comma, pipe, or whitespace separated list of component tokens.
    pub fn parse_list(spec: &str) -> Result<Self, String> {
        let mut mask = PsiComponent::empty();
        for token in spec
            .split(|c| matches!(c, ',' | '|' | ';'))
            .flat_map(|segment| segment.split_whitespace())
        {
            if token.is_empty() {
                continue;
            }
            mask |= PsiComponent::from_token(token)?;
        }
        if mask.is_empty() {
            Ok(PsiComponent::defaults())
        } else {
            Ok(mask)
        }
    }
}

impl fmt::Display for PsiComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl FromStr for PsiComponent {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PsiComponent::from_token(s)
    }
}

#[derive(Clone, Debug)]
pub struct PsiConfig {
    pub enabled: bool,
    pub components: PsiComponent,
    pub weights: HashMap<PsiComponent, f32>,
    pub ema_alpha: f32,
    pub sample_rate: u32,
    pub thresholds: HashMap<PsiComponent, f32>,
}

impl PsiConfig {
    /// Builds an automated configuration derived from scheduler hints.
    pub fn automated(hint: PsiAutomationHint) -> Self {
        let depth = hint.depth().max(1);
        let ema_alpha = if depth <= 24 {
            0.28
        } else if depth <= 96 {
            0.22
        } else {
            0.16
        };
        let sample_rate = if depth <= 48 {
            1
        } else if depth <= 128 {
            2
        } else {
            4
        };
        let mut cfg = PsiConfig::default();
        cfg.enabled = true;
        cfg.components = PsiComponent::defaults() | PsiComponent::BAND_ENERGY;
        cfg.ema_alpha = ema_alpha;
        cfg.sample_rate = sample_rate.max(1);
        cfg.weights.insert(PsiComponent::LOSS, 1.0);
        cfg.weights.insert(PsiComponent::GRAD_NORM, 0.35);
        cfg.weights.insert(
            PsiComponent::UPDATE_RATIO,
            (depth as f32).recip().sqrt().clamp(0.2, 0.6),
        );
        cfg.weights
            .insert(PsiComponent::ACT_DRIFT, hint.drift_weight.clamp(0.1, 0.4));
        cfg.weights.insert(
            PsiComponent::BAND_ENERGY,
            (hint.band_focus / depth as f32).clamp(0.1, 0.5),
        );
        cfg
    }

    /// Applies the Spiral-derived tuning adjustments to the configuration.
    pub fn apply_spiral_tuning(&mut self, tuning: &PsiSpiralTuning) {
        if tuning.required_components.is_empty() {
            // Nothing to extend in the component mask; continue with weight updates.
        } else {
            self.components |= tuning.required_components;
        }

        for (component, delta) in &tuning.weight_increments {
            let entry = self.weights.entry(*component).or_insert(0.0);
            *entry = (*entry + *delta).clamp(0.0, 10.0);
        }

        for (component, shift) in &tuning.threshold_shifts {
            let entry = self.thresholds.entry(*component).or_insert(0.0);
            *entry += *shift;
        }

        if tuning.stability_score < 0.25 {
            self.sample_rate = self.sample_rate.max(1);
        } else if tuning.stability_score < 0.6 {
            self.sample_rate = self.sample_rate.max(2);
        } else {
            self.sample_rate = self.sample_rate.max(3);
        }
    }
}

impl Default for PsiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            components: PsiComponent::empty(),
            weights: HashMap::new(),
            ema_alpha: 0.0,
            sample_rate: 0,
            thresholds: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PsiInput {
    pub loss: f32,
    pub grad_l2: f32,
    pub update_ratio: f32,
    pub act_drift: f32,
    pub attn_entropy: f32,
    pub band_energy: f32,
}

#[derive(Clone, Debug)]
pub struct PsiReading {
    pub total: f32,
    pub breakdown: HashMap<PsiComponent, f32>,
    pub step: u64,
}

#[derive(Clone, Debug)]
pub enum PsiEvent {
    ThresholdCross {
        component: PsiComponent,
        value: f32,
        threshold: f32,
        up: bool,
        step: u64,
    },
}

/// Advisory emitted when projecting Spiral dynamics health into PSI channels.
#[derive(Clone, Debug, PartialEq)]
pub struct PsiSpiralAdvisory {
    /// Hopf regime classification derived from the cubic coefficient.
    pub regime: HopfRegime,
    /// Linear growth rate at the origin.
    pub mu_eff0: f32,
    /// Cubic coefficient α₃ after the center-manifold correction.
    pub alpha3: f32,
    /// Signed audit/container gap κa/τ − σ_s ρ / λ.
    pub audit_container_gap: f32,
    /// Dimensionless audit gain cluster κ a / (τ ν).
    pub audit_cluster: f32,
    /// Dimensionless container cluster σ_s ρ / (λ ν).
    pub container_cluster: f32,
}

impl PsiSpiralAdvisory {
    /// Builds an advisory from the Spiral dynamics metrics.
    pub fn from_metrics(metrics: &PsiSpiralMetrics) -> Self {
        Self {
            regime: metrics.hopf.regime,
            mu_eff0: metrics.hopf.mu_eff0 as f32,
            alpha3: metrics.hopf.alpha3 as f32,
            audit_container_gap: metrics.balance.difference as f32,
            audit_cluster: metrics.dimensionless.audit_cluster as f32,
            container_cluster: metrics.dimensionless.container_cluster as f32,
        }
    }

    /// Normalised stability score in \([0, 1]\) derived from the Hopf data.
    pub fn stability_score(&self) -> f32 {
        let linear_margin = (-self.mu_eff0).max(0.0);
        let cubic_margin = self.alpha3.max(0.0);
        let base = (linear_margin + 0.5 * cubic_margin).tanh();
        let regime_penalty = match self.regime {
            HopfRegime::Supercritical => 0.0,
            HopfRegime::Degenerate => 0.2,
            HopfRegime::Subcritical => 0.4,
        };
        let audit_penalty = (self.audit_container_gap.max(0.0)).tanh() * 0.5;
        (base * (1.0 - regime_penalty) - audit_penalty).clamp(0.0, 1.0)
    }

    /// Returns `true` when the advisory flags a destabilising audit surplus.
    pub fn audit_overbias(&self) -> bool {
        self.regime == HopfRegime::Subcritical || self.audit_container_gap > 0.0
    }

    /// Magnitude of the recommended container reinforcement in PSI weights.
    pub fn container_reinforcement(&self) -> f32 {
        (self.audit_container_gap.max(0.0) + (-self.mu_eff0).max(0.0))
            .tanh()
            .clamp(0.0, 1.0)
    }

    /// Produces a telemetry tuning plan matching the Spiral health advisory.
    pub fn tuning(&self) -> PsiSpiralTuning {
        let mut required_components = PsiComponent::empty();
        let mut weight_increments = HashMap::new();
        let mut threshold_shifts = HashMap::new();

        let reinforcement = self.container_reinforcement();
        if reinforcement > 0.0 {
            required_components |= PsiComponent::ACT_DRIFT | PsiComponent::BAND_ENERGY;
            weight_increments.insert(PsiComponent::ACT_DRIFT, 0.2 * reinforcement);
            weight_increments.insert(PsiComponent::BAND_ENERGY, 0.15 * reinforcement);
        }

        let audit_bias = self.audit_container_gap.max(0.0);
        if audit_bias > 0.0 {
            required_components |= PsiComponent::UPDATE_RATIO | PsiComponent::GRAD_NORM;
            weight_increments
                .entry(PsiComponent::UPDATE_RATIO)
                .and_modify(|w| *w += 0.1 * audit_bias)
                .or_insert(0.1 * audit_bias);
            threshold_shifts.insert(
                PsiComponent::GRAD_NORM,
                -0.08 * (audit_bias + reinforcement),
            );
        }

        if self.regime == HopfRegime::Subcritical {
            required_components |= PsiComponent::LOSS | PsiComponent::ATTN_ENTROPY;
            threshold_shifts
                .entry(PsiComponent::LOSS)
                .and_modify(|v| *v -= 0.1)
                .or_insert(-0.1 - 0.1 * reinforcement);
            weight_increments
                .entry(PsiComponent::ATTN_ENTROPY)
                .and_modify(|v| *v += 0.12)
                .or_insert(0.12 + 0.1 * reinforcement);
        }

        PsiSpiralTuning {
            stability_score: self.stability_score(),
            required_components,
            weight_increments,
            threshold_shifts,
        }
    }
}

/// Convenience helper that derives a Spiral advisory directly from the
/// parameter tuple used in `psi_spiral_metrics`.
pub fn psi_spiral_advisory_from_params(
    mu0: f64,
    gamma: f64,
    omega: f64,
    nu: f64,
    kappa: f64,
    a: f64,
    tau: f64,
    theta: f64,
    sigma_s: f64,
    rho: f64,
    lambda: f64,
    c_max: f64,
) -> Option<PsiSpiralAdvisory> {
    let metrics = crate::theory::spiral_dynamics::psi_spiral_metrics(
        mu0, gamma, omega, nu, kappa, a, tau, theta, sigma_s, rho, lambda, c_max,
    )?;
    Some(PsiSpiralAdvisory::from_metrics(&metrics))
}

/// Computes the PSI tuning prescription derived from the Spiral parameters.
pub fn psi_spiral_tuning_from_params(
    mu0: f64,
    gamma: f64,
    omega: f64,
    nu: f64,
    kappa: f64,
    a: f64,
    tau: f64,
    theta: f64,
    sigma_s: f64,
    rho: f64,
    lambda: f64,
    c_max: f64,
) -> Option<PsiSpiralTuning> {
    psi_spiral_advisory_from_params(
        mu0, gamma, omega, nu, kappa, a, tau, theta, sigma_s, rho, lambda, c_max,
    )
    .map(|advisory| advisory.tuning())
}

/// Bundle of PSI telemetry adjustments derived from a Spiral advisory.
#[derive(Clone, Debug, PartialEq)]
pub struct PsiSpiralTuning {
    pub stability_score: f32,
    pub required_components: PsiComponent,
    pub weight_increments: HashMap<PsiComponent, f32>,
    pub threshold_shifts: HashMap<PsiComponent, f32>,
}

impl PsiSpiralTuning {
    pub fn is_neutral(&self) -> bool {
        self.required_components.is_empty()
            && self
                .weight_increments
                .values()
                .all(|delta| delta.abs() < f32::EPSILON)
            && self
                .threshold_shifts
                .values()
                .all(|delta| delta.abs() < f32::EPSILON)
    }
}

#[derive(Clone, Debug)]
pub struct PsiMeter {
    cfg: PsiConfig,
    ema: HashMap<PsiComponent, f32>,
    step: u64,
}

impl PsiMeter {
    pub fn new(mut cfg: PsiConfig) -> Self {
        if cfg.components.is_empty() {
            cfg.components = PsiComponent::defaults();
        }
        if cfg.weights.is_empty() {
            cfg.weights.insert(PsiComponent::LOSS, 1.0);
            cfg.weights.insert(PsiComponent::GRAD_NORM, 0.3);
            cfg.weights.insert(PsiComponent::UPDATE_RATIO, 0.4);
            cfg.weights.insert(PsiComponent::ACT_DRIFT, 0.2);
        }
        if cfg.ema_alpha <= 0.0 || cfg.ema_alpha >= 1.0 {
            cfg.ema_alpha = 0.2;
        }
        if cfg.sample_rate == 0 {
            cfg.sample_rate = 1;
        }
        Self {
            cfg,
            ema: HashMap::new(),
            step: 0,
        }
    }

    #[inline]
    fn take_component(&self, c: PsiComponent, x: &PsiInput) -> f32 {
        match c {
            PsiComponent::LOSS => x.loss.abs(),
            PsiComponent::GRAD_NORM => (1.0 + x.grad_l2).ln(),
            PsiComponent::UPDATE_RATIO => x.update_ratio,
            PsiComponent::ACT_DRIFT => x.act_drift,
            PsiComponent::ATTN_ENTROPY => x.attn_entropy,
            PsiComponent::BAND_ENERGY => x.band_energy,
            _ => 0.0,
        }
    }

    pub fn update(&mut self, input: &PsiInput) -> (PsiReading, Vec<PsiEvent>) {
        self.step += 1;
        let mut events = Vec::new();
        let mut breakdown = self.ema.clone();

        if !self.cfg.enabled || (self.step as u32) % self.cfg.sample_rate != 0 {
            let total = self
                .cfg
                .weights
                .iter()
                .map(|(component, weight)| {
                    weight * breakdown.get(component).copied().unwrap_or(0.0)
                })
                .sum();
            return (
                PsiReading {
                    total,
                    breakdown,
                    step: self.step,
                },
                events,
            );
        }

        let alpha = self.cfg.ema_alpha;
        let mut total = 0.0;

        for &comp in [
            PsiComponent::LOSS,
            PsiComponent::GRAD_NORM,
            PsiComponent::UPDATE_RATIO,
            PsiComponent::ACT_DRIFT,
            PsiComponent::ATTN_ENTROPY,
            PsiComponent::BAND_ENERGY,
        ]
        .iter()
        {
            if !self.cfg.components.contains(comp) {
                continue;
            }
            let raw = self.take_component(comp, input);
            let prev = *self.ema.get(&comp).unwrap_or(&raw);
            let ema = alpha * raw + (1.0 - alpha) * prev;
            self.ema.insert(comp, ema);
            breakdown.insert(comp, ema);

            let weight = *self.cfg.weights.get(&comp).unwrap_or(&0.0);
            total += weight * ema;

            if let Some(&threshold) = self.cfg.thresholds.get(&comp) {
                let up = ema > threshold && prev <= threshold;
                let down = ema < threshold && prev >= threshold;
                if up || down {
                    events.push(PsiEvent::ThresholdCross {
                        component: comp,
                        value: ema,
                        threshold,
                        up,
                        step: self.step,
                    });
                }
            }
        }

        (
            PsiReading {
                total,
                breakdown,
                step: self.step,
            },
            events,
        )
    }

    pub fn config(&self) -> &PsiConfig {
        &self.cfg
    }
}

/// Hint produced by the roundtable scheduler to seed ψ metering defaults.
#[derive(Clone, Copy, Debug, Default)]
pub struct PsiAutomationHint {
    pub above: u32,
    pub here: u32,
    pub beneath: u32,
    pub band_focus: f32,
    pub drift_weight: f32,
}

impl PsiAutomationHint {
    pub fn depth(&self) -> u32 {
        self.above + self.here + self.beneath
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spiral_advisory_tracks_audit_bias() {
        let metrics = crate::theory::spiral_dynamics::psi_spiral_metrics(
            -0.05, 0.6, 1.0, 0.9, 0.7, 0.8, 1.2, 0.2, 0.5, 0.4, 1.1, 0.35,
        )
        .unwrap();
        let advisory = PsiSpiralAdvisory::from_metrics(&metrics);
        assert!(advisory.audit_overbias());
        assert!(advisory.container_reinforcement() > 0.0);
    }

    #[test]
    fn spiral_tuning_biases_weights_and_components() {
        let advisory = psi_spiral_advisory_from_params(
            -0.05, 0.6, 1.0, 0.9, 0.7, 0.8, 1.2, 0.2, 0.5, 0.4, 1.1, 0.35,
        )
        .unwrap();
        let tuning = advisory.tuning();
        assert!(tuning.stability_score >= 0.0 && tuning.stability_score <= 1.0);
        assert!(tuning.required_components.contains(PsiComponent::ACT_DRIFT));
        assert!(tuning
            .weight_increments
            .get(&PsiComponent::ACT_DRIFT)
            .is_some());
        let mut cfg = PsiConfig::default();
        cfg.enabled = true;
        cfg.components = PsiComponent::LOSS;
        cfg.weights.insert(PsiComponent::LOSS, 1.0);
        cfg.apply_spiral_tuning(&tuning);
        assert!(cfg.components.contains(PsiComponent::ACT_DRIFT));
        assert!(cfg.weights.get(&PsiComponent::ACT_DRIFT).is_some());
        assert!(cfg.sample_rate >= 1);
    }

    #[test]
    fn ema_basic() {
        let mut cfg = PsiConfig::default();
        cfg.enabled = true;
        cfg.components = PsiComponent::LOSS;
        cfg.ema_alpha = 0.5;
        let mut meter = PsiMeter::new(cfg);
        let (reading, events) = meter.update(&PsiInput {
            loss: 1.0,
            ..PsiInput::default()
        });
        assert!(events.is_empty());
        let loss = *reading.breakdown.get(&PsiComponent::LOSS).unwrap();
        assert!((loss - 1.0).abs() < 1e-6);
        let (reading, _) = meter.update(&PsiInput {
            loss: 3.0,
            ..PsiInput::default()
        });
        let loss = *reading.breakdown.get(&PsiComponent::LOSS).unwrap();
        assert!((loss - 2.0).abs() < 1e-6);
    }

    #[test]
    fn threshold_cross_up_and_down() {
        let mut cfg = PsiConfig::default();
        cfg.enabled = true;
        cfg.components = PsiComponent::GRAD_NORM;
        cfg.ema_alpha = 0.5;
        cfg.thresholds
            .insert(PsiComponent::GRAD_NORM, (1.0 + 1.0f32.sqrt()).ln());
        let mut meter = PsiMeter::new(cfg);
        let (_, events) = meter.update(&PsiInput {
            grad_l2: 0.0,
            ..PsiInput::default()
        });
        assert!(events.is_empty());
        let (_, events) = meter.update(&PsiInput {
            grad_l2: 10.0,
            ..PsiInput::default()
        });
        assert_eq!(events.len(), 1);
        match &events[0] {
            PsiEvent::ThresholdCross { up, .. } => assert!(*up),
        }
        let (_, events) = meter.update(&PsiInput {
            grad_l2: 0.0,
            ..PsiInput::default()
        });
        assert_eq!(events.len(), 1);
        match &events[0] {
            PsiEvent::ThresholdCross { up, .. } => assert!(!*up),
        }
    }

    #[test]
    fn disabled_short_circuits() {
        let mut cfg = PsiConfig::default();
        cfg.enabled = false;
        cfg.components = PsiComponent::LOSS;
        let mut meter = PsiMeter::new(cfg);
        let (reading, events) = meter.update(&PsiInput {
            loss: 42.0,
            ..PsiInput::default()
        });
        assert!(events.is_empty());
        assert!(reading.breakdown.is_empty());
        assert_eq!(reading.total, 0.0);
    }

    #[test]
    fn parse_component_lists() {
        let mask = PsiComponent::parse_list("Loss, Grad Update").unwrap();
        assert!(mask.contains(PsiComponent::LOSS));
        assert!(mask.contains(PsiComponent::GRAD_NORM));
        assert!(mask.contains(PsiComponent::UPDATE_RATIO));
        assert!(!mask.contains(PsiComponent::ACT_DRIFT));
    }

    #[test]
    fn psi_meter_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PsiMeter>();
    }
}
