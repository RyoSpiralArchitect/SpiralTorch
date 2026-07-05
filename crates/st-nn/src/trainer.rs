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

#[cfg(feature = "selfsup")]
pub mod selfsup;

use crate::cloud::CloudTargetSummary;
use crate::dataset::DataLoader;
use crate::execution::{push_backend_policy, BackendPolicy};
use crate::gnn::spiralk::{GraphConsensusBridge, GraphConsensusDigest};
use crate::gnn::RoundtableBandSignal;
#[cfg(feature = "golden")]
use crate::golden::{GoldenBlackcatPulse, GoldenCooperativeDirective, GoldenCouncilSnapshot};
#[cfg(feature = "psi")]
use crate::language::{DesirePsiBridge, DesirePsiSummary};
use crate::language::{
    DesireRoundtableBridge, DesireRoundtableSummary, DesireTelemetryBundle, DesireTrainerBridge,
    DesireTrainerSummary,
};
use crate::loss::Loss;
use crate::module::Module;
use crate::optim::{LocalLearningRateAdapter, SpectralLrAdapter, SpectralLrAdapterState};
use crate::plan::RankPlanner;
use crate::roundtable::{
    simulate_proposal_locally, BlackcatModerator, BlackcatScore, DistConfig, GlobalProposal,
    HeurOpKind, HeurOpLog, MetaConductor, ModeratorMinutes, OutcomeBand, RoundtableGnnBridge,
    RoundtableNode,
};
use crate::schedule::{BandEnergy, GradientBands, RoundtableConfig, RoundtableSchedule};
use crate::zspace_coherence::{
    CoherenceDiagnostics, CoherenceLabel, CoherenceObservation, ZSpaceTraceEvent,
};
use crate::{PureResult, Tensor, TensorError};
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, SeedableRng};
use serde_json::Value;
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
use st_core::backend::unison_heuristics::RankKind;
use st_core::distributed::{AccumulatorSyncError, AccumulatorSynchronizer};
use st_core::ecosystem::{
    CloudConnector, ConnectorEvent, DistributionSummary, EcosystemRegistry, MetricSample,
    RankPlanSummary, RoundtableConfigSummary, RoundtableSummary,
};
#[cfg(feature = "collapse")]
use st_core::engine::collapse_drive::{CollapseConfig, CollapseDrive, DriveCmd};
use st_core::ops::rank_entry::RankPlan;
use st_core::plugin::{global_registry, PluginEvent};
use st_core::runtime::autopilot::Autopilot;
use st_core::runtime::blackcat::{BlackCatRuntime, BlackcatRuntimeStats, StepMetrics};
#[cfg(feature = "collapse")]
use st_core::telemetry::hub::CollapsePulse;
use st_core::telemetry::hub::{self, LoopbackEnvelope, SoftlogicZFeedback};
#[cfg(feature = "psi")]
use st_core::telemetry::psi::{PsiComponent, PsiConfig, PsiInput, PsiMeter, PsiReading};
#[cfg(feature = "psychoid")]
use st_core::telemetry::psychoid::{PsychoidConfig, PsychoidEvent, PsychoidMeter, PsychoidReading};
use st_core::telemetry::region_visualizer::{
    RegionHeatmapCell, RegionHeatmapHistory, RegionHeatmapSnapshot,
};
use st_core::telemetry::xai_report::AttributionReport;
use st_core::telemetry::zspace_region::{
    ZSpaceRadiusBand, ZSpaceRegionDescriptor, ZSpaceRegionKey, ZSpaceSpinBand,
};
use st_core::theory::zpulse::ZScale;
use st_tensor::{
    set_tensor_op_meta_observer, topos::OpenCartesianTopos, GradientSummary, TensorOpMetaEvent,
    TensorOpMetaObserver,
};
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

#[cfg(test)]
fn tensor_meta_observer_test_lock() -> std::sync::MutexGuard<'static, ()> {
    tensor_meta_observer_lock()
}

fn tensor_meta_observer_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
thread_local! {
    static STRICT_GPU_TEST_OVERRIDE: std::cell::RefCell<Option<bool>> =
        const { std::cell::RefCell::new(None) };
}

/// Adaptive curvature controller that nudges the trainer's hyperbolic geometry
/// towards the pressure window observed in recent gradients.
#[derive(Debug, Clone)]
pub struct CurvatureScheduler {
    min_curvature: f32,
    max_curvature: f32,
    target_pressure: f32,
    tolerance: f32,
    step: f32,
    kp: f32,
    alpha: f32,
    current: f32,
    ema_pressure: Option<f32>,
    ema_pressure2: Option<f32>,
    stability_threshold: f32,
    stability_boost: f32,
    stable_steps: u32,
    dither_strength: f32,
    dither_period: u32,
    dither_sign: f32,
}

const CURVATURE_PRESSURE_MAX: f32 = 1.0e19;

impl CurvatureScheduler {
    /// Builds a scheduler anchored to the provided curvature range and target
    /// gradient pressure. `initial` is clamped into `[min_curvature,
    /// max_curvature]` and the range itself is normalised so both bounds remain
    /// negative.
    pub fn new(initial: f32, min_curvature: f32, max_curvature: f32, target_pressure: f32) -> Self {
        let mut min_curvature = min_curvature.min(-1e-6);
        let mut max_curvature = max_curvature.min(-1e-6);
        if min_curvature > max_curvature {
            core::mem::swap(&mut min_curvature, &mut max_curvature);
        }
        let current = initial.clamp(min_curvature, max_curvature).min(-1e-6);
        Self {
            min_curvature,
            max_curvature,
            target_pressure: target_pressure.max(0.0),
            tolerance: 0.05,
            step: 0.05,
            kp: 1.0,
            alpha: 0.2,
            current,
            ema_pressure: None,
            ema_pressure2: None,
            stability_threshold: 1.0e-3,
            stability_boost: 0.5,
            stable_steps: 0,
            dither_strength: 0.15,
            dither_period: 8,
            dither_sign: 1.0,
        }
    }

    /// Adjusts the maximum step a single observation may move the curvature by.
    pub fn with_step(mut self, step: f32) -> Self {
        if step.is_finite() && step > 0.0 {
            self.step = step;
        }
        self
    }

    /// Adjusts the proportional gain used to convert pressure error into curvature delta.
    pub fn with_proportional_gain(mut self, gain: f32) -> Self {
        self.set_proportional_gain(gain);
        self
    }

    /// Adjusts the tolerated pressure band before the curvature is nudged.
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        if tolerance.is_finite() && tolerance >= 0.0 {
            self.tolerance = tolerance;
        }
        self
    }

    /// Adjusts the stability threshold used to detect "settled" pressure conditions.
    pub fn with_stability_threshold(mut self, threshold: f32) -> Self {
        self.set_stability_threshold(threshold);
        self
    }

    /// Adjusts the stability boost applied to the proportional gain when pressure settles.
    pub fn with_stability_boost(mut self, boost: f32) -> Self {
        self.set_stability_boost(boost);
        self
    }

    /// Enables a small curvature dither when pressure remains stable within tolerance.
    pub fn with_dither(mut self, strength: f32, period: u32) -> Self {
        self.set_dither(strength, period);
        self
    }

    /// Applies environment overrides to the scheduler in-place.
    ///
    /// Supported variables:
    /// - `SPIRAL_CURVATURE_STEP`
    /// - `SPIRAL_CURVATURE_TOLERANCE`
    /// - `SPIRAL_CURVATURE_ALPHA`
    /// - `SPIRAL_CURVATURE_KP`
    /// - `SPIRAL_CURVATURE_STABILITY_THRESHOLD`
    /// - `SPIRAL_CURVATURE_STABILITY_BOOST`
    /// - `SPIRAL_CURVATURE_DITHER_STRENGTH`
    /// - `SPIRAL_CURVATURE_DITHER_PERIOD`
    pub fn apply_env_overrides(&mut self) {
        if let Ok(value) = env::var("SPIRAL_CURVATURE_STEP") {
            if let Ok(step) = value.parse::<f32>() {
                self.set_step(step);
            }
        }
        if let Ok(value) = env::var("SPIRAL_CURVATURE_TOLERANCE") {
            if let Ok(tolerance) = value.parse::<f32>() {
                self.set_tolerance(tolerance);
            }
        }
        if let Ok(value) = env::var("SPIRAL_CURVATURE_ALPHA") {
            if let Ok(alpha) = value.parse::<f32>() {
                self.set_smoothing(alpha);
            }
        }
        if let Ok(value) = env::var("SPIRAL_CURVATURE_KP") {
            if let Ok(kp) = value.parse::<f32>() {
                self.set_proportional_gain(kp);
            }
        }
        if let Ok(value) = env::var("SPIRAL_CURVATURE_STABILITY_THRESHOLD") {
            if let Ok(threshold) = value.parse::<f32>() {
                self.set_stability_threshold(threshold);
            }
        }
        if let Ok(value) = env::var("SPIRAL_CURVATURE_STABILITY_BOOST") {
            if let Ok(boost) = value.parse::<f32>() {
                self.set_stability_boost(boost);
            }
        }

        let mut dither_strength = self.dither_strength;
        let mut dither_period = self.dither_period;
        let mut changed = false;
        if let Ok(value) = env::var("SPIRAL_CURVATURE_DITHER_STRENGTH") {
            if let Ok(strength) = value.parse::<f32>() {
                dither_strength = strength;
                changed = true;
            }
        }
        if let Ok(value) = env::var("SPIRAL_CURVATURE_DITHER_PERIOD") {
            if let Ok(period) = value.parse::<u32>() {
                dither_period = period;
                changed = true;
            }
        }
        if changed {
            self.set_dither(dither_strength, dither_period);
        }
    }

    /// Applies environment overrides and returns `self` for fluent construction.
    pub fn with_env_overrides(mut self) -> Self {
        self.apply_env_overrides();
        self
    }

    /// Adjusts the smoothing factor applied to the pressure EMA.
    pub fn with_smoothing(mut self, alpha: f32) -> Self {
        if alpha.is_finite() && alpha > 0.0 {
            self.alpha = alpha.clamp(0.01, 1.0);
        }
        self
    }

    /// Returns the proportional gain used to compute curvature deltas.
    pub fn proportional_gain(&self) -> f32 {
        self.kp
    }

    /// Returns the curvature currently enforced by the scheduler.
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Returns the minimum curvature allowed by the scheduler.
    pub fn min_curvature(&self) -> f32 {
        self.min_curvature
    }

    /// Returns the maximum curvature allowed by the scheduler.
    pub fn max_curvature(&self) -> f32 {
        self.max_curvature
    }

    /// Returns the configured target pressure.
    pub fn target_pressure(&self) -> f32 {
        self.target_pressure
    }

    /// Returns the maximum curvature delta applied per observation.
    pub fn step_size(&self) -> f32 {
        self.step
    }

    /// Returns the stability threshold applied to pressure variance estimates.
    pub fn stability_threshold(&self) -> f32 {
        self.stability_threshold
    }

    /// Returns the stability boost applied to the proportional gain.
    pub fn stability_boost(&self) -> f32 {
        self.stability_boost
    }

    /// Returns the dither strength expressed as a fraction of the step size.
    pub fn dither_strength(&self) -> f32 {
        self.dither_strength
    }

    /// Returns the minimum stable observation count before applying dither.
    pub fn dither_period(&self) -> u32 {
        self.dither_period
    }

    /// Returns the tolerated pressure band before adjustments.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Returns the smoothing factor applied to the pressure EMA.
    pub fn smoothing(&self) -> f32 {
        self.alpha
    }

    /// Returns the last smoothed pressure observation, if available.
    pub fn last_pressure(&self) -> Option<f32> {
        self.ema_pressure
    }

    /// Returns the estimated pressure variance (EMA(p²) - EMA(p)²), if available.
    pub fn last_pressure_variance(&self) -> Option<f32> {
        let mean = self.ema_pressure?;
        let second = self.ema_pressure2?;
        if !mean.is_finite() || !second.is_finite() {
            return None;
        }
        let variance = (f64::from(second) - f64::from(mean) * f64::from(mean)).max(0.0);
        Some(variance.min(f64::from(f32::MAX)) as f32)
    }

    /// Returns the estimated relative variance (var / (mean² + eps)), if available.
    pub fn last_pressure_rel_variance(&self) -> Option<f32> {
        let mean = self.ema_pressure?;
        let var = self.last_pressure_variance()?;
        if !mean.is_finite() || !var.is_finite() {
            return None;
        }
        let denom = f64::from(mean) * f64::from(mean) + 1.0e-6;
        let rel = f64::from(var) / denom;
        if rel.is_finite() {
            Some(rel.min(f64::from(f32::MAX)) as f32)
        } else {
            None
        }
    }

    /// Synchronises the scheduler with an externally adjusted curvature and
    /// clears the pressure history so subsequent observations start fresh.
    pub fn sync(&mut self, curvature: f32) {
        self.current = curvature
            .clamp(self.min_curvature, self.max_curvature)
            .min(-1e-6);
        self.ema_pressure = None;
        self.ema_pressure2 = None;
        self.stable_steps = 0;
        self.dither_sign = 1.0;
    }

    /// Adjusts the curvature bounds while keeping the current value within range.
    pub fn set_bounds(&mut self, min_curvature: f32, max_curvature: f32) {
        let mut min_curvature = min_curvature.min(-1e-6);
        let mut max_curvature = max_curvature.min(-1e-6);
        if min_curvature > max_curvature {
            core::mem::swap(&mut min_curvature, &mut max_curvature);
        }
        self.min_curvature = min_curvature;
        self.max_curvature = max_curvature;
        self.current = self
            .current
            .clamp(self.min_curvature, self.max_curvature)
            .min(-1e-6);
    }

    /// Adjusts the target pressure for future observations.
    pub fn set_target_pressure(&mut self, target_pressure: f32) {
        self.target_pressure = target_pressure.max(0.0);
    }

    /// Adjusts the maximum step a single observation may move the curvature by.
    pub fn set_step(&mut self, step: f32) {
        if step.is_finite() && step > 0.0 {
            self.step = step;
        }
    }

    /// Adjusts the proportional gain used to convert pressure error into curvature delta.
    pub fn set_proportional_gain(&mut self, gain: f32) {
        if gain.is_finite() && gain >= 0.0 {
            self.kp = gain;
        }
    }

    /// Adjusts the tolerated pressure band before the curvature is nudged.
    pub fn set_tolerance(&mut self, tolerance: f32) {
        if tolerance.is_finite() && tolerance >= 0.0 {
            self.tolerance = tolerance;
        }
    }

    /// Adjusts the stability threshold used to detect low-variance pressure.
    pub fn set_stability_threshold(&mut self, threshold: f32) {
        if threshold.is_finite() && threshold > 0.0 {
            self.stability_threshold = threshold;
        }
    }

    /// Adjusts the stability boost applied to the proportional gain.
    pub fn set_stability_boost(&mut self, boost: f32) {
        if boost.is_finite() && boost >= 0.0 {
            self.stability_boost = boost;
        }
    }

    /// Enables a small curvature dither when pressure remains stable within tolerance.
    pub fn set_dither(&mut self, strength: f32, period: u32) {
        if strength.is_finite() && strength >= 0.0 {
            self.dither_strength = strength.clamp(0.0, 1.0);
        }
        if period > 0 {
            self.dither_period = period.max(1);
        }
    }

    /// Adjusts the smoothing factor applied to the pressure EMA.
    pub fn set_smoothing(&mut self, alpha: f32) {
        if alpha.is_finite() && alpha > 0.0 {
            self.alpha = alpha.clamp(0.01, 1.0);
        }
    }

    /// Records a gradient summary returning the raw/smoothed pressure alongside
    /// the curvature chosen for the next step.
    pub fn observe(&mut self, summary: GradientSummary) -> CurvatureDecision {
        let raw_pressure = summary.mean_abs();
        self.observe_pressure(raw_pressure)
    }

    /// Records a raw pressure observation without requiring a full gradient summary.
    pub fn observe_pressure(&mut self, raw_pressure: f32) -> CurvatureDecision {
        let raw_pressure = if raw_pressure.is_finite() {
            raw_pressure.max(0.0).min(CURVATURE_PRESSURE_MAX)
        } else {
            0.0
        };
        let alpha = if self.alpha.is_finite() && self.alpha > 0.0 {
            self.alpha.clamp(0.01, 1.0)
        } else {
            1.0
        };
        let smoothed = match self.ema_pressure.filter(|value| value.is_finite()) {
            Some(prev) => prev + alpha * (raw_pressure - prev),
            None => raw_pressure,
        }
        .max(0.0)
        .min(CURVATURE_PRESSURE_MAX);
        self.ema_pressure = Some(smoothed);
        let raw_sq =
            (f64::from(raw_pressure) * f64::from(raw_pressure)).min(f64::from(f32::MAX)) as f32;
        let smoothed_sq = match self.ema_pressure2.filter(|value| value.is_finite()) {
            Some(prev) => prev + alpha * (raw_sq - prev),
            None => raw_sq,
        }
        .max(0.0)
        .min(f32::MAX);
        self.ema_pressure2 = Some(smoothed_sq);
        let variance =
            (f64::from(smoothed_sq) - f64::from(smoothed) * f64::from(smoothed)).max(0.0);
        let variance = variance.min(f64::from(f32::MAX)) as f32;
        let denom = f64::from(smoothed) * f64::from(smoothed) + 1.0e-6;
        let rel_var = (f64::from(variance) / denom).min(f64::from(f32::MAX)) as f32;
        let error = smoothed - self.target_pressure;
        let within_band = error.abs() <= self.tolerance;

        let mut curvature = self.current;
        let mut changed = false;
        let stable = within_band && rel_var.is_finite() && rel_var <= self.stability_threshold;
        if stable {
            self.stable_steps = self.stable_steps.saturating_add(1);
        } else {
            self.stable_steps = 0;
        }

        let mut delta = 0.0f32;
        if !within_band {
            let stability = if rel_var.is_finite() && self.stability_threshold > 0.0 {
                (1.0 - (rel_var / self.stability_threshold).clamp(0.0, 1.0)).max(0.0)
            } else {
                0.0
            };
            let kp = self.kp * (1.0 + self.stability_boost * stability);
            delta = (kp * error).clamp(-self.step, self.step);
        } else if stable
            && self.dither_strength > 0.0
            && self.dither_period > 0
            && self.stable_steps >= self.dither_period
        {
            delta = self.dither_sign * self.dither_strength * self.step;
            self.dither_sign = -self.dither_sign;
            self.stable_steps = 0;
        }

        if delta.abs() > f32::EPSILON {
            let next = (self.current + delta)
                .clamp(self.min_curvature, self.max_curvature)
                .min(-1e-6);
            if (next - self.current).abs() > f32::EPSILON {
                curvature = next;
                changed = true;
                self.current = curvature;
            }
        }

        CurvatureDecision {
            raw_pressure,
            smoothed_pressure: smoothed,
            curvature,
            changed,
        }
    }
}

/// Decision emitted by [`CurvatureScheduler::observe`].
#[derive(Debug, Clone, Copy)]
pub struct CurvatureDecision {
    pub raw_pressure: f32,
    pub smoothed_pressure: f32,
    pub curvature: f32,
    pub changed: bool,
}

/// Last curvature update captured by the trainer.
#[derive(Debug, Clone, Copy, Default)]
pub struct CurvatureMetrics {
    pub raw_pressure: f32,
    pub smoothed_pressure: f32,
    pub curvature: f32,
}

impl From<CurvatureDecision> for CurvatureMetrics {
    fn from(decision: CurvatureDecision) -> Self {
        Self {
            raw_pressure: decision.raw_pressure,
            smoothed_pressure: decision.smoothed_pressure,
            curvature: decision.curvature,
        }
    }
}

/// Coherence statistics required by [`SpectralLearningRatePolicy`].
#[derive(Debug, Clone, Copy)]
pub struct CoherenceSignal {
    dominant_channel: Option<usize>,
    preserved_channels: usize,
    mean_coherence: f32,
    z_bias: f32,
    energy_ratio: f32,
    entropy: f32,
    label: CoherenceLabel,
    repaired_non_finite_weights: usize,
    repaired_negative_weights: usize,
    pre_discard_repaired_non_finite: usize,
    pre_discard_repaired_negative: usize,
}

impl CoherenceSignal {
    pub fn dominant_channel(&self) -> Option<usize> {
        self.dominant_channel
    }

    pub fn preserved_channels(&self) -> usize {
        self.preserved_channels
    }

    pub fn mean_coherence(&self) -> f32 {
        self.mean_coherence
    }

    pub fn z_bias(&self) -> f32 {
        self.z_bias
    }

    pub fn energy_ratio(&self) -> f32 {
        self.energy_ratio
    }

    pub fn coherence_entropy(&self) -> f32 {
        self.entropy
    }

    pub fn label(&self) -> CoherenceLabel {
        self.label
    }

    pub fn repaired_non_finite_weights(&self) -> usize {
        self.repaired_non_finite_weights
    }

    pub fn repaired_negative_weights(&self) -> usize {
        self.repaired_negative_weights
    }

    pub fn repaired_weights_total(&self) -> usize {
        self.repaired_non_finite_weights
            .saturating_add(self.repaired_negative_weights)
    }

    pub fn pre_discard_repaired_non_finite(&self) -> usize {
        self.pre_discard_repaired_non_finite
    }

    pub fn pre_discard_repaired_negative(&self) -> usize {
        self.pre_discard_repaired_negative
    }

    pub fn pre_discard_repairs_total(&self) -> usize {
        self.pre_discard_repaired_non_finite
            .saturating_add(self.pre_discard_repaired_negative)
    }

    pub fn repairs_total(&self) -> usize {
        self.repaired_weights_total()
            .saturating_add(self.pre_discard_repairs_total())
    }

    fn label_from_str(label: &str) -> CoherenceLabel {
        match label.trim() {
            "symmetric_pulse" => CoherenceLabel::SymmetricPulse,
            "cascade_imbalance" => CoherenceLabel::CascadeImbalance,
            "diffuse_drift" => CoherenceLabel::DiffuseDrift,
            _ => CoherenceLabel::Background,
        }
    }

    fn from_zspace_trace_event(event: &ZSpaceTraceEvent) -> Option<Self> {
        let ZSpaceTraceEvent::Aggregated { diagnostics, .. } = event else {
            return None;
        };
        Some(Self {
            dominant_channel: diagnostics.dominant_channel,
            preserved_channels: diagnostics.preserved_channels.max(1),
            mean_coherence: diagnostics.mean_coherence,
            z_bias: diagnostics.z_bias,
            energy_ratio: diagnostics.energy_ratio,
            entropy: diagnostics.entropy,
            label: Self::label_from_str(&diagnostics.label),
            repaired_non_finite_weights: diagnostics.repaired_non_finite_weights,
            repaired_negative_weights: diagnostics.repaired_negative_weights,
            pre_discard_repaired_non_finite: diagnostics.pre_discard_repaired_non_finite,
            pre_discard_repaired_negative: diagnostics.pre_discard_repaired_negative,
        })
    }
}

impl From<&CoherenceDiagnostics> for CoherenceSignal {
    fn from(diagnostics: &CoherenceDiagnostics) -> Self {
        Self {
            dominant_channel: diagnostics.dominant_channel(),
            preserved_channels: diagnostics.preserved_channels().max(1),
            mean_coherence: diagnostics.mean_coherence(),
            z_bias: diagnostics.z_bias(),
            energy_ratio: diagnostics.energy_ratio(),
            entropy: diagnostics.coherence_entropy(),
            label: diagnostics.observation().lift_to_label(),
            repaired_non_finite_weights: diagnostics.repaired_non_finite_weights(),
            repaired_negative_weights: diagnostics.repaired_negative_weights(),
            pre_discard_repaired_non_finite: diagnostics
                .pre_discard()
                .map(|telemetry| telemetry.repaired_non_finite())
                .unwrap_or(0),
            pre_discard_repaired_negative: diagnostics
                .pre_discard()
                .map(|telemetry| telemetry.repaired_negative())
                .unwrap_or(0),
        }
    }
}

/// Listens for `ZSpaceTrace` plugin events and lifts coherence diagnostics into a signal.
pub struct ZSpaceTraceCoherenceBridge {
    bus: st_core::plugin::PluginEventBus,
    subscription_id: usize,
    latest: Arc<Mutex<Option<CoherenceSignal>>>,
}

impl ZSpaceTraceCoherenceBridge {
    pub fn subscribe() -> Self {
        let bus = global_registry().event_bus().clone();
        let latest: Arc<Mutex<Option<CoherenceSignal>>> = Arc::new(Mutex::new(None));
        let latest_clone = Arc::clone(&latest);
        let subscription_id = bus.subscribe(
            "ZSpaceTrace",
            Arc::new(move |event: &PluginEvent| {
                let Some(payload) = event.downcast_data::<serde_json::Value>() else {
                    return;
                };
                let Ok(trace_event) = serde_json::from_value::<ZSpaceTraceEvent>(payload.clone())
                else {
                    return;
                };
                let Some(signal) = CoherenceSignal::from_zspace_trace_event(&trace_event) else {
                    return;
                };
                let mut guard = latest_clone
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *guard = Some(signal);
            }),
        );
        Self {
            bus,
            subscription_id,
            latest,
        }
    }

    pub fn drain(&self) -> Option<CoherenceSignal> {
        let mut guard = self
            .latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.take()
    }
}

impl Drop for ZSpaceTraceCoherenceBridge {
    fn drop(&mut self) {
        let _ = self.bus.unsubscribe("ZSpaceTrace", self.subscription_id);
    }
}

#[derive(Debug, Default)]
struct TensorBackendStepTrace {
    total: usize,
    fallbacks: usize,
    meta_events: usize,
    meta_null_values: usize,
    meta_non_finite_strings: usize,
    by_backend: HashMap<String, usize>,
    by_kernel_backend: HashMap<String, usize>,
    by_op: HashMap<String, usize>,
    by_op_backend: HashMap<String, usize>,
    by_op_kernel_backend: HashMap<String, usize>,
    requested_wgpu_hits: usize,
    requested_wgpu_runtime_fallbacks: usize,
    requested_wgpu_hits_by_op_backend: HashMap<String, usize>,
    requested_wgpu_runtime_fallbacks_by_op_backend: HashMap<String, usize>,
    requested_wgpu_component_hits: usize,
    requested_wgpu_component_fallbacks: usize,
    requested_wgpu_component_hits_by_op_backend: HashMap<String, usize>,
    requested_wgpu_component_fallbacks_by_op_backend: HashMap<String, usize>,
    embedding_tokens: usize,
    embedding_unique_token_indices: usize,
    embedding_repeated_token_indices: usize,
    embedding_non_finite_tokens: usize,
    embedding_rounded_tokens: usize,
    embedding_clamped_low_tokens: usize,
    embedding_clamped_high_tokens: usize,
    backend_policy_events: usize,
    backend_policy_status: HashMap<String, usize>,
    backend_policy_source: HashMap<String, usize>,
    backend_policy_wgpu_choices: usize,
    backend_policy_unison_choices: usize,
    backend_policy_kdsl_env_events: usize,
    backend_policy_kdsl_kv_events: usize,
    backend_policy_kv_soft_events: usize,
    backend_policy_wasm_tuner_events: usize,
    backend_policy_tensor_util_routes: usize,
    backend_policy_wgpu_last_workgroup: Option<f64>,
    backend_policy_wgpu_last_lanes: Option<f64>,
    backend_policy_wgpu_last_compaction_tile: Option<f64>,
    backend_policy_wgpu_last_fft_radix: Option<f64>,
    backend_policy_wgpu_last_fft_segments: Option<f64>,
    backend_policy_wgpu_last_override_count: Option<f64>,
    backend_policy_unison_last_candidate_count: Option<f64>,
    backend_policy_unison_last_best_score: Option<f64>,
    backend_policy_unison_last_baseline_score: Option<f64>,
    backend_policy_unison_last_wgpu_generated_score: Option<f64>,
    backend_policy_unison_last_wgpu_generated_score_delta: Option<f64>,
    backend_policy_tensor_util_last_values: Option<f64>,
    backend_policy_tensor_util_last_threshold: Option<f64>,
    lstm_forward_estimated_gate_activation_ops: usize,
    lstm_forward_estimated_gate_activation_cpu_debt_ops: usize,
    lstm_forward_estimated_gate_activation_wgpu_ops: usize,
    lstm_backward_estimated_gate_activation_ops: usize,
    lstm_backward_estimated_gate_activation_cpu_debt_ops: usize,
    lstm_backward_estimated_gate_activation_wgpu_ops: usize,
    lstm_backward_estimated_bptt_ops: usize,
    lstm_backward_estimated_bptt_cpu_debt_ops: usize,
    lstm_backward_estimated_bptt_wgpu_ops: usize,
    lstm_backward_estimated_bptt_gate_derivative_ops: usize,
    lstm_backward_estimated_bptt_cell_recurrence_ops: usize,
    lstm_backward_estimated_bptt_state_carry_ops: usize,
    lstm_backward_estimated_bptt_scan_steps: usize,
}

impl TensorBackendStepTrace {
    fn record_sub_backend(
        &mut self,
        op: &str,
        requested_backend: Option<&str>,
        field: &'static str,
        sub_backend: &str,
        count_fallback: bool,
    ) {
        let sub_backend = backend_metric_fragment(sub_backend);
        if sub_backend == "auto" {
            return;
        }
        let field = field.strip_suffix("_backend").unwrap_or(field);
        *self
            .by_op_backend
            .entry(format!("{op}_{field}_{sub_backend}"))
            .or_default() += 1;
        if count_fallback && requested_backend == Some("wgpu") {
            let key = format!("{op}_{field}_{sub_backend}");
            if sub_backend == "wgpu" {
                self.requested_wgpu_component_hits =
                    self.requested_wgpu_component_hits.saturating_add(1);
                *self
                    .requested_wgpu_component_hits_by_op_backend
                    .entry(key)
                    .or_default() += 1;
            } else if !is_metadata_only_backend(&sub_backend) {
                self.requested_wgpu_component_fallbacks =
                    self.requested_wgpu_component_fallbacks.saturating_add(1);
                *self
                    .requested_wgpu_component_fallbacks_by_op_backend
                    .entry(key)
                    .or_default() += 1;
            }
        }
        if count_fallback {
            if let Some(requested) = requested_backend {
                if requested != "auto"
                    && requested != sub_backend
                    && !is_metadata_only_backend(&sub_backend)
                {
                    self.fallbacks = self.fallbacks.saturating_add(1);
                }
            }
        }
    }

    fn record(&mut self, event: &TensorOpMetaEvent) {
        self.meta_events = self.meta_events.saturating_add(1);
        let mut meta_health = TensorMetaHealth::default();
        meta_health.record(&event.data);
        self.meta_null_values = self
            .meta_null_values
            .saturating_add(meta_health.null_values);
        self.meta_non_finite_strings = self
            .meta_non_finite_strings
            .saturating_add(meta_health.non_finite_strings);

        self.record_backend_policy_event(event);

        let Some(backend) = event.data.get("backend").and_then(|value| value.as_str()) else {
            return;
        };
        let kernel_backend = metric_fragment(backend);
        let backend = backend_metric_fragment(backend);
        let op = metric_fragment(event.op_name);
        self.total = self.total.saturating_add(1);
        *self.by_backend.entry(backend.clone()).or_default() += 1;
        *self
            .by_kernel_backend
            .entry(kernel_backend.clone())
            .or_default() += 1;
        *self.by_op.entry(op.clone()).or_default() += 1;
        *self
            .by_op_backend
            .entry(format!("{op}_{backend}"))
            .or_default() += 1;
        *self
            .by_op_kernel_backend
            .entry(format!("{op}_{kernel_backend}"))
            .or_default() += 1;

        let requested_backend = event
            .data
            .get("requested_backend")
            .and_then(|value| value.as_str())
            .map(backend_metric_fragment);
        if let Some(requested) = requested_backend.as_deref() {
            if requested == "wgpu" {
                let op_backend = format!("{op}_{backend}");
                if backend == "wgpu" {
                    self.requested_wgpu_hits = self.requested_wgpu_hits.saturating_add(1);
                    *self
                        .requested_wgpu_hits_by_op_backend
                        .entry(op_backend)
                        .or_default() += 1;
                } else if is_cpu_runtime_backend(&backend) && is_wgpu_runtime_fallback(&event.data)
                {
                    self.requested_wgpu_runtime_fallbacks =
                        self.requested_wgpu_runtime_fallbacks.saturating_add(1);
                    *self
                        .requested_wgpu_runtime_fallbacks_by_op_backend
                        .entry(op_backend)
                        .or_default() += 1;
                }
            }
            if requested != "auto" && requested != backend && !is_metadata_only_backend(&backend) {
                self.fallbacks = self.fallbacks.saturating_add(1);
            }
        }
        for (field, count_fallback) in [
            ("input_gradient_backend", true),
            ("input_gradient_reduction_backend", true),
            ("gradient_reduction_backend", true),
            ("affine_gradient_backend", true),
            ("normalization_backend", true),
            ("input_projection_backend", true),
            ("bias_backend", true),
            ("recurrent_backend", true),
            ("gate_activation_backend", true),
            ("bptt_backend", true),
            ("bptt_scan_backend", false),
            ("bptt_gate_derivative_backend", true),
            ("bptt_cell_recurrence_backend", true),
            ("bptt_state_carry_backend", true),
            ("raw_parameter_gradient_backend", true),
            ("parameter_gradient_reduction_backend", true),
            ("bias_gradient_backend", true),
            ("parameter_gradient_scale_backend", true),
            ("broadcast_backend", true),
            ("activation_backend", true),
            ("preactivation_backend", true),
            ("geometry_backend", true),
            ("mask_backend", true),
            ("rng_backend", false),
            ("deterministic_backend", true),
            ("gradient_scale_backend", true),
            ("merge_backend", true),
            ("accumulation_backend", true),
            ("normalise_backend", true),
            ("rewrite_backend", true),
            ("projection_backend", true),
            ("projection_gradient_backend", true),
            ("saturation_gradient_backend", true),
            ("softmax_backend", true),
            ("exp_backend", true),
            ("sanitize_backend", true),
            ("distribution_scale_backend", true),
            ("semantic_inference_backend", true),
            ("semantic_sparse_scan_backend", true),
            ("semantic_accumulation_backend", true),
            ("semantic_sanitize_backend", true),
            ("window_energy_backend", true),
            ("fusion_accumulation_backend", true),
            ("marginal_scan_backend", true),
            ("marginal_sum_backend", true),
            ("row_scan_backend", true),
            ("row_sum_backend", true),
            ("state_sum_backend", true),
            ("precision_backend", true),
            ("reduction_backend", true),
            ("covariance_centering_backend", true),
            ("covariance_accumulation_backend", true),
            ("low_rank_projection_backend", true),
            ("psd_projection_backend", true),
        ] {
            if let Some(sub_backend) = event.data.get(field).and_then(|value| value.as_str()) {
                self.record_sub_backend(
                    &op,
                    requested_backend.as_deref(),
                    field,
                    sub_backend,
                    count_fallback,
                );
            }
        }

        if matches!(event.op_name, "embedding_forward" | "embedding_backward") {
            self.embedding_tokens = self
                .embedding_tokens
                .saturating_add(meta_usize(&event.data, "tokens"));
            self.embedding_unique_token_indices = self
                .embedding_unique_token_indices
                .saturating_add(meta_usize(&event.data, "unique_token_indices"));
            self.embedding_repeated_token_indices = self
                .embedding_repeated_token_indices
                .saturating_add(meta_usize(&event.data, "repeated_token_indices"));
            self.embedding_non_finite_tokens = self
                .embedding_non_finite_tokens
                .saturating_add(meta_usize(&event.data, "non_finite_tokens"));
            self.embedding_rounded_tokens = self
                .embedding_rounded_tokens
                .saturating_add(meta_usize(&event.data, "rounded_tokens"));
            self.embedding_clamped_low_tokens = self
                .embedding_clamped_low_tokens
                .saturating_add(meta_usize(&event.data, "clamped_low_tokens"));
            self.embedding_clamped_high_tokens = self
                .embedding_clamped_high_tokens
                .saturating_add(meta_usize(&event.data, "clamped_high_tokens"));
        }
        if event.op_name == "lstm_forward" {
            let gate_activation_ops = meta_usize(&event.data, "estimated_gate_activation_ops");
            self.lstm_forward_estimated_gate_activation_ops = self
                .lstm_forward_estimated_gate_activation_ops
                .saturating_add(gate_activation_ops);
            let gate_backend = event
                .data
                .get("gate_activation_backend")
                .and_then(|value| value.as_str())
                .map(backend_metric_fragment)
                .unwrap_or_else(|| "cpu".to_string());
            if gate_backend == "wgpu" {
                self.lstm_forward_estimated_gate_activation_wgpu_ops = self
                    .lstm_forward_estimated_gate_activation_wgpu_ops
                    .saturating_add(gate_activation_ops);
            } else {
                self.lstm_forward_estimated_gate_activation_cpu_debt_ops = self
                    .lstm_forward_estimated_gate_activation_cpu_debt_ops
                    .saturating_add(gate_activation_ops);
            }
        }
        if event.op_name == "lstm_backward" {
            let gate_activation_ops = meta_usize(&event.data, "estimated_gate_activation_ops");
            self.lstm_backward_estimated_gate_activation_ops = self
                .lstm_backward_estimated_gate_activation_ops
                .saturating_add(gate_activation_ops);
            let gate_backend = event
                .data
                .get("gate_activation_backend")
                .and_then(|value| value.as_str())
                .map(backend_metric_fragment)
                .unwrap_or_else(|| "cpu".to_string());
            if gate_backend == "wgpu" {
                self.lstm_backward_estimated_gate_activation_wgpu_ops = self
                    .lstm_backward_estimated_gate_activation_wgpu_ops
                    .saturating_add(gate_activation_ops);
            } else {
                self.lstm_backward_estimated_gate_activation_cpu_debt_ops = self
                    .lstm_backward_estimated_gate_activation_cpu_debt_ops
                    .saturating_add(gate_activation_ops);
            }
            let bptt_ops = meta_usize(&event.data, "estimated_bptt_ops");
            self.lstm_backward_estimated_bptt_ops = self
                .lstm_backward_estimated_bptt_ops
                .saturating_add(bptt_ops);
            let bptt_backend = event
                .data
                .get("bptt_backend")
                .or_else(|| event.data.get("bptt_scan_backend"))
                .and_then(|value| value.as_str())
                .map(backend_metric_fragment);
            let bptt_cpu_debt_ops = if event.data.get("estimated_bptt_cpu_debt_ops").is_some() {
                meta_usize(&event.data, "estimated_bptt_cpu_debt_ops")
            } else if bptt_backend.as_deref() == Some("cpu") {
                bptt_ops
            } else {
                0
            };
            let bptt_wgpu_ops = if event.data.get("estimated_bptt_wgpu_ops").is_some() {
                meta_usize(&event.data, "estimated_bptt_wgpu_ops")
            } else if bptt_backend.as_deref() == Some("wgpu") {
                bptt_ops
            } else {
                0
            };
            self.lstm_backward_estimated_bptt_cpu_debt_ops = self
                .lstm_backward_estimated_bptt_cpu_debt_ops
                .saturating_add(bptt_cpu_debt_ops);
            self.lstm_backward_estimated_bptt_wgpu_ops = self
                .lstm_backward_estimated_bptt_wgpu_ops
                .saturating_add(bptt_wgpu_ops);
            self.lstm_backward_estimated_bptt_gate_derivative_ops = self
                .lstm_backward_estimated_bptt_gate_derivative_ops
                .saturating_add(meta_usize(
                    &event.data,
                    "estimated_bptt_gate_derivative_ops",
                ));
            self.lstm_backward_estimated_bptt_cell_recurrence_ops = self
                .lstm_backward_estimated_bptt_cell_recurrence_ops
                .saturating_add(meta_usize(
                    &event.data,
                    "estimated_bptt_cell_recurrence_ops",
                ));
            self.lstm_backward_estimated_bptt_state_carry_ops = self
                .lstm_backward_estimated_bptt_state_carry_ops
                .saturating_add(meta_usize(&event.data, "estimated_bptt_state_carry_ops"));
            self.lstm_backward_estimated_bptt_scan_steps = self
                .lstm_backward_estimated_bptt_scan_steps
                .saturating_add(meta_usize(&event.data, "estimated_bptt_scan_steps"));
        }
    }

    fn record_backend_policy_event(&mut self, event: &TensorOpMetaEvent) {
        let op = metric_fragment(event.op_name);
        let tracked = matches!(
            event.op_name,
            "wgpu_heuristic_choice"
                | "unison_rank_choice"
                | "kdsl_env_bridge"
                | "kdsl_kv_bridge"
                | "kv_consensus_soft_rules"
                | "wasm_tuner_choice"
                | "backend_resolution"
                | "backend_device_report"
                | "temporal_spectral_fusion"
                | "tensor_util_route"
        );
        if !tracked {
            return;
        }

        self.backend_policy_events = self.backend_policy_events.saturating_add(1);
        if let Some(status) = meta_str_fragment(&event.data, "status") {
            *self
                .backend_policy_status
                .entry(format!("{op}_{status}"))
                .or_default() += 1;
        }
        if let Some(source) = meta_str_fragment(&event.data, "choice_source") {
            *self
                .backend_policy_source
                .entry(format!("{op}_{source}"))
                .or_default() += 1;
        }

        match event.op_name {
            "wgpu_heuristic_choice" => {
                self.backend_policy_wgpu_choices =
                    self.backend_policy_wgpu_choices.saturating_add(1);
                self.backend_policy_wgpu_last_workgroup = meta_f64(&event.data, "workgroup");
                self.backend_policy_wgpu_last_lanes = meta_f64(&event.data, "lanes");
                self.backend_policy_wgpu_last_compaction_tile =
                    meta_f64(&event.data, "compaction_tile");
                self.backend_policy_wgpu_last_fft_radix = meta_f64(&event.data, "fft_radix");
                self.backend_policy_wgpu_last_fft_segments = meta_f64(&event.data, "fft_segments");
                self.backend_policy_wgpu_last_override_count =
                    meta_f64(&event.data, "override_count");
            }
            "unison_rank_choice" => {
                self.backend_policy_unison_choices =
                    self.backend_policy_unison_choices.saturating_add(1);
                self.backend_policy_unison_last_candidate_count =
                    meta_f64(&event.data, "candidate_count");
                self.backend_policy_unison_last_best_score = meta_f64(&event.data, "best_score");
                self.backend_policy_unison_last_baseline_score =
                    meta_f64(&event.data, "baseline_score");
                self.backend_policy_unison_last_wgpu_generated_score =
                    meta_f64(&event.data, "wgpu_generated_score");
                self.backend_policy_unison_last_wgpu_generated_score_delta =
                    meta_f64(&event.data, "wgpu_generated_score_delta");
            }
            "kdsl_env_bridge" => {
                self.backend_policy_kdsl_env_events =
                    self.backend_policy_kdsl_env_events.saturating_add(1);
            }
            "kdsl_kv_bridge" => {
                self.backend_policy_kdsl_kv_events =
                    self.backend_policy_kdsl_kv_events.saturating_add(1);
            }
            "kv_consensus_soft_rules" => {
                self.backend_policy_kv_soft_events =
                    self.backend_policy_kv_soft_events.saturating_add(1);
            }
            "wasm_tuner_choice" => {
                self.backend_policy_wasm_tuner_events =
                    self.backend_policy_wasm_tuner_events.saturating_add(1);
            }
            "tensor_util_route" => {
                self.backend_policy_tensor_util_routes =
                    self.backend_policy_tensor_util_routes.saturating_add(1);
                self.backend_policy_tensor_util_last_values = meta_f64(&event.data, "values");
                self.backend_policy_tensor_util_last_threshold = meta_f64(&event.data, "threshold");
            }
            _ => {}
        }
    }

    fn write_extra(self, extra: &mut HashMap<String, f64>) {
        if self.meta_events > 0 {
            extra.insert("tensor_meta_events".to_string(), self.meta_events as f64);
            extra.insert(
                "tensor_meta_null_values".to_string(),
                self.meta_null_values as f64,
            );
            extra.insert(
                "tensor_meta_non_finite_strings".to_string(),
                self.meta_non_finite_strings as f64,
            );
            let sentinels = self
                .meta_null_values
                .saturating_add(self.meta_non_finite_strings);
            extra.insert(
                "tensor_meta_non_finite_sentinels".to_string(),
                sentinels as f64,
            );
            if sentinels > 0 {
                extra.insert("tensor_meta_non_finite_detected".to_string(), 1.0);
            }
        }

        if self.total > 0 {
            extra.insert("tensor_ops_total".to_string(), self.total as f64);
            extra.insert(
                "tensor_backend_fallbacks".to_string(),
                self.fallbacks as f64,
            );
            for (backend, count) in self.by_backend {
                extra.insert(format!("tensor_backend_{backend}"), count as f64);
            }
            for (backend, count) in self.by_kernel_backend {
                extra.insert(format!("tensor_kernel_backend_{backend}"), count as f64);
            }
            for (op, count) in self.by_op {
                extra.insert(format!("tensor_op_{op}"), count as f64);
            }
            for (op_backend, count) in self.by_op_backend {
                extra.insert(format!("tensor_op_backend_{op_backend}"), count as f64);
            }
            for (op_backend, count) in self.by_op_kernel_backend {
                extra.insert(
                    format!("tensor_op_kernel_backend_{op_backend}"),
                    count as f64,
                );
            }
            let requested_wgpu_total = self
                .requested_wgpu_hits
                .saturating_add(self.requested_wgpu_runtime_fallbacks);
            if requested_wgpu_total > 0 {
                extra.insert(
                    "tensor_backend_requested_wgpu_hits".to_string(),
                    self.requested_wgpu_hits as f64,
                );
                extra.insert(
                    "tensor_backend_requested_wgpu_runtime_fallbacks".to_string(),
                    self.requested_wgpu_runtime_fallbacks as f64,
                );
                for (op_backend, count) in self.requested_wgpu_hits_by_op_backend {
                    extra.insert(
                        format!("tensor_op_backend_requested_wgpu_hit_{op_backend}"),
                        count as f64,
                    );
                }
                for (op_backend, count) in self.requested_wgpu_runtime_fallbacks_by_op_backend {
                    extra.insert(
                        format!("tensor_op_backend_wgpu_runtime_fallback_{op_backend}"),
                        count as f64,
                    );
                }
            }
            let requested_wgpu_component_total = self
                .requested_wgpu_component_hits
                .saturating_add(self.requested_wgpu_component_fallbacks);
            if requested_wgpu_component_total > 0 {
                extra.insert(
                    "tensor_backend_requested_wgpu_component_hits".to_string(),
                    self.requested_wgpu_component_hits as f64,
                );
                extra.insert(
                    "tensor_backend_requested_wgpu_component_fallbacks".to_string(),
                    self.requested_wgpu_component_fallbacks as f64,
                );
                for (op_backend, count) in self.requested_wgpu_component_hits_by_op_backend {
                    extra.insert(
                        format!("tensor_op_backend_requested_wgpu_component_hit_{op_backend}"),
                        count as f64,
                    );
                }
                for (op_backend, count) in self.requested_wgpu_component_fallbacks_by_op_backend {
                    extra.insert(
                        format!("tensor_op_backend_requested_wgpu_component_fallback_{op_backend}"),
                        count as f64,
                    );
                }
            }
        }

        if self.embedding_tokens > 0 {
            extra.insert(
                "tensor_embedding_tokens".to_string(),
                self.embedding_tokens as f64,
            );
            extra.insert(
                "tensor_embedding_unique_token_indices".to_string(),
                self.embedding_unique_token_indices as f64,
            );
            extra.insert(
                "tensor_embedding_repeated_token_indices".to_string(),
                self.embedding_repeated_token_indices as f64,
            );
            extra.insert(
                "tensor_embedding_non_finite_tokens".to_string(),
                self.embedding_non_finite_tokens as f64,
            );
            extra.insert(
                "tensor_embedding_rounded_tokens".to_string(),
                self.embedding_rounded_tokens as f64,
            );
            extra.insert(
                "tensor_embedding_clamped_low_tokens".to_string(),
                self.embedding_clamped_low_tokens as f64,
            );
            extra.insert(
                "tensor_embedding_clamped_high_tokens".to_string(),
                self.embedding_clamped_high_tokens as f64,
            );
            let repairs = self
                .embedding_non_finite_tokens
                .saturating_add(self.embedding_rounded_tokens)
                .saturating_add(self.embedding_clamped_low_tokens)
                .saturating_add(self.embedding_clamped_high_tokens);
            extra.insert(
                "tensor_embedding_token_repairs_total".to_string(),
                repairs as f64,
            );
            if repairs > 0 {
                extra.insert("tensor_embedding_token_repair_detected".to_string(), 1.0);
            }
        }

        if self.lstm_forward_estimated_gate_activation_ops > 0
            || self.lstm_backward_estimated_gate_activation_ops > 0
            || self.lstm_backward_estimated_bptt_ops > 0
        {
            let gate_activation_ops = self
                .lstm_forward_estimated_gate_activation_ops
                .saturating_add(self.lstm_backward_estimated_gate_activation_ops);
            let gate_activation_cpu_debt_ops = self
                .lstm_forward_estimated_gate_activation_cpu_debt_ops
                .saturating_add(self.lstm_backward_estimated_gate_activation_cpu_debt_ops);
            let gate_activation_wgpu_ops = self
                .lstm_forward_estimated_gate_activation_wgpu_ops
                .saturating_add(self.lstm_backward_estimated_gate_activation_wgpu_ops);
            let cpu_debt_ops = gate_activation_cpu_debt_ops
                .saturating_add(self.lstm_backward_estimated_bptt_cpu_debt_ops);
            extra.insert(
                "lstm_estimated_gate_activation_ops".to_string(),
                gate_activation_ops as f64,
            );
            extra.insert(
                "lstm_estimated_gate_activation_cpu_debt_ops".to_string(),
                gate_activation_cpu_debt_ops as f64,
            );
            extra.insert(
                "lstm_estimated_gate_activation_wgpu_ops".to_string(),
                gate_activation_wgpu_ops as f64,
            );
            extra.insert(
                "lstm_estimated_cpu_debt_ops".to_string(),
                cpu_debt_ops as f64,
            );
            extra.insert(
                "lstm_forward_estimated_gate_activation_ops".to_string(),
                self.lstm_forward_estimated_gate_activation_ops as f64,
            );
            extra.insert(
                "lstm_forward_estimated_gate_activation_cpu_debt_ops".to_string(),
                self.lstm_forward_estimated_gate_activation_cpu_debt_ops as f64,
            );
            extra.insert(
                "lstm_forward_estimated_gate_activation_wgpu_ops".to_string(),
                self.lstm_forward_estimated_gate_activation_wgpu_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_gate_activation_ops".to_string(),
                self.lstm_backward_estimated_gate_activation_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_gate_activation_cpu_debt_ops".to_string(),
                self.lstm_backward_estimated_gate_activation_cpu_debt_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_gate_activation_wgpu_ops".to_string(),
                self.lstm_backward_estimated_gate_activation_wgpu_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_bptt_ops".to_string(),
                self.lstm_backward_estimated_bptt_ops as f64,
            );
            extra.insert(
                "lstm_estimated_bptt_cpu_debt_ops".to_string(),
                self.lstm_backward_estimated_bptt_cpu_debt_ops as f64,
            );
            extra.insert(
                "lstm_estimated_bptt_wgpu_ops".to_string(),
                self.lstm_backward_estimated_bptt_wgpu_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_bptt_cpu_debt_ops".to_string(),
                self.lstm_backward_estimated_bptt_cpu_debt_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_bptt_wgpu_ops".to_string(),
                self.lstm_backward_estimated_bptt_wgpu_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_bptt_gate_derivative_ops".to_string(),
                self.lstm_backward_estimated_bptt_gate_derivative_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_bptt_cell_recurrence_ops".to_string(),
                self.lstm_backward_estimated_bptt_cell_recurrence_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_bptt_state_carry_ops".to_string(),
                self.lstm_backward_estimated_bptt_state_carry_ops as f64,
            );
            extra.insert(
                "lstm_backward_estimated_bptt_scan_steps".to_string(),
                self.lstm_backward_estimated_bptt_scan_steps as f64,
            );
        }

        if self.backend_policy_events > 0 {
            extra.insert(
                "backend_policy_events".to_string(),
                self.backend_policy_events as f64,
            );
            extra.insert(
                "backend_policy_wgpu_choices".to_string(),
                self.backend_policy_wgpu_choices as f64,
            );
            extra.insert(
                "backend_policy_unison_choices".to_string(),
                self.backend_policy_unison_choices as f64,
            );
            extra.insert(
                "backend_policy_kdsl_env_events".to_string(),
                self.backend_policy_kdsl_env_events as f64,
            );
            extra.insert(
                "backend_policy_kdsl_kv_events".to_string(),
                self.backend_policy_kdsl_kv_events as f64,
            );
            extra.insert(
                "backend_policy_kv_soft_events".to_string(),
                self.backend_policy_kv_soft_events as f64,
            );
            extra.insert(
                "backend_policy_wasm_tuner_events".to_string(),
                self.backend_policy_wasm_tuner_events as f64,
            );
            extra.insert(
                "backend_policy_tensor_util_routes".to_string(),
                self.backend_policy_tensor_util_routes as f64,
            );
            for (status, count) in self.backend_policy_status {
                extra.insert(format!("backend_policy_status_{status}"), count as f64);
            }
            for (source, count) in self.backend_policy_source {
                extra.insert(format!("backend_policy_source_{source}"), count as f64);
            }
            insert_optional_extra(
                extra,
                "backend_policy_wgpu_last_workgroup",
                self.backend_policy_wgpu_last_workgroup,
            );
            insert_optional_extra(
                extra,
                "backend_policy_wgpu_last_lanes",
                self.backend_policy_wgpu_last_lanes,
            );
            insert_optional_extra(
                extra,
                "backend_policy_wgpu_last_compaction_tile",
                self.backend_policy_wgpu_last_compaction_tile,
            );
            insert_optional_extra(
                extra,
                "backend_policy_wgpu_last_fft_radix",
                self.backend_policy_wgpu_last_fft_radix,
            );
            insert_optional_extra(
                extra,
                "backend_policy_wgpu_last_fft_segments",
                self.backend_policy_wgpu_last_fft_segments,
            );
            insert_optional_extra(
                extra,
                "backend_policy_wgpu_last_override_count",
                self.backend_policy_wgpu_last_override_count,
            );
            insert_optional_extra(
                extra,
                "backend_policy_unison_last_candidate_count",
                self.backend_policy_unison_last_candidate_count,
            );
            insert_optional_extra(
                extra,
                "backend_policy_unison_last_best_score",
                self.backend_policy_unison_last_best_score,
            );
            insert_optional_extra(
                extra,
                "backend_policy_unison_last_baseline_score",
                self.backend_policy_unison_last_baseline_score,
            );
            insert_optional_extra(
                extra,
                "backend_policy_unison_last_wgpu_generated_score",
                self.backend_policy_unison_last_wgpu_generated_score,
            );
            insert_optional_extra(
                extra,
                "backend_policy_unison_last_wgpu_generated_score_delta",
                self.backend_policy_unison_last_wgpu_generated_score_delta,
            );
            insert_optional_extra(
                extra,
                "backend_policy_tensor_util_last_values",
                self.backend_policy_tensor_util_last_values,
            );
            insert_optional_extra(
                extra,
                "backend_policy_tensor_util_last_threshold",
                self.backend_policy_tensor_util_last_threshold,
            );
        }
    }

    fn validate_expected_backend(&self, expected: BackendKind) -> PureResult<()> {
        let expected_label = expected.as_str();
        let kernel_total = self
            .by_backend
            .iter()
            .filter(|(backend, _)| !is_metadata_only_backend(backend))
            .map(|(_, count)| *count)
            .sum::<usize>();
        if kernel_total == 0 {
            return Err(TensorError::BackendFailure {
                backend: expected_label,
                message: format!(
                    "trainer requested {expected_label} with SPIRALTORCH_STRICT_GPU, but no tensor kernel backend metadata was emitted"
                ),
            });
        }
        let mismatched = self
            .by_backend
            .iter()
            .filter(|(backend, _)| {
                backend.as_str() != expected_label && !is_metadata_only_backend(backend)
            })
            .map(|(backend, count)| format!("{backend}:{count}"))
            .collect::<Vec<_>>();
        if mismatched.is_empty() {
            return Ok(());
        }
        let mut observed = self
            .by_backend
            .iter()
            .map(|(backend, count)| format!("{backend}:{count}"))
            .collect::<Vec<_>>();
        observed.sort();
        Err(TensorError::BackendFailure {
            backend: expected_label,
            message: format!(
                "trainer requested {expected_label} with SPIRALTORCH_STRICT_GPU, but tensor kernels used [{}]; mismatched [{}]",
                observed.join(", "),
                mismatched.join(", ")
            ),
        })
    }
}

/// Tensor backend counters aggregated across one training epoch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct EpochTensorBackendStats {
    pub ops_total: usize,
    pub fallbacks: usize,
    pub meta_events: usize,
    pub meta_non_finite_sentinels: usize,
    pub backend_wgpu: usize,
    pub backend_cuda: usize,
    pub backend_hip: usize,
    pub backend_cpu: usize,
    pub backend_cpu_simd: usize,
    pub backend_f64_cpu: usize,
    pub backend_faer: usize,
    pub backend_naive: usize,
    pub backend_other: usize,
    pub kernel_backend_wgpu_dense: usize,
    pub kernel_backend_simd: usize,
    pub kernel_backend_other: usize,
    pub requested_wgpu_hits: usize,
    pub requested_wgpu_runtime_fallbacks: usize,
    pub requested_wgpu_component_hits: usize,
    pub requested_wgpu_component_fallbacks: usize,
    pub embedding_tokens: usize,
    pub embedding_unique_token_indices: usize,
    pub embedding_repeated_token_indices: usize,
    pub embedding_token_repairs: usize,
}

impl EpochTensorBackendStats {
    fn accumulate_trace(&mut self, trace: &TensorBackendStepTrace) {
        self.ops_total = self.ops_total.saturating_add(trace.total);
        self.fallbacks = self.fallbacks.saturating_add(trace.fallbacks);
        self.requested_wgpu_hits = self
            .requested_wgpu_hits
            .saturating_add(trace.requested_wgpu_hits);
        self.requested_wgpu_runtime_fallbacks = self
            .requested_wgpu_runtime_fallbacks
            .saturating_add(trace.requested_wgpu_runtime_fallbacks);
        self.requested_wgpu_component_hits = self
            .requested_wgpu_component_hits
            .saturating_add(trace.requested_wgpu_component_hits);
        self.requested_wgpu_component_fallbacks = self
            .requested_wgpu_component_fallbacks
            .saturating_add(trace.requested_wgpu_component_fallbacks);
        self.meta_events = self.meta_events.saturating_add(trace.meta_events);
        self.meta_non_finite_sentinels = self.meta_non_finite_sentinels.saturating_add(
            trace
                .meta_null_values
                .saturating_add(trace.meta_non_finite_strings),
        );
        self.embedding_tokens = self.embedding_tokens.saturating_add(trace.embedding_tokens);
        self.embedding_unique_token_indices = self
            .embedding_unique_token_indices
            .saturating_add(trace.embedding_unique_token_indices);
        self.embedding_repeated_token_indices = self
            .embedding_repeated_token_indices
            .saturating_add(trace.embedding_repeated_token_indices);
        self.embedding_token_repairs = self.embedding_token_repairs.saturating_add(
            trace
                .embedding_non_finite_tokens
                .saturating_add(trace.embedding_rounded_tokens)
                .saturating_add(trace.embedding_clamped_low_tokens)
                .saturating_add(trace.embedding_clamped_high_tokens),
        );
        for (backend, count) in trace.by_backend.iter() {
            match backend.as_str() {
                "wgpu" => self.backend_wgpu = self.backend_wgpu.saturating_add(*count),
                "cuda" => self.backend_cuda = self.backend_cuda.saturating_add(*count),
                "hip" => self.backend_hip = self.backend_hip.saturating_add(*count),
                "cpu" => self.backend_cpu = self.backend_cpu.saturating_add(*count),
                "cpu_simd" => self.backend_cpu_simd = self.backend_cpu_simd.saturating_add(*count),
                "f64_cpu" => self.backend_f64_cpu = self.backend_f64_cpu.saturating_add(*count),
                "faer" => self.backend_faer = self.backend_faer.saturating_add(*count),
                "naive" => self.backend_naive = self.backend_naive.saturating_add(*count),
                _ => self.backend_other = self.backend_other.saturating_add(*count),
            }
        }
        for (backend, count) in trace.by_kernel_backend.iter() {
            match backend.as_str() {
                "wgpu_dense" => {
                    self.kernel_backend_wgpu_dense =
                        self.kernel_backend_wgpu_dense.saturating_add(*count)
                }
                "simd" => {
                    self.kernel_backend_simd = self.kernel_backend_simd.saturating_add(*count)
                }
                "wgpu" | "cuda" | "hip" | "cpu" | "cpu_simd" | "faer" | "naive" => {}
                _ => self.kernel_backend_other = self.kernel_backend_other.saturating_add(*count),
            }
        }
    }

    /// Writes this summary using the same metric vocabulary as trainer traces.
    pub fn write_extra(&self, extra: &mut HashMap<String, f64>) {
        extra.insert("epoch_tensor_ops_total".to_string(), self.ops_total as f64);
        extra.insert(
            "epoch_tensor_backend_fallbacks".to_string(),
            self.fallbacks as f64,
        );
        extra.insert(
            "epoch_tensor_backend_requested_wgpu_hits".to_string(),
            self.requested_wgpu_hits as f64,
        );
        extra.insert(
            "epoch_tensor_backend_requested_wgpu_runtime_fallbacks".to_string(),
            self.requested_wgpu_runtime_fallbacks as f64,
        );
        extra.insert(
            "epoch_tensor_backend_requested_wgpu_component_hits".to_string(),
            self.requested_wgpu_component_hits as f64,
        );
        extra.insert(
            "epoch_tensor_backend_requested_wgpu_component_fallbacks".to_string(),
            self.requested_wgpu_component_fallbacks as f64,
        );
        extra.insert(
            "epoch_tensor_meta_events".to_string(),
            self.meta_events as f64,
        );
        extra.insert(
            "epoch_tensor_meta_non_finite_sentinels".to_string(),
            self.meta_non_finite_sentinels as f64,
        );
        extra.insert(
            "epoch_tensor_backend_wgpu".to_string(),
            self.backend_wgpu as f64,
        );
        extra.insert(
            "epoch_tensor_backend_cuda".to_string(),
            self.backend_cuda as f64,
        );
        extra.insert(
            "epoch_tensor_backend_hip".to_string(),
            self.backend_hip as f64,
        );
        extra.insert(
            "epoch_tensor_backend_cpu".to_string(),
            self.backend_cpu as f64,
        );
        extra.insert(
            "epoch_tensor_backend_cpu_simd".to_string(),
            self.backend_cpu_simd as f64,
        );
        extra.insert(
            "epoch_tensor_backend_f64_cpu".to_string(),
            self.backend_f64_cpu as f64,
        );
        extra.insert(
            "epoch_tensor_backend_faer".to_string(),
            self.backend_faer as f64,
        );
        extra.insert(
            "epoch_tensor_backend_naive".to_string(),
            self.backend_naive as f64,
        );
        extra.insert(
            "epoch_tensor_backend_other".to_string(),
            self.backend_other as f64,
        );
        extra.insert(
            "epoch_tensor_kernel_backend_wgpu_dense".to_string(),
            self.kernel_backend_wgpu_dense as f64,
        );
        extra.insert(
            "epoch_tensor_kernel_backend_simd".to_string(),
            self.kernel_backend_simd as f64,
        );
        extra.insert(
            "epoch_tensor_kernel_backend_other".to_string(),
            self.kernel_backend_other as f64,
        );
        extra.insert(
            "epoch_tensor_embedding_tokens".to_string(),
            self.embedding_tokens as f64,
        );
        extra.insert(
            "epoch_tensor_embedding_unique_token_indices".to_string(),
            self.embedding_unique_token_indices as f64,
        );
        extra.insert(
            "epoch_tensor_embedding_repeated_token_indices".to_string(),
            self.embedding_repeated_token_indices as f64,
        );
        extra.insert(
            "epoch_tensor_embedding_token_repairs".to_string(),
            self.embedding_token_repairs as f64,
        );
    }
}

fn meta_usize(data: &Value, key: &str) -> usize {
    data.get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0)
}

fn meta_f64(data: &Value, key: &str) -> Option<f64> {
    data.get(key)
        .and_then(|value| value.as_f64())
        .filter(|value| value.is_finite())
}

fn meta_str_fragment(data: &Value, key: &str) -> Option<String> {
    data.get(key)
        .and_then(|value| value.as_str())
        .map(metric_fragment)
}

fn insert_optional_extra(extra: &mut HashMap<String, f64>, key: &str, value: Option<f64>) {
    if let Some(value) = value.filter(|value| value.is_finite()) {
        extra.insert(key.to_string(), value);
    }
}

fn insert_tensor_shape_extra(
    extra: &mut HashMap<String, f64>,
    prefix: &str,
    shape: (usize, usize),
) {
    extra.insert(format!("{prefix}_rows"), shape.0 as f64);
    extra.insert(format!("{prefix}_cols"), shape.1 as f64);
    extra.insert(
        format!("{prefix}_values"),
        shape.0.saturating_mul(shape.1) as f64,
    );
}

#[derive(Debug, Default)]
struct ParameterValueSnapshot {
    params: usize,
    values: Vec<Vec<f32>>,
}

#[derive(Debug, Default)]
struct ParameterUpdateStats {
    params: usize,
    values: usize,
    before_l2_sq: f64,
    after_l2_sq: f64,
    delta_l1: f64,
    delta_l2_sq: f64,
    delta_linf: f64,
    non_finite_values: usize,
    active_params: usize,
    zero_update_params: usize,
    max_update_l2: f64,
    max_update_l2_index: Option<usize>,
    max_update_l2_values: usize,
    max_update_l2_ratio: f64,
    max_update_ratio_l2: f64,
    max_update_ratio_index: Option<usize>,
}

impl ParameterUpdateStats {
    fn write_extra(&self, extra: &mut HashMap<String, f64>) {
        let before_l2 = self.before_l2_sq.sqrt();
        let after_l2 = self.after_l2_sq.sqrt();
        let delta_l2 = self.delta_l2_sq.sqrt();
        let ratio = delta_l2 / before_l2.max(1.0e-12);
        extra.insert("optim_param_update_params".to_string(), self.params as f64);
        extra.insert("optim_param_update_values".to_string(), self.values as f64);
        extra.insert("optim_param_l2_before".to_string(), before_l2);
        extra.insert("optim_param_l2_after".to_string(), after_l2);
        extra.insert("optim_param_update_l1".to_string(), self.delta_l1);
        extra.insert("optim_param_update_l2".to_string(), delta_l2);
        extra.insert("optim_param_update_linf".to_string(), self.delta_linf);
        extra.insert("optim_param_update_ratio_l2".to_string(), ratio);
        extra.insert(
            "optim_param_update_active_params".to_string(),
            self.active_params as f64,
        );
        extra.insert(
            "optim_param_update_zero_params".to_string(),
            self.zero_update_params as f64,
        );
        extra.insert(
            "optim_param_update_zero_param_ratio".to_string(),
            if self.params == 0 {
                0.0
            } else {
                self.zero_update_params as f64 / self.params as f64
            },
        );
        extra.insert("optim_param_update_max_l2".to_string(), self.max_update_l2);
        extra.insert(
            "optim_param_update_max_l2_values".to_string(),
            self.max_update_l2_values as f64,
        );
        extra.insert(
            "optim_param_update_max_l2_ratio".to_string(),
            self.max_update_l2_ratio,
        );
        extra.insert(
            "optim_param_update_max_ratio_l2".to_string(),
            self.max_update_ratio_l2,
        );
        if let Some(index) = self.max_update_l2_index {
            extra.insert("optim_param_update_max_l2_index".to_string(), index as f64);
        }
        if let Some(index) = self.max_update_ratio_index {
            extra.insert(
                "optim_param_update_max_ratio_index".to_string(),
                index as f64,
            );
        }
        extra.insert(
            "optim_param_update_non_finite_values".to_string(),
            self.non_finite_values as f64,
        );
        if delta_l2 > 0.0 {
            extra.insert("optim_param_update_detected".to_string(), 1.0);
        }
        if self.non_finite_values > 0 {
            extra.insert("optim_param_update_non_finite_detected".to_string(), 1.0);
        }
    }
}

#[derive(Debug, Default)]
struct TensorValueHealth {
    total: usize,
    finite: usize,
    non_finite: usize,
    nan: usize,
    infinite: usize,
    finite_l1: f64,
    finite_l2_sq: f64,
    finite_linf: f32,
}

impl TensorValueHealth {
    fn from_tensor(tensor: &Tensor) -> Self {
        let mut health = Self::default();
        health.record_values(tensor.data());
        health
    }

    fn record_values(&mut self, values: &[f32]) {
        for &value in values {
            self.total = self.total.saturating_add(1);
            if value.is_finite() {
                let abs = value.abs();
                self.finite = self.finite.saturating_add(1);
                self.finite_l1 += abs as f64;
                self.finite_l2_sq += (value as f64) * (value as f64);
                self.finite_linf = self.finite_linf.max(abs);
            } else {
                self.non_finite = self.non_finite.saturating_add(1);
                if value.is_nan() {
                    self.nan = self.nan.saturating_add(1);
                } else if value.is_infinite() {
                    self.infinite = self.infinite.saturating_add(1);
                }
            }
        }
    }

    fn non_finite_ratio(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.non_finite as f64 / self.total as f64
        }
    }

    fn write_extra(&self, extra: &mut HashMap<String, f64>, prefix: &str) {
        if self.total == 0 {
            return;
        }
        extra.insert(format!("{prefix}_values_total"), self.total as f64);
        extra.insert(format!("{prefix}_finite_values"), self.finite as f64);
        extra.insert(
            format!("{prefix}_non_finite_values"),
            self.non_finite as f64,
        );
        extra.insert(
            format!("{prefix}_non_finite_ratio"),
            self.non_finite_ratio(),
        );
        extra.insert(format!("{prefix}_nan_values"), self.nan as f64);
        extra.insert(format!("{prefix}_infinite_values"), self.infinite as f64);
        extra.insert(format!("{prefix}_l1_finite"), self.finite_l1);
        extra.insert(format!("{prefix}_l2_finite"), self.finite_l2_sq.sqrt());
        extra.insert(format!("{prefix}_linf_finite"), self.finite_linf as f64);
        if self.non_finite > 0 {
            extra.insert(format!("{prefix}_non_finite_detected"), 1.0);
        }
    }
}

fn capture_parameter_value_snapshot<M: Module + ?Sized>(
    module: &M,
) -> PureResult<ParameterValueSnapshot> {
    let mut snapshot = ParameterValueSnapshot::default();
    module.visit_parameters(&mut |param| {
        snapshot.params = snapshot.params.saturating_add(1);
        snapshot.values.push(param.value().data().to_vec());
        Ok(())
    })?;
    Ok(snapshot)
}

fn collect_parameter_update_stats<M: Module + ?Sized>(
    module: &M,
    snapshot: &ParameterValueSnapshot,
) -> PureResult<ParameterUpdateStats> {
    let mut stats = ParameterUpdateStats::default();
    let mut index = 0usize;
    module.visit_parameters(&mut |param| {
        let Some(before) = snapshot.values.get(index) else {
            return Err(TensorError::ShapeMismatch {
                left: (snapshot.values.len(), 1),
                right: (index.saturating_add(1), 1),
            });
        };
        let after = param.value().data();
        if before.len() != after.len() {
            return Err(TensorError::ShapeMismatch {
                left: (before.len(), 1),
                right: (after.len(), 1),
            });
        }
        stats.params = stats.params.saturating_add(1);
        let param_index = index;
        let mut param_before_l2_sq = 0.0f64;
        let mut param_delta_l2_sq = 0.0f64;
        for (&before_value, &after_value) in before.iter().zip(after.iter()) {
            stats.values = stats.values.saturating_add(1);
            if before_value.is_finite() {
                let value = before_value as f64;
                stats.before_l2_sq += value * value;
                param_before_l2_sq += value * value;
            } else {
                stats.non_finite_values = stats.non_finite_values.saturating_add(1);
            }
            if after_value.is_finite() {
                let value = after_value as f64;
                stats.after_l2_sq += value * value;
            } else {
                stats.non_finite_values = stats.non_finite_values.saturating_add(1);
            }
            let delta = after_value - before_value;
            if delta.is_finite() {
                let value = delta as f64;
                stats.delta_l1 += value.abs();
                stats.delta_l2_sq += value * value;
                param_delta_l2_sq += value * value;
                stats.delta_linf = stats.delta_linf.max(value.abs());
            } else {
                stats.non_finite_values = stats.non_finite_values.saturating_add(1);
            }
        }
        let param_update_l2 = param_delta_l2_sq.sqrt();
        let param_update_ratio_l2 = param_update_l2 / param_before_l2_sq.sqrt().max(1.0e-12);
        if param_update_l2 > 0.0 {
            stats.active_params = stats.active_params.saturating_add(1);
        } else {
            stats.zero_update_params = stats.zero_update_params.saturating_add(1);
        }
        if stats
            .max_update_l2_index
            .map(|_| param_update_l2 > stats.max_update_l2)
            .unwrap_or(true)
        {
            stats.max_update_l2 = param_update_l2;
            stats.max_update_l2_index = Some(param_index);
            stats.max_update_l2_values = after.len();
            stats.max_update_l2_ratio = param_update_ratio_l2;
        }
        if stats
            .max_update_ratio_index
            .map(|_| param_update_ratio_l2 > stats.max_update_ratio_l2)
            .unwrap_or(true)
        {
            stats.max_update_ratio_l2 = param_update_ratio_l2;
            stats.max_update_ratio_index = Some(param_index);
        }
        index = index.saturating_add(1);
        Ok(())
    })?;
    if index != snapshot.params {
        return Err(TensorError::ShapeMismatch {
            left: (snapshot.params, 1),
            right: (index, 1),
        });
    }
    Ok(stats)
}

#[derive(Debug, Default)]
struct TensorMetaHealth {
    null_values: usize,
    non_finite_strings: usize,
}

impl TensorMetaHealth {
    fn record(&mut self, value: &Value) {
        match value {
            Value::Null => {
                self.null_values = self.null_values.saturating_add(1);
            }
            Value::Bool(_) | Value::Number(_) => {}
            Value::String(text) => {
                if is_non_finite_sentinel(text) {
                    self.non_finite_strings = self.non_finite_strings.saturating_add(1);
                }
            }
            Value::Array(values) => {
                for value in values {
                    self.record(value);
                }
            }
            Value::Object(values) => {
                for value in values.values() {
                    self.record(value);
                }
            }
        }
    }
}

fn is_non_finite_sentinel(text: &str) -> bool {
    let trimmed = text.trim();
    matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "nan" | "+nan" | "-nan" | "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity"
    )
}

fn validate_trainer_scalar(label: &'static str, value: f32) -> PureResult<f32> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(value)
}

fn validate_trainer_slice(label: &'static str, values: &[f32]) -> PureResult<()> {
    for value in values.iter().copied() {
        validate_trainer_scalar(label, value)?;
    }
    Ok(())
}

fn validate_trainer_tensor(label: &'static str, tensor: &Tensor) -> PureResult<()> {
    validate_trainer_slice(label, tensor.data())
}

fn trainer_tensor_sum(label: &'static str, tensor: &Tensor) -> PureResult<f32> {
    validate_trainer_tensor(label, tensor)?;
    let sum = tensor
        .data()
        .iter()
        .fold(0.0f64, |acc, value| acc + f64::from(*value));
    if !sum.is_finite() || sum.abs() > f64::from(f32::MAX) {
        let value = if sum.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(sum as f32)
}

fn validate_band_energy_for_trainer(energy: &BandEnergy) -> PureResult<()> {
    validate_trainer_scalar("trainer_band_energy_above", energy.above)?;
    validate_trainer_scalar("trainer_band_energy_here", energy.here)?;
    validate_trainer_scalar("trainer_band_energy_beneath", energy.beneath)?;
    validate_trainer_scalar("trainer_band_energy_drift", energy.drift)?;
    validate_trainer_scalar(
        "trainer_band_spectral_sheet_confidence",
        energy.spectral.sheet_confidence,
    )?;
    validate_trainer_scalar("trainer_band_spectral_curvature", energy.spectral.curvature)?;
    validate_trainer_scalar("trainer_band_spectral_spin", energy.spectral.spin)?;
    validate_trainer_scalar("trainer_band_spectral_energy", energy.spectral.energy)?;
    Ok(())
}

fn validate_band_weights_for_trainer(weights: (f32, f32, f32)) -> PureResult<()> {
    validate_trainer_scalar("trainer_band_weight_above", weights.0)?;
    validate_trainer_scalar("trainer_band_weight_here", weights.1)?;
    validate_trainer_scalar("trainer_band_weight_beneath", weights.2)?;
    Ok(())
}

#[derive(Debug, Default)]
struct GradientHealth {
    total: usize,
    finite: usize,
    non_finite: usize,
    nan: usize,
    infinite: usize,
    finite_l1: f64,
    finite_l2_sq: f64,
    finite_linf: f32,
    first_non_finite: Option<f32>,
}

impl GradientHealth {
    fn record_values(&mut self, values: &[f32]) {
        for &value in values {
            self.total = self.total.saturating_add(1);
            if value.is_finite() {
                let abs = value.abs();
                self.finite = self.finite.saturating_add(1);
                self.finite_l1 += abs as f64;
                self.finite_l2_sq += (value as f64) * (value as f64);
                self.finite_linf = self.finite_linf.max(abs);
            } else {
                self.non_finite = self.non_finite.saturating_add(1);
                if self.first_non_finite.is_none() {
                    self.first_non_finite = Some(value);
                }
                if value.is_nan() {
                    self.nan = self.nan.saturating_add(1);
                } else if value.is_infinite() {
                    self.infinite = self.infinite.saturating_add(1);
                }
            }
        }
    }

    fn non_finite_ratio(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.non_finite as f64 / self.total as f64
        }
    }

    fn write_extra(&self, extra: &mut HashMap<String, f64>) {
        if self.total == 0 {
            return;
        }
        extra.insert("grad_values_total".to_string(), self.total as f64);
        extra.insert("grad_values_finite".to_string(), self.finite as f64);
        extra.insert("grad_values_non_finite".to_string(), self.non_finite as f64);
        extra.insert(
            "grad_values_non_finite_ratio".to_string(),
            self.non_finite_ratio(),
        );
        extra.insert("grad_values_nan".to_string(), self.nan as f64);
        extra.insert("grad_values_infinite".to_string(), self.infinite as f64);
        extra.insert("grad_l1_finite".to_string(), self.finite_l1);
        extra.insert("grad_l2_finite".to_string(), self.finite_l2_sq.sqrt());
        extra.insert("grad_linf_finite".to_string(), self.finite_linf as f64);
        if self.non_finite > 0 {
            extra.insert("grad_non_finite_detected".to_string(), 1.0);
        }
    }

    fn ensure_finite(&self, label: &'static str) -> PureResult<()> {
        if let Some(value) = self.first_non_finite {
            return Err(TensorError::NonFiniteValue { label, value });
        }
        Ok(())
    }
}

fn write_backend_policy_extra(policy: BackendPolicy, extra: &mut HashMap<String, f64>) {
    extra.insert(
        format!(
            "tensor_policy_device_{}",
            metric_fragment(policy.device_backend_label())
        ),
        1.0,
    );
    extra.insert(
        format!(
            "tensor_policy_matmul_{}",
            metric_fragment(policy.matmul_backend_label())
        ),
        1.0,
    );
    extra.insert(
        format!(
            "tensor_policy_prepacked_matmul_{}",
            metric_fragment(policy.prepacked_matmul_backend_label())
        ),
        1.0,
    );
    extra.insert(
        format!(
            "tensor_policy_layer_norm_{}",
            metric_fragment(policy.layer_norm_backend_label())
        ),
        1.0,
    );
    extra.insert(
        format!(
            "tensor_policy_attention_{}",
            metric_fragment(policy.attention_backend_label())
        ),
        1.0,
    );
    extra.insert(
        format!(
            "tensor_policy_softmax_{}",
            metric_fragment(policy.softmax_backend_label())
        ),
        1.0,
    );
}

fn write_coherence_repair_extra(
    signal: Option<&CoherenceSignal>,
    extra: &mut HashMap<String, f64>,
) {
    let Some(signal) = signal else {
        return;
    };
    extra.insert(
        "coherence_repaired_non_finite_weights".to_string(),
        signal.repaired_non_finite_weights() as f64,
    );
    extra.insert(
        "coherence_repaired_negative_weights".to_string(),
        signal.repaired_negative_weights() as f64,
    );
    extra.insert(
        "coherence_repaired_weights_total".to_string(),
        signal.repaired_weights_total() as f64,
    );
    extra.insert(
        "coherence_pre_discard_repaired_non_finite".to_string(),
        signal.pre_discard_repaired_non_finite() as f64,
    );
    extra.insert(
        "coherence_pre_discard_repaired_negative".to_string(),
        signal.pre_discard_repaired_negative() as f64,
    );
    extra.insert(
        "coherence_pre_discard_repairs_total".to_string(),
        signal.pre_discard_repairs_total() as f64,
    );
    extra.insert(
        "coherence_repairs_total".to_string(),
        signal.repairs_total() as f64,
    );
    if signal.repairs_total() > 0 {
        extra.insert("coherence_repaired_detected".to_string(), 1.0);
    }
}

struct TensorOpMetaStepCollector {
    _observer_lock: std::sync::MutexGuard<'static, ()>,
    trace: Arc<Mutex<TensorBackendStepTrace>>,
    previous: Option<TensorOpMetaObserver>,
}

impl TensorOpMetaStepCollector {
    fn install() -> Self {
        let observer_lock = tensor_meta_observer_lock();
        let trace = Arc::new(Mutex::new(TensorBackendStepTrace::default()));
        let trace_capture = Arc::clone(&trace);
        let previous_slot: Arc<Mutex<Option<TensorOpMetaObserver>>> = Arc::new(Mutex::new(None));
        let previous_capture = Arc::clone(&previous_slot);
        let observer: TensorOpMetaObserver = Arc::new(move |event: &TensorOpMetaEvent| {
            trace_capture
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .record(event);
            let previous = previous_capture
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            if let Some(previous) = previous {
                previous(event);
            }
        });
        let previous = set_tensor_op_meta_observer(Some(observer));
        *previous_slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = previous.clone();
        Self {
            _observer_lock: observer_lock,
            trace,
            previous,
        }
    }

    fn clear(&self) {
        *self
            .trace
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = TensorBackendStepTrace::default();
    }

    fn drain(&self) -> TensorBackendStepTrace {
        std::mem::take(
            &mut *self
                .trace
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }
}

impl Drop for TensorOpMetaStepCollector {
    fn drop(&mut self) {
        set_tensor_op_meta_observer(self.previous.take());
    }
}

fn metric_fragment(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    let mut last_underscore = false;
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn backend_metric_fragment(label: &str) -> String {
    let fragment = metric_fragment(label);
    match fragment.as_str() {
        "wgpu_dense" => "wgpu".to_string(),
        "simd" => "cpu_simd".to_string(),
        _ => fragment,
    }
}

fn is_metadata_only_backend(label: &str) -> bool {
    matches!(
        label,
        "composite" | "hybrid" | "view" | "semantic_bridge_window_distribution"
    )
}

fn is_cpu_runtime_backend(label: &str) -> bool {
    matches!(
        label,
        "cpu"
            | "cpu_eigen"
            | "cpu_simd"
            | "f64_cpu"
            | "faer"
            | "naive"
            | "probability_cpu"
            | "semantic_cpu"
            | "topos_cpu"
    )
}

fn is_wgpu_runtime_fallback(data: &Value) -> bool {
    let Some(fallback) = data.get("fallback") else {
        return false;
    };
    if fallback.get("from").and_then(Value::as_str) != Some("wgpu") {
        return false;
    }
    if fallback.get("reason").and_then(Value::as_str) == Some("runtime_unavailable") {
        return true;
    }
    let Some(message) = fallback.get("message").and_then(Value::as_str) else {
        return false;
    };
    message.contains("no suitable WGPU adapter")
        || message.contains("failed to initialize WGPU")
        || message.contains("WGPU backend not available")
}

fn strict_gpu_path() -> bool {
    #[cfg(test)]
    {
        if let Some(value) = STRICT_GPU_TEST_OVERRIDE.with(|slot| *slot.borrow()) {
            return value;
        }
    }
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn strict_expected_tensor_backend(caps: DeviceCaps) -> Option<BackendKind> {
    if !strict_gpu_path() {
        return None;
    }
    match caps.backend {
        BackendKind::Wgpu | BackendKind::Cuda | BackendKind::Hip => Some(caps.backend),
        BackendKind::Mps | BackendKind::Cpu => None,
    }
}

/// High-level orchestrator that keeps hypergrad, SpiralK, and module updates aligned.
pub struct ModuleTrainer {
    epoch: usize,
    planner: RankPlanner,
    curvature: f32,
    hyper_learning_rate: f32,
    fallback_learning_rate: f32,
    grad_clip_max_norm: Option<f32>,
    real_learning_rate: Option<f32>,
    blackcat: Option<BlackCatRuntime>,
    blackcat_moderator: Option<BlackcatModerator>,
    autopilot: Option<Autopilot>,
    band_weight_fn: Option<BandWeightFn>,
    text_infusion: Option<TextInfusionConfig>,
    injector_enabled: bool,
    distribution: Option<RoundtableNode>,
    meta_conductor: Option<MetaConductor>,
    heur_log: HeurOpLog,
    rewrite_budget: Option<RewriteBudget>,
    softlogic: SoftLogicFlex,
    spectral_adapter: SpectralLrAdapter,
    accumulator_synchronizer: Option<Arc<dyn AccumulatorSynchronizer>>,
    last_accumulator_sync: TrainerAccumulatorSyncStats,
    desire_bridge: Option<DesireTrainerBridge>,
    desire_roundtable_bridge: Option<DesireRoundtableBridge>,
    last_desire_roundtable_summary: Option<DesireRoundtableSummary>,
    #[cfg(feature = "psi")]
    desire_psi_bridge: Option<DesirePsiBridge>,
    graph_bridge: Option<GraphConsensusBridge>,
    graph_pending: Option<GraphConsensusDigest>,
    graph_last_hint: Option<String>,
    gnn_roundtable_bridge: Option<RoundtableGnnBridge>,
    gnn_last_roundtable_signal: Option<RoundtableBandSignal>,
    curvature_scheduler: Option<CurvatureScheduler>,
    last_curvature_metrics: Option<CurvatureMetrics>,
    loss_strategy: LossStrategy,
    spectral_policy: Option<SpectralLearningRatePolicy>,
    coherence_bridge: Option<ZSpaceTraceCoherenceBridge>,
    pending_coherence: Option<CoherenceSignal>,
    last_spectral_metrics: Option<SpectralAdjustmentMetrics>,
    last_band_energy: Option<BandEnergy>,
    phase_turnover_spike_threshold: f32,
    phase_loss_ema_alpha: f32,
    phase_loss_spike_ratio: f32,
    phase_drift_spike_threshold: f32,
    phase_last_label: Option<CoherenceLabel>,
    phase_last_turnover: Option<f32>,
    phase_last_band: Option<u8>,
    phase_last_drift_abs: Option<f32>,
    phase_loss_ema: Option<f32>,
    phase_loss_spiking: bool,
    #[cfg(feature = "golden")]
    golden_pulse: Option<GoldenBlackcatPulse>,
    #[cfg(feature = "golden")]
    golden_directive: Option<GoldenCooperativeDirective>,
    #[cfg(feature = "golden")]
    golden_council: Option<GoldenCouncilSnapshot>,
    #[cfg(feature = "psi")]
    psi: Option<PsiMeter>,
    #[cfg(feature = "psychoid")]
    psychoid: Option<PsychoidMeter>,
    #[cfg(feature = "psychoid")]
    psychoid_log: bool,
    #[cfg(feature = "collapse")]
    collapse: Option<CollapseDrive>,
}

impl core::fmt::Debug for ModuleTrainer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ModuleTrainer(curv={},lr_h={},lr_f={},lr_r={:?})",
            self.curvature,
            self.hyper_learning_rate,
            self.fallback_learning_rate,
            self.real_learning_rate
        )
    }
}

/// Copyable optimizer-facing trainer state for traces and future checkpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainerOptimizerState {
    pub epoch: usize,
    pub curvature: f32,
    pub hyper_learning_rate: f32,
    pub fallback_learning_rate: f32,
    pub real_learning_rate: Option<f32>,
    pub grad_clip_max_norm: Option<f32>,
    pub spectral_policy_enabled: bool,
    pub curvature_scheduler_enabled: bool,
    pub training_device_enabled: bool,
    pub training_rank: usize,
    pub training_world_size: usize,
    pub spectral_adapter: SpectralLrAdapterState,
}

impl TrainerOptimizerState {
    fn write_extra(&self, extra: &mut HashMap<String, f64>) {
        extra.insert("optim_state_epoch".to_string(), self.epoch as f64);
        extra.insert("optim_state_curvature".to_string(), self.curvature as f64);
        extra.insert(
            "optim_state_hyper_lr".to_string(),
            self.hyper_learning_rate as f64,
        );
        extra.insert(
            "optim_state_fallback_lr".to_string(),
            self.fallback_learning_rate as f64,
        );
        extra.insert(
            "optim_state_realgrad_enabled".to_string(),
            if self.real_learning_rate.is_some() {
                1.0
            } else {
                0.0
            },
        );
        if let Some(rate) = self.real_learning_rate {
            extra.insert("optim_state_real_lr".to_string(), rate as f64);
        }
        extra.insert(
            "optim_state_grad_clip_enabled".to_string(),
            if self.grad_clip_max_norm.is_some() {
                1.0
            } else {
                0.0
            },
        );
        if let Some(limit) = self.grad_clip_max_norm {
            extra.insert("optim_state_grad_clip_max_norm".to_string(), limit as f64);
        }
        extra.insert(
            "optim_state_spectral_policy_enabled".to_string(),
            if self.spectral_policy_enabled {
                1.0
            } else {
                0.0
            },
        );
        extra.insert(
            "optim_state_curvature_scheduler_enabled".to_string(),
            if self.curvature_scheduler_enabled {
                1.0
            } else {
                0.0
            },
        );
        extra.insert(
            "optim_state_training_device_enabled".to_string(),
            if self.training_device_enabled {
                1.0
            } else {
                0.0
            },
        );
        extra.insert(
            "optim_state_training_rank".to_string(),
            self.training_rank as f64,
        );
        extra.insert(
            "optim_state_training_world_size".to_string(),
            self.training_world_size as f64,
        );
        extra.insert(
            "optim_state_adapter_sheet_hint".to_string(),
            self.spectral_adapter.sheet_hint as f64,
        );
        extra.insert(
            "optim_state_adapter_curvature_target".to_string(),
            self.spectral_adapter.curvature_target as f64,
        );
        extra.insert(
            "optim_state_adapter_avg_curvature".to_string(),
            self.spectral_adapter.avg_curvature as f64,
        );
        extra.insert(
            "optim_state_adapter_avg_spin".to_string(),
            self.spectral_adapter.avg_spin as f64,
        );
        extra.insert(
            "optim_state_adapter_avg_energy".to_string(),
            self.spectral_adapter.avg_energy as f64,
        );
        extra.insert(
            "optim_state_adapter_min_scale".to_string(),
            self.spectral_adapter.min_scale as f64,
        );
        extra.insert(
            "optim_state_adapter_max_scale".to_string(),
            self.spectral_adapter.max_scale as f64,
        );
    }
}

/// Summary of the most recent accumulator synchronization pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrainerAccumulatorSyncStats {
    pub enabled: bool,
    pub rank: usize,
    pub world_size: usize,
    pub buffers: usize,
    pub values: usize,
}

impl TrainerAccumulatorSyncStats {
    fn disabled() -> Self {
        Self {
            enabled: false,
            rank: 0,
            world_size: 1,
            buffers: 0,
            values: 0,
        }
    }

    fn write_extra(&self, extra: &mut HashMap<String, f64>) {
        extra.insert(
            "optim_accumulator_sync_enabled".to_string(),
            if self.enabled { 1.0 } else { 0.0 },
        );
        extra.insert("optim_accumulator_sync_rank".to_string(), self.rank as f64);
        extra.insert(
            "optim_accumulator_sync_world_size".to_string(),
            self.world_size as f64,
        );
        extra.insert(
            "optim_accumulator_sync_buffers".to_string(),
            self.buffers as f64,
        );
        extra.insert(
            "optim_accumulator_sync_values".to_string(),
            self.values as f64,
        );
    }
}

/// Function pointer used to convert band energy into Above/Here/Beneath weights.
pub type BandWeightFn = fn(BandEnergy) -> (f32, f32, f32);

/// Controls how often the trainer injects [`Module::infuse_text`] calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextInfusionEvery {
    /// Infuse once at the start of the first epoch only.
    Once,
    /// Infuse once at the start of each epoch (after clearing accumulators).
    Epoch,
    /// Infuse before every optimisation step (each batch).
    Batch,
}

/// Controls whether infused text is blended into the loss update or applied as a
/// separate optimiser step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextInfusionMode {
    /// Add infused gradients to the current loss update before stepping.
    Blend,
    /// Apply infused gradients in a dedicated optimiser step.
    Separate,
}

#[derive(Clone, Debug)]
struct TextInfusionConfig {
    text: String,
    every: TextInfusionEvery,
    mode: TextInfusionMode,
}

fn append_cloud_targets(metadata: &mut HashMap<String, String>, targets: &[CloudConnector]) {
    CloudTargetSummary::from_targets(targets).extend_map(metadata);
}

/// Configures how regional Z-space weights influence the aggregated loss.
#[derive(Clone, Debug)]
pub struct RegionLossWeights {
    default: f32,
    overrides: HashMap<ZSpaceRegionKey, f32>,
}

impl RegionLossWeights {
    /// Creates a weight table with the provided default multiplier.
    pub fn new(default: f32) -> Self {
        Self {
            default: default.max(0.0),
            overrides: HashMap::new(),
        }
    }

    /// Inserts or replaces the multiplier for the supplied region key.
    pub fn with_override(mut self, key: ZSpaceRegionKey, weight: f32) -> Self {
        self.set_override(key, weight);
        self
    }

    /// Updates the multiplier for the supplied region key.
    pub fn set_override(&mut self, key: ZSpaceRegionKey, weight: f32) {
        self.overrides.insert(key, weight.max(0.0));
    }

    /// Looks up the multiplier for the given key.
    pub fn weight_for(&self, key: ZSpaceRegionKey) -> f32 {
        self.overrides
            .get(&key)
            .copied()
            .unwrap_or(self.default)
            .max(0.0)
    }

    /// Returns the default multiplier applied when no overrides match.
    pub fn default(&self) -> f32 {
        self.default
    }
}

impl Default for RegionLossWeights {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Optional thresholds controlling when region weights apply.
#[derive(Clone, Debug, Default)]
pub struct RegionLossCondition {
    min_spin_magnitude: Option<f32>,
    min_radius: Option<f32>,
}

impl RegionLossCondition {
    /// Requires the absolute spin alignment to exceed the provided threshold.
    pub fn with_min_spin_magnitude(mut self, threshold: f32) -> Self {
        if threshold > 0.0 {
            self.min_spin_magnitude = Some(threshold.min(1.0));
        }
        self
    }

    /// Requires the normalised radius to exceed the provided threshold.
    pub fn with_min_radius(mut self, threshold: f32) -> Self {
        if threshold > 0.0 {
            self.min_radius = Some(threshold.clamp(0.0, 1.0));
        }
        self
    }

    fn satisfied(&self, descriptor: &ZSpaceRegionDescriptor) -> bool {
        if let Some(threshold) = self.min_spin_magnitude {
            if descriptor.spin_alignment.abs() < threshold {
                return false;
            }
        }
        if let Some(threshold) = self.min_radius {
            if descriptor.normalized_radius < threshold {
                return false;
            }
        }
        true
    }

    /// Returns the minimum spin magnitude threshold, if configured.
    pub fn min_spin_magnitude(&self) -> Option<f32> {
        self.min_spin_magnitude
    }

    /// Returns the minimum radius threshold, if configured.
    pub fn min_radius(&self) -> Option<f32> {
        self.min_radius
    }
}

/// Adaptive policy that adjusts regional multipliers based on observed losses.
#[derive(Clone, Debug)]
pub struct AdaptiveRegionWeighting {
    learning_rate: f32,
    min_samples: u32,
    min_multiplier: f32,
    max_multiplier: f32,
    epsilon: f32,
    global_loss: Option<f32>,
    entries: HashMap<ZSpaceRegionKey, AdaptiveRegionEntry>,
}

impl AdaptiveRegionWeighting {
    /// Builds a policy with sensible defaults for smoothing and bounds.
    pub fn new() -> Self {
        Self {
            learning_rate: 0.1,
            min_samples: 4,
            min_multiplier: 0.25,
            max_multiplier: 4.0,
            epsilon: 1e-4,
            global_loss: None,
            entries: HashMap::new(),
        }
    }

    /// Sets the learning rate used for EMA updates.
    pub fn with_learning_rate(mut self, rate: f32) -> Self {
        if rate.is_finite() && rate > 0.0 {
            self.learning_rate = rate.clamp(0.01, 1.0);
        }
        self
    }

    /// Sets the minimum sample count required before adaptation kicks in.
    pub fn with_min_samples(mut self, samples: u32) -> Self {
        if samples > 0 {
            self.min_samples = samples;
        }
        self
    }

    /// Sets the allowed multiplier bounds.
    pub fn with_bounds(mut self, min: f32, max: f32) -> Self {
        if min.is_finite() && max.is_finite() && min > 0.0 && max >= min {
            self.min_multiplier = min;
            self.max_multiplier = max;
        }
        self
    }

    /// Returns the last recorded global loss EMA.
    pub fn global_loss(&self) -> Option<f32> {
        self.global_loss
    }

    /// Returns the adaptive entry tracked for a specific region key.
    pub fn entry_for(&self, key: ZSpaceRegionKey) -> Option<&AdaptiveRegionEntry> {
        self.entries.get(&key)
    }

    fn observe_background(&mut self, loss: f32) {
        let magnitude = loss.max(self.epsilon);
        let reference = self.global_loss.unwrap_or(magnitude);
        let next = reference + self.learning_rate * (magnitude - reference);
        self.global_loss = Some(next);
    }

    fn observe_region(&mut self, key: ZSpaceRegionKey, loss: f32) -> f32 {
        let magnitude = loss.max(self.epsilon);
        let reference = self.global_loss.unwrap_or(magnitude);
        let entry = self.entries.entry(key).or_default();
        entry.samples = entry.samples.saturating_add(1);
        if entry.samples == 1 {
            entry.ema_loss = magnitude;
        } else {
            entry.ema_loss += self.learning_rate * (magnitude - entry.ema_loss);
        }
        let mut multiplier = 1.0;
        if entry.samples >= self.min_samples && reference > self.epsilon {
            multiplier =
                (entry.ema_loss / reference).clamp(self.min_multiplier, self.max_multiplier);
        }
        entry.multiplier = multiplier;
        let next_global = reference + self.learning_rate * (magnitude - reference);
        self.global_loss = Some(next_global);
        multiplier
    }
}

impl Default for AdaptiveRegionWeighting {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveRegionEntry {
    ema_loss: f32,
    samples: u32,
    multiplier: f32,
}

impl AdaptiveRegionEntry {
    fn new() -> Self {
        Self {
            ema_loss: 0.0,
            samples: 0,
            multiplier: 1.0,
        }
    }

    /// Returns the EMA of the regional loss.
    pub fn ema_loss(&self) -> f32 {
        self.ema_loss
    }

    /// Returns the number of samples accumulated for this region.
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Returns the most recent adaptive multiplier.
    pub fn multiplier(&self) -> f32 {
        self.multiplier
    }
}

impl Default for AdaptiveRegionEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete configuration describing how regional multipliers are applied.
#[derive(Clone, Debug)]
pub struct RegionLossConfig {
    weights: RegionLossWeights,
    condition: RegionLossCondition,
    adaptive: Option<AdaptiveRegionWeighting>,
    history: Option<RegionHeatmapHistory>,
}

impl RegionLossConfig {
    /// Creates a configuration with the supplied weight table and default condition.
    pub fn new(weights: RegionLossWeights) -> Self {
        Self {
            weights,
            condition: RegionLossCondition::default(),
            adaptive: None,
            history: None,
        }
    }

    /// Updates the condition controlling when region weights are used.
    pub fn with_condition(mut self, condition: RegionLossCondition) -> Self {
        self.condition = condition;
        self
    }

    /// Attaches an adaptive weighting policy to the configuration.
    pub fn with_adaptive(mut self, adaptive: AdaptiveRegionWeighting) -> Self {
        self.adaptive = Some(adaptive);
        self
    }

    /// Enables temporal aggregation over a rolling window of snapshots.
    pub fn with_history_window(mut self, window: usize) -> Self {
        if window >= 2 {
            self.history = Some(RegionHeatmapHistory::new(window));
        }
        self
    }

    fn region_factor(
        &mut self,
        feedback: &SoftlogicZFeedback,
        base_loss: f32,
    ) -> Option<(f32, ZSpaceRegionDescriptor)> {
        let descriptor = match feedback.region_descriptor() {
            Some(descriptor) => descriptor,
            None => {
                if let Some(adaptive) = self.adaptive.as_mut() {
                    adaptive.observe_background(base_loss.abs());
                }
                return None;
            }
        };
        if !self.condition.satisfied(&descriptor) {
            if let Some(adaptive) = self.adaptive.as_mut() {
                adaptive.observe_background(base_loss.abs());
            }
            return None;
        }
        let factor = self.weights.weight_for(descriptor.key());
        let mut factor = factor;
        if let Some(adaptive) = self.adaptive.as_mut() {
            let multiplier = adaptive.observe_region(descriptor.key(), base_loss.abs());
            factor *= multiplier;
        }
        Some((factor, descriptor))
    }

    /// Returns a reference to the underlying weights.
    pub fn weights(&self) -> &RegionLossWeights {
        &self.weights
    }

    /// Returns the configured condition.
    pub fn condition(&self) -> &RegionLossCondition {
        &self.condition
    }

    /// Returns the adaptive weighting policy, if one is configured.
    pub fn adaptive(&self) -> Option<&AdaptiveRegionWeighting> {
        self.adaptive.as_ref()
    }

    /// Returns the configured history buffer, if enabled.
    pub fn history(&self) -> Option<&RegionHeatmapHistory> {
        self.history.as_ref()
    }

    /// Captures a snapshot of all region multipliers for visualization.
    pub fn snapshot(&self, highlight: Option<ZSpaceRegionDescriptor>) -> RegionHeatmapSnapshot {
        let mut snapshot = RegionHeatmapSnapshot::new(self.weights.default())
            .with_highlight(highlight)
            .with_condition(
                self.condition.min_spin_magnitude(),
                self.condition.min_radius(),
            );
        if let Some(adaptive) = self.adaptive.as_ref() {
            snapshot = snapshot.with_global_loss(adaptive.global_loss());
        }
        for spin in ZSpaceSpinBand::values() {
            for radius in ZSpaceRadiusBand::values() {
                let key = ZSpaceRegionKey::new(spin, radius);
                let base_weight = self.weights.weight_for(key);
                let (adaptive_multiplier, samples, ema_loss) = if let Some(adaptive) =
                    self.adaptive.as_ref()
                {
                    adaptive
                        .entry_for(key)
                        .map(|entry| (entry.multiplier(), entry.samples(), Some(entry.ema_loss())))
                        .unwrap_or((1.0, 0, None))
                } else {
                    (1.0, 0, None)
                };
                snapshot.insert_cell(RegionHeatmapCell::new(
                    key,
                    base_weight,
                    adaptive_multiplier,
                    samples,
                    ema_loss,
                ));
            }
        }
        snapshot
    }

    /// Builds a bundle of reports, updating the history buffer when enabled.
    pub fn reports(
        &mut self,
        highlight: Option<ZSpaceRegionDescriptor>,
        step: u64,
    ) -> RegionReportBundle {
        let snapshot = self.snapshot(highlight);
        let mut bundle = RegionReportBundle {
            snapshot: Some(snapshot.clone().into_report()),
            ..RegionReportBundle::default()
        };
        if let Some(history) = self.history.as_mut() {
            history.push(step, snapshot);
            bundle.trend = history.delta_report();
            bundle.volatility = history.volatility_report();
        }
        bundle
    }

    /// Builds an attribution report summarising the configured region weights.
    pub fn visualize(&self, highlight: Option<ZSpaceRegionDescriptor>) -> AttributionReport {
        self.snapshot(highlight).into_report()
    }
}

/// Loss aggregation strategies supported by [`ModuleTrainer`].
#[derive(Clone, Debug, Default)]
pub enum LossStrategy {
    /// Baseline behaviour using band weights only.
    #[default]
    Baseline,
    /// Applies additional region weights extracted from Softlogic feedback.
    Region(RegionLossConfig),
}

impl LossStrategy {
    fn region_factor(
        &mut self,
        feedback: &SoftlogicZFeedback,
        base_loss: f32,
    ) -> Option<(f32, ZSpaceRegionDescriptor)> {
        match self {
            LossStrategy::Baseline => None,
            LossStrategy::Region(config) => config.region_factor(feedback, base_loss),
        }
    }

    fn region_reports(
        &mut self,
        highlight: Option<ZSpaceRegionDescriptor>,
        step: u64,
    ) -> RegionReportBundle {
        match self {
            LossStrategy::Baseline => RegionReportBundle::default(),
            LossStrategy::Region(config) => config.reports(highlight, step),
        }
    }
}

/// Container bundling the available region visualization reports.
#[derive(Default, Clone, Debug)]
pub struct RegionReportBundle {
    pub snapshot: Option<AttributionReport>,
    pub trend: Option<AttributionReport>,
    pub volatility: Option<AttributionReport>,
}

/// Configuration describing how SoftLogic adjusts gradient band weighting.
///
/// Environment overrides (all optional):
/// - `SPIRAL_SOFTLOGIC_INERTIA`
/// - `SPIRAL_SOFTLOGIC_INERTIA_MIN`
/// - `SPIRAL_SOFTLOGIC_INERTIA_DRIFT_K`
/// - `SPIRAL_SOFTLOGIC_INERTIA_Z_K`
/// - `SPIRAL_SOFTLOGIC_DRIFT_GAIN`
/// - `SPIRAL_SOFTLOGIC_PSI_GAIN`
/// - `SPIRAL_SOFTLOGIC_LOSS_GAIN`
/// - `SPIRAL_SOFTLOGIC_FLOOR`
/// - `SPIRAL_SOFTLOGIC_SCALE_GAIN`
/// - `SPIRAL_SOFTLOGIC_REGION_GAIN`
/// - `SPIRAL_SOFTLOGIC_REGION_FACTOR_GAIN`
/// - `SPIRAL_SOFTLOGIC_ENERGY_EQUALIZE_GAIN`
/// - `SPIRAL_SOFTLOGIC_MEAN_NORMALIZE_GAIN`
/// - `SPIRAL_SOFTLOGIC_ENERGY_EQUALIZE_AUTO`
/// - `SPIRAL_SOFTLOGIC_MEAN_NORMALIZE_AUTO`
#[derive(Debug, Clone, Copy)]
pub struct SoftLogicConfig {
    pub inertia: f32,
    pub inertia_min: f32,
    pub inertia_drift_k: f32,
    pub inertia_z_k: f32,
    pub drift_gain: f32,
    pub psi_gain: f32,
    pub loss_gain: f32,
    pub floor: f32,
    pub scale_gain: f32,
    pub region_gain: f32,
    pub region_factor_gain: f32,
    pub energy_equalize_gain: f32,
    pub mean_normalize_gain: f32,
    pub energy_equalize_auto: f32,
    pub mean_normalize_auto: f32,
}

impl Default for SoftLogicConfig {
    fn default() -> Self {
        Self {
            inertia: 0.65,
            inertia_min: 0.15,
            inertia_drift_k: 0.6,
            inertia_z_k: 0.2,
            drift_gain: 0.25,
            psi_gain: 0.5,
            loss_gain: 0.35,
            floor: 0.25,
            scale_gain: 0.2,
            region_gain: 0.15,
            region_factor_gain: 0.35,
            energy_equalize_gain: 0.0,
            mean_normalize_gain: 0.0,
            energy_equalize_auto: 0.0,
            mean_normalize_auto: 0.0,
        }
    }
}

impl SoftLogicConfig {
    pub fn clamp_inplace(&mut self) {
        if !self.inertia.is_finite() {
            self.inertia = 0.65;
        }
        if !self.inertia_min.is_finite() {
            self.inertia_min = 0.15;
        }
        if !self.inertia_drift_k.is_finite() {
            self.inertia_drift_k = 0.6;
        }
        if !self.inertia_z_k.is_finite() {
            self.inertia_z_k = 0.2;
        }
        if !self.drift_gain.is_finite() {
            self.drift_gain = 0.25;
        }
        if !self.psi_gain.is_finite() {
            self.psi_gain = 0.5;
        }
        if !self.loss_gain.is_finite() {
            self.loss_gain = 0.35;
        }
        if !self.floor.is_finite() {
            self.floor = 0.25;
        }
        if !self.scale_gain.is_finite() {
            self.scale_gain = 0.2;
        }
        if !self.region_gain.is_finite() {
            self.region_gain = 0.15;
        }
        if !self.region_factor_gain.is_finite() {
            self.region_factor_gain = 0.35;
        }
        if !self.energy_equalize_gain.is_finite() {
            self.energy_equalize_gain = 0.0;
        }
        if !self.mean_normalize_gain.is_finite() {
            self.mean_normalize_gain = 0.0;
        }
        if !self.energy_equalize_auto.is_finite() {
            self.energy_equalize_auto = 0.0;
        }
        if !self.mean_normalize_auto.is_finite() {
            self.mean_normalize_auto = 0.0;
        }

        self.inertia = self.inertia.clamp(0.0, 0.95);
        self.inertia_min = self.inertia_min.clamp(0.0, 0.95).min(self.inertia);
        self.inertia_drift_k = self.inertia_drift_k.clamp(0.0, 4.0);
        self.inertia_z_k = self.inertia_z_k.clamp(0.0, 2.0);
        self.drift_gain = self.drift_gain.clamp(0.0, 1.0);
        self.psi_gain = self.psi_gain.clamp(0.0, 2.0);
        self.loss_gain = self.loss_gain.clamp(0.0, 1.5);
        self.floor = self.floor.clamp(0.05, 1.0);
        self.scale_gain = self.scale_gain.clamp(0.0, 1.5);
        self.region_gain = self.region_gain.clamp(0.0, 1.5);
        self.region_factor_gain = self.region_factor_gain.clamp(0.0, 2.0);
        self.energy_equalize_gain = self.energy_equalize_gain.clamp(0.0, 1.0);
        self.mean_normalize_gain = self.mean_normalize_gain.clamp(0.0, 1.0);
        self.energy_equalize_auto = self.energy_equalize_auto.clamp(0.0, 1.0);
        self.mean_normalize_auto = self.mean_normalize_auto.clamp(0.0, 1.0);
    }

    /// Applies environment overrides to the configuration in-place.
    pub fn apply_env_overrides(&mut self) {
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_INERTIA") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.inertia = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_INERTIA_MIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.inertia_min = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_INERTIA_DRIFT_K") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.inertia_drift_k = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_INERTIA_Z_K") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.inertia_z_k = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_DRIFT_GAIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.drift_gain = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_PSI_GAIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.psi_gain = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_LOSS_GAIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.loss_gain = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_FLOOR") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.floor = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_SCALE_GAIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.scale_gain = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_REGION_GAIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.region_gain = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_REGION_FACTOR_GAIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.region_factor_gain = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_ENERGY_EQUALIZE_GAIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.energy_equalize_gain = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_MEAN_NORMALIZE_GAIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.mean_normalize_gain = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_ENERGY_EQUALIZE_AUTO") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.energy_equalize_auto = parsed;
            }
        }
        if let Ok(value) = env::var("SPIRAL_SOFTLOGIC_MEAN_NORMALIZE_AUTO") {
            if let Ok(parsed) = value.parse::<f32>() {
                self.mean_normalize_auto = parsed;
            }
        }

        self.clamp_inplace();
    }

    /// Applies environment overrides and returns `self` for fluent construction.
    pub fn with_env_overrides(mut self) -> Self {
        self.apply_env_overrides();
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SoftLogicAdaptiveMetrics {
    energy_equalize_gain_eff: f32,
    energy_equalize_strength: f32,
    energy_equalize_over: f32,
    energy_equalize_state: f32,
    mean_normalize_gain_eff: f32,
    mean_normalize_need: f32,
    mean_normalize_state: f32,
    mean_target: f32,
}

#[derive(Debug, Clone)]
struct SoftLogicFlex {
    config: SoftLogicConfig,
    last_weights: (f32, f32, f32),
    last_z: f32,
    last_feedback: Option<SoftlogicZFeedback>,
    last_inertia: f32,
    last_region: Option<ZSpaceRegionDescriptor>,
    last_region_factor: f32,
    last_region_scale: (f32, f32, f32),
    equalize_state: f32,
    mean_normalize_state: f32,
    adaptive_metrics: SoftLogicAdaptiveMetrics,
    pending_events: Vec<String>,
    equalize_guard_on: bool,
    equalize_clamp_on: bool,
    normalize_on: bool,
}

impl SoftLogicFlex {
    fn new() -> Self {
        Self::from_config(SoftLogicConfig::default().with_env_overrides())
    }

    fn from_config(mut config: SoftLogicConfig) -> Self {
        config.clamp_inplace();
        let mut flex = Self {
            config,
            last_weights: (1.0, 1.0, 1.0),
            last_z: 0.0,
            last_feedback: None,
            last_inertia: 0.0,
            last_region: None,
            last_region_factor: 1.0,
            last_region_scale: (1.0, 1.0, 1.0),
            equalize_state: 1.0,
            mean_normalize_state: 1.0,
            adaptive_metrics: SoftLogicAdaptiveMetrics::default(),
            pending_events: Vec::new(),
            equalize_guard_on: false,
            equalize_clamp_on: false,
            normalize_on: false,
        };
        flex.last_inertia = flex.config.inertia;
        flex
    }

    fn config(&self) -> SoftLogicConfig {
        self.config
    }

    fn set_config(&mut self, mut config: SoftLogicConfig) {
        config.clamp_inplace();
        self.config = config;
        self.last_inertia = self.config.inertia;
        self.equalize_state = 1.0;
        self.mean_normalize_state = 1.0;
        self.adaptive_metrics = SoftLogicAdaptiveMetrics::default();
        self.pending_events.clear();
        self.equalize_guard_on = false;
        self.equalize_clamp_on = false;
        self.normalize_on = false;
    }

    fn adaptive_metrics(&self) -> SoftLogicAdaptiveMetrics {
        self.adaptive_metrics
    }

    fn record_region_feedback(&mut self, descriptor: ZSpaceRegionDescriptor, factor: f32) {
        self.last_region = Some(descriptor);
        self.last_region_factor = if factor.is_finite() {
            factor.max(0.0)
        } else {
            1.0
        };
    }

    fn clear_region_feedback(&mut self) {
        self.last_region = None;
        self.last_region_factor = 1.0;
        self.last_region_scale = (1.0, 1.0, 1.0);
    }

    fn last_inertia(&self) -> f32 {
        self.last_inertia
    }

    fn last_region_scale(&self) -> (f32, f32, f32) {
        self.last_region_scale
    }

    fn effective_inertia(&mut self, band_energy: &BandEnergy) -> f32 {
        let drift_drive = (band_energy.drift.abs() * self.config.inertia_drift_k).tanh();
        let z_drive = self
            .last_feedback
            .as_ref()
            .map(|feedback| feedback.z_signal.abs())
            .unwrap_or(self.last_z.abs())
            * self.config.inertia_z_k;
        let spectral_curvature = band_energy.spectral_curvature();
        let spectral_diffusion = (1.0 - band_energy.spectral.spin.abs()).clamp(0.0, 1.0);
        let spectral_drive =
            (0.18 * spectral_curvature + 0.12 * spectral_diffusion).clamp(0.0, 0.3);
        let mut adapt = (drift_drive + z_drive + spectral_drive).clamp(0.0, 0.9);
        if self.config.energy_equalize_auto > 0.0 {
            let above_abs = band_energy.above.abs();
            let here_abs = band_energy.here.abs();
            let beneath_abs = band_energy.beneath.abs();
            let norm = (above_abs + here_abs + beneath_abs).max(1e-4);
            let max_share = above_abs.max(here_abs).max(beneath_abs) / norm;
            let energy_drive = ((max_share - (1.0 / 3.0)) / (2.0 / 3.0)).clamp(0.0, 1.0);
            adapt =
                (adapt + energy_drive * 0.35 * self.config.energy_equalize_auto).clamp(0.0, 0.9);
        }
        let inertia = (self.config.inertia * (1.0 - adapt)).clamp(self.config.inertia_min, 0.95);
        self.last_inertia = inertia;
        inertia
    }

    fn region_steering(&mut self, z_bias: f32) -> (f32, f32, f32) {
        let Some(descriptor) = self.last_region else {
            self.last_region_scale = (1.0, 1.0, 1.0);
            return self.last_region_scale;
        };
        if self.config.region_gain <= 0.0 {
            self.last_region_scale = (1.0, 1.0, 1.0);
            return self.last_region_scale;
        }
        let factor_boost = if self.last_region_factor.is_finite() && self.last_region_factor > 1.0 {
            1.0 + self.config.region_factor_gain * (self.last_region_factor - 1.0).clamp(0.0, 3.0)
        } else {
            1.0
        };
        let gain = (self.config.region_gain * factor_boost).clamp(0.0, 0.9);
        let radius_weight = match descriptor.key().radius {
            ZSpaceRadiusBand::Core => 0.25,
            ZSpaceRadiusBand::Mantle => 0.6,
            ZSpaceRadiusBand::Edge => 1.0,
        };
        let intensity = (gain * radius_weight).clamp(0.0, 0.9);
        let preferred_band = match descriptor.key().spin {
            ZSpaceSpinBand::Leading => 0,
            ZSpaceSpinBand::Trailing => 2,
            ZSpaceSpinBand::Neutral => {
                if matches!(descriptor.key().radius, ZSpaceRadiusBand::Edge) {
                    if z_bias >= 0.0 {
                        0
                    } else {
                        2
                    }
                } else {
                    1
                }
            }
        };
        let other = (1.0 - 0.5 * intensity).max(0.1);
        self.last_region_scale = match preferred_band {
            0 => (1.0 + intensity, other, other),
            1 => (other, 1.0 + intensity, other),
            _ => (other, other, 1.0 + intensity),
        };
        self.last_region_scale
    }

    fn prepare_weights(&mut self, band_energy: &BandEnergy) -> (f32, f32, f32) {
        let inertia = self.effective_inertia(band_energy);
        self.adaptive_metrics = SoftLogicAdaptiveMetrics::default();
        let above_abs = band_energy.above.abs();
        let here_abs = band_energy.here.abs();
        let beneath_abs = band_energy.beneath.abs();
        let norm = (above_abs + here_abs + beneath_abs).max(1e-4);
        let asymmetry = (band_energy.above - band_energy.beneath) / norm;
        let drift_term = band_energy.drift.tanh();
        let spectral_focus = band_energy.spectral_focus();
        let spectral_curvature = band_energy.spectral_curvature();
        let spectral_spin = band_energy.spectral.spin.clamp(-1.0, 1.0);
        let spectral_stability = band_energy.spectral_stability();
        let z_bias = self
            .last_feedback
            .as_ref()
            .map(|feedback| feedback.z_signal)
            .unwrap_or(self.last_z);
        let mut target_above =
            1.0 + (asymmetry * self.config.drift_gain) + (z_bias * self.config.psi_gain);
        let mut target_here = 1.0
            + ((band_energy.here / norm) - (band_energy.drift.abs() / norm))
                * self.config.loss_gain;
        let mut target_beneath =
            1.0 - (asymmetry * self.config.drift_gain) - (z_bias * self.config.psi_gain)
                + (-drift_term * self.config.drift_gain * 0.5);

        if spectral_focus > 0.0 {
            let edge_gain = 0.25 * self.config.drift_gain * spectral_focus;
            target_above *= 1.0 + edge_gain * spectral_spin.max(0.0);
            target_beneath *= 1.0 + edge_gain * (-spectral_spin).max(0.0);
            target_here *= 1.0 + 0.2 * self.config.loss_gain * spectral_stability;
            target_here *= 1.0 - 0.08 * spectral_focus * spectral_curvature;
        }

        if self.config.scale_gain > 0.0 {
            if let Some(scale) = self
                .last_feedback
                .as_ref()
                .and_then(|feedback| feedback.scale)
            {
                let bias = (-scale.log_radius).tanh();
                let explore = bias.max(0.0);
                let settle = (-bias).max(0.0);
                target_above *= 1.0 + self.config.scale_gain * explore;
                target_here *= 1.0 + self.config.scale_gain * 0.5 * (explore - settle);
                target_beneath *= 1.0 + self.config.scale_gain * settle;
            }
        }

        let region_scale = self.region_steering(z_bias);
        target_above *= region_scale.0;
        target_here *= region_scale.1;
        target_beneath *= region_scale.2;

        if self.config.energy_equalize_gain > 0.0 {
            let target_share = 1.0 / 3.0;
            let eps = 1e-4;
            let pa = (above_abs / norm).max(eps);
            let ph = (here_abs / norm).max(eps);
            let pb = (beneath_abs / norm).max(eps);
            let ratio_max = 9.0;
            let ra_raw = target_share / pa;
            let rh_raw = target_share / ph;
            let rb_raw = target_share / pb;
            let ra = ra_raw.clamp(1.0 / ratio_max, ratio_max);
            let rh = rh_raw.clamp(1.0 / ratio_max, ratio_max);
            let rb = rb_raw.clamp(1.0 / ratio_max, ratio_max);
            let raw_max = ra_raw.max(rh_raw).max(rb_raw);
            let ln_ratio_max = ratio_max.ln().max(1e-4);
            let strength = if raw_max.is_finite() && raw_max > 0.0 {
                (raw_max.ln() / ln_ratio_max).clamp(0.0, 2.0)
            } else if raw_max.is_sign_positive() && raw_max.is_infinite() {
                2.0
            } else {
                0.0
            };
            let strength_clamped = strength.clamp(0.0, 1.0);
            let over = ((strength_clamped - 0.85) / 0.15).clamp(0.0, 1.0);
            let desired_guard = (1.0 - 0.75 * over).clamp(0.05, 1.0);

            let gain_base = self.config.energy_equalize_gain.clamp(0.0, 1.0);
            let auto = self.config.energy_equalize_auto.clamp(0.0, 1.0);
            let gain = if auto > 0.0 {
                let update = (auto * (1.0 - inertia)).clamp(0.0, 1.0);
                self.equalize_state =
                    Self::lerp(self.equalize_state, desired_guard, update).clamp(0.0, 1.0);
                let guard_on = desired_guard < 0.999;
                let clamp_on = ra_raw > ratio_max || rh_raw > ratio_max || rb_raw > ratio_max;
                if guard_on != self.equalize_guard_on {
                    self.pending_events.push(if guard_on {
                        "equalize.guard.on".to_string()
                    } else {
                        "equalize.guard.off".to_string()
                    });
                    self.equalize_guard_on = guard_on;
                }
                if clamp_on != self.equalize_clamp_on {
                    self.pending_events.push(if clamp_on {
                        "equalize.clamp.on".to_string()
                    } else {
                        "equalize.clamp.off".to_string()
                    });
                    self.equalize_clamp_on = clamp_on;
                }
                gain_base * self.equalize_state
            } else {
                self.equalize_state = 1.0;
                self.equalize_guard_on = false;
                self.equalize_clamp_on = false;
                gain_base
            };

            self.adaptive_metrics.energy_equalize_gain_eff = gain;
            self.adaptive_metrics.energy_equalize_strength = strength_clamped;
            self.adaptive_metrics.energy_equalize_over = over;
            self.adaptive_metrics.energy_equalize_state = self.equalize_state;

            target_above *= ra.powf(gain);
            target_here *= rh.powf(gain);
            target_beneath *= rb.powf(gain);
        }

        let mut target = (
            target_above.clamp(self.config.floor, 3.0),
            target_here.clamp(self.config.floor, 2.5),
            target_beneath.clamp(self.config.floor, 3.0),
        );

        if self.config.mean_normalize_gain > 0.0 {
            let mean = (target.0 + target.1 + target.2) / 3.0;
            if mean.is_finite() && mean > 0.0 {
                self.adaptive_metrics.mean_target = mean;
                let gain_base = self.config.mean_normalize_gain.clamp(0.0, 1.0);
                let auto = self.config.mean_normalize_auto.clamp(0.0, 1.0);
                let mean_dev = mean.ln().abs();
                let need = (mean_dev / 0.4).clamp(0.0, 1.0);
                let gain = if auto > 0.0 {
                    let update = (auto * (1.0 - inertia)).clamp(0.0, 1.0);
                    self.mean_normalize_state =
                        Self::lerp(self.mean_normalize_state, need, update).clamp(0.0, 1.0);
                    let normalize_on = need > 0.25;
                    if normalize_on != self.normalize_on {
                        self.pending_events.push(if normalize_on {
                            "normalize.on".to_string()
                        } else {
                            "normalize.off".to_string()
                        });
                        self.normalize_on = normalize_on;
                    }
                    gain_base * self.mean_normalize_state
                } else {
                    self.mean_normalize_state = 1.0;
                    self.normalize_on = false;
                    gain_base
                };
                let norm_factor = 1.0 / mean;
                let blend = (1.0 - gain) + gain * norm_factor;
                target.0 = (target.0 * blend).clamp(self.config.floor, 3.0);
                target.1 = (target.1 * blend).clamp(self.config.floor, 2.5);
                target.2 = (target.2 * blend).clamp(self.config.floor, 3.0);

                self.adaptive_metrics.mean_normalize_gain_eff = gain;
                self.adaptive_metrics.mean_normalize_need = need;
                self.adaptive_metrics.mean_normalize_state = self.mean_normalize_state;
            }
        }
        self.last_weights = (
            Self::lerp(self.last_weights.0, target.0, 1.0 - inertia),
            Self::lerp(self.last_weights.1, target.1, 1.0 - inertia),
            Self::lerp(self.last_weights.2, target.2, 1.0 - inertia),
        );
        self.last_weights
    }

    fn observe(
        &mut self,
        band_energy: &BandEnergy,
        weighted_loss: f32,
        psi_total: Option<f32>,
        scale_hint: Option<ZScale>,
    ) -> SoftlogicZFeedback {
        let inertia = self.effective_inertia(band_energy);
        let psi_total = psi_total.unwrap_or(0.0);
        let total = (band_energy.above + band_energy.here + band_energy.beneath).max(1e-4);
        let asym = (band_energy.above - band_energy.beneath) / total;
        let drift = band_energy.drift;
        let raw_signal = 0.6 * (psi_total - weighted_loss) + 0.3 * asym + 0.1 * drift;
        let z_signal = raw_signal.tanh();
        self.last_z = Self::lerp(self.last_z, z_signal, 1.0 - inertia);
        let feedback = SoftlogicZFeedback {
            psi_total,
            weighted_loss,
            band_energy: (band_energy.above, band_energy.here, band_energy.beneath),
            drift,
            z_signal: self.last_z,
            scale: scale_hint,
            events: std::mem::take(&mut self.pending_events),
            attributions: Vec::new(),
            elliptic: None,
        };
        let mut feedback = feedback;
        if feedback.elliptic.is_none() {
            if let Some(last) = self
                .last_feedback
                .as_ref()
                .and_then(|sample| sample.elliptic)
            {
                feedback.elliptic = Some(last);
            }
        }
        self.last_feedback = Some(feedback.clone());
        feedback
    }

    fn lerp(current: f32, target: f32, factor: f32) -> f32 {
        current + (target - current) * factor
    }
}

#[derive(Debug, Clone)]
pub struct SpectralLearningRatePolicy {
    smoothing: f32,
    event_smoothing: f32,
    turnover_smoothing: f32,
    coherence_gain: f32,
    curvature_gain: f32,
    sheet_gain: f32,
    spin_gain: f32,
    radius_gain: f32,
    energy_gain: f32,
    phase_gain: f32,
    stuck_phase_gain: f32,
    max_phase_gain: f32,
    stuck_turnover_threshold: f32,
    dominant_turnover: f32,
    last_dominant: Option<usize>,
    last_label: Option<CoherenceLabel>,
    min_lr_scale: f32,
    max_lr_scale: f32,
    max_lr_step: f32,
    min_band_scale: f32,
    max_band_scale: f32,
    band_state: (f32, f32, f32),
    lr_state: f32,
    applied_lr_scale: f32,
    local_lr_state: (f32, f32, f32),
}

impl SpectralLearningRatePolicy {
    pub fn new() -> Self {
        Self {
            smoothing: 0.2,
            event_smoothing: 0.75,
            turnover_smoothing: 0.15,
            coherence_gain: 0.5,
            curvature_gain: 0.15,
            sheet_gain: 0.6,
            spin_gain: 0.4,
            radius_gain: 0.35,
            energy_gain: 0.25,
            phase_gain: 0.25,
            stuck_phase_gain: 0.35,
            max_phase_gain: 0.65,
            stuck_turnover_threshold: 0.12,
            dominant_turnover: 0.0,
            last_dominant: None,
            last_label: None,
            min_lr_scale: 0.1,
            max_lr_scale: 6.0,
            max_lr_step: 1.4,
            min_band_scale: 0.5,
            max_band_scale: 2.75,
            band_state: (1.0, 1.0, 1.0),
            lr_state: 1.0,
            applied_lr_scale: 1.0,
            local_lr_state: (1.0, 1.0, 1.0),
        }
    }

    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        if smoothing.is_finite() && smoothing > 0.0 {
            self.smoothing = smoothing.clamp(1.0e-3, 1.0);
        }
        self
    }

    /// Overrides the smoothing factor used when a coherence label transition is detected.
    pub fn with_event_smoothing(mut self, smoothing: f32) -> Self {
        if smoothing.is_finite() && smoothing > 0.0 {
            self.event_smoothing = smoothing.clamp(1.0e-3, 1.0);
        }
        self
    }

    /// Overrides the smoothing factor used to track dominant channel turnover.
    pub fn with_turnover_smoothing(mut self, smoothing: f32) -> Self {
        if smoothing.is_finite() && smoothing > 0.0 {
            self.turnover_smoothing = smoothing.clamp(1.0e-3, 1.0);
        }
        self
    }

    /// Adjusts the base phase gain used to bias band scales towards a dominant "seat".
    pub fn with_phase_gain(mut self, gain: f32) -> Self {
        if gain.is_finite() && gain >= 0.0 {
            self.phase_gain = gain;
        }
        self
    }

    /// Adjusts the extra phase gain applied when dominant channel turnover drops below threshold.
    pub fn with_stuck_phase_gain(mut self, gain: f32) -> Self {
        if gain.is_finite() && gain >= 0.0 {
            self.stuck_phase_gain = gain;
        }
        self
    }

    /// Adjusts the turnover threshold used to detect "stuck" coherence.
    pub fn with_stuck_turnover_threshold(mut self, threshold: f32) -> Self {
        if threshold.is_finite() && threshold >= 0.0 {
            self.stuck_turnover_threshold = threshold;
        }
        self
    }

    pub fn with_coherence_gain(mut self, gain: f32) -> Self {
        if gain.is_finite() {
            self.coherence_gain = gain.max(0.0);
        }
        self
    }

    pub fn with_sheet_gain(mut self, gain: f32) -> Self {
        if gain.is_finite() {
            self.sheet_gain = gain.max(0.0);
        }
        self
    }

    pub fn with_spin_gain(mut self, gain: f32) -> Self {
        if gain.is_finite() {
            self.spin_gain = gain.max(0.0);
        }
        self
    }

    pub fn with_radius_gain(mut self, gain: f32) -> Self {
        if gain.is_finite() {
            self.radius_gain = gain.max(0.0);
        }
        self
    }

    pub fn with_energy_gain(mut self, gain: f32) -> Self {
        if gain.is_finite() {
            self.energy_gain = gain.max(0.0);
        }
        self
    }

    pub fn with_lr_bounds(mut self, min: f32, max: f32) -> Self {
        if min.is_finite() && max.is_finite() && min > 0.0 && max > min {
            self.min_lr_scale = min;
            self.max_lr_scale = max;
        }
        self
    }

    pub fn with_band_bounds(mut self, min: f32, max: f32) -> Self {
        if min.is_finite() && max.is_finite() && min > 0.0 && max > min {
            self.min_band_scale = min;
            self.max_band_scale = max;
        }
        self
    }

    pub fn with_max_lr_step(mut self, max_step: f32) -> Self {
        if max_step.is_finite() && max_step > 0.0 {
            self.max_lr_step = max_step.max(1.0);
        }
        self
    }

    /// Applies environment overrides to the policy in-place.
    ///
    /// Supported variables (all optional):
    /// - `SPIRAL_SPECTRAL_POLICY_SMOOTHING`
    /// - `SPIRAL_SPECTRAL_POLICY_EVENT_SMOOTHING`
    /// - `SPIRAL_SPECTRAL_POLICY_TURNOVER_SMOOTHING`
    /// - `SPIRAL_SPECTRAL_POLICY_PHASE_GAIN`
    /// - `SPIRAL_SPECTRAL_POLICY_STUCK_PHASE_GAIN`
    /// - `SPIRAL_SPECTRAL_POLICY_MAX_PHASE_GAIN`
    /// - `SPIRAL_SPECTRAL_POLICY_STUCK_TURNOVER_THRESHOLD`
    /// - `SPIRAL_SPECTRAL_POLICY_LR_MIN` / `SPIRAL_SPECTRAL_POLICY_LR_MAX`
    /// - `SPIRAL_SPECTRAL_POLICY_BAND_MIN` / `SPIRAL_SPECTRAL_POLICY_BAND_MAX`
    pub fn apply_env_overrides(&mut self) {
        if let Ok(value) = env::var("SPIRAL_SPECTRAL_POLICY_SMOOTHING") {
            if let Ok(smoothing) = value.parse::<f32>() {
                if smoothing.is_finite() && smoothing > 0.0 {
                    self.smoothing = smoothing.clamp(1.0e-3, 1.0);
                }
            }
        }
        if let Ok(value) = env::var("SPIRAL_SPECTRAL_POLICY_EVENT_SMOOTHING") {
            if let Ok(smoothing) = value.parse::<f32>() {
                if smoothing.is_finite() && smoothing > 0.0 {
                    self.event_smoothing = smoothing.clamp(1.0e-3, 1.0);
                }
            }
        }
        if let Ok(value) = env::var("SPIRAL_SPECTRAL_POLICY_TURNOVER_SMOOTHING") {
            if let Ok(smoothing) = value.parse::<f32>() {
                if smoothing.is_finite() && smoothing > 0.0 {
                    self.turnover_smoothing = smoothing.clamp(1.0e-3, 1.0);
                }
            }
        }
        if let Ok(value) = env::var("SPIRAL_SPECTRAL_POLICY_PHASE_GAIN") {
            if let Ok(gain) = value.parse::<f32>() {
                if gain.is_finite() && gain >= 0.0 {
                    self.phase_gain = gain;
                }
            }
        }
        if let Ok(value) = env::var("SPIRAL_SPECTRAL_POLICY_STUCK_PHASE_GAIN") {
            if let Ok(gain) = value.parse::<f32>() {
                if gain.is_finite() && gain >= 0.0 {
                    self.stuck_phase_gain = gain;
                }
            }
        }
        if let Ok(value) = env::var("SPIRAL_SPECTRAL_POLICY_MAX_PHASE_GAIN") {
            if let Ok(gain) = value.parse::<f32>() {
                if gain.is_finite() && gain >= 0.0 {
                    self.max_phase_gain = gain;
                }
            }
        }
        if let Ok(value) = env::var("SPIRAL_SPECTRAL_POLICY_STUCK_TURNOVER_THRESHOLD") {
            if let Ok(threshold) = value.parse::<f32>() {
                if threshold.is_finite() && threshold >= 0.0 {
                    self.stuck_turnover_threshold = threshold.clamp(0.0, 1.0);
                }
            }
        }
        let lr_min = env::var("SPIRAL_SPECTRAL_POLICY_LR_MIN")
            .ok()
            .and_then(|value| value.parse::<f32>().ok());
        let lr_max = env::var("SPIRAL_SPECTRAL_POLICY_LR_MAX")
            .ok()
            .and_then(|value| value.parse::<f32>().ok());
        if let (Some(min), Some(max)) = (lr_min, lr_max) {
            if min.is_finite() && max.is_finite() && min > 0.0 && max > min {
                self.min_lr_scale = min;
                self.max_lr_scale = max;
            }
        }
        let band_min = env::var("SPIRAL_SPECTRAL_POLICY_BAND_MIN")
            .ok()
            .and_then(|value| value.parse::<f32>().ok());
        let band_max = env::var("SPIRAL_SPECTRAL_POLICY_BAND_MAX")
            .ok()
            .and_then(|value| value.parse::<f32>().ok());
        if let (Some(min), Some(max)) = (band_min, band_max) {
            if min.is_finite() && max.is_finite() && min > 0.0 && max > min {
                self.min_band_scale = min;
                self.max_band_scale = max;
            }
        }
        self.max_lr_step = self.max_lr_step.max(1.0);
        self.max_phase_gain = self.max_phase_gain.max(0.0);
        self.phase_gain = self.phase_gain.max(0.0);
        self.stuck_phase_gain = self.stuck_phase_gain.max(0.0);
    }

    /// Applies environment overrides and returns `self` for fluent construction.
    pub fn with_env_overrides(mut self) -> Self {
        self.apply_env_overrides();
        self
    }

    /// Returns the last semantic coherence label that influenced the policy.
    pub fn last_coherence_label(&self) -> Option<CoherenceLabel> {
        self.last_label
    }

    /// Returns the EMA of dominant channel changes (0 = locked, 1 = swapping every step).
    pub fn dominant_turnover(&self) -> f32 {
        self.dominant_turnover
    }

    pub fn observe(
        &mut self,
        diagnostics: Option<&CoherenceDiagnostics>,
        curvature: f32,
        band_energy: &BandEnergy,
    ) -> Option<SpectralAdjustment> {
        let signal = diagnostics.map(CoherenceSignal::from);
        self.observe_signal(signal.as_ref(), curvature, band_energy)
    }

    pub fn observe_signal(
        &mut self,
        diagnostics: Option<&CoherenceSignal>,
        curvature: f32,
        band_energy: &BandEnergy,
    ) -> Option<SpectralAdjustment> {
        let diagnostics = diagnostics?;
        let label = diagnostics.label();
        let preserved = diagnostics.preserved_channels().max(1);
        let dominant = diagnostics
            .dominant_channel()
            .unwrap_or(0)
            .min(preserved - 1);
        let dominant_changed = self
            .last_dominant
            .map(|previous| previous != dominant)
            .unwrap_or(false);
        self.dominant_turnover = Self::smooth(
            self.dominant_turnover,
            if dominant_changed { 1.0 } else { 0.0 },
            self.turnover_smoothing,
        )
        .clamp(0.0, 1.0);
        self.last_dominant = Some(dominant);
        let label_changed = self
            .last_label
            .map(|previous| previous != label)
            .unwrap_or(false);
        self.last_label = Some(label);
        let smoothing = if label_changed {
            self.event_smoothing.max(self.smoothing)
        } else {
            self.smoothing
        };
        let sheet_ratio = (dominant as f32 + 0.5) / preserved as f32;
        let sheet_bias = (sheet_ratio - 0.5) * self.sheet_gain;
        let spin = diagnostics.z_bias().clamp(-1.0, 1.0);
        let spin_bias = spin * self.spin_gain;
        let spectral_radius = diagnostics.mean_coherence().abs();
        let entropy = diagnostics.coherence_entropy().max(0.0);
        let energy_ratio = diagnostics.energy_ratio().max(0.0);
        let spectral_pressure = energy_ratio * (1.0 - entropy.tanh());
        let curvature_term = if curvature < 0.0 {
            (-curvature).sqrt() * self.curvature_gain
        } else {
            0.0
        };
        let radius_bias = (1.0 - spectral_radius.tanh()).clamp(0.0, 1.0) * self.radius_gain;

        let base_above = 1.0
            + sheet_bias.max(0.0)
            + spin_bias.max(0.0)
            + curvature_term * 0.25
            + radius_bias * 0.5;
        let base_here = 1.0
            + (1.0 - sheet_bias.abs()).max(0.0) * 0.5 * self.sheet_gain
            + curvature_term * 0.5
            + (1.0 - radius_bias) * 0.25;
        let base_beneath = 1.0
            + (-sheet_bias).max(0.0)
            + (-spin_bias).max(0.0)
            + curvature_term * 0.25
            + radius_bias * 0.5;

        let energy_total =
            (band_energy.above.abs() + band_energy.here.abs() + band_energy.beneath.abs())
                .max(1e-5);
        let energy_bias = (
            (band_energy.above / energy_total).clamp(-1.0, 1.0),
            (band_energy.here / energy_total).clamp(-1.0, 1.0),
            (band_energy.beneath / energy_total).clamp(-1.0, 1.0),
        );

        let label_lr_scale = match label {
            CoherenceLabel::Background => {
                if self.dominant_turnover < self.stuck_turnover_threshold {
                    1.05
                } else {
                    1.0
                }
            }
            CoherenceLabel::SymmetricPulse => 0.9,
            CoherenceLabel::CascadeImbalance => 0.7,
            CoherenceLabel::DiffuseDrift => 1.15,
        };
        let label_band_bias = match label {
            CoherenceLabel::CascadeImbalance => (0.92, 1.15, 0.92),
            CoherenceLabel::SymmetricPulse => (0.96, 1.06, 0.96),
            _ => (1.0, 1.0, 1.0),
        };
        let mut phase_gain = match label {
            CoherenceLabel::DiffuseDrift => self.phase_gain * 1.15,
            CoherenceLabel::CascadeImbalance => 0.0,
            CoherenceLabel::SymmetricPulse => self.phase_gain * 0.35,
            CoherenceLabel::Background => self.phase_gain,
        };
        if self.dominant_turnover < self.stuck_turnover_threshold
            && !matches!(label, CoherenceLabel::CascadeImbalance)
        {
            phase_gain += self.stuck_phase_gain;
        }
        phase_gain = phase_gain.clamp(0.0, self.max_phase_gain);
        let phase = dominant % 3;
        let phase_mul = match phase {
            0 => (
                1.0 + phase_gain,
                1.0 - 0.5 * phase_gain,
                1.0 - 0.5 * phase_gain,
            ),
            1 => (
                1.0 - 0.5 * phase_gain,
                1.0 + phase_gain,
                1.0 - 0.5 * phase_gain,
            ),
            _ => (
                1.0 - 0.5 * phase_gain,
                1.0 - 0.5 * phase_gain,
                1.0 + phase_gain,
            ),
        };

        let mut target_band = (
            (base_above * (1.0 + energy_bias.0 * self.energy_gain))
                .clamp(self.min_band_scale, self.max_band_scale),
            (base_here * (1.0 + energy_bias.1 * self.energy_gain))
                .clamp(self.min_band_scale, self.max_band_scale),
            (base_beneath * (1.0 + energy_bias.2 * self.energy_gain))
                .clamp(self.min_band_scale, self.max_band_scale),
        );
        target_band.0 = (target_band.0 * phase_mul.0 * label_band_bias.0)
            .clamp(self.min_band_scale, self.max_band_scale);
        target_band.1 = (target_band.1 * phase_mul.1 * label_band_bias.1)
            .clamp(self.min_band_scale, self.max_band_scale);
        target_band.2 = (target_band.2 * phase_mul.2 * label_band_bias.2)
            .clamp(self.min_band_scale, self.max_band_scale);

        self.band_state.0 = Self::smooth(self.band_state.0, target_band.0, smoothing);
        self.band_state.1 = Self::smooth(self.band_state.1, target_band.1, smoothing);
        self.band_state.2 = Self::smooth(self.band_state.2, target_band.2, smoothing);
        target_band = self.band_state;

        let mean_band = (target_band.0 + target_band.1 + target_band.2) / 3.0;
        let coherence_boost = 1.0 + spectral_pressure * self.coherence_gain;
        let mut target_lr = (mean_band * coherence_boost * label_lr_scale)
            .clamp(self.min_lr_scale, self.max_lr_scale);
        self.lr_state = Self::smooth(self.lr_state, target_lr, smoothing);
        target_lr = self.lr_state.clamp(self.min_lr_scale, self.max_lr_scale);

        let mut target_local = (
            (target_band.0 / mean_band).clamp(self.min_band_scale, self.max_band_scale),
            (target_band.1 / mean_band).clamp(self.min_band_scale, self.max_band_scale),
            (target_band.2 / mean_band).clamp(self.min_band_scale, self.max_band_scale),
        );
        self.local_lr_state.0 = Self::smooth(self.local_lr_state.0, target_local.0, smoothing);
        self.local_lr_state.1 = Self::smooth(self.local_lr_state.1, target_local.1, smoothing);
        self.local_lr_state.2 = Self::smooth(self.local_lr_state.2, target_local.2, smoothing);
        target_local = self.local_lr_state;

        let mut lr_multiplier = 1.0;
        if self.applied_lr_scale.is_finite() && self.applied_lr_scale > 0.0 {
            let max_lr_step = if label_changed {
                (self.max_lr_step * 1.25).max(self.max_lr_step)
            } else {
                self.max_lr_step
            };
            lr_multiplier =
                (target_lr / self.applied_lr_scale).clamp(1.0 / max_lr_step, max_lr_step);
        }
        self.applied_lr_scale =
            (self.applied_lr_scale * lr_multiplier).clamp(self.min_lr_scale, self.max_lr_scale);

        Some(SpectralAdjustment {
            band_scale: target_band,
            lr_multiplier,
            lr_scale: self.applied_lr_scale,
            local_lr: target_local,
            metrics: SpectralAdjustmentMetrics {
                absolute_lr_scale: self.applied_lr_scale,
                sheet_index: diagnostics.dominant_channel().map(|idx| idx as u32),
                sheet_count: preserved as u32,
                spin_alignment: spin,
                spectral_radius,
                spectral_entropy: entropy,
                spectral_pressure,
                energy_ratio,
                band_scale: target_band,
                local_lr: target_local,
            },
        })
    }

    fn smooth(previous: f32, target: f32, alpha: f32) -> f32 {
        if !previous.is_finite() {
            return target;
        }
        previous + (target - previous) * alpha
    }
}

impl Default for SpectralLearningRatePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SpectralAdjustment {
    pub band_scale: (f32, f32, f32),
    pub lr_multiplier: f32,
    pub lr_scale: f32,
    pub local_lr: (f32, f32, f32),
    pub metrics: SpectralAdjustmentMetrics,
}

#[derive(Debug, Clone)]
pub struct SpectralAdjustmentMetrics {
    pub absolute_lr_scale: f32,
    pub sheet_index: Option<u32>,
    pub sheet_count: u32,
    pub spin_alignment: f32,
    pub spectral_radius: f32,
    pub spectral_entropy: f32,
    pub spectral_pressure: f32,
    pub energy_ratio: f32,
    pub band_scale: (f32, f32, f32),
    pub local_lr: (f32, f32, f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainerSpectralMetricsSource {
    BandEnergy,
    CoherenceAdjustment,
    Combined,
}

impl TrainerSpectralMetricsSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BandEnergy => "band_energy",
            Self::CoherenceAdjustment => "coherence_adjustment",
            Self::Combined => "combined",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainerSpectralMetrics {
    pub source: TrainerSpectralMetricsSource,
    pub turnover: Option<f32>,
    pub label: Option<CoherenceLabel>,
    pub adjustment: Option<SpectralAdjustmentMetrics>,
    pub band_energy: Option<BandEnergy>,
}

#[derive(Debug, Clone)]
struct RewriteBudget {
    per_epoch: u32,
    cooldown: u32,
    used_this_epoch: u32,
    cooldown_left: u32,
}

impl RewriteBudget {
    fn new(per_epoch: u32, cooldown: u32) -> Self {
        Self {
            per_epoch: per_epoch.max(1),
            cooldown,
            used_this_epoch: 0,
            cooldown_left: 0,
        }
    }

    fn begin_epoch(&mut self) {
        if self.cooldown_left > 0 {
            self.cooldown_left -= 1;
        }
        self.used_this_epoch = 0;
    }

    fn try_consume(&mut self, amount: u32) -> bool {
        if amount == 0 {
            return true;
        }
        if self.cooldown_left > 0 {
            return false;
        }
        if self.used_this_epoch.saturating_add(amount) > self.per_epoch {
            self.used_this_epoch = self.per_epoch;
            if self.cooldown > 0 {
                self.cooldown_left = self.cooldown;
            }
            return false;
        }
        self.used_this_epoch += amount;
        if self.used_this_epoch >= self.per_epoch && self.cooldown > 0 {
            self.cooldown_left = self.cooldown;
        }
        true
    }
}

impl ModuleTrainer {
    /// Creates a new trainer with the provided device capabilities and learning rates.
    pub fn new(
        caps: DeviceCaps,
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
    ) -> Self {
        #[cfg(feature = "psi")]
        let psi = Self::init_psi_meter();

        let mut spectral_adapter = SpectralLrAdapter::default().with_sheet_hint(8);
        spectral_adapter.set_curvature_target(curvature);

        let phase_turnover_spike_threshold =
            env::var("SPIRAL_TRAINER_PHASE_TURNOVER_SPIKE_THRESHOLD")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
                .unwrap_or(0.35);
        let phase_loss_ema_alpha = env::var("SPIRAL_TRAINER_PHASE_LOSS_EMA_ALPHA")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0 && *value <= 1.0)
            .unwrap_or(0.12);
        let phase_loss_spike_ratio = env::var("SPIRAL_TRAINER_PHASE_LOSS_SPIKE_RATIO")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(|value| value.clamp(0.0, 10.0))
            .unwrap_or(0.25);
        let phase_drift_spike_threshold = env::var("SPIRAL_TRAINER_PHASE_DRIFT_SPIKE_THRESHOLD")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(|value| value.max(0.0))
            .unwrap_or(0.6);

        Self {
            epoch: 0,
            planner: RankPlanner::new(caps),
            curvature,
            hyper_learning_rate,
            fallback_learning_rate,
            grad_clip_max_norm: None,
            real_learning_rate: None,
            blackcat: None,
            blackcat_moderator: None,
            autopilot: None,
            band_weight_fn: None,
            text_infusion: None,
            injector_enabled: false,
            distribution: None,
            meta_conductor: None,
            heur_log: HeurOpLog::default(),
            rewrite_budget: None,
            softlogic: SoftLogicFlex::new(),
            spectral_adapter,
            accumulator_synchronizer: None,
            last_accumulator_sync: TrainerAccumulatorSyncStats::disabled(),
            desire_bridge: None,
            desire_roundtable_bridge: None,
            last_desire_roundtable_summary: None,
            #[cfg(feature = "psi")]
            desire_psi_bridge: None,
            graph_bridge: None,
            graph_pending: None,
            graph_last_hint: None,
            gnn_roundtable_bridge: None,
            gnn_last_roundtable_signal: None,
            curvature_scheduler: None,
            last_curvature_metrics: None,
            loss_strategy: LossStrategy::default(),
            spectral_policy: None,
            coherence_bridge: None,
            pending_coherence: None,
            last_spectral_metrics: None,
            last_band_energy: None,
            phase_turnover_spike_threshold,
            phase_loss_ema_alpha,
            phase_loss_spike_ratio,
            phase_drift_spike_threshold,
            phase_last_label: None,
            phase_last_turnover: None,
            phase_last_band: None,
            phase_last_drift_abs: None,
            phase_loss_ema: None,
            phase_loss_spiking: false,
            #[cfg(feature = "golden")]
            golden_pulse: None,
            #[cfg(feature = "golden")]
            golden_directive: None,
            #[cfg(feature = "golden")]
            golden_council: None,
            #[cfg(feature = "psi")]
            psi,
            #[cfg(feature = "psychoid")]
            psychoid: None,
            #[cfg(feature = "psychoid")]
            psychoid_log: false,
            #[cfg(feature = "collapse")]
            collapse: None,
        }
    }

    /// Enables the graph consensus feedback loop by attaching a bridge that
    /// drains graph flow telemetry after each optimisation step.
    pub fn enable_graph_feedback(&mut self, bridge: GraphConsensusBridge) {
        self.graph_bridge = Some(bridge);
        self.graph_pending = None;
    }

    /// Enables roundtable-driven adjustments for GNN modules.
    pub fn enable_gnn_roundtable_bridge(&mut self, bridge: RoundtableGnnBridge) {
        self.gnn_roundtable_bridge = Some(bridge);
        self.gnn_last_roundtable_signal = None;
    }

    /// Disables any previously attached GNN roundtable bridge.
    pub fn disable_gnn_roundtable_bridge(&mut self) {
        self.gnn_roundtable_bridge = None;
        self.gnn_last_roundtable_signal = None;
    }

    /// Returns the most recent roundtable signal broadcast to the GNN.
    pub fn gnn_roundtable_signal(&self) -> Option<RoundtableBandSignal> {
        self.gnn_last_roundtable_signal.clone()
    }

    /// Enables desire telemetry feedback so automation and training can share
    /// aggregated summaries without bespoke glue.
    pub fn enable_desire_pipeline(&mut self, bridge: DesireTrainerBridge) {
        self.desire_bridge = Some(bridge);
    }

    /// Enables the roundtable desire braid so Z-space impulses can steer the
    /// A/B/C consensus without bespoke glue.
    pub fn enable_desire_roundtable_bridge(&mut self, bridge: DesireRoundtableBridge) {
        self.desire_roundtable_bridge = Some(bridge);
    }

    /// Bundles desire trainer, roundtable, and ψ bridges so experiments can
    /// attach every compatible telemetry stream in one call.
    pub fn enable_desire_telemetry(&mut self, bundle: &DesireTelemetryBundle) {
        if let Some(bridge) = bundle.trainer_bridge() {
            self.enable_desire_pipeline(bridge.clone());
        }
        if let Some(bridge) = bundle.roundtable_bridge() {
            self.enable_desire_roundtable_bridge(bridge.clone());
        }
        #[cfg(feature = "psi")]
        if let Some(bridge) = bundle.psi_bridge() {
            self.enable_desire_psi_bridge(bridge.clone());
        }
    }

    /// Clears any attached roundtable desire bridge.
    pub fn disable_desire_roundtable_bridge(&mut self) {
        self.desire_roundtable_bridge = None;
        self.last_desire_roundtable_summary = None;
    }

    /// Returns a new trainer with the provided loss strategy.
    pub fn with_loss_strategy(mut self, strategy: LossStrategy) -> Self {
        self.loss_strategy = strategy;
        self
    }

    /// Installs a loss strategy controlling how band losses are aggregated.
    pub fn set_loss_strategy(&mut self, strategy: LossStrategy) {
        self.loss_strategy = strategy;
    }

    /// Configures the trainer to apply region-aware loss weights.
    pub fn enable_region_loss(&mut self, config: RegionLossConfig) {
        self.loss_strategy = LossStrategy::Region(config);
    }

    /// Restores the baseline loss aggregation strategy.
    pub fn disable_region_loss(&mut self) {
        self.loss_strategy = LossStrategy::Baseline;
    }

    /// Returns the currently configured loss strategy.
    pub fn loss_strategy(&self) -> &LossStrategy {
        &self.loss_strategy
    }

    /// Returns the active SoftLogic configuration.
    pub fn softlogic_config(&self) -> SoftLogicConfig {
        self.softlogic.config()
    }

    /// Installs a new SoftLogic configuration (clamped to safe ranges).
    pub fn set_softlogic_config(&mut self, config: SoftLogicConfig) {
        self.softlogic.set_config(config);
    }

    /// Resets SoftLogic state and reloads environment overrides.
    pub fn reset_softlogic(&mut self) {
        self.softlogic = SoftLogicFlex::new();
    }

    /// Surfaces the qualitative coherence observation so schedulers can react.
    pub fn coherence_observation(
        &self,
        diagnostics: &CoherenceDiagnostics,
    ) -> CoherenceObservation {
        diagnostics.observation()
    }

    /// Converts coherence observations into semantic labels.
    pub fn interpret_coherence(&self, diagnostics: &CoherenceDiagnostics) -> CoherenceLabel {
        self.coherence_observation(diagnostics).lift_to_label()
    }

    /// Installs a curvature scheduler so the trainer can adapt its hyperbolic
    /// geometry based on recent gradient pressure observations.
    pub fn enable_curvature_scheduler(&mut self, mut scheduler: CurvatureScheduler) {
        scheduler.apply_env_overrides();
        scheduler.sync(self.curvature);
        self.curvature_scheduler = Some(scheduler);
        self.last_curvature_metrics = None;
        self.spectral_adapter.set_curvature_target(self.curvature);
    }

    /// Disables any configured curvature scheduler and clears cached metrics.
    pub fn disable_curvature_scheduler(&mut self) {
        self.curvature_scheduler = None;
        self.last_curvature_metrics = None;
        self.spectral_adapter.set_curvature_target(self.curvature);
    }

    /// Returns the most recently recorded curvature metrics emitted by the
    /// scheduler, when available.
    pub fn curvature_metrics(&self) -> Option<CurvatureMetrics> {
        self.last_curvature_metrics
    }

    /// Enables the spectral learning rate adaptation policy driven by Z-space coherence.
    pub fn enable_spectral_learning_rate(&mut self, mut policy: SpectralLearningRatePolicy) {
        policy.apply_env_overrides();
        self.spectral_policy = Some(policy);
        self.last_spectral_metrics = None;
        self.last_band_energy = None;
        self.phase_last_label = None;
        self.phase_last_turnover = None;
        self.enable_zspace_trace_coherence_bridge();
    }

    /// Disables the spectral learning rate policy.
    pub fn disable_spectral_learning_rate(&mut self) {
        self.spectral_policy = None;
        self.last_spectral_metrics = None;
        self.last_band_energy = None;
        self.phase_last_label = None;
        self.phase_last_turnover = None;
    }

    /// Publishes fresh coherence diagnostics that will be consumed by the spectral policy.
    pub fn push_coherence_diagnostics(&mut self, diagnostics: CoherenceDiagnostics) {
        self.pending_coherence = Some(CoherenceSignal::from(&diagnostics));
    }

    /// Enables automatic coherence injection by listening to `ZSpaceTrace` plugin events.
    pub fn enable_zspace_trace_coherence_bridge(&mut self) {
        if self.coherence_bridge.is_none() {
            self.coherence_bridge = Some(ZSpaceTraceCoherenceBridge::subscribe());
        }
    }

    /// Disables any configured ZSpaceTrace coherence bridge.
    pub fn disable_zspace_trace_coherence_bridge(&mut self) {
        self.coherence_bridge = None;
    }

    /// Returns the last recorded spectral metrics, including the most recent
    /// roundtable band snapshot even when coherence adaptation did not fire.
    pub fn spectral_metrics(&self) -> Option<TrainerSpectralMetrics> {
        if self.spectral_policy.is_none() && self.last_spectral_metrics.is_none() {
            return None;
        }
        let adjustment = self.last_spectral_metrics.clone();
        let band_energy = self.last_band_energy;
        let source = match (adjustment.is_some(), band_energy.is_some()) {
            (true, true) => TrainerSpectralMetricsSource::Combined,
            (true, false) => TrainerSpectralMetricsSource::CoherenceAdjustment,
            (false, true) => TrainerSpectralMetricsSource::BandEnergy,
            (false, false) => return None,
        };
        Some(TrainerSpectralMetrics {
            source,
            turnover: self.phase_last_turnover,
            label: self.phase_last_label,
            adjustment,
            band_energy,
        })
    }

    #[cfg(feature = "psi")]
    pub fn enable_desire_psi_bridge(&mut self, bridge: DesirePsiBridge) {
        self.desire_psi_bridge = Some(bridge);
    }

    /// Returns the SpiralK hint generated from the most recently applied graph
    /// digest, if any.
    pub fn graph_hint(&self) -> Option<&str> {
        self.graph_last_hint.as_deref()
    }

    /// Returns the last drained roundtable desire summary, if available.
    pub fn desire_roundtable_summary(&self) -> Option<DesireRoundtableSummary> {
        self.last_desire_roundtable_summary.clone()
    }

    #[cfg(feature = "psi")]
    fn init_psi_meter() -> Option<PsiMeter> {
        let enabled = env::var("SPIRAL_PSI")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on"
                )
            })
            .unwrap_or(false);
        if !enabled {
            return None;
        }

        let mut cfg = PsiConfig {
            enabled: true,
            components: PsiComponent::defaults(),
            ..Default::default()
        };

        if let Ok(spec) = env::var("SPIRAL_PSI_COMPONENTS") {
            if let Ok(mask) = PsiComponent::parse_list(&spec) {
                cfg.components = mask;
            }
        }

        if let Ok(alpha_str) = env::var("SPIRAL_PSI_ALPHA") {
            if let Ok(alpha) = alpha_str.parse::<f32>() {
                cfg.ema_alpha = alpha.clamp(1.0e-3, 0.999);
            }
        }

        if let Ok(rate_str) = env::var("SPIRAL_PSI_SAMPLE_RATE") {
            if let Ok(rate) = rate_str.parse::<u32>() {
                cfg.sample_rate = rate.max(1);
            }
        }

        for (var, component) in [
            ("SPIRAL_PSI_WEIGHT_LOSS", PsiComponent::LOSS),
            ("SPIRAL_PSI_WEIGHT_GRAD", PsiComponent::GRAD_NORM),
            ("SPIRAL_PSI_WEIGHT_UPDATE", PsiComponent::UPDATE_RATIO),
            ("SPIRAL_PSI_WEIGHT_ACT", PsiComponent::ACT_DRIFT),
            ("SPIRAL_PSI_WEIGHT_ATTN", PsiComponent::ATTN_ENTROPY),
            ("SPIRAL_PSI_WEIGHT_BAND", PsiComponent::BAND_ENERGY),
        ] {
            if let Ok(weight_str) = env::var(var) {
                if let Ok(weight) = weight_str.parse::<f32>() {
                    cfg.weights.insert(component, weight);
                }
            }
        }

        for (var, component) in [
            ("SPIRAL_PSI_TH_LOSS", PsiComponent::LOSS),
            ("SPIRAL_PSI_TH_GRAD", PsiComponent::GRAD_NORM),
            ("SPIRAL_PSI_TH_UPDATE", PsiComponent::UPDATE_RATIO),
            ("SPIRAL_PSI_TH_ACT", PsiComponent::ACT_DRIFT),
            ("SPIRAL_PSI_TH_ATTN", PsiComponent::ATTN_ENTROPY),
            ("SPIRAL_PSI_TH_BAND", PsiComponent::BAND_ENERGY),
        ] {
            if let Ok(threshold_str) = env::var(var) {
                if let Ok(threshold) = threshold_str.parse::<f32>() {
                    cfg.thresholds.insert(component, threshold);
                }
            }
        }

        Some(PsiMeter::new(cfg))
    }

    /// Attaches the BlackCat runtime so contextual rewards update after each step.
    pub fn with_blackcat(mut self, runtime: BlackCatRuntime) -> Self {
        self.blackcat = Some(runtime);
        self
    }

    /// Installs a Blackcat moderator that seats between local and distributed consensus.
    pub fn install_blackcat_moderator(&mut self, threshold: f32, participants: usize) {
        self.blackcat_moderator = Some(BlackcatModerator::with_default_runtime(
            threshold.max(0.1),
            participants.max(1),
        ));
        self.meta_conductor = None;
    }

    /// Installs a moderator with a custom runtime configuration.
    pub fn install_blackcat_moderator_with_runtime(
        &mut self,
        runtime: BlackCatRuntime,
        threshold: f32,
        participants: usize,
    ) {
        self.blackcat_moderator = Some(BlackcatModerator::new(
            runtime,
            threshold.max(0.1),
            participants.max(1),
        ));
        self.meta_conductor = None;
    }

    /// Clears any configured moderator.
    pub fn clear_blackcat_moderator(&mut self) {
        self.blackcat_moderator = None;
    }

    /// Attaches an Autopilot runtime for contextual kernel selection.
    pub fn with_autopilot(mut self, autopilot: Autopilot) -> Self {
        self.autopilot = Some(autopilot);
        self
    }

    /// Enables per-band reweighting of the loss/gradient.
    pub fn set_band_weights(&mut self, weight_fn: BandWeightFn) {
        self.band_weight_fn = Some(weight_fn);
    }

    /// Connects the trainer to a distributed roundtable node.
    pub fn configure_distribution(&mut self, config: DistConfig) {
        let mut metadata = HashMap::new();
        metadata.insert("node_id".to_string(), config.node_id.clone());
        metadata.insert("mode".to_string(), config.mode.as_str().to_string());
        metadata.insert(
            "push_interval_ms".to_string(),
            config
                .push_interval
                .as_millis()
                .min(u64::MAX as u128)
                .to_string(),
        );
        metadata.insert(
            "summary_window".to_string(),
            config.summary_window.to_string(),
        );
        if !config.meta_endpoints.is_empty() {
            metadata.insert(
                "meta_endpoints".to_string(),
                config.meta_endpoints.join(","),
            );
        }

        append_cloud_targets(&mut metadata, &config.cloud_targets);
        self.log_connector_event("configure_distribution", metadata);
        self.distribution = Some(RoundtableNode::new(config));
    }

    /// Removes any configured distribution node.
    pub fn clear_distribution(&mut self) {
        if let Some(node) = self.distribution.as_mut() {
            node.drain();
        }
        if self.distribution.is_some() {
            self.log_connector_event("clear_distribution", HashMap::new());
        }
        self.distribution = None;
    }

    /// Returns the currently configured distribution node, if any.
    pub fn distribution_config(&self) -> Option<&DistConfig> {
        self.distribution.as_ref().map(|node| node.config())
    }

    /// Installs a meta-layer conductor so this trainer can aggregate remote summaries.
    pub fn install_meta_conductor(&mut self, threshold: f32, participants: usize) {
        self.blackcat_moderator = None;
        self.meta_conductor = Some(MetaConductor::new(threshold.max(0.1), participants.max(1)));
    }

    /// Returns the current heuristics op-log.
    pub fn heuristics_log(&self) -> &HeurOpLog {
        &self.heur_log
    }

    /// Merges the provided heuristics log into the trainer's local log.
    pub fn merge_heuristics_log(&mut self, log: &HeurOpLog) {
        self.heur_log.merge(log);
    }

    /// Returns the latest moderator minutes captured by Blackcat.
    pub fn blackcat_minutes(&self) -> Vec<ModeratorMinutes> {
        self.blackcat_moderator
            .as_ref()
            .map(|m| m.minutes().to_vec())
            .unwrap_or_default()
    }

    /// Returns the aggregated scoreboard derived from the local Blackcat moderator.
    pub fn blackcat_scoreboard(&self) -> Vec<BlackcatScore> {
        self.blackcat_moderator
            .as_ref()
            .map(|m| m.scoreboard())
            .unwrap_or_default()
    }

    /// Returns aggregated stats tracked by the embedded Blackcat runtime, when available.
    pub fn blackcat_runtime_stats(&self) -> Option<BlackcatRuntimeStats> {
        self.blackcat.as_ref().map(|rt| rt.stats())
    }

    /// Replays moderator minutes so the embedded Blackcat runtime can stay aligned.
    pub fn sync_blackcat_minutes(&mut self, minutes: &[ModeratorMinutes]) {
        if let Some(moderator) = self.blackcat_moderator.as_mut() {
            moderator.absorb_minutes(minutes);
        }
    }

    #[cfg(feature = "golden")]
    /// Applies a cooperative pulse emitted by the Golden retriever.
    pub fn apply_blackcat_pulse(&mut self, pulse: &GoldenBlackcatPulse) {
        self.golden_pulse = Some(pulse.clone());
        self.golden_directive = None;
        if let Some(node) = self.distribution.as_mut() {
            let base_interval = node.config().push_interval;
            let base_window = node.config().summary_window.max(1);
            let directive = pulse.directive(base_interval, base_window);
            node.retune(directive.push_interval, directive.summary_window);
            if directive.reinforcement_weight > 0.1 {
                self.injector_enabled = true;
            }
            self.golden_directive = Some(directive);
        } else if pulse.reinforcement_weight > 0.1 || pulse.optimization_gain > 0.1 {
            self.injector_enabled = true;
        }
    }

    #[cfg(feature = "golden")]
    pub(crate) fn record_golden_council(&mut self, snapshot: &GoldenCouncilSnapshot) {
        self.golden_council = Some(snapshot.clone());
    }

    #[cfg(feature = "golden")]
    /// Returns the most recent cooperative pulse applied to this trainer.
    pub fn last_blackcat_pulse(&self) -> Option<&GoldenBlackcatPulse> {
        self.golden_pulse.as_ref()
    }

    #[cfg(feature = "golden")]
    /// Returns the latest cooperative directive derived from the Golden pulse.
    pub fn last_blackcat_directive(&self) -> Option<&GoldenCooperativeDirective> {
        self.golden_directive.as_ref()
    }

    #[cfg(feature = "golden")]
    /// Returns the latest self-rewrite council snapshot received from GoldenRetriever.
    pub fn last_golden_council_snapshot(&self) -> Option<&GoldenCouncilSnapshot> {
        self.golden_council.as_ref()
    }

    #[cfg(feature = "golden")]
    /// Returns a clone of the latest council snapshot for downstream mutation.
    pub fn last_council(&self) -> Option<GoldenCouncilSnapshot> {
        self.golden_council.clone()
    }

    /// Configures how many rewrite operations may be applied per epoch and the cooldown required
    /// before new rewrites are accepted.
    pub fn set_rewrite_budget(&mut self, per_epoch: u32, cooldown: u32) {
        if per_epoch == 0 {
            self.rewrite_budget = None;
        } else {
            self.rewrite_budget = Some(RewriteBudget::new(per_epoch, cooldown));
        }
    }

    /// Clears any registered band weighting rule.
    pub fn clear_band_weights(&mut self) {
        self.band_weight_fn = None;
    }

    /// Enables or disables the adaptive injector heuristics.
    pub fn enable_injector(&mut self, on: bool) {
        self.injector_enabled = on;
    }

    /// Returns the underlying planner.
    pub fn planner(&self) -> &RankPlanner {
        &self.planner
    }

    /// Produces a roundtable schedule for the provided output dimensions.
    ///
    /// When an Autopilot runtime is attached the planner suggestions are
    /// overridden with the latest contextual picks before the schedule is
    /// emitted.
    pub fn roundtable(
        &mut self,
        rows: u32,
        cols: u32,
        config: RoundtableConfig,
    ) -> RoundtableSchedule {
        let mut schedule = RoundtableSchedule::new(&self.planner, rows, cols, config);
        let mut autopilot_picks: Option<std::collections::HashMap<String, String>> = None;
        if self.autopilot.is_some() {
            let depth = schedule.above().k + schedule.here().k + schedule.beneath().k;
            let device_load = self.estimate_device_load();
            if let Some(ap) = self.autopilot.as_mut() {
                let context = ap.build_context(rows, cols, depth, device_load, &[]);
                let picks = ap.suggest(context).clone();
                if !picks.is_empty() {
                    schedule.apply_knob_overrides(&picks);
                }
                autopilot_picks = Some(picks);
            }
        }
        let bus = global_registry().event_bus();
        if bus.has_listeners("RoundtablePlanned") {
            let encode_plan = |plan: &RankPlan| {
                serde_json::json!({
                    "kind": plan.kind.as_str(),
                    "rows": plan.rows,
                    "cols": plan.cols,
                    "k": plan.k,
                    "choice": {
                        "subgroup": plan.choice.subgroup,
                        "use_2ce": plan.choice.use_2ce,
                        "wg": plan.choice.wg,
                        "kl": plan.choice.kl,
                        "ch": plan.choice.ch,
                        "tile": plan.choice.tile,
                        "ctile": plan.choice.ctile,
                        "fft_tile": plan.choice.fft_tile,
                        "fft_radix": plan.choice.fft_radix,
                        "fft_segments": plan.choice.fft_segments,
                    }
                })
            };
            bus.publish(&PluginEvent::custom(
                "RoundtablePlanned",
                serde_json::json!({
                    "rows": rows,
                    "cols": cols,
                    "config": {
                        "top_k": config.top_k,
                        "mid_k": config.mid_k,
                        "bottom_k": config.bottom_k,
                        "here_tolerance": config.here_tolerance,
                    },
                    "autopilot_enabled": self.autopilot.is_some(),
                    "autopilot_picks": autopilot_picks,
                    "bands": {
                        "above": encode_plan(schedule.above()),
                        "here": encode_plan(schedule.here()),
                        "beneath": encode_plan(schedule.beneath()),
                    },
                }),
            ));
        }
        self.emit_roundtable_summary(rows, cols, config, &schedule);
        schedule
    }

    fn emit_roundtable_summary(
        &self,
        rows: u32,
        cols: u32,
        config: RoundtableConfig,
        schedule: &RoundtableSchedule,
    ) {
        let pipeline = crate::language::LanguagePipeline::builder("module_trainer")
            .with_tag("component", "module_trainer")
            .build();
        pipeline.record_roundtable(
            rows,
            cols,
            config,
            schedule,
            self.autopilot.is_some(),
            self.distribution.as_ref(),
        );
        let cfg_summary = {
            #[allow(unused_mut)]
            let mut summary = RoundtableConfigSummary::new(
                config.top_k,
                config.mid_k,
                config.bottom_k,
                config.here_tolerance,
            );
            #[cfg(feature = "psychoid")]
            {
                summary
                    .extras
                    .insert("psychoid".to_string(), config.psychoid_enabled);
                if config.psychoid_log {
                    summary.extras.insert("psychoid_log".to_string(), true);
                }
            }
            #[cfg(feature = "psi")]
            {
                summary.extras.insert("psi".to_string(), config.psi_enabled);
            }
            #[cfg(feature = "collapse")]
            {
                summary
                    .extras
                    .insert("collapse".to_string(), config.collapse_enabled);
            }
            summary
        };

        let plans = vec![
            Self::summarize_rank_plan(schedule.above()),
            Self::summarize_rank_plan(schedule.here()),
            Self::summarize_rank_plan(schedule.beneath()),
        ];

        let distribution = self.distribution.as_ref().map(|node| {
            let cfg = node.config();
            DistributionSummary {
                node_id: cfg.node_id.clone(),
                mode: cfg.mode.as_str().to_string(),
                summary_window: cfg.summary_window,
                push_interval_ms: cfg.push_interval.as_millis().min(u64::MAX as u128) as u64,
                meta_endpoints: cfg.meta_endpoints.clone(),
                cloud_targets: cfg.cloud_targets.clone(),
            }
        });

        let summary = RoundtableSummary {
            rows,
            cols,
            config: cfg_summary,
            plans,
            autopilot_enabled: self.autopilot.is_some(),
            distribution,
            issued_at: SystemTime::now(),
        };

        let registry = EcosystemRegistry::global();
        let autopilot_tag = summary.autopilot_enabled.to_string();
        let distribution_mode = summary.distribution.as_ref().map(|d| d.mode.clone());
        let tag_sample = |sample: MetricSample| {
            let mut sample = sample.with_tag("autopilot", autopilot_tag.as_str());
            if let Some(mode) = &distribution_mode {
                sample = sample.with_tag("distribution_mode", mode.clone());
            }
            sample
        };

        registry.record_metric(tag_sample(
            MetricSample::new("roundtable.rows", rows as f64).with_unit("rows"),
        ));
        registry.record_metric(tag_sample(
            MetricSample::new("roundtable.cols", cols as f64).with_unit("cols"),
        ));
        registry.record_metric(tag_sample(
            MetricSample::new(
                "roundtable.autopilot",
                if summary.autopilot_enabled { 1.0 } else { 0.0 },
            )
            .with_unit("flag"),
        ));
        registry.record_metric(tag_sample(
            MetricSample::new("roundtable.config.top_k", config.top_k as f64).with_unit("items"),
        ));
        registry.record_metric(tag_sample(
            MetricSample::new("roundtable.config.mid_k", config.mid_k as f64).with_unit("items"),
        ));
        registry.record_metric(tag_sample(
            MetricSample::new("roundtable.config.bottom_k", config.bottom_k as f64)
                .with_unit("items"),
        ));
        registry.record_metric(tag_sample(
            MetricSample::new(
                "roundtable.config.here_tolerance",
                config.here_tolerance as f64,
            )
            .with_unit("ratio"),
        ));

        let plan_summaries = [
            ("above", schedule.above()),
            ("here", schedule.here()),
            ("beneath", schedule.beneath()),
        ];
        for (band, plan) in plan_summaries {
            let tag_band = |sample: MetricSample| tag_sample(sample.with_tag("band", band));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.rows", plan.rows as f64).with_unit("rows"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.cols", plan.cols as f64).with_unit("cols"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.k", plan.k as f64).with_unit("items"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.workgroup", plan.choice.wg as f64)
                    .with_unit("threads"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.lanes", plan.choice.kl as f64)
                    .with_unit("lanes"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.channel_stride", plan.choice.ch as f64)
                    .with_unit("stride"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.tile", plan.choice.tile as f64)
                    .with_unit("tile"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.compaction_tile", plan.choice.ctile as f64)
                    .with_unit("tile"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new(
                    "roundtable.band.subgroup",
                    if plan.choice.subgroup { 1.0 } else { 0.0 },
                )
                .with_unit("flag"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.fft_tile", plan.choice.fft_tile as f64)
                    .with_unit("tile"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new("roundtable.band.fft_radix", plan.choice.fft_radix as f64)
                    .with_unit("radix"),
            ));
            registry.record_metric(tag_band(
                MetricSample::new(
                    "roundtable.band.fft_segments",
                    plan.choice.fft_segments as f64,
                )
                .with_unit("segments"),
            ));
        }

        registry.record_roundtable(summary);
    }

    fn summarize_rank_plan(plan: &RankPlan) -> RankPlanSummary {
        let mut summary = RankPlanSummary::new(plan.kind, plan.rows, plan.cols, plan.k);
        summary.workgroup = plan.choice.wg;
        summary.lanes = plan.choice.kl;
        summary.channel_stride = plan.choice.ch;
        summary.tile = plan.choice.tile;
        summary.compaction_tile = plan.choice.ctile;
        summary.subgroup = plan.choice.subgroup;
        summary.fft_tile = plan.choice.fft_tile;
        summary.fft_radix = plan.choice.fft_radix;
        summary.fft_segments = plan.choice.fft_segments;
        summary
    }

    fn log_connector_event(&self, stage: &str, metadata: HashMap<String, String>) {
        EcosystemRegistry::global().record_connector(ConnectorEvent {
            name: "module_trainer".to_string(),
            stage: stage.to_string(),
            metadata,
            issued_at: SystemTime::now(),
        });
    }

    /// Returns the fallback Euclidean learning rate.
    pub fn fallback_learning_rate(&self) -> f32 {
        self.fallback_learning_rate
    }

    fn accumulator_synchronizer_snapshot(&self) -> (bool, usize, usize) {
        if let Some(device) = self.accumulator_synchronizer.as_ref() {
            return (true, device.rank(), device.world_size());
        }
        (false, 0, 1)
    }

    /// Returns a copyable optimizer-facing state snapshot.
    pub fn optimizer_state(&self) -> TrainerOptimizerState {
        let (training_device_enabled, training_rank, training_world_size) =
            self.accumulator_synchronizer_snapshot();
        TrainerOptimizerState {
            epoch: self.epoch,
            curvature: self.curvature,
            hyper_learning_rate: self.hyper_learning_rate,
            fallback_learning_rate: self.fallback_learning_rate,
            real_learning_rate: self.real_learning_rate,
            grad_clip_max_norm: self.grad_clip_max_norm,
            spectral_policy_enabled: self.spectral_policy.is_some(),
            curvature_scheduler_enabled: self.curvature_scheduler.is_some(),
            training_device_enabled,
            training_rank,
            training_world_size,
            spectral_adapter: self.spectral_adapter.state(),
        }
    }

    /// Returns details for the most recent accumulator synchronization pass.
    pub fn last_accumulator_sync(&self) -> TrainerAccumulatorSyncStats {
        self.last_accumulator_sync
    }

    /// Installs a training device used to synchronize parameter accumulators before updates.
    pub fn set_training_device<D>(&mut self, device: D)
    where
        D: AccumulatorSynchronizer + 'static,
    {
        self.accumulator_synchronizer = Some(Arc::new(device));
    }

    /// Installs a shared training device used to synchronize parameter accumulators before updates.
    pub fn set_training_device_arc(&mut self, device: Arc<dyn AccumulatorSynchronizer>) {
        self.accumulator_synchronizer = Some(device);
    }

    /// Clears any configured training device and returns to local-only updates.
    pub fn clear_training_device(&mut self) {
        self.accumulator_synchronizer = None;
        self.last_accumulator_sync = TrainerAccumulatorSyncStats::disabled();
    }

    /// Returns the curvature used for hypergrad preparation.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the learning rate used for hypergrad updates.
    pub fn hyper_learning_rate(&self) -> f32 {
        self.hyper_learning_rate
    }

    /// Multiplies all learning rates (hypergrad, fallback, optional realgrad) by `factor`.
    ///
    /// This also scales any per-parameter learning rates so optimiser state stays consistent.
    pub fn mul_learning_rate<M: Module + ?Sized>(
        &mut self,
        module: &mut M,
        factor: f32,
    ) -> PureResult<()> {
        self.scale_learning_rates(module, factor)
    }

    /// Returns the optional gradient clipping threshold.
    pub fn grad_clip_max_norm(&self) -> Option<f32> {
        self.grad_clip_max_norm
    }

    /// Sets the gradient clipping threshold (global L2 norm).
    ///
    /// Passing a non-positive or non-finite value disables clipping.
    pub fn set_grad_clip_max_norm(&mut self, max_norm: f32) {
        if max_norm.is_finite() && max_norm > 0.0 {
            self.grad_clip_max_norm = Some(max_norm);
        } else {
            self.grad_clip_max_norm = None;
        }
    }

    /// Disables any previously configured gradient clipping.
    pub fn clear_grad_clip_max_norm(&mut self) {
        self.grad_clip_max_norm = None;
    }

    /// Returns the Euclidean realgrad learning rate, when enabled.
    pub fn real_learning_rate(&self) -> Option<f32> {
        self.real_learning_rate
    }

    /// Enables Euclidean realgrad accumulation with the provided learning rate.
    pub fn with_realgrad(mut self, learning_rate: f32) -> Self {
        self.enable_realgrad(learning_rate);
        self
    }

    /// Enables Euclidean realgrad accumulation with the provided learning rate.
    pub fn enable_realgrad(&mut self, learning_rate: f32) {
        if learning_rate.is_finite() && learning_rate > 0.0 {
            self.real_learning_rate = Some(learning_rate);
        }
    }

    /// Disables the optional realgrad accumulation pathway.
    pub fn disable_realgrad(&mut self) {
        self.real_learning_rate = None;
    }

    /// Configures the trainer to broadcast a text infusion signal through the module.
    pub fn set_text_infusion(
        &mut self,
        text: impl Into<String>,
        every: TextInfusionEvery,
        mode: TextInfusionMode,
    ) -> PureResult<()> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(TensorError::EmptyInput("text_infusion"));
        }
        self.text_infusion = Some(TextInfusionConfig { text, every, mode });
        Ok(())
    }

    /// Disables any previously configured text infusion signal.
    pub fn clear_text_infusion(&mut self) {
        self.text_infusion = None;
    }

    /// Attaches hypergrad tapes to all parameters of the provided module.
    pub fn prepare<M: Module + ?Sized>(&self, module: &mut M) -> PureResult<()> {
        module.attach_hypergrad(self.curvature, self.hyper_learning_rate)?;
        if let Some(rate) = self.real_learning_rate {
            module.attach_realgrad(rate)?;
        }
        Ok(())
    }

    /// Attaches hypergrad tapes with an explicit topos shared across parameters.
    pub fn prepare_with_topos<M: Module + ?Sized>(
        &self,
        module: &mut M,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        module.attach_hypergrad_with_topos(self.curvature, self.hyper_learning_rate, topos)?;
        if let Some(rate) = self.real_learning_rate {
            module.attach_realgrad(rate)?;
        }
        Ok(())
    }

    /// Clears accumulated gradients or hypergrad buffers.
    pub fn zero<M: Module + ?Sized>(&self, module: &mut M) -> PureResult<()> {
        module.zero_accumulators()
    }

    /// Applies the parameter updates using either the hypergrad tape or the fallback rate.
    pub fn step<M: Module + ?Sized>(&mut self, module: &mut M) -> PureResult<()> {
        self.synchronize_parameter_accumulators(module)?;
        if let Some(limit) = self.grad_clip_max_norm {
            self.clip_grad_global_norm(module, limit)?;
        }
        let adapter: &mut dyn LocalLearningRateAdapter = &mut self.spectral_adapter;
        module.apply_step_with_adapter(self.fallback_learning_rate, Some(adapter))
    }

    /// Runs a full epoch over the provided iterator of `(input, target)` pairs.
    pub fn train_epoch<M, L, I>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        batches: I,
        schedule: &RoundtableSchedule,
    ) -> PureResult<EpochStats>
    where
        M: Module + ?Sized,
        L: Loss + ?Sized,
        I: IntoIterator,
        I::Item: IntoBatch,
    {
        self.epoch = self.epoch.saturating_add(1);
        let epoch = self.epoch;
        global_registry()
            .event_bus()
            .publish(&PluginEvent::EpochStart { epoch });
        if let Some(budget) = self.rewrite_budget.as_mut() {
            budget.begin_epoch();
        }
        self.zero(module)?;
        module.train()?;
        if let Some(infusion) = self.text_infusion.clone() {
            let should_apply = match infusion.every {
                TextInfusionEvery::Once => epoch == 1,
                TextInfusionEvery::Epoch => true,
                TextInfusionEvery::Batch => false,
            };
            if should_apply {
                module.infuse_text(&infusion.text)?;
                if infusion.mode == TextInfusionMode::Separate {
                    self.step(module)?;
                }
                if matches!(infusion.every, TextInfusionEvery::Once) {
                    self.text_infusion = None;
                }
            }
        }
        self.phase_last_label = None;
        self.phase_last_turnover = None;
        self.phase_last_band = None;
        self.phase_last_drift_abs = None;
        self.phase_loss_ema = None;
        self.phase_loss_spiking = false;
        self.last_band_energy = None;
        self.last_spectral_metrics = None;
        #[cfg(feature = "psi")]
        self.bootstrap_psi(schedule);
        #[cfg(feature = "psychoid")]
        self.bootstrap_psychoid(schedule);
        #[cfg(feature = "collapse")]
        self.bootstrap_collapse(schedule);
        let mut total_loss = 0.0f32;
        let mut steps = 0usize;
        let backend_policy = BackendPolicy::from_device_caps(self.planner.device_caps());
        let strict_expected_backend = strict_expected_tensor_backend(self.planner.device_caps());
        let trace_trainer_steps = global_registry().event_bus().has_listeners("TrainerStep");
        let trace_parameter_updates = trace_trainer_steps;
        let tensor_trace = Some(TensorOpMetaStepCollector::install());
        let mut epoch_tensor_backend = EpochTensorBackendStats::default();
        for batch in batches.into_iter() {
            if let Some(trace) = tensor_trace.as_ref() {
                trace.clear();
            }
            let (input, target) = batch.into_batch()?;
            validate_trainer_tensor("trainer_batch_input", &input)?;
            validate_trainer_tensor("trainer_batch_target", &target)?;
            let input_shape = input.shape();
            let target_shape = target.shape();
            let graph_adjustment = self.graph_pending.take();
            self.graph_last_hint = graph_adjustment
                .as_ref()
                .and_then(|digest| digest.spiralk_script.clone());
            let step_start = Instant::now();
            if let Some(rt) = self.blackcat.as_mut() {
                rt.begin_step();
            }
            let device_load = self.estimate_device_load();
            if let Some(ap) = self.autopilot.as_mut() {
                let (rows, cols) = input.shape();
                let depth = schedule.above().k + schedule.here().k + schedule.beneath().k;
                let context = ap.build_context(rows as u32, cols as u32, depth, device_load, &[]);
                let _ = ap.suggest(context);
            }
            let _backend_policy_guard = push_backend_policy(backend_policy);
            let prediction = module.forward(&input)?;
            validate_trainer_tensor("trainer_prediction", &prediction)?;
            let prediction_shape = prediction.shape();
            let loss_value = loss.forward(&prediction, &target)?;
            validate_trainer_tensor("trainer_loss", &loss_value)?;
            let loss_shape = loss_value.shape();
            let step_loss = trainer_tensor_sum("trainer_step_loss", &loss_value)?;
            let grad_output = loss.backward(&prediction, &target)?;
            validate_trainer_tensor("trainer_grad_output", &grad_output)?;
            let grad_output_shape = grad_output.shape();
            let mut band_energy = schedule.band_energy(&grad_output)?;
            validate_band_energy_for_trainer(&band_energy)?;
            let baseline_band_energy = band_energy;
            let mut desire_impulse = None;
            if let Some(bridge) = self.desire_roundtable_bridge.as_ref() {
                if let Some(impulse) = bridge.impulse()? {
                    band_energy.above *= impulse.multipliers.0;
                    band_energy.here *= impulse.multipliers.1;
                    band_energy.beneath *= impulse.multipliers.2;
                    band_energy.drift += impulse.drift;
                    desire_impulse = Some(impulse);
                }
            }
            if let Some(ref digest) = graph_adjustment {
                band_energy.above *= digest.multipliers.0;
                band_energy.here *= digest.multipliers.1;
                band_energy.beneath *= digest.multipliers.2;
            }
            if let Some(rt) = self.blackcat.as_ref() {
                band_energy.drift = rt.frac_penalty() as f32;
            }
            validate_band_energy_for_trainer(&band_energy)?;
            let mut roundtable_signal = RoundtableBandSignal::from_schedule(schedule, band_energy);
            if let Some(bridge) = self.gnn_roundtable_bridge.as_ref() {
                roundtable_signal = bridge.publish(roundtable_signal.clone())?;
            }
            self.gnn_last_roundtable_signal = Some(roundtable_signal.clone());
            let mut bands: GradientBands = schedule.split(&grad_output)?;
            let mut weights = self.softlogic.prepare_weights(&band_energy);
            if let Some(ref impulse) = desire_impulse {
                weights.0 *= impulse.multipliers.0;
                weights.1 *= impulse.multipliers.1;
                weights.2 *= impulse.multipliers.2;
            }
            if let Some(ref digest) = graph_adjustment {
                weights.0 *= digest.multipliers.0;
                weights.1 *= digest.multipliers.1;
                weights.2 *= digest.multipliers.2;
            }
            if let Some(f) = self.band_weight_fn {
                let override_weights = f(band_energy);
                weights.0 *= override_weights.0;
                weights.1 *= override_weights.1;
                weights.2 *= override_weights.2;
            }
            validate_band_weights_for_trainer(weights)?;
            let mut spectral_used = false;
            let mut spectral_extra: Option<SpectralAdjustmentMetrics> = None;
            if self.pending_coherence.is_none() {
                if let Some(bridge) = self.coherence_bridge.as_ref() {
                    if let Some(signal) = bridge.drain() {
                        self.pending_coherence = Some(signal);
                    }
                }
            }
            let coherence_snapshot = self.pending_coherence.take();
            let coherence_label = coherence_snapshot
                .as_ref()
                .map(|diagnostics| diagnostics.label());
            let mut spectral_label_code = coherence_label.map(|label| match label {
                CoherenceLabel::Background => 0.0,
                CoherenceLabel::SymmetricPulse => 1.0,
                CoherenceLabel::CascadeImbalance => 2.0,
                CoherenceLabel::DiffuseDrift => 3.0,
            });
            let mut spectral_turnover = None;
            let mut spectral_adjustment: Option<SpectralAdjustment> = None;
            if let Some(policy) = self.spectral_policy.as_mut() {
                spectral_adjustment = policy.observe_signal(
                    coherence_snapshot.as_ref(),
                    self.curvature,
                    &band_energy,
                );
                spectral_turnover = Some(policy.dominant_turnover());
                if let Some(label) = policy.last_coherence_label() {
                    spectral_label_code = Some(match label {
                        CoherenceLabel::Background => 0.0,
                        CoherenceLabel::SymmetricPulse => 1.0,
                        CoherenceLabel::CascadeImbalance => 2.0,
                        CoherenceLabel::DiffuseDrift => 3.0,
                    });
                }
            }
            if let Some(adjustment) = spectral_adjustment {
                weights.0 *= adjustment.band_scale.0;
                weights.1 *= adjustment.band_scale.1;
                weights.2 *= adjustment.band_scale.2;
                if adjustment.lr_multiplier.is_finite()
                    && (adjustment.lr_multiplier - 1.0).abs() > 1.0e-3
                {
                    self.scale_learning_rates(module, adjustment.lr_multiplier)?;
                }
                spectral_extra = Some(adjustment.metrics.clone());
                self.last_spectral_metrics = Some(adjustment.metrics);
                spectral_used = coherence_snapshot.is_some();
            }
            if coherence_snapshot.is_some() && !spectral_used {
                self.pending_coherence = coherence_snapshot;
            }
            validate_band_weights_for_trainer(weights)?;
            bands.scale_inplace(weights.0, weights.1, weights.2)?;
            for band in bands.iter() {
                validate_trainer_tensor("trainer_scaled_band_gradient", band)?;
            }
            let weight_mean = validate_trainer_scalar(
                "trainer_band_weight_mean",
                (weights.0 + weights.1 + weights.2) / 3.0,
            )?;
            let weighted_loss_base = validate_trainer_scalar(
                "trainer_weighted_loss_base",
                step_loss * weight_mean.max(0.0),
            )?;
            let mut weighted_loss = weighted_loss_base;
            let mut extra = HashMap::new();
            insert_tensor_shape_extra(&mut extra, "batch_input", input_shape);
            insert_tensor_shape_extra(&mut extra, "batch_target", target_shape);
            insert_tensor_shape_extra(&mut extra, "batch_prediction", prediction_shape);
            insert_tensor_shape_extra(&mut extra, "batch_loss", loss_shape);
            insert_tensor_shape_extra(&mut extra, "batch_grad_output", grad_output_shape);
            if trace_trainer_steps {
                TensorValueHealth::from_tensor(&input).write_extra(&mut extra, "batch_input");
                TensorValueHealth::from_tensor(&target).write_extra(&mut extra, "batch_target");
                TensorValueHealth::from_tensor(&prediction)
                    .write_extra(&mut extra, "batch_prediction");
                TensorValueHealth::from_tensor(&loss_value).write_extra(&mut extra, "batch_loss");
                TensorValueHealth::from_tensor(&grad_output)
                    .write_extra(&mut extra, "batch_grad_output");
            }
            write_backend_policy_extra(backend_policy, &mut extra);
            write_coherence_repair_extra(coherence_snapshot.as_ref(), &mut extra);
            extra.insert("softlogic_w_above".to_string(), weights.0 as f64);
            extra.insert("softlogic_w_here".to_string(), weights.1 as f64);
            extra.insert("softlogic_w_beneath".to_string(), weights.2 as f64);
            extra.insert(
                "softlogic_inertia".to_string(),
                self.softlogic.last_inertia() as f64,
            );
            let region_scale = self.softlogic.last_region_scale();
            extra.insert(
                "softlogic_region_scale_above".to_string(),
                region_scale.0 as f64,
            );
            extra.insert(
                "softlogic_region_scale_here".to_string(),
                region_scale.1 as f64,
            );
            extra.insert(
                "softlogic_region_scale_beneath".to_string(),
                region_scale.2 as f64,
            );
            let softlogic_config = self.softlogic.config();
            let softlogic_metrics = self.softlogic.adaptive_metrics();
            extra.insert(
                "softlogic_energy_equalize_gain".to_string(),
                softlogic_config.energy_equalize_gain as f64,
            );
            extra.insert(
                "softlogic_energy_equalize_auto".to_string(),
                softlogic_config.energy_equalize_auto as f64,
            );
            extra.insert(
                "softlogic_energy_equalize_gain_eff".to_string(),
                softlogic_metrics.energy_equalize_gain_eff as f64,
            );
            extra.insert(
                "softlogic_energy_equalize_strength".to_string(),
                softlogic_metrics.energy_equalize_strength as f64,
            );
            extra.insert(
                "softlogic_energy_equalize_over".to_string(),
                softlogic_metrics.energy_equalize_over as f64,
            );
            extra.insert(
                "softlogic_energy_equalize_state".to_string(),
                softlogic_metrics.energy_equalize_state as f64,
            );
            extra.insert(
                "softlogic_mean_normalize_gain".to_string(),
                softlogic_config.mean_normalize_gain as f64,
            );
            extra.insert(
                "softlogic_mean_normalize_auto".to_string(),
                softlogic_config.mean_normalize_auto as f64,
            );
            extra.insert(
                "softlogic_mean_normalize_gain_eff".to_string(),
                softlogic_metrics.mean_normalize_gain_eff as f64,
            );
            extra.insert(
                "softlogic_mean_normalize_need".to_string(),
                softlogic_metrics.mean_normalize_need as f64,
            );
            extra.insert(
                "softlogic_mean_normalize_state".to_string(),
                softlogic_metrics.mean_normalize_state as f64,
            );
            extra.insert(
                "softlogic_mean_target".to_string(),
                softlogic_metrics.mean_target as f64,
            );
            if let Some(label) = spectral_label_code {
                extra.insert("spectral_label".to_string(), label);
            }
            if let Some(turnover) = spectral_turnover {
                extra.insert("spectral_turnover".to_string(), turnover as f64);
            }
            if let Some(metrics) = spectral_extra {
                extra.insert(
                    "spectral_lr_scale".to_string(),
                    metrics.absolute_lr_scale as f64,
                );
                extra.insert(
                    "spectral_band_above".to_string(),
                    metrics.band_scale.0 as f64,
                );
                extra.insert(
                    "spectral_band_here".to_string(),
                    metrics.band_scale.1 as f64,
                );
                extra.insert(
                    "spectral_band_beneath".to_string(),
                    metrics.band_scale.2 as f64,
                );
                extra.insert(
                    "spectral_local_lr_above".to_string(),
                    metrics.local_lr.0 as f64,
                );
                extra.insert(
                    "spectral_local_lr_here".to_string(),
                    metrics.local_lr.1 as f64,
                );
                extra.insert(
                    "spectral_local_lr_beneath".to_string(),
                    metrics.local_lr.2 as f64,
                );
                extra.insert("spectral_spin".to_string(), metrics.spin_alignment as f64);
                extra.insert(
                    "spectral_radius".to_string(),
                    metrics.spectral_radius as f64,
                );
                extra.insert(
                    "spectral_entropy".to_string(),
                    metrics.spectral_entropy as f64,
                );
                extra.insert(
                    "spectral_pressure".to_string(),
                    metrics.spectral_pressure as f64,
                );
                extra.insert(
                    "spectral_energy_ratio".to_string(),
                    metrics.energy_ratio as f64,
                );
                if let Some(index) = metrics.sheet_index {
                    extra.insert("spectral_sheet_index".to_string(), index as f64);
                }
                extra.insert(
                    "spectral_sheet_count".to_string(),
                    metrics.sheet_count as f64,
                );
            } else {
                self.last_spectral_metrics = None;
            }
            if let Some(ref impulse) = desire_impulse {
                extra.insert(
                    "desire_roundtable_multiplier_above".to_string(),
                    impulse.multipliers.0 as f64,
                );
                extra.insert(
                    "desire_roundtable_multiplier_here".to_string(),
                    impulse.multipliers.1 as f64,
                );
                extra.insert(
                    "desire_roundtable_multiplier_beneath".to_string(),
                    impulse.multipliers.2 as f64,
                );
                extra.insert("desire_roundtable_drift".to_string(), impulse.drift as f64);
            }
            if let Some(bridge) = self.desire_bridge.as_ref() {
                if let Some(summary) = bridge.drain_summary()? {
                    Self::insert_desire_summary(&mut extra, &summary);
                }
            }
            if let Some(bridge) = self.desire_roundtable_bridge.as_ref() {
                if let Some(summary) = bridge.drain_summary()? {
                    Self::insert_desire_roundtable_summary(&mut extra, &summary);
                    self.last_desire_roundtable_summary = Some(summary);
                }
            }
            #[cfg(feature = "psi")]
            if let Some(bridge) = self.desire_psi_bridge.as_ref() {
                if let Some(summary) = bridge.drain_summary()? {
                    Self::insert_desire_psi_summary(&mut extra, &summary);
                }
            }
            if let Some(ref digest) = graph_adjustment {
                extra.insert("graph_share".to_string(), digest.barycentric[3] as f64);
                extra.insert(
                    "graph_multiplier_above".to_string(),
                    digest.multipliers.0 as f64,
                );
                extra.insert(
                    "graph_multiplier_here".to_string(),
                    digest.multipliers.1 as f64,
                );
                extra.insert(
                    "graph_multiplier_beneath".to_string(),
                    digest.multipliers.2 as f64,
                );
                extra.insert("graph_layers".to_string(), digest.layer_count() as f64);
            }
            module.apply_roundtable_band(&roundtable_signal)?;
            let backward_result = module.backward_bands(&input, &bands);
            module.clear_roundtable_band()?;
            if let Err(error) = backward_result {
                self.zero(module)?;
                return Err(error);
            }
            let gradient_health = Self::collect_gradient_health(module)?;
            gradient_health.write_extra(&mut extra);
            let gradient_error = gradient_health
                .ensure_finite("trainer_gradient_accumulator")
                .err();
            if let Some(error) = gradient_error {
                self.zero(module)?;
                return Err(error);
            }
            if let Some(bridge) = self.graph_bridge.as_ref() {
                self.graph_pending = bridge.digest(&baseline_band_energy)?;
            }
            if let Some(trace) = tensor_trace.as_ref() {
                let step_trace = trace.drain();
                if let Some(expected) = strict_expected_backend {
                    step_trace.validate_expected_backend(expected)?;
                }
                epoch_tensor_backend.accumulate_trace(&step_trace);
                step_trace.write_extra(&mut extra);
            }
            #[cfg(feature = "psychoid")]
            let mut psychoid_events = 0usize;
            #[cfg(feature = "psi")]
            let mut psi_snapshot: Option<PsiReading> = None;
            #[cfg(feature = "psi")]
            {
                let curvature_pos = self
                    .curvature_metrics()
                    .map(|metrics| metrics.curvature.max(0.0))
                    .unwrap_or(0.0);
                if let Some(meter) = self.psi.as_mut() {
                    let grad_l2 = Self::collect_grad_l2(module)?;
                    let act_drift = module.psi_probe().unwrap_or(0.0);
                    let input_snapshot = PsiInput {
                        loss: step_loss.abs(),
                        grad_l2,
                        update_ratio: 0.0,
                        act_drift,
                        attn_entropy: 0.0,
                        band_energy: band_energy.l1() + band_energy.drift.abs(),
                        curvature_pos,
                    };
                    let (reading, events) = meter.update(&input_snapshot);
                    psi_snapshot = Some(reading.clone());
                    hub::set_last_psi(&reading);
                    hub::set_last_psi_events(&events);
                    extra.insert("psi_total".to_string(), reading.total as f64);
                    for (component, value) in reading.breakdown.iter() {
                        let key = format!("psi_{}", component);
                        extra.insert(key, *value as f64);
                    }
                    extra.insert("psi_loss".to_string(), input_snapshot.loss as f64);
                    extra.insert("psi_grad_l2".to_string(), input_snapshot.grad_l2 as f64);
                    extra.insert(
                        "psi_update_ratio".to_string(),
                        input_snapshot.update_ratio as f64,
                    );
                    extra.insert("psi_act_drift".to_string(), input_snapshot.act_drift as f64);
                    extra.insert(
                        "psi_band_energy".to_string(),
                        input_snapshot.band_energy as f64,
                    );
                    extra.insert("psi_events".to_string(), events.len() as f64);
                }
            }
            #[cfg(feature = "psychoid")]
            {
                if let Some(meter) = self.psychoid.as_mut() {
                    if let Some(sample) = module.psychoid_sample(&input, &prediction) {
                        if let Some((reading, events)) = meter.observe(sample) {
                            hub::set_last_psychoid(&reading);
                            if self.psychoid_log {
                                Self::log_psychoid(&reading, &events);
                            }
                            extra.insert("psychoid_cti".to_string(), reading.cti as f64);
                            for (metric, value) in reading.raw.iter() {
                                extra.insert(
                                    format!("psychoid_raw_{}", metric.to_lowercase()),
                                    *value as f64,
                                );
                            }
                            for (metric, value) in reading.z_scores.iter() {
                                extra.insert(
                                    format!("psychoid_z_{}", metric.to_lowercase()),
                                    *value as f64,
                                );
                            }
                            psychoid_events = events.len();
                        }
                    }
                }
            }
            #[cfg(feature = "collapse")]
            if let Some(reading) = psi_snapshot.as_ref() {
                let command = self.collapse.as_mut().map(|driver| driver.update(reading));
                if let Some(command) = command {
                    match &command {
                        DriveCmd::Collapse {
                            grad_scale,
                            max_norm,
                            lr_decay,
                            ..
                        } => {
                            if *grad_scale < 0.999 {
                                self.apply_grad_scale(module, *grad_scale)?;
                            }
                            if let Some(limit) = max_norm {
                                self.clip_grad_global_norm(module, *limit)?;
                            }
                            if let Some(decay) = lr_decay {
                                let factor = (1.0 - decay).clamp(0.1, 1.0);
                                self.optimizer_mul_lr(module, factor)?;
                            }
                        }
                        DriveCmd::Bloom { lr_mul, .. } => {
                            if *lr_mul > 1.0 {
                                self.optimizer_mul_lr(module, *lr_mul)?;
                            }
                        }
                        DriveCmd::None => {}
                    }
                    if !matches!(command, DriveCmd::None) {
                        let loop_signal = hub::get_chrono_loop();
                        hub::set_collapse_pulse(CollapsePulse {
                            step: reading.step,
                            total: reading.total,
                            command,
                            loop_signal,
                        });
                    }
                }
            }
            let psi_total_opt: Option<f32> = {
                #[cfg(feature = "psi")]
                {
                    psi_snapshot.as_ref().map(|reading| reading.total.max(0.0))
                }
                #[cfg(not(feature = "psi"))]
                {
                    None
                }
            };
            let scale_hint = hub::get_softlogic_z().and_then(|feedback| feedback.scale);
            let mut z_feedback =
                self.softlogic
                    .observe(&band_energy, weighted_loss_base, psi_total_opt, scale_hint);
            if z_feedback.elliptic.is_none() {
                if let Some(previous) = hub::get_softlogic_z() {
                    if let Some(sample) = previous.elliptic {
                        z_feedback.elliptic = Some(sample);
                    }
                }
            }
            hub::set_softlogic_z(z_feedback.clone());
            extra.insert("softlogic_z".to_string(), z_feedback.z_signal as f64);
            for event in z_feedback.events.iter() {
                let trimmed = event.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let key = format!("softlogic_event_{}", trimmed.replace(['.', '-'], "_"));
                extra.insert(key, 1.0);
            }
            let mut region_highlight = None;
            if let Some((region_factor, region_descriptor)) = self
                .loss_strategy
                .region_factor(&z_feedback, weighted_loss_base)
            {
                validate_trainer_scalar("trainer_loss_region_factor", region_factor)?;
                let factor = region_factor.max(0.0);
                if factor > 0.0 {
                    weighted_loss =
                        validate_trainer_scalar("trainer_weighted_loss", weighted_loss * factor)?;
                }
                extra.insert("loss_region_factor".to_string(), factor as f64);
                extra.insert(
                    "region_spin_alignment".to_string(),
                    region_descriptor.spin_alignment as f64,
                );
                extra.insert(
                    "region_normalized_radius".to_string(),
                    region_descriptor.normalized_radius as f64,
                );
                self.softlogic
                    .record_region_feedback(region_descriptor, factor);
                region_highlight = Some(region_descriptor);
            } else {
                self.softlogic.clear_region_feedback();
            }
            let region_bundle = self
                .loss_strategy
                .region_reports(region_highlight, steps as u64);
            match region_bundle.snapshot {
                Some(report) => hub::set_region_loss_report(report),
                None => hub::clear_region_loss_report(),
            }
            match region_bundle.trend {
                Some(report) => hub::set_region_loss_trend_report(report),
                None => hub::clear_region_loss_trend_report(),
            }
            match region_bundle.volatility {
                Some(report) => hub::set_region_loss_volatility_report(report),
                None => hub::clear_region_loss_volatility_report(),
            }
            validate_trainer_scalar("trainer_weighted_loss", weighted_loss)?;
            total_loss = validate_trainer_scalar("trainer_total_loss", total_loss + weighted_loss)?;
            extra.insert("loss_weighted_base".to_string(), weighted_loss_base as f64);
            let mut loop_broadcasted = false;
            if let Some(node) = self.distribution.as_mut() {
                let outcome = OutcomeBand::from_weights(
                    band_energy.above,
                    band_energy.here,
                    band_energy.beneath,
                );
                let plan = match outcome {
                    OutcomeBand::Above => schedule.above(),
                    OutcomeBand::Here => schedule.here(),
                    OutcomeBand::Beneath => schedule.beneath(),
                };
                let signature = plan_signature(plan, outcome);
                let script_hint = plan.choice.to_unison_script(plan.kind).replace('\n', "; ");
                if let Some(summary) = node.record_decision(
                    signature,
                    script_hint,
                    plan.kind,
                    outcome,
                    (1.0 / (1.0 + weighted_loss.abs())).clamp(0.0, 1.0),
                    psi_total_opt,
                    (band_energy.above, band_energy.here, band_energy.beneath),
                    band_energy.drift,
                    z_feedback.z_signal,
                ) {
                    if let Some(moderator) = self.blackcat_moderator.as_mut() {
                        let outcome = moderator.ingest(summary.clone());
                        if let Some(proposal) = outcome.proposal {
                            let (accepted, preview) =
                                simulate_proposal_locally(&proposal, &mut self.heur_log);
                            if accepted {
                                self.apply_proposal(&proposal, preview)?;
                            }
                        }
                    } else if let Some(conductor) = self.meta_conductor.as_mut() {
                        if let Some(proposal) = conductor.ingest(summary.clone()) {
                            let (accepted, preview) =
                                simulate_proposal_locally(&proposal, &mut self.heur_log);
                            if accepted {
                                self.apply_proposal(&proposal, preview)?;
                            }
                        }
                    }
                    if let Some(loop_signal) = hub::get_chrono_loop() {
                        let collapse_hint = psi_total_opt
                            .filter(|value| *value > 0.0)
                            .or((summary.mean_psi > 0.0).then_some(summary.mean_psi));
                        let support = (summary.support + summary.mean_score).max(0.1);
                        let envelope = LoopbackEnvelope::new(loop_signal)
                            .with_source(summary.node_id.clone())
                            .with_support(support)
                            .with_collapse_total(collapse_hint)
                            .with_z_signal(Some(z_feedback.z_signal))
                            .with_script_hint(Some(summary.script_hint.clone()));
                        hub::push_loopback_envelope(envelope);
                        loop_broadcasted = true;
                    }
                }
            }
            if !loop_broadcasted {
                if let Some(loop_signal) = hub::get_chrono_loop() {
                    let envelope = LoopbackEnvelope::new(loop_signal)
                        .with_support(1.0)
                        .with_collapse_total(psi_total_opt.filter(|value| *value > 0.0))
                        .with_z_signal(Some(z_feedback.z_signal));
                    hub::push_loopback_envelope(envelope);
                }
            }
            let batch_infusion = self
                .text_infusion
                .clone()
                .filter(|infusion| infusion.every == TextInfusionEvery::Batch);
            if let Some(infusion) = batch_infusion
                .as_ref()
                .filter(|infusion| infusion.mode == TextInfusionMode::Blend)
            {
                module.infuse_text(&infusion.text)?;
            }
            let curvature_summary = if self.curvature_scheduler.is_some() {
                Some(Self::collect_gradient_summary(module)?)
            } else {
                None
            };
            let update_snapshot = if trace_parameter_updates {
                Some(capture_parameter_value_snapshot(module)?)
            } else {
                None
            };
            let optim_step_fallback_lr = self.fallback_learning_rate;
            let optim_step_hyper_lr = self.hyper_learning_rate;
            let optim_step_real_lr = self.real_learning_rate;
            let optim_step_grad_clip = self.grad_clip_max_norm;
            let optim_state_before_step = self.optimizer_state();
            self.step(module)?;
            if let Some(snapshot) = update_snapshot.as_ref() {
                collect_parameter_update_stats(module, snapshot)?.write_extra(&mut extra);
            }
            optim_state_before_step.write_extra(&mut extra);
            self.last_accumulator_sync.write_extra(&mut extra);
            extra.insert(
                "optim_step_fallback_lr".to_string(),
                optim_step_fallback_lr as f64,
            );
            extra.insert(
                "optim_step_hyper_lr".to_string(),
                optim_step_hyper_lr as f64,
            );
            extra.insert(
                "optim_realgrad_enabled".to_string(),
                if optim_step_real_lr.is_some() {
                    1.0
                } else {
                    0.0
                },
            );
            if let Some(rate) = optim_step_real_lr {
                extra.insert("optim_step_real_lr".to_string(), rate as f64);
            }
            if let Some(limit) = optim_step_grad_clip {
                extra.insert("optim_step_grad_clip_max_norm".to_string(), limit as f64);
            }
            if let Some(summary) = curvature_summary {
                if let Some(decision) = self.apply_curvature_scheduler(module, summary)? {
                    extra.insert(
                        "curvature_pressure".to_string(),
                        decision.raw_pressure as f64,
                    );
                    extra.insert(
                        "curvature_pressure_ema".to_string(),
                        decision.smoothed_pressure as f64,
                    );
                    extra.insert("curvature_value".to_string(), decision.curvature as f64);
                    if decision.changed {
                        extra.insert("curvature_adjusted".to_string(), 1.0);
                    }
                    if let Some(scheduler) = self.curvature_scheduler.as_ref() {
                        extra.insert(
                            "curvature_kp".to_string(),
                            scheduler.proportional_gain() as f64,
                        );
                        extra.insert("curvature_step".to_string(), scheduler.step_size() as f64);
                        extra.insert(
                            "curvature_tolerance".to_string(),
                            scheduler.tolerance() as f64,
                        );
                        extra.insert(
                            "curvature_dither_strength".to_string(),
                            scheduler.dither_strength() as f64,
                        );
                        extra.insert(
                            "curvature_dither_period".to_string(),
                            scheduler.dither_period() as f64,
                        );
                        if let Some(var) = scheduler.last_pressure_variance() {
                            extra.insert("curvature_pressure_var".to_string(), var as f64);
                        }
                        if let Some(rel_var) = scheduler.last_pressure_rel_variance() {
                            extra.insert("curvature_pressure_rel_var".to_string(), rel_var as f64);
                        }
                    }
                }
            }
            if let Some(infusion) = batch_infusion
                .as_ref()
                .filter(|infusion| infusion.mode == TextInfusionMode::Separate)
            {
                module.infuse_text(&infusion.text)?;
                self.step(module)?;
            }
            self.zero(module)?;
            steps += 1;

            let phase_label = self
                .spectral_policy
                .as_ref()
                .and_then(|policy| policy.last_coherence_label())
                .or(coherence_label);
            let mut phase_label_change = None;
            if let Some(label) = phase_label {
                if let Some(previous) = self.phase_last_label {
                    if previous != label {
                        phase_label_change = Some((previous, label));
                    }
                }
                self.phase_last_label = Some(label);
            }

            let mut turnover_spike = None;
            if let Some(turnover) = spectral_turnover {
                if let Some(previous) = self.phase_last_turnover {
                    if previous <= self.phase_turnover_spike_threshold
                        && turnover > self.phase_turnover_spike_threshold
                    {
                        turnover_spike = Some((previous, turnover));
                    }
                }
                self.phase_last_turnover = Some(turnover);
            }

            let mut band_shift = None;
            let band_abs = [
                baseline_band_energy.above.abs(),
                baseline_band_energy.here.abs(),
                baseline_band_energy.beneath.abs(),
            ];
            let mut dominant_band = 0usize;
            for idx in 1..band_abs.len() {
                if band_abs[idx] > band_abs[dominant_band] {
                    dominant_band = idx;
                }
            }
            let dominant_band = dominant_band as u8;
            if let Some(previous) = self.phase_last_band {
                if previous != dominant_band {
                    band_shift = Some((previous, dominant_band));
                }
            }
            self.phase_last_band = Some(dominant_band);

            let mut drift_spike = None;
            let drift_abs = band_energy.drift.abs();
            if drift_abs.is_finite() {
                if let Some(previous) = self.phase_last_drift_abs {
                    if previous <= self.phase_drift_spike_threshold
                        && drift_abs > self.phase_drift_spike_threshold
                    {
                        drift_spike = Some((previous, drift_abs));
                    }
                }
                self.phase_last_drift_abs = Some(drift_abs);
            }

            let mut loss_spike = None;
            let loss_value = weighted_loss.abs();
            if loss_value.is_finite() {
                let ema = if let Some(previous) = self.phase_loss_ema {
                    previous + self.phase_loss_ema_alpha * (loss_value - previous)
                } else {
                    loss_value
                };
                self.phase_loss_ema = Some(ema);
                let threshold = ema * (1.0 + self.phase_loss_spike_ratio);
                let spiking = threshold.is_finite() && loss_value > threshold;
                if spiking && !self.phase_loss_spiking {
                    loss_spike = Some((loss_value, ema, threshold));
                }
                self.phase_loss_spiking = spiking;
            }

            let bus = global_registry().event_bus();
            if bus.has_listeners("TrainerPhase") {
                let label_code = |label: CoherenceLabel| -> u8 {
                    match label {
                        CoherenceLabel::Background => 0,
                        CoherenceLabel::SymmetricPulse => 1,
                        CoherenceLabel::CascadeImbalance => 2,
                        CoherenceLabel::DiffuseDrift => 3,
                    }
                };

                if let Some((from, to)) = phase_label_change {
                    bus.publish(&PluginEvent::custom(
                        "TrainerPhase",
                        serde_json::json!({
                            "epoch": epoch,
                            "step": steps,
                            "kind": "label_change",
                            "from": {"code": label_code(from), "label": from.to_string()},
                            "to": {"code": label_code(to), "label": to.to_string()},
                            "turnover": spectral_turnover.map(|v| v as f64),
                            "curvature": self.curvature as f64,
                        }),
                    ));
                }

                if let Some((prev, now)) = turnover_spike {
                    bus.publish(&PluginEvent::custom(
                        "TrainerPhase",
                        serde_json::json!({
                            "epoch": epoch,
                            "step": steps,
                            "kind": "turnover_spike",
                            "turnover_prev": prev as f64,
                            "turnover": now as f64,
                            "threshold": self.phase_turnover_spike_threshold as f64,
                            "label": phase_label.map(|label| label.to_string()),
                            "label_code": phase_label.map(label_code),
                            "curvature": self.curvature as f64,
                        }),
                    ));
                }

                let band_name = |code: u8| -> &'static str {
                    match code {
                        0 => "above",
                        1 => "here",
                        2 => "beneath",
                        _ => "unknown",
                    }
                };

                if let Some((from, to)) = band_shift {
                    bus.publish(&PluginEvent::custom(
                        "TrainerPhase",
                        serde_json::json!({
                            "epoch": epoch,
                            "step": steps,
                            "kind": "band_shift",
                            "from": {"code": from, "band": band_name(from)},
                            "to": {"code": to, "band": band_name(to)},
                            "baseline_energy": {
                                "above": baseline_band_energy.above as f64,
                                "here": baseline_band_energy.here as f64,
                                "beneath": baseline_band_energy.beneath as f64,
                                "drift": baseline_band_energy.drift as f64,
                            },
                            "energy": {
                                "above": band_energy.above as f64,
                                "here": band_energy.here as f64,
                                "beneath": band_energy.beneath as f64,
                                "drift": band_energy.drift as f64,
                            },
                            "turnover": spectral_turnover.map(|v| v as f64),
                            "label": phase_label.map(|label| label.to_string()),
                            "label_code": phase_label.map(label_code),
                            "curvature": self.curvature as f64,
                        }),
                    ));
                }

                if let Some((prev, now)) = drift_spike {
                    bus.publish(&PluginEvent::custom(
                        "TrainerPhase",
                        serde_json::json!({
                            "epoch": epoch,
                            "step": steps,
                            "kind": "drift_spike",
                            "drift": band_energy.drift as f64,
                            "drift_abs_prev": prev as f64,
                            "drift_abs": now as f64,
                            "threshold": self.phase_drift_spike_threshold as f64,
                            "turnover": spectral_turnover.map(|v| v as f64),
                            "label": phase_label.map(|label| label.to_string()),
                            "label_code": phase_label.map(label_code),
                            "curvature": self.curvature as f64,
                        }),
                    ));
                }

                if let Some((loss, ema, threshold)) = loss_spike {
                    bus.publish(&PluginEvent::custom(
                        "TrainerPhase",
                        serde_json::json!({
                            "epoch": epoch,
                            "step": steps,
                            "kind": "loss_spike",
                            "loss_weighted": loss as f64,
                            "loss_ema": ema as f64,
                            "threshold": threshold as f64,
                            "ratio": self.phase_loss_spike_ratio as f64,
                            "turnover": spectral_turnover.map(|v| v as f64),
                            "label": phase_label.map(|label| label.to_string()),
                            "label_code": phase_label.map(label_code),
                            "curvature": self.curvature as f64,
                        }),
                    ));
                }
            }

            let elapsed_ms = if let Some(rt) = self.blackcat.as_ref() {
                rt.elapsed_since_begin()
                    .unwrap_or_else(|| Duration::from_secs_f64(0.0))
                    .as_secs_f64()
                    * 1_000.0
            } else {
                step_start.elapsed().as_secs_f64() * 1_000.0
            };
            extra.insert("band_above".to_string(), band_energy.above as f64);
            extra.insert("band_here".to_string(), band_energy.here as f64);
            extra.insert("band_beneath".to_string(), band_energy.beneath as f64);
            extra.insert("band_drift".to_string(), band_energy.drift as f64);
            extra.insert(
                "band_sheet_confidence".to_string(),
                band_energy.spectral.sheet_confidence as f64,
            );
            extra.insert("band_spin".to_string(), band_energy.spectral.spin as f64);
            extra.insert(
                "band_curvature".to_string(),
                band_energy.spectral.curvature as f64,
            );
            extra.insert(
                "band_mean_energy".to_string(),
                band_energy.spectral.energy as f64,
            );
            self.last_band_energy = Some(band_energy);
            if !step_loss.is_finite() {
                extra.insert("step_loss_non_finite".to_string(), 1.0);
            }
            if !weighted_loss.is_finite() {
                extra.insert("loss_weighted_non_finite".to_string(), 1.0);
            }
            extra.insert("step_loss".to_string(), step_loss as f64);
            extra.insert("loss_weighted".to_string(), weighted_loss as f64);
            #[cfg(feature = "psychoid")]
            {
                extra.insert("psychoid_events".to_string(), psychoid_events as f64);
            }
            let metrics = StepMetrics {
                step_time_ms: elapsed_ms,
                mem_peak_mb: 0.0,
                retry_rate: 0.0,
                extra,
            };
            bus.publish(&PluginEvent::custom(
                "TrainerStep",
                serde_json::json!({
                    "epoch": epoch,
                    "step": steps,
                    "metrics": {
                        "step_time_ms": metrics.step_time_ms,
                        "mem_peak_mb": metrics.mem_peak_mb,
                        "retry_rate": metrics.retry_rate,
                        "extra": &metrics.extra,
                    },
                }),
            ));
            if let Some(ap) = self.autopilot.as_mut() {
                ap.report(&metrics);
            }
            if let Some(rt) = self.blackcat.as_mut() {
                let reward = rt.post_step(&metrics);
                if reward > 0.0 {
                    let plan = schedule.above();
                    let script = plan
                        .choice
                        .to_unison_script(RankKind::TopK)
                        .replace('\n', "; ");
                    let _ = rt.try_adopt_soft(&script, 1, 1, 0.5);
                }
            }
        }
        let stats = EpochStats {
            batches: steps,
            total_loss,
            average_loss: if steps == 0 {
                0.0
            } else {
                total_loss / steps as f32
            },
            tensor_backend: epoch_tensor_backend,
        };
        global_registry()
            .event_bus()
            .publish(&PluginEvent::EpochEnd {
                epoch,
                loss: stats.average_loss,
            });
        Ok(stats)
    }

    /// Evaluates an epoch without accumulating gradients or stepping parameters.
    pub fn evaluate_epoch<M, L, I>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        batches: I,
    ) -> PureResult<EpochStats>
    where
        M: Module + ?Sized,
        L: Loss + ?Sized,
        I: IntoIterator,
        I::Item: IntoBatch,
    {
        module.eval()?;
        let result = (|| {
            let mut total_loss = 0.0f32;
            let mut steps = 0usize;
            let backend_policy = BackendPolicy::from_device_caps(self.planner.device_caps());
            for batch in batches.into_iter() {
                let (input, target) = batch.into_batch()?;
                validate_trainer_tensor("trainer_eval_input", &input)?;
                validate_trainer_tensor("trainer_eval_target", &target)?;
                let _backend_policy_guard = push_backend_policy(backend_policy);
                let prediction = module.forward(&input)?;
                validate_trainer_tensor("trainer_eval_prediction", &prediction)?;
                let loss_value = loss.forward(&prediction, &target)?;
                validate_trainer_tensor("trainer_eval_loss", &loss_value)?;
                let step_loss = trainer_tensor_sum("trainer_eval_step_loss", &loss_value)?;
                total_loss =
                    validate_trainer_scalar("trainer_eval_total_loss", total_loss + step_loss)?;
                steps += 1;
            }
            let average_loss = if steps == 0 {
                0.0
            } else {
                validate_trainer_scalar("trainer_eval_average_loss", total_loss / steps as f32)?
            };
            Ok(EpochStats {
                batches: steps,
                total_loss,
                average_loss,
                tensor_backend: EpochTensorBackendStats::default(),
            })
        })();
        let restore = module.train();
        match (result, restore) {
            (Ok(stats), Ok(())) => Ok(stats),
            (Err(err), _) => Err(err),
            (Ok(_), Err(err)) => Err(err),
        }
    }

    /// Runs a reusable multi-epoch training loop with optional validation.
    pub fn train_epochs<M, L, B>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_batches: &[B],
        validation_batches: Option<&[B]>,
        schedule: &RoundtableSchedule,
        config: TrainingRunConfig,
    ) -> PureResult<TrainingRunReport>
    where
        M: Module + ?Sized,
        L: Loss + ?Sized,
        B: Clone + IntoBatch,
    {
        self.train_epochs_from_factory(
            module,
            loss,
            |epoch_idx| batches_for_epoch(train_batches, config, epoch_idx),
            |_| validation_batches.map(|batches| batches.to_vec()),
            schedule,
            config,
        )
    }

    /// Runs a reusable multi-epoch training loop from [`DataLoader`] inputs.
    pub fn train_epochs_loader<M, L>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        train_loader: &DataLoader,
        validation_loader: Option<&DataLoader>,
        schedule: &RoundtableSchedule,
        config: TrainingRunConfig,
    ) -> PureResult<TrainingRunReport>
    where
        M: Module + ?Sized,
        L: Loss + ?Sized,
    {
        self.train_epochs_from_factory(
            module,
            loss,
            |epoch_idx| loader_for_epoch(train_loader, config, epoch_idx),
            |_| validation_loader.cloned(),
            schedule,
            config,
        )
    }

    fn train_epochs_from_factory<M, L, FT, FV, TI, VI>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        mut train_batches_for_epoch: FT,
        mut validation_batches_for_epoch: FV,
        schedule: &RoundtableSchedule,
        config: TrainingRunConfig,
    ) -> PureResult<TrainingRunReport>
    where
        M: Module + ?Sized,
        L: Loss + ?Sized,
        FT: FnMut(usize) -> TI,
        TI: IntoIterator,
        TI::Item: IntoBatch,
        FV: FnMut(usize) -> Option<VI>,
        VI: IntoIterator,
        VI::Item: IntoBatch,
    {
        let mut history = Vec::with_capacity(config.epochs());
        let mut best_score: Option<f32> = None;
        let mut best_epoch_index = None;
        let mut best_state = None;
        let mut stale_epochs = 0usize;
        let mut stopped_early = false;

        for epoch_idx in 0..config.epochs() {
            let train =
                self.train_epoch(module, loss, train_batches_for_epoch(epoch_idx), schedule)?;
            let validation = match validation_batches_for_epoch(epoch_idx) {
                Some(batches) => Some(self.evaluate_epoch(module, loss, batches)?),
                None => None,
            };
            let score = validation
                .as_ref()
                .map(|stats| stats.average_loss)
                .unwrap_or(train.average_loss);
            let improved = score.is_finite()
                && best_score
                    .map(|best| score + config.min_delta() < best)
                    .unwrap_or(true);
            if improved {
                best_score = Some(score);
                best_epoch_index = Some(history.len());
                if config.restore_best() {
                    best_state = Some(module.state_dict()?);
                }
                stale_epochs = 0;
            } else {
                stale_epochs = stale_epochs.saturating_add(1);
            }

            history.push(TrainingEpochStats {
                epoch: epoch_idx + 1,
                train,
                validation,
                score,
                improved,
            });

            if let Some(patience) = config.validation_patience() {
                if stale_epochs > patience {
                    stopped_early = true;
                    break;
                }
            }
        }

        let restored_best = if config.restore_best() {
            if let Some(state) = best_state.as_ref() {
                module.load_state_dict(state)?;
                true
            } else {
                false
            }
        } else {
            false
        };

        Ok(TrainingRunReport::new(
            history,
            best_epoch_index,
            stopped_early,
            restored_best,
        ))
    }

    fn apply_curvature_scheduler<M: Module + ?Sized>(
        &mut self,
        module: &mut M,
        summary: GradientSummary,
    ) -> PureResult<Option<CurvatureDecision>> {
        let Some(scheduler) = self.curvature_scheduler.as_mut() else {
            return Ok(None);
        };
        let decision = scheduler.observe(summary);
        self.last_curvature_metrics = Some(decision.into());
        if decision.changed {
            Self::retune_hypergrads(module, decision.curvature, self.hyper_learning_rate)?;
            self.curvature = decision.curvature;
            self.spectral_adapter
                .set_curvature_target(decision.curvature);
        }
        Ok(Some(decision))
    }

    fn estimate_device_load(&self) -> f64 {
        let caps = self.planner.device_caps();
        caps.occupancy_score(caps.max_workgroup) as f64
    }

    fn collect_gradient_summary<M: Module + ?Sized>(module: &M) -> PureResult<GradientSummary> {
        let mut accumulator = CurvatureGradientAccumulator::default();
        module.visit_parameters(&mut |param| {
            let mut accounted = false;
            if let Some(tape) = param.hypergrad() {
                accumulator.accumulate(tape.summary());
                accounted = true;
            }
            if let Some(tape) = param.realgrad() {
                accumulator.accumulate(tape.summary());
                accounted = true;
            }
            if !accounted {
                if let Some(grad) = param.gradient() {
                    accumulator.accumulate(GradientSummary::from_slice(grad.data()));
                }
            }
            Ok(())
        })?;
        Ok(accumulator.finish())
    }

    fn collect_gradient_health<M: Module + ?Sized>(module: &M) -> PureResult<GradientHealth> {
        let mut health = GradientHealth::default();
        module.visit_parameters(&mut |param| {
            let mut accounted = false;
            if let Some(tape) = param.hypergrad() {
                health.record_values(tape.gradient());
                accounted = true;
            }
            if let Some(tape) = param.realgrad() {
                health.record_values(tape.gradient());
                accounted = true;
            }
            if !accounted {
                if let Some(grad) = param.gradient() {
                    health.record_values(grad.data());
                }
            }
            Ok(())
        })?;
        Ok(health)
    }

    fn retune_hypergrads<M: Module + ?Sized>(
        module: &mut M,
        curvature: f32,
        learning_rate: f32,
    ) -> PureResult<()> {
        module.visit_parameters_mut(&mut |param| {
            if let Some(tape) = param.hypergrad_mut() {
                tape.retune(curvature, learning_rate)?;
            }
            Ok(())
        })
    }

    fn insert_desire_summary(target: &mut HashMap<String, f64>, summary: &DesireTrainerSummary) {
        target.insert("desire_steps".to_string(), summary.total as f64);
        target.insert(
            "desire_phase_observation".to_string(),
            summary.observation as f64,
        );
        target.insert(
            "desire_phase_injection".to_string(),
            summary.injection as f64,
        );
        target.insert(
            "desire_phase_integration".to_string(),
            summary.integration as f64,
        );
        target.insert(
            "desire_mean_entropy".to_string(),
            summary.mean_entropy as f64,
        );
        target.insert(
            "desire_mean_temperature".to_string(),
            summary.mean_temperature as f64,
        );
        target.insert(
            "desire_mean_penalty".to_string(),
            summary.mean_penalty as f64,
        );
        target.insert("desire_mean_alpha".to_string(), summary.mean_alpha as f64);
        target.insert("desire_mean_beta".to_string(), summary.mean_beta as f64);
        target.insert("desire_mean_gamma".to_string(), summary.mean_gamma as f64);
        target.insert("desire_mean_lambda".to_string(), summary.mean_lambda as f64);
        target.insert("desire_triggers".to_string(), summary.triggers as f64);
        target.insert(
            "desire_trigger_mean_penalty".to_string(),
            summary.trigger_mean_penalty as f64,
        );
        target.insert(
            "desire_trigger_mean_entropy".to_string(),
            summary.trigger_mean_entropy as f64,
        );
        target.insert(
            "desire_trigger_mean_temperature".to_string(),
            summary.trigger_mean_temperature as f64,
        );
        target.insert(
            "desire_trigger_mean_samples".to_string(),
            summary.trigger_mean_samples as f64,
        );
    }

    fn insert_desire_roundtable_summary(
        target: &mut HashMap<String, f64>,
        summary: &DesireRoundtableSummary,
    ) {
        target.insert("desire_roundtable_steps".to_string(), summary.steps as f64);
        target.insert(
            "desire_roundtable_triggers".to_string(),
            summary.triggers as f64,
        );
        target.insert(
            "desire_roundtable_mean_entropy".to_string(),
            summary.mean_entropy as f64,
        );
        target.insert(
            "desire_roundtable_mean_temperature".to_string(),
            summary.mean_temperature as f64,
        );
        target.insert(
            "desire_roundtable_mean_alpha".to_string(),
            summary.mean_alpha as f64,
        );
        target.insert(
            "desire_roundtable_mean_beta".to_string(),
            summary.mean_beta as f64,
        );
        target.insert(
            "desire_roundtable_mean_gamma".to_string(),
            summary.mean_gamma as f64,
        );
        target.insert(
            "desire_roundtable_mean_lambda".to_string(),
            summary.mean_lambda as f64,
        );
        target.insert(
            "desire_roundtable_mean_above".to_string(),
            summary.mean_above as f64,
        );
        target.insert(
            "desire_roundtable_mean_here".to_string(),
            summary.mean_here as f64,
        );
        target.insert(
            "desire_roundtable_mean_beneath".to_string(),
            summary.mean_beneath as f64,
        );
        target.insert(
            "desire_roundtable_mean_drift".to_string(),
            summary.mean_drift as f64,
        );
    }

    #[cfg(feature = "psi")]
    fn insert_desire_psi_summary(target: &mut HashMap<String, f64>, summary: &DesirePsiSummary) {
        target.insert("desire_psi_steps".to_string(), summary.steps as f64);
        target.insert("desire_psi_triggers".to_string(), summary.triggers as f64);
        target.insert("desire_psi_samples".to_string(), summary.psi_samples as f64);
        target.insert(
            "desire_psi_mean_total".to_string(),
            summary.mean_psi_total as f64,
        );
        target.insert(
            "desire_psi_mean_entropy".to_string(),
            summary.mean_entropy as f64,
        );
        target.insert(
            "desire_psi_mean_temperature".to_string(),
            summary.mean_temperature as f64,
        );
        target.insert(
            "desire_psi_mean_penalty".to_string(),
            summary.mean_hypergrad_penalty as f64,
        );
        target.insert(
            "desire_psi_mean_z".to_string(),
            summary.mean_z_signal as f64,
        );
        for (component, mean) in summary.component_means.iter() {
            let key = format!("desire_psi_component_{}_mean", component);
            target.insert(key, *mean as f64);
        }
        for (component, (up, down)) in summary.threshold_crossings.iter() {
            let up_key = format!("desire_psi_threshold_{}_up", component);
            let down_key = format!("desire_psi_threshold_{}_down", component);
            target.insert(up_key, *up as f64);
            target.insert(down_key, *down as f64);
        }
    }

    #[cfg(feature = "psi")]
    fn bootstrap_psi(&mut self, schedule: &RoundtableSchedule) {
        if self.psi.is_some() || !schedule.psi_enabled() {
            return;
        }
        let cfg = PsiConfig::automated(schedule.psi_hint());
        self.psi = Some(PsiMeter::new(cfg));
    }

    #[cfg(feature = "psychoid")]
    fn bootstrap_psychoid(&mut self, schedule: &RoundtableSchedule) {
        if self.psychoid.is_some() || !schedule.psychoid_enabled() {
            return;
        }
        let cfg = PsychoidConfig::default();
        self.psychoid = Some(PsychoidMeter::new(cfg));
        self.psychoid_log = schedule.psychoid_log();
    }

    fn scale_learning_rates<M: Module + ?Sized>(
        &mut self,
        module: &mut M,
        factor: f32,
    ) -> PureResult<()> {
        if !factor.is_finite() || factor <= 0.0 {
            return Ok(());
        }
        let next_fallback_learning_rate =
            Self::checked_scaled_learning_rate(self.fallback_learning_rate, factor)?;
        let next_hyper_learning_rate =
            Self::checked_scaled_learning_rate(self.hyper_learning_rate, factor)?;
        let next_real_learning_rate = self
            .real_learning_rate
            .map(|rate| Self::checked_scaled_learning_rate(rate, factor))
            .transpose()?;
        self.fallback_learning_rate = next_fallback_learning_rate;
        self.hyper_learning_rate = next_hyper_learning_rate;
        if let Some(rate) = self.real_learning_rate.as_mut() {
            *rate = next_real_learning_rate.expect("real learning rate");
        }
        self.spectral_adapter.on_global_scale(factor);
        module.visit_parameters_mut(&mut |param| {
            param.scale_learning_rate(factor);
            Ok(())
        })
    }

    fn checked_scaled_learning_rate(rate: f32, factor: f32) -> PureResult<f32> {
        let next = rate * factor;
        if next <= 0.0 || !next.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: next });
        }
        Ok(next)
    }

    fn map_accumulator_sync_error(error: AccumulatorSyncError) -> TensorError {
        TensorError::BackendFailure {
            backend: "accumulator_synchronizer",
            message: error.to_string(),
        }
    }

    fn synchronize_parameter_accumulators<M: Module + ?Sized>(
        &mut self,
        module: &mut M,
    ) -> PureResult<()> {
        let Some(device) = self.accumulator_synchronizer.as_ref().cloned() else {
            self.last_accumulator_sync = TrainerAccumulatorSyncStats::disabled();
            return Ok(());
        };
        let mut stats = TrainerAccumulatorSyncStats {
            enabled: true,
            rank: device.rank(),
            world_size: device.world_size(),
            buffers: 0,
            values: 0,
        };
        module.visit_parameters_mut(&mut |param| {
            let buffers = param.synchronize_accumulators_with(|gradient| {
                stats.values += gradient.len();
                device
                    .synchronize_accumulators(gradient)
                    .map_err(Self::map_accumulator_sync_error)
            })?;
            stats.buffers += buffers;
            Ok(())
        })?;
        self.last_accumulator_sync = stats;
        Ok(())
    }

    #[cfg(feature = "collapse")]
    fn bootstrap_collapse(&mut self, schedule: &RoundtableSchedule) {
        if self.collapse.is_some() || !schedule.collapse_enabled() {
            return;
        }
        let cfg = CollapseConfig::automated(schedule.psi_hint());
        self.collapse = Some(CollapseDrive::new(cfg));
    }

    fn apply_grad_scale<M: Module + ?Sized>(&self, module: &mut M, scale: f32) -> PureResult<()> {
        if (scale - 1.0).abs() <= f32::EPSILON {
            return Ok(());
        }
        module.visit_parameters_mut(&mut |param| {
            param.scale_accumulators_with_backend_policy(scale)?;
            Ok(())
        })
    }

    fn clip_grad_global_norm<M: Module + ?Sized>(
        &self,
        module: &mut M,
        max_norm: f32,
    ) -> PureResult<()> {
        if max_norm <= 0.0 {
            return Ok(());
        }
        let mut total = 0.0f64;
        module.visit_parameters(&mut |param| {
            total += param.accumulators_norm_sq();
            Ok(())
        })?;
        let norm = total.sqrt() as f32;
        if norm <= max_norm || norm <= f32::EPSILON {
            return Ok(());
        }
        let scale = (max_norm / norm).clamp(0.0, 1.0);
        self.apply_grad_scale(module, scale)
    }

    #[cfg(feature = "collapse")]
    fn optimizer_mul_lr<M: Module + ?Sized>(
        &mut self,
        module: &mut M,
        factor: f32,
    ) -> PureResult<()> {
        self.scale_learning_rates(module, factor)
    }

    #[cfg(feature = "psi")]
    fn collect_grad_l2<M: Module + ?Sized>(module: &M) -> PureResult<f32> {
        let mut sum = 0.0f64;
        module.visit_parameters(&mut |param| {
            if let Some(tape) = param.hypergrad() {
                for &value in tape.gradient().iter() {
                    let v = value as f64;
                    sum += v * v;
                }
            }
            if let Some(tape) = param.realgrad() {
                for &value in tape.gradient().iter() {
                    let v = value as f64;
                    sum += v * v;
                }
            }
            if param.hypergrad().is_none() && param.realgrad().is_none() {
                if let Some(grad) = param.gradient() {
                    for &value in grad.data().iter() {
                        let v = value as f64;
                        sum += v * v;
                    }
                }
            }
            Ok(())
        })?;
        Ok((sum).sqrt() as f32)
    }

    fn apply_proposal(
        &mut self,
        proposal: &GlobalProposal,
        preview_metrics: HashMap<String, f32>,
    ) -> PureResult<()> {
        let _ = preview_metrics;
        let rewrite_ops = proposal
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    op.kind,
                    HeurOpKind::AppendSoft { .. } | HeurOpKind::Retract { .. }
                )
            })
            .count()
            .min(u32::MAX as usize) as u32;
        if let Some(budget) = self.rewrite_budget.as_mut() {
            if !budget.try_consume(rewrite_ops) {
                return Ok(());
            }
        }
        for op in &proposal.ops {
            let mut applied = op.clone();
            applied.issued_at = SystemTime::now();
            self.heur_log.append(applied);
        }
        Ok(())
    }

    #[cfg(feature = "psychoid")]
    fn log_psychoid(reading: &PsychoidReading, events: &[PsychoidEvent]) {
        println!(
            "[psychoid] step={} cti={:.4} raw={{D:{:.3} S:{:.3} C:{:.3} K:{:.3} H:{:.3}}}",
            reading.step,
            reading.cti,
            reading.raw.get("D").copied().unwrap_or(0.0),
            reading.raw.get("S").copied().unwrap_or(0.0),
            reading.raw.get("C").copied().unwrap_or(0.0),
            reading.raw.get("K").copied().unwrap_or(0.0),
            reading.raw.get("H").copied().unwrap_or(0.0)
        );
        for event in events {
            match event {
                PsychoidEvent::DreamPass { step, cti } => {
                    println!("[psychoid-event] step={} dream-pass cti={:.4}", step, cti);
                }
                PsychoidEvent::DreamExport {
                    step,
                    diary,
                    symbols,
                } => {
                    println!(
                        "[psychoid-event] step={} dream-export symbols={:?} diary=\"{}\"",
                        step, symbols, diary
                    );
                }
            }
        }
    }
}

fn outcome_label(outcome: OutcomeBand) -> &'static str {
    match outcome {
        OutcomeBand::Above => "above",
        OutcomeBand::Here => "here",
        OutcomeBand::Beneath => "beneath",
    }
}

fn plan_signature(plan: &st_core::ops::rank_entry::RankPlan, outcome: OutcomeBand) -> String {
    format!(
        "{:?}:{}x{}:k{}:{}",
        plan.kind,
        plan.rows,
        plan.cols,
        plan.k,
        outcome_label(outcome)
    )
}

fn epoch_shuffle_seed(config: TrainingRunConfig, epoch_index: usize) -> Option<u64> {
    config
        .epoch_shuffle_seed()
        .map(|seed| seed.wrapping_add(epoch_index as u64))
}

fn batches_for_epoch<B: Clone>(
    batches: &[B],
    config: TrainingRunConfig,
    epoch_index: usize,
) -> Vec<B> {
    let mut items = batches.to_vec();
    if let Some(seed) = epoch_shuffle_seed(config, epoch_index) {
        let mut rng = StdRng::seed_from_u64(seed);
        items.shuffle(&mut rng);
    }
    items
}

fn loader_for_epoch(
    loader: &DataLoader,
    config: TrainingRunConfig,
    epoch_index: usize,
) -> DataLoader {
    let loader = loader.clone();
    match epoch_shuffle_seed(config, epoch_index) {
        Some(seed) => loader.shuffle(seed),
        None => loader,
    }
}

/// Metrics captured while running [`ModuleTrainer::train_epoch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochStats {
    pub batches: usize,
    pub total_loss: f32,
    pub average_loss: f32,
    pub tensor_backend: EpochTensorBackendStats,
}

/// Configuration for running several training epochs with optional validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingRunConfig {
    epochs: usize,
    validation_patience: Option<usize>,
    min_delta: f32,
    epoch_shuffle_seed: Option<u64>,
    restore_best: bool,
}

impl TrainingRunConfig {
    /// Creates a multi-epoch training configuration.
    pub fn new(epochs: usize) -> Self {
        Self {
            epochs,
            validation_patience: None,
            min_delta: 0.0,
            epoch_shuffle_seed: None,
            restore_best: false,
        }
    }

    /// Sets how many non-improving epochs are tolerated before stopping.
    pub fn with_validation_patience(mut self, patience: Option<usize>) -> Self {
        self.validation_patience = patience;
        self
    }

    /// Sets the minimum score improvement required to reset patience.
    pub fn with_min_delta(mut self, min_delta: f32) -> Self {
        self.min_delta = min_delta.max(0.0);
        self
    }

    /// Sets the seed used to deterministically reshuffle training batches each epoch.
    pub fn with_epoch_shuffle_seed(mut self, seed: Option<u64>) -> Self {
        self.epoch_shuffle_seed = seed;
        self
    }

    /// Restores the best recorded parameter state before returning the report.
    pub fn with_restore_best(mut self, restore_best: bool) -> Self {
        self.restore_best = restore_best;
        self
    }

    /// Returns the number of epochs requested.
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// Returns the optional early-stop patience.
    pub fn validation_patience(&self) -> Option<usize> {
        self.validation_patience
    }

    /// Returns the minimum score improvement required to count as better.
    pub fn min_delta(&self) -> f32 {
        self.min_delta
    }

    /// Returns the seed used to reshuffle training batches per epoch, when enabled.
    pub fn epoch_shuffle_seed(&self) -> Option<u64> {
        self.epoch_shuffle_seed
    }

    /// Returns whether the run should restore the best recorded parameter state.
    pub fn restore_best(&self) -> bool {
        self.restore_best
    }
}

impl Default for TrainingRunConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Metrics captured for one epoch inside a multi-epoch training run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingEpochStats {
    pub epoch: usize,
    pub train: EpochStats,
    pub validation: Option<EpochStats>,
    pub score: f32,
    pub improved: bool,
}

/// Aggregated report emitted by [`ModuleTrainer::train_epochs`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRunReport {
    pub epochs: Vec<TrainingEpochStats>,
    pub best_epoch_index: Option<usize>,
    pub stopped_early: bool,
    pub restored_best: bool,
}

impl TrainingRunReport {
    /// Builds a report from the captured epoch history.
    pub fn new(
        epochs: Vec<TrainingEpochStats>,
        best_epoch_index: Option<usize>,
        stopped_early: bool,
        restored_best: bool,
    ) -> Self {
        Self {
            epochs,
            best_epoch_index,
            stopped_early,
            restored_best,
        }
    }

    /// Returns the best epoch, using validation loss when available.
    pub fn best_epoch(&self) -> Option<&TrainingEpochStats> {
        self.best_epoch_index
            .and_then(|index| self.epochs.get(index))
    }

    /// Returns the last recorded epoch.
    pub fn last_epoch(&self) -> Option<&TrainingEpochStats> {
        self.epochs.last()
    }

    /// Returns the number of epochs that actually ran.
    pub fn epochs_run(&self) -> usize {
        self.epochs.len()
    }
}

#[derive(Default)]
struct CurvatureGradientAccumulator {
    l1: f64,
    sum: f64,
    sum_squares: f64,
    sum_cubes: f64,
    sum_quartic: f64,
    linf: f32,
    count: usize,
    min: f32,
    max: f32,
    positive: usize,
    negative: usize,
    near_zero: usize,
}

impl CurvatureGradientAccumulator {
    fn accumulate(&mut self, summary: GradientSummary) {
        self.l1 += summary.l1() as f64;
        self.sum += summary.sum() as f64;
        self.sum_squares += summary.sum_squares() as f64;
        self.sum_cubes += summary.sum_cubes() as f64;
        self.sum_quartic += summary.sum_quartic() as f64;
        self.linf = self.linf.max(summary.linf());
        self.count += summary.count();
        if self.count == summary.count() {
            self.min = summary.min();
            self.max = summary.max();
        } else {
            self.min = self.min.min(summary.min());
            self.max = self.max.max(summary.max());
        }
        self.positive += summary.positive_count();
        self.negative += summary.negative_count();
        self.near_zero += summary.near_zero_count();
    }

    fn finish(self) -> GradientSummary {
        if self.count == 0 {
            GradientSummary::default()
        } else {
            GradientSummary::from_extended_moments(
                self.l1 as f32,
                self.sum as f32,
                self.sum_squares as f32,
                self.sum_cubes as f32,
                self.sum_quartic as f32,
                self.linf,
                self.count,
            )
            .with_support(
                self.min,
                self.max,
                self.positive,
                self.negative,
                self.near_zero,
            )
        }
    }
}

/// Helper trait that allows [`ModuleTrainer::train_epoch`] to accept both raw
/// `(Tensor, Tensor)` batches and fallible [`PureResult`] batches produced by
/// the [`dataset::DataLoader`] surface.
pub trait IntoBatch {
    fn into_batch(self) -> PureResult<(Tensor, Tensor)>;
}

impl IntoBatch for (Tensor, Tensor) {
    fn into_batch(self) -> PureResult<(Tensor, Tensor)> {
        Ok(self)
    }
}

impl IntoBatch for PureResult<(Tensor, Tensor)> {
    fn into_batch(self) -> PureResult<(Tensor, Tensor)> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::gnn::{GraphActivation, GraphContext, GraphLayerSpec, ZSpaceGraphNetworkBuilder};
    use crate::language::{
        constant, warmup, ConceptHint, DesireAutomation, DesireLagrangian, DesirePipeline,
        DesireTrainerBridge, DesireTriggerBuffer, RepressionField, SemanticBridge, SparseKernel,
        SymbolGeometry, TemperatureController,
    };
    use crate::layers::linear::Linear;
    use crate::layers::sequential::Sequential;
    use crate::layers::wave_gate::WaveGate;
    use crate::loss::MeanSquaredError;
    use crate::module::Parameter;
    use crate::roundtable::RoundtableGnnBridge;
    use crate::roundtable::{HeurOp, HeurOpKind};
    use crate::schedule::RoundtableConfig;
    #[cfg(feature = "golden")]
    use crate::CouncilEvidence;
    use st_core::runtime::autopilot::{AutoConfig, AutoMode};
    use st_core::runtime::blackcat::{bandit::SoftBanditMode, zmeta::ZMetaParams, ChoiceGroups};
    use st_core::telemetry::hub::{SoftlogicEllipticSample, SoftlogicZFeedback};
    use st_core::telemetry::xai::GraphFlowTracer;
    use st_core::telemetry::zspace_region::{ZSpaceRadiusBand, ZSpaceRegionKey, ZSpaceSpinBand};
    use st_core::theory::zpulse::ZSource;
    use st_tensor::topos::OpenCartesianTopos;
    use std::collections::HashMap;
    use std::num::NonZeroUsize;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant, SystemTime};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock available")
    }

    struct StrictGpuOverrideRestore {
        previous: Option<bool>,
    }

    impl Drop for StrictGpuOverrideRestore {
        fn drop(&mut self) {
            let previous = self.previous;
            super::STRICT_GPU_TEST_OVERRIDE.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }

    fn with_strict_gpu_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _lock = env_lock();
        let override_value = value.map(|value| matches!(value, "1" | "true" | "TRUE"));
        let previous = super::STRICT_GPU_TEST_OVERRIDE.with(|slot| slot.replace(override_value));
        let _restore = StrictGpuOverrideRestore { previous };
        f()
    }

    fn build_language_geometry() -> SymbolGeometry {
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

    fn build_language_semantics() -> SemanticBridge {
        use std::collections::HashSet;

        let log_pi = vec![
            vec![(0, (0.65f32).ln()), (1, (0.35f32).ln())],
            vec![(0, (0.4f32).ln()), (1, (0.6f32).ln())],
        ];
        let row = vec![1.0, 1.0];
        let col = vec![1.0, 1.0];
        let anchors = HashSet::new();
        let concept_kernel =
            SparseKernel::from_rows(vec![vec![(0, 1.0)], vec![(1, 1.0)]], 1e-6).unwrap();
        SemanticBridge::new(log_pi, row, col, anchors, 1e-6, concept_kernel).unwrap()
    }

    fn build_language_automation() -> DesireAutomation {
        let geometry = build_language_geometry();
        let repression = RepressionField::new(vec![0.05, 0.15]).unwrap();
        let semantics = build_language_semantics();
        let controller = TemperatureController::new(1.0, 0.8, 0.4, 0.4, 1.6);
        let desire = DesireLagrangian::new(geometry, repression, semantics, controller)
            .unwrap()
            .with_alpha_schedule(warmup(0.0, 0.2, 1))
            .with_beta_schedule(warmup(0.0, 0.1, 1))
            .with_gamma_schedule(constant(0.04))
            .with_lambda_schedule(constant(0.02))
            .with_observation_horizon(Some(1))
            .with_integration_horizon(Some(2));
        let cfg = st_core::config::self_rewrite::SelfRewriteCfg {
            score_thresh: 0.0,
            min_samples: 2,
            cooldown_sec: 0,
        };
        DesireAutomation::new(desire, cfg)
    }

    struct FixedGradientModule {
        param: Parameter,
        grad_value: f32,
    }

    impl FixedGradientModule {
        fn new(grad_value: f32) -> Self {
            let tensor = Tensor::zeros(1, 1).unwrap();
            let param = Parameter::new("weight", tensor);
            Self { param, grad_value }
        }
    }

    struct IdentityModule;

    impl Module for IdentityModule {
        fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
            Ok(input.clone())
        }

        fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
            Ok(grad_output.clone())
        }

        fn visit_parameters(
            &self,
            _visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }

        fn visit_parameters_mut(
            &mut self,
            _visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }
    }

    struct TrainingFlagModule {
        training: bool,
    }

    impl TrainingFlagModule {
        fn new() -> Self {
            Self { training: true }
        }

        fn training(&self) -> bool {
            self.training
        }
    }

    impl Module for TrainingFlagModule {
        fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
            Ok(input.clone())
        }

        fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
            Ok(grad_output.clone())
        }

        fn visit_parameters(
            &self,
            _visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }

        fn visit_parameters_mut(
            &mut self,
            _visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            Ok(())
        }

        fn set_training(&mut self, training: bool) -> PureResult<()> {
            self.training = training;
            Ok(())
        }
    }

    struct ConstantLoss {
        value: f32,
    }

    impl ConstantLoss {
        fn new(value: f32) -> Self {
            Self { value }
        }
    }

    impl Loss for ConstantLoss {
        fn forward(&mut self, prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
            let len = prediction.data().len();
            Tensor::from_vec(
                prediction.shape().0,
                prediction.shape().1,
                vec![self.value; len],
            )
        }

        fn backward(&mut self, prediction: &Tensor, _target: &Tensor) -> PureResult<Tensor> {
            Tensor::zeros(prediction.shape().0, prediction.shape().1)
        }
    }

    impl Module for FixedGradientModule {
        fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
            Ok(input.clone())
        }

        fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
            let update = Tensor::from_vec(1, 1, vec![self.grad_value])?;
            self.param.accumulate_euclidean(&update)?;
            Ok(grad_output.clone())
        }

        fn visit_parameters(
            &self,
            visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            visitor(&self.param)
        }

        fn visit_parameters_mut(
            &mut self,
            visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            visitor(&mut self.param)
        }
    }

    struct NonFiniteAccumulatorModule {
        param: Parameter,
        value: f32,
    }

    impl NonFiniteAccumulatorModule {
        fn new(value: f32) -> Self {
            Self {
                param: Parameter::new("poison", Tensor::zeros(1, 1).unwrap()),
                value,
            }
        }

        fn value(&self) -> &Tensor {
            self.param.value()
        }

        fn gradients_are_zero(&self) -> bool {
            self.param
                .hypergrad()
                .map(|tape| tape.gradient().iter().all(|value| *value == 0.0))
                .unwrap_or(false)
        }
    }

    impl Module for NonFiniteAccumulatorModule {
        fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
            Ok(input.clone())
        }

        fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
            if let Some(tape) = self.param.hypergrad_mut() {
                if let Some(slot) = tape.gradient_mut().first_mut() {
                    *slot = self.value;
                }
            } else {
                let update = Tensor::from_vec(1, 1, vec![self.value])?;
                self.param.accumulate_euclidean(&update)?;
            }
            Ok(grad_output.clone())
        }

        fn visit_parameters(
            &self,
            visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            visitor(&self.param)
        }

        fn visit_parameters_mut(
            &mut self,
            visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            visitor(&mut self.param)
        }
    }

    struct SpectralGradientModule {
        param: Parameter,
        grad: Vec<f32>,
    }

    impl SpectralGradientModule {
        fn new(grad: Vec<f32>) -> Self {
            let cols = grad.len();
            let param = Parameter::new("spectral", Tensor::zeros(1, cols).unwrap());
            Self { param, grad }
        }

        fn accumulate(&mut self) {
            let update = Tensor::from_vec(1, self.grad.len(), self.grad.clone()).unwrap();
            self.param.accumulate_euclidean(&update).unwrap();
        }

        fn weights(&self) -> &Tensor {
            self.param.value()
        }
    }

    impl Module for SpectralGradientModule {
        fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
            Ok(input.clone())
        }

        fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
            let update = Tensor::from_vec(1, self.grad.len(), self.grad.clone()).unwrap();
            self.param.accumulate_euclidean(&update)?;
            Ok(grad_output.clone())
        }

        fn visit_parameters(
            &self,
            visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            visitor(&self.param)
        }

        fn visit_parameters_mut(
            &mut self,
            visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
        ) -> PureResult<()> {
            visitor(&mut self.param)
        }
    }

    #[derive(Debug)]
    struct ScalingTrainingDevice {
        factor: f32,
        calls: Arc<Mutex<usize>>,
        rank: usize,
        world_size: usize,
    }

    impl ScalingTrainingDevice {
        fn new(factor: f32, calls: Arc<Mutex<usize>>) -> Self {
            Self {
                factor,
                calls,
                rank: 1,
                world_size: 2,
            }
        }
    }

    impl AccumulatorSynchronizer for ScalingTrainingDevice {
        fn rank(&self) -> usize {
            self.rank
        }

        fn world_size(&self) -> usize {
            self.world_size
        }

        fn synchronize_accumulators(
            &self,
            gradients: &mut [f32],
        ) -> Result<(), AccumulatorSyncError> {
            *self
                .calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) += 1;
            for value in gradients {
                *value *= self.factor;
            }
            Ok(())
        }
    }

    #[test]
    fn trainer_attaches_and_steps() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut layer = Linear::new("fc", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let input = crate::Tensor::from_vec(1, 2, vec![1.0, -1.0]).unwrap();
        let target = crate::Tensor::from_vec(1, 1, vec![0.5]).unwrap();
        let out = layer.forward(&input).unwrap();
        let grad = out.sub(&target).unwrap();
        let _ = layer.backward(&input, &grad).unwrap();
        trainer.step(&mut layer).unwrap();
        assert!(trainer.planner().topk(64, 128, 32).k > 0);
    }

    #[test]
    fn trainer_rejects_non_finite_batch_input_before_forward() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = Linear::new("finite_gate_input", 2, 1).unwrap();
        trainer.prepare(&mut model).unwrap();
        let before = model.weight().value().clone();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(1, 2, vec![f32::NAN, 1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )];
        let mut loss = MeanSquaredError::new();

        let err = trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .expect_err("trainer should reject non-finite batch input");

        match err {
            TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "trainer_batch_input");
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(model.weight().value().data(), before.data());
    }

    #[test]
    fn trainer_rejects_loss_sum_overflow_before_backward() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = IdentityModule;
        let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(1, 2, vec![0.0, 0.0]).unwrap(),
            Tensor::from_vec(1, 2, vec![0.0, 0.0]).unwrap(),
        )];
        let mut loss = ConstantLoss::new(f32::MAX);

        let err = trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .expect_err("trainer should reject overflowed step loss");

        match err {
            TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "trainer_step_loss");
                assert!(value.is_infinite());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn trainer_rejects_non_finite_band_weight_before_replay() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        trainer.set_band_weights(|_| (f32::NAN, 1.0, 1.0));
        let mut model = Linear::new("finite_gate_band", 2, 1).unwrap();
        trainer.prepare(&mut model).unwrap();
        let before = model.weight().value().clone();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(1, 2, vec![0.5, -0.25]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.75]).unwrap(),
        )];
        let mut loss = MeanSquaredError::new();

        let err = trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .expect_err("trainer should reject non-finite band weight");

        match err {
            TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "trainer_band_weight_above");
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(model.weight().value().data(), before.data());
    }

    #[test]
    fn trainer_rejects_non_finite_accumulator_before_optimizer_step() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = NonFiniteAccumulatorModule::new(f32::NAN);
        trainer.prepare(&mut model).unwrap();
        let before = model.value().clone();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
        )];
        let mut loss = MeanSquaredError::new();

        let err = trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .expect_err("trainer should reject non-finite accumulators before stepping");

        match err {
            TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "trainer_gradient_accumulator");
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(model.value().data(), before.data());
        assert!(model.gradients_are_zero());
    }

    #[test]
    fn evaluate_epoch_rejects_non_finite_input_and_restores_training_mode() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = TrainingFlagModule::new();
        let dataset = vec![(
            Tensor::from_vec(1, 1, vec![f32::NAN]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )];
        let mut loss = MeanSquaredError::new();

        let err = trainer
            .evaluate_epoch(&mut model, &mut loss, dataset)
            .expect_err("evaluation should reject non-finite inputs");

        match err {
            TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "trainer_eval_input");
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(model.training());
    }

    #[test]
    fn evaluate_epoch_rejects_loss_sum_overflow() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = IdentityModule;
        let dataset = vec![(
            Tensor::from_vec(1, 2, vec![0.0, 0.0]).unwrap(),
            Tensor::from_vec(1, 2, vec![0.0, 0.0]).unwrap(),
        )];
        let mut loss = ConstantLoss::new(f32::MAX);

        let err = trainer
            .evaluate_epoch(&mut model, &mut loss, dataset)
            .expect_err("evaluation should reject overflowed loss sums");

        match err {
            TensorError::NonFiniteValue { label, value } => {
                assert_eq!(label, "trainer_eval_step_loss");
                assert!(value.is_infinite());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn trainer_attaches_realgrad_when_enabled() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01).with_realgrad(0.02);
        let mut layer = Linear::new("fc", 2, 1).unwrap();
        trainer.prepare(&mut layer).unwrap();
        let mut saw_parameter = false;
        layer
            .visit_parameters(&mut |param| {
                saw_parameter = true;
                assert!(param.realgrad().is_some());
                Ok(())
            })
            .unwrap();
        assert!(saw_parameter);
    }

    #[test]
    fn trainer_prepares_with_topos_for_wave_gate() {
        let caps = DeviceCaps::wgpu(64, true, 512);
        let mut trainer = ModuleTrainer::new(caps, -0.9, 0.06, 0.02);
        let encoder_curvature = trainer.curvature();
        let topos = OpenCartesianTopos::new(encoder_curvature, 1e-6, 1e4, 512, 16384).unwrap();
        let mut gate = WaveGate::with_topos(
            "wg",
            8,
            st_tensor::LanguageWaveEncoder::new(encoder_curvature, 0.7).unwrap(),
            topos.clone(),
        )
        .unwrap();
        trainer.prepare_with_topos(&mut gate, topos).unwrap();
        let input =
            Tensor::from_vec(1, 8, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]).unwrap();
        let grad_out = gate.forward(&input).unwrap();
        let _ = gate.backward(&input, &grad_out).unwrap();
        trainer.step(&mut gate).unwrap();
        assert!(gate.gate().value().squared_l2_norm() > 0.0);
    }

    #[test]
    fn trainer_prepare_with_topos_attaches_realgrad() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -0.95, 0.05, 0.01).with_realgrad(0.015);
        let mut layer = Linear::new("fc", 3, 2).unwrap();
        let topos = OpenCartesianTopos::new(trainer.curvature(), 1e-6, 1e4, 128, 4096).unwrap();
        trainer.prepare_with_topos(&mut layer, topos).unwrap();
        layer
            .visit_parameters(&mut |param| {
                assert!(param.realgrad().is_some());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn spectral_adapter_scales_local_learning_rate() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let grad = vec![0.8f32, -0.4, 0.6, -0.2];
        let mut module = SpectralGradientModule::new(grad.clone());
        module.accumulate();

        let mut probe = SpectralLrAdapter::default().with_sheet_hint(8);
        probe.set_curvature_target(trainer.curvature());
        let features = st_core::ops::zspace_round::SpectralFeatureSample::from_slice(
            &grad,
            probe.sheet_hint(),
        )
        .unwrap();
        let factor = probe.scale_factor("spectral", &features);

        trainer.step(&mut module).unwrap();

        let lr = trainer.fallback_learning_rate();
        for (value, expected_grad) in module.weights().data().iter().zip(grad.iter()) {
            let expected = -lr * factor * expected_grad;
            assert!(
                (value - expected).abs() < 1e-4,
                "value {value} vs expected {expected}"
            );
        }
    }

    #[test]
    fn trainer_optimizer_state_tracks_lr_scale_and_adapter_reset() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01).with_realgrad(0.02);
        trainer.set_grad_clip_max_norm(0.5);
        let mut module = SpectralGradientModule::new(vec![0.8, -0.4, 0.6, -0.2]);
        module.accumulate();

        trainer.step(&mut module).unwrap();
        let before_scale = trainer.optimizer_state();
        assert_eq!(before_scale.epoch, 0);
        assert_eq!(before_scale.real_learning_rate, Some(0.02));
        assert_eq!(before_scale.grad_clip_max_norm, Some(0.5));
        assert!(before_scale.spectral_adapter.avg_energy > 0.0);

        trainer.mul_learning_rate(&mut module, 0.5).unwrap();
        let after_scale = trainer.optimizer_state();
        assert!((after_scale.fallback_learning_rate - 0.005).abs() < 1e-8);
        assert!((after_scale.hyper_learning_rate - 0.025).abs() < 1e-8);
        assert!((after_scale.real_learning_rate.unwrap_or(0.0) - 0.01).abs() < 1e-8);
        assert_eq!(after_scale.spectral_adapter.avg_curvature, 0.0);
        assert_eq!(after_scale.spectral_adapter.avg_spin, 0.0);
        assert_eq!(after_scale.spectral_adapter.avg_energy, 0.0);
    }

    #[test]
    fn trainer_rejects_overflow_lr_scale_without_mutating_state() {
        let caps = DeviceCaps::cpu();
        let mut trainer =
            ModuleTrainer::new(caps, -1.0, f32::MAX, f32::MAX).with_realgrad(f32::MAX);
        let mut module = SpectralGradientModule::new(vec![0.8, -0.4, 0.6, -0.2]);
        let before = trainer.optimizer_state();

        let err = trainer.mul_learning_rate(&mut module, 2.0).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonPositiveLearningRate { rate } if !rate.is_finite()
        ));
        assert_eq!(trainer.optimizer_state(), before);
    }

    #[test]
    fn trainer_synchronizes_accumulators_before_apply_step() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let calls = Arc::new(Mutex::new(0usize));
        trainer.set_training_device(ScalingTrainingDevice::new(0.0, Arc::clone(&calls)));
        let mut module = SpectralGradientModule::new(vec![0.8, -0.4, 0.6, -0.2]);
        module.accumulate();

        trainer.step(&mut module).unwrap();

        assert_eq!(
            *calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            1
        );
        assert!(module.weights().data().iter().all(|value| *value == 0.0));
        let sync = trainer.last_accumulator_sync();
        assert!(sync.enabled);
        assert_eq!(sync.rank, 1);
        assert_eq!(sync.world_size, 2);
        assert_eq!(sync.buffers, 1);
        assert_eq!(sync.values, 4);
        let state = trainer.optimizer_state();
        assert!(state.training_device_enabled);
        assert_eq!(state.training_rank, 1);
        assert_eq!(state.training_world_size, 2);
    }

    #[test]
    fn trainer_rejects_non_finite_synchronized_accumulator_without_updating_weights() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let calls = Arc::new(Mutex::new(0usize));
        trainer.set_training_device(ScalingTrainingDevice::new(
            f32::INFINITY,
            Arc::clone(&calls),
        ));
        let mut module = SpectralGradientModule::new(vec![0.8, -0.4, 0.6, -0.2]);
        module.accumulate();
        let grad_before = module.param.gradient().unwrap().clone();

        let err = trainer.step(&mut module).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "synchronized_accumulator",
                value,
            } if !value.is_finite()
        ));
        assert_eq!(
            *calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            1
        );
        assert!(module.weights().data().iter().all(|value| *value == 0.0));
        assert_eq!(*module.param.gradient().unwrap(), grad_before);
    }

    #[test]
    fn trainer_runs_epoch_with_roundtable_schedule() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.1, 0.05, 0.01);
        let mut model = Sequential::new();
        model.push(Linear::new("lin", 2, 1).unwrap());
        trainer.prepare(&mut model).unwrap();

        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
        ];

        let mut loss = MeanSquaredError::new();
        let stats = trainer
            .train_epoch(&mut model, &mut loss, dataset.clone(), &schedule)
            .unwrap();
        assert_eq!(stats.batches, dataset.len());
        assert!(stats.total_loss.is_finite());

        // Ensure the model parameters changed by running another batch and checking the outputs.
        let input = Tensor::from_vec(1, 2, vec![1.0, 1.0]).unwrap();
        let before = model.forward(&input).unwrap();
        trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();
        let after = model.forward(&input).unwrap();
        assert_ne!(before.data(), after.data());
    }

    #[test]
    fn trainer_train_epochs_tracks_validation_history() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = Sequential::new();
        model.push(Linear::new("fit_lin", 2, 1).unwrap());
        trainer.prepare(&mut model).unwrap();

        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let train_batches = vec![
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
        ];
        let validation_batches = vec![(
            Tensor::from_vec(1, 2, vec![0.5, 0.5]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
        )];

        let mut loss = MeanSquaredError::new();
        let report = trainer
            .train_epochs(
                &mut model,
                &mut loss,
                &train_batches,
                Some(validation_batches.as_slice()),
                &schedule,
                TrainingRunConfig::new(3),
            )
            .unwrap();

        assert_eq!(report.epochs_run(), 3);
        assert!(report.best_epoch_index.is_some());
        assert!(!report.stopped_early);
        assert!(!report.restored_best);
        assert!(report.epochs.iter().all(|epoch| epoch.validation.is_some()));
        assert!(report.best_epoch().expect("best epoch").score.is_finite());
    }

    #[test]
    fn trainer_train_epochs_can_stop_on_validation_patience() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = Sequential::new();
        model.push(Linear::new("early_stop_lin", 2, 1).unwrap());
        trainer.prepare(&mut model).unwrap();

        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let batches = vec![(
            Tensor::from_vec(1, 2, vec![0.25, 0.75]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
        )];
        let mut loss = MeanSquaredError::new();

        let report = trainer
            .train_epochs(
                &mut model,
                &mut loss,
                &batches,
                Some(batches.as_slice()),
                &schedule,
                TrainingRunConfig::new(5)
                    .with_validation_patience(Some(0))
                    .with_min_delta(100.0),
            )
            .unwrap();

        assert_eq!(report.epochs_run(), 2);
        assert_eq!(report.best_epoch_index, Some(0));
        assert!(report.stopped_early);
        assert!(!report.epochs[1].improved);
    }

    #[test]
    fn trainer_train_epochs_can_restore_best_state() {
        let mut schedule_trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 0.05, 0.01);
        let schedule = schedule_trainer.roundtable(1, 1, RoundtableConfig::default());
        let batches = vec![(
            Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )];

        let mut expected_trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 0.05, 0.01);
        let mut expected = FixedGradientModule::new(1.0);
        expected_trainer.prepare(&mut expected).unwrap();
        let mut expected_loss = ConstantLoss::new(1.0);
        expected_trainer
            .train_epoch(
                &mut expected,
                &mut expected_loss,
                batches.clone(),
                &schedule,
            )
            .unwrap();
        let expected_state = expected.state_dict().unwrap();

        let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 0.05, 0.01);
        let mut module = FixedGradientModule::new(1.0);
        trainer.prepare(&mut module).unwrap();
        let mut loss = ConstantLoss::new(1.0);
        let report = trainer
            .train_epochs(
                &mut module,
                &mut loss,
                &batches,
                Some(batches.as_slice()),
                &schedule,
                TrainingRunConfig::new(5)
                    .with_validation_patience(Some(0))
                    .with_min_delta(100.0)
                    .with_restore_best(true),
            )
            .unwrap();

        assert_eq!(report.epochs_run(), 2);
        assert_eq!(report.best_epoch_index, Some(0));
        assert!(report.stopped_early);
        assert!(report.restored_best);
        assert_eq!(module.state_dict().unwrap(), expected_state);
    }

    #[test]
    fn trainer_train_epochs_loader_rebuilds_epoch_batches() {
        let mut trainer = ModuleTrainer::new(DeviceCaps::cpu(), -1.0, 0.05, 0.01);
        let mut model = Sequential::new();
        model.push(Linear::new("loader_fit", 1, 1).unwrap());
        trainer.prepare(&mut model).unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let train = Dataset::from_vec(vec![
            (
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 1, vec![2.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![2.0]).unwrap(),
            ),
        ]);
        let validation = Dataset::from_vec(vec![(
            Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.5]).unwrap(),
        )]);
        let train_loader = train.loader().batched(1);
        let validation_loader = validation.loader().batched(1);
        let mut loss = MeanSquaredError::new();

        let report = trainer
            .train_epochs_loader(
                &mut model,
                &mut loss,
                &train_loader,
                Some(&validation_loader),
                &schedule,
                TrainingRunConfig::new(3)
                    .with_epoch_shuffle_seed(Some(7))
                    .with_restore_best(true),
            )
            .unwrap();

        assert_eq!(report.epochs_run(), 3);
        assert!(report.restored_best);
        assert!(report.epochs.iter().all(|epoch| epoch.train.batches == 3));
        assert!(report
            .epochs
            .iter()
            .all(|epoch| epoch.validation.map(|stats| stats.batches) == Some(1)));
    }

    #[test]
    fn trainer_spectral_metrics_capture_band_snapshot_without_coherence_events() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.1, 0.05, 0.01);
        trainer.enable_spectral_learning_rate(SpectralLearningRatePolicy::default());
        let mut model = Sequential::new();
        model.push(Linear::new("lin", 2, 3).unwrap());
        trainer.prepare(&mut model).unwrap();

        let schedule = trainer.roundtable(
            1,
            3,
            RoundtableConfig {
                top_k: 1,
                mid_k: 1,
                bottom_k: 1,
                here_tolerance: 1e-6,
                ..RoundtableConfig::default()
            },
        );
        let dataset = vec![
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 3, vec![1.0, 0.5, -0.5]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![1.0, -0.5]).unwrap(),
                Tensor::from_vec(1, 3, vec![0.25, -0.75, 0.5]).unwrap(),
            ),
        ];

        let mut loss = MeanSquaredError::new();
        trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();

        let metrics = trainer.spectral_metrics().expect("spectral metrics");
        assert_eq!(metrics.source, TrainerSpectralMetricsSource::BandEnergy);
        assert!(metrics.adjustment.is_none());
        assert_eq!(metrics.turnover, Some(0.0));
        let band = metrics.band_energy.expect("band energy snapshot");
        assert!(band.l1().is_finite());
        assert!(band.l1() > 0.0);
        assert!((0.0..=1.0).contains(&band.spectral.sheet_confidence));
    }

    #[test]
    fn roundtable_applies_autopilot_overrides() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut groups = HashMap::new();
        groups.insert("wg".to_string(), vec!["64".to_string()]);
        groups.insert("here.tile".to_string(), vec!["42".to_string()]);
        groups.insert("beneath.subgroup".to_string(), vec!["false".to_string()]);
        let runtime = BlackCatRuntime::new(
            ZMetaParams::default(),
            ChoiceGroups { groups },
            4,
            SoftBanditMode::TS,
            None,
        );
        let autopilot = Autopilot::new(
            caps,
            AutoConfig {
                feat_dim: 4,
                mode: AutoMode::Auto,
                ..AutoConfig::default()
            },
            runtime,
        );
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01).with_autopilot(autopilot);
        let schedule = trainer.roundtable(8, 16, RoundtableConfig::default());
        assert_eq!(schedule.above().choice.wg, 64);
        assert_eq!(schedule.here().choice.tile, 42);
        assert!(!schedule.beneath().choice.subgroup);
    }

    #[test]
    fn region_loss_strategy_scales_total_loss() {
        let elliptic = SoftlogicEllipticSample {
            curvature_radius: 1.0,
            geodesic_radius: 0.9,
            normalized_radius: 0.9,
            spin_alignment: 0.9,
            sheet_index: 1,
            sheet_position: 0.0,
            normal_bias: 0.0,
            sheet_count: 4,
            topological_sector: 2,
            homology_index: 0,
            rotor_field: [0.0; 3],
            flow_vector: [0.0; 3],
            curvature_tensor: [[0.0; 3]; 3],
            resonance_heat: 0.0,
            noise_density: 0.0,
            quaternion: [0.0; 4],
            rotation: [0.0; 9],
            lie_log: [0.0; 3],
            rotor_transport: [0.0; 3],
        };
        let seed_feedback = SoftlogicZFeedback {
            psi_total: 0.0,
            weighted_loss: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_signal: 0.0,
            scale: None,
            events: Vec::new(),
            attributions: vec![(ZSource::Microlocal, 1.0)],
            elliptic: Some(elliptic),
        };

        let weights = RegionLossWeights::new(1.0).with_override(
            ZSpaceRegionKey::new(ZSpaceSpinBand::Leading, ZSpaceRadiusBand::Edge),
            2.0,
        );
        let config = RegionLossConfig::new(weights);
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01)
            .with_loss_strategy(LossStrategy::Region(config));
        trainer.softlogic.last_feedback = Some(seed_feedback);
        let mut module = IdentityModule;
        trainer.prepare(&mut module).unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
        )];
        let mut loss = ConstantLoss::new(1.5);
        let stats = trainer
            .train_epoch(&mut module, &mut loss, dataset, &schedule)
            .unwrap();
        assert!((stats.total_loss - 3.0).abs() < 1e-6);
    }

    #[test]
    fn adaptive_region_weighting_amplifies_high_loss_regions() {
        let weights = RegionLossWeights::new(1.0);
        let adaptive = AdaptiveRegionWeighting::new()
            .with_learning_rate(1.0)
            .with_min_samples(1)
            .with_bounds(0.5, 8.0);
        let mut config = RegionLossConfig::new(weights)
            .with_adaptive(adaptive)
            .with_history_window(4);

        let elliptic = SoftlogicEllipticSample {
            curvature_radius: 1.0,
            geodesic_radius: 0.9,
            normalized_radius: 0.9,
            spin_alignment: 0.9,
            sheet_index: 1,
            sheet_position: 0.0,
            normal_bias: 0.0,
            sheet_count: 4,
            topological_sector: 2,
            homology_index: 0,
            rotor_field: [0.0; 3],
            flow_vector: [0.0; 3],
            curvature_tensor: [[0.0; 3]; 3],
            resonance_heat: 0.0,
            noise_density: 0.0,
            quaternion: [0.0; 4],
            rotation: [0.0; 9],
            lie_log: [0.0; 3],
            rotor_transport: [0.0; 3],
        };
        let feedback = SoftlogicZFeedback {
            psi_total: 0.0,
            weighted_loss: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_signal: 0.0,
            scale: None,
            events: Vec::new(),
            attributions: vec![(ZSource::Microlocal, 1.0)],
            elliptic: Some(elliptic),
        };

        // Seed a low baseline loss before observing a large spike.
        let _ = config.region_factor(&feedback, 0.1);
        let result = config.region_factor(&feedback, 2.0).unwrap();
        assert!(result.0 > 1.0);
        let first_bundle = config.reports(feedback.region_descriptor(), 0);
        let report = first_bundle
            .snapshot
            .expect("snapshot report should always be present");
        assert_eq!(report.shape(), (3, 3));
        assert!(report.metadata.extras.contains_key("highlight"));
        let second_bundle = config.reports(feedback.region_descriptor(), 1);
        assert!(second_bundle.trend.is_some());
        assert!(second_bundle.volatility.is_some());
    }

    #[test]
    fn trainer_enforces_rewrite_budget() {
        let caps = DeviceCaps::wgpu(16, true, 128);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        trainer.set_rewrite_budget(1, 1);
        if let Some(budget) = trainer.rewrite_budget.as_mut() {
            budget.begin_epoch();
        }

        let op = HeurOp {
            origin: "test".to_string(),
            kind: HeurOpKind::AppendSoft {
                script: "k:topk(2)".to_string(),
                weight: 1.0,
            },
            issued_at: SystemTime::now(),
        };
        let proposal = GlobalProposal {
            proposal_id: "proposal-test".to_string(),
            ops: vec![op],
            evidence: Vec::new(),
        };

        let initial = trainer.heuristics_log().entries().len();
        trainer
            .apply_proposal(&proposal, HashMap::new())
            .expect("first rewrite allowed");
        let after_first = trainer.heuristics_log().entries().len();
        assert_eq!(after_first, initial + proposal.ops.len());

        trainer
            .apply_proposal(&proposal, HashMap::new())
            .expect("second rewrite ignored");
        let after_second = trainer.heuristics_log().entries().len();
        assert_eq!(after_second, after_first);

        if let Some(budget) = trainer.rewrite_budget.as_mut() {
            budget.begin_epoch();
        }
        trainer
            .apply_proposal(&proposal, HashMap::new())
            .expect("rewrite allowed after cooldown");
        let after_third = trainer.heuristics_log().entries().len();
        assert_eq!(after_third, after_first + proposal.ops.len());
    }

    #[test]
    fn trainer_exposes_blackcat_runtime_stats() {
        let caps = DeviceCaps::wgpu(16, true, 128);
        let mut groups = HashMap::new();
        groups.insert("tile".to_string(), vec!["a".to_string(), "b".to_string()]);
        let runtime = BlackCatRuntime::new(
            ZMetaParams::default(),
            ChoiceGroups { groups },
            4,
            SoftBanditMode::TS,
            None,
        );
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01).with_blackcat(runtime);
        if let Some(rt) = trainer.blackcat.as_mut() {
            rt.begin_step();
            let mut extra = HashMap::new();
            extra.insert("grad_norm".into(), 0.4);
            let metrics = StepMetrics {
                step_time_ms: 10.0,
                mem_peak_mb: 256.0,
                retry_rate: 0.05,
                extra,
            };
            let _ = rt.post_step(&metrics);
        }
        let stats = trainer
            .blackcat_runtime_stats()
            .expect("runtime stats available");
        assert_eq!(stats.steps, 1);
        assert!(stats.step_time_ms_ema > 0.0);
        assert_eq!(stats.extras.get("grad_norm").cloned().unwrap(), 0.4);
    }

    #[test]
    fn trainer_step_trace_includes_tensor_backend_counters() {
        let bus = global_registry().event_bus().clone();
        let events = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let captured = events.clone();
        let subscription_id = bus.subscribe(
            "TrainerStep",
            Arc::new(move |event: &PluginEvent| {
                if let Some(payload) = event.downcast_data::<serde_json::Value>() {
                    captured
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(payload.clone());
                }
            }),
        );

        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = Linear::new("trace_linear", 2, 1).unwrap();
        trainer.prepare(&mut model).unwrap();
        let schedule = trainer.roundtable(2, 1, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(2, 2, vec![0.5, -1.0, 1.5, 0.25]).unwrap(),
            Tensor::from_vec(2, 1, vec![0.1, -0.2]).unwrap(),
        )];
        let mut loss = MeanSquaredError::new();
        let result = trainer.train_epoch(&mut model, &mut loss, dataset, &schedule);
        let _ = bus.unsubscribe("TrainerStep", subscription_id);
        result.unwrap();

        let events = events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let event = events
            .iter()
            .rev()
            .find(|event| {
                event
                    .get("metrics")
                    .and_then(|metrics| metrics.get("extra"))
                    .and_then(|extra| extra.as_object())
                    .is_some_and(|extra| {
                        extra
                            .get("tensor_op_mse_loss_forward")
                            .and_then(|value| value.as_f64())
                            .unwrap_or(0.0)
                            >= 1.0
                            && extra
                                .get("batch_input_rows")
                                .and_then(|value| value.as_f64())
                                == Some(2.0)
                            && extra
                                .get("batch_input_values_total")
                                .and_then(|value| value.as_f64())
                                == Some(4.0)
                            && extra
                                .get("tensor_policy_device_cpu")
                                .and_then(|value| value.as_f64())
                                == Some(1.0)
                    })
            })
            .expect("trainer step trace event for this test");
        let extra = event
            .get("metrics")
            .and_then(|metrics| metrics.get("extra"))
            .and_then(|extra| extra.as_object())
            .expect("TrainerStep metrics.extra");
        assert_eq!(
            extra
                .get("batch_input_rows")
                .and_then(|value| value.as_f64()),
            Some(2.0)
        );
        assert_eq!(
            extra
                .get("batch_input_cols")
                .and_then(|value| value.as_f64()),
            Some(2.0)
        );
        assert_eq!(
            extra
                .get("batch_input_values")
                .and_then(|value| value.as_f64()),
            Some(4.0)
        );
        assert_eq!(
            extra
                .get("batch_target_rows")
                .and_then(|value| value.as_f64()),
            Some(2.0)
        );
        assert_eq!(
            extra
                .get("batch_target_cols")
                .and_then(|value| value.as_f64()),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("batch_prediction_rows")
                .and_then(|value| value.as_f64()),
            Some(2.0)
        );
        assert_eq!(
            extra
                .get("batch_prediction_cols")
                .and_then(|value| value.as_f64()),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("batch_loss_values")
                .and_then(|value| value.as_f64()),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("batch_grad_output_values")
                .and_then(|value| value.as_f64()),
            Some(2.0)
        );
        assert_eq!(
            extra
                .get("batch_input_values_total")
                .and_then(|value| value.as_f64()),
            Some(4.0)
        );
        assert_eq!(
            extra
                .get("batch_input_non_finite_values")
                .and_then(|value| value.as_f64()),
            Some(0.0)
        );
        assert_eq!(
            extra
                .get("batch_prediction_values_total")
                .and_then(|value| value.as_f64()),
            Some(2.0)
        );
        assert_eq!(
            extra
                .get("batch_prediction_non_finite_values")
                .and_then(|value| value.as_f64()),
            Some(0.0)
        );
        assert!(
            extra
                .get("batch_prediction_l2_finite")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0
        );
        assert_eq!(
            extra
                .get("batch_loss_non_finite_values")
                .and_then(|value| value.as_f64()),
            Some(0.0)
        );
        assert!(
            extra
                .get("batch_grad_output_l2_finite")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0
        );
        assert_eq!(
            extra
                .get("optim_param_update_params")
                .and_then(|value| value.as_f64()),
            Some(2.0)
        );
        assert_eq!(
            extra
                .get("optim_param_update_values")
                .and_then(|value| value.as_f64()),
            Some(3.0)
        );
        assert!(
            extra
                .get("optim_param_update_l2")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0
        );
        assert!(
            extra
                .get("optim_param_update_ratio_l2")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0
        );
        assert!(
            extra
                .get("optim_param_update_active_params")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                >= 1.0
        );
        assert!(
            extra
                .get("optim_param_update_zero_params")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                <= 1.0
        );
        assert!(
            extra
                .get("optim_param_update_max_l2")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0
        );
        assert!(
            extra
                .get("optim_param_update_max_l2_ratio")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0
        );
        assert!(
            extra
                .get("optim_param_update_max_ratio_l2")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0
        );
        assert!(
            extra
                .get("optim_param_update_max_l2_index")
                .and_then(|value| value.as_f64())
                .unwrap_or(99.0)
                < 2.0
        );
        assert_eq!(
            extra
                .get("optim_param_update_non_finite_values")
                .and_then(|value| value.as_f64()),
            Some(0.0)
        );
        let fallback_lr = extra
            .get("optim_step_fallback_lr")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0);
        assert!((fallback_lr - 0.01).abs() < 1.0e-8);
        let hyper_lr = extra
            .get("optim_step_hyper_lr")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0);
        assert!((hyper_lr - 0.05).abs() < 1.0e-8);
        let state_fallback_lr = extra
            .get("optim_state_fallback_lr")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0);
        assert!((state_fallback_lr - 0.01).abs() < 1.0e-8);
        let state_hyper_lr = extra
            .get("optim_state_hyper_lr")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0);
        assert!((state_hyper_lr - 0.05).abs() < 1.0e-8);
        assert_eq!(
            extra
                .get("optim_state_realgrad_enabled")
                .and_then(|value| value.as_f64()),
            Some(0.0)
        );
        assert_eq!(
            extra
                .get("optim_state_adapter_sheet_hint")
                .and_then(|value| value.as_f64()),
            Some(8.0)
        );
        assert_eq!(
            extra
                .get("optim_accumulator_sync_enabled")
                .and_then(|value| value.as_f64()),
            Some(0.0)
        );
        assert_eq!(
            extra
                .get("optim_accumulator_sync_world_size")
                .and_then(|value| value.as_f64()),
            Some(1.0)
        );
        assert!(
            extra
                .get("tensor_ops_total")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                >= 1.0
        );
        assert!(
            extra
                .get("grad_values_total")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0
        );
        assert_eq!(
            extra
                .get("grad_values_non_finite")
                .and_then(|value| value.as_f64()),
            Some(0.0)
        );
        assert!(
            extra
                .get("tensor_op_matmul_prepacked")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                >= 1.0
        );
        assert!(
            extra
                .get("tensor_op_matmul_prepacked_bias")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                >= 1.0
        );
        assert!(
            extra
                .get("tensor_op_sum_axis0_scaled")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                >= 1.0
        );
        assert!(
            extra
                .get("tensor_op_scale")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                >= 1.0
        );
        assert!(
            extra
                .get("tensor_op_add_scaled")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                >= 1.0
        );
        assert!(
            extra
                .get("tensor_op_mse_loss_forward")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                >= 1.0
        );
        assert!(
            extra
                .get("tensor_op_mse_loss_backward")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                >= 1.0
        );
        assert_eq!(
            extra
                .get("tensor_policy_device_cpu")
                .and_then(|value| value.as_f64()),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_policy_matmul_auto")
                .and_then(|value| value.as_f64()),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_policy_prepacked_matmul_auto")
                .and_then(|value| value.as_f64()),
            Some(1.0)
        );
        assert!(extra
            .keys()
            .any(|key| key.starts_with("tensor_policy_layer_norm_")));
        assert!(extra
            .keys()
            .any(|key| key.starts_with("tensor_policy_attention_")));
        assert!(extra
            .keys()
            .any(|key| key.starts_with("tensor_policy_softmax_")));
        assert!(extra
            .keys()
            .any(|key| key.starts_with("tensor_backend_") && key != "tensor_backend_fallbacks"));
    }

    #[test]
    fn train_epoch_returns_tensor_backend_summary_without_trace_listener() {
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = Linear::new("epoch_trace_linear", 2, 1).unwrap();
        trainer.prepare(&mut model).unwrap();
        let schedule = trainer.roundtable(2, 1, RoundtableConfig::default());
        let dataset = vec![(
            Tensor::from_vec(2, 2, vec![0.25, -0.5, 0.75, 1.0]).unwrap(),
            Tensor::from_vec(2, 1, vec![0.0, 0.5]).unwrap(),
        )];
        let mut loss = MeanSquaredError::new();

        let stats = trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();

        assert_eq!(stats.batches, 1);
        assert!(stats.tensor_backend.ops_total >= 1);
        assert!(stats.tensor_backend.meta_events >= stats.tensor_backend.ops_total);
        assert_eq!(stats.tensor_backend.fallbacks, 0);
        assert!(stats.tensor_backend.backend_cpu >= 2);
        assert!(
            stats.tensor_backend.backend_cpu
                + stats.tensor_backend.backend_cpu_simd
                + stats.tensor_backend.backend_faer
                + stats.tensor_backend.backend_naive
                + stats.tensor_backend.backend_wgpu
                + stats.tensor_backend.backend_cuda
                + stats.tensor_backend.backend_hip
                + stats.tensor_backend.backend_other
                >= 1
        );
    }

    #[test]
    fn tensor_backend_trace_counts_non_finite_metadata_sentinels() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "scaled_dot_attention",
            data: serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "scale": null,
                "kernel": {
                    "diagnostic": "NaN",
                    "fallbacks": ["inf", "ok"],
                },
            }),
        });

        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(extra.get("tensor_meta_events").copied(), Some(1.0));
        assert_eq!(extra.get("tensor_meta_null_values").copied(), Some(1.0));
        assert_eq!(
            extra.get("tensor_meta_non_finite_strings").copied(),
            Some(2.0)
        );
        assert_eq!(
            extra.get("tensor_meta_non_finite_sentinels").copied(),
            Some(3.0)
        );
        assert_eq!(
            extra.get("tensor_meta_non_finite_detected").copied(),
            Some(1.0)
        );
    }

    #[test]
    fn tensor_backend_trace_surfaces_embedding_token_repairs() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "embedding_forward",
            data: serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "tokens": 5,
                "unique_token_indices": 3,
                "repeated_token_indices": 2,
                "non_finite_tokens": 1,
                "rounded_tokens": 1,
                "clamped_low_tokens": 1,
                "clamped_high_tokens": 0,
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "embedding_backward",
            data: serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "tokens": 5,
                "unique_token_indices": 3,
                "repeated_token_indices": 2,
                "non_finite_tokens": 1,
                "rounded_tokens": 1,
                "clamped_low_tokens": 1,
                "clamped_high_tokens": 1,
            }),
        });

        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(extra.get("tensor_embedding_tokens").copied(), Some(10.0));
        assert_eq!(
            extra.get("tensor_embedding_unique_token_indices").copied(),
            Some(6.0)
        );
        assert_eq!(
            extra
                .get("tensor_embedding_repeated_token_indices")
                .copied(),
            Some(4.0)
        );
        assert_eq!(
            extra.get("tensor_embedding_non_finite_tokens").copied(),
            Some(2.0)
        );
        assert_eq!(
            extra.get("tensor_embedding_rounded_tokens").copied(),
            Some(2.0)
        );
        assert_eq!(
            extra.get("tensor_embedding_clamped_low_tokens").copied(),
            Some(2.0)
        );
        assert_eq!(
            extra.get("tensor_embedding_clamped_high_tokens").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra.get("tensor_embedding_token_repairs_total").copied(),
            Some(7.0)
        );
        assert_eq!(
            extra.get("tensor_embedding_token_repair_detected").copied(),
            Some(1.0)
        );
    }

    #[test]
    fn tensor_backend_trace_records_coherence_measurement_fallbacks() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "coherence_measure_phases",
            data: serde_json::json!({
                "backend": "cpu",
                "requested_backend": "webgpu",
                "kind": "coherence_reduction",
                "rows": 2,
                "cols": 128,
                "channels": 2,
                "accelerated_requested": true,
            }),
        });

        let mut epoch = EpochTensorBackendStats::default();
        epoch.accumulate_trace(&trace);
        assert_eq!(epoch.ops_total, 1);
        assert_eq!(epoch.backend_cpu, 1);
        assert_eq!(epoch.fallbacks, 1);

        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(extra.get("tensor_ops_total").copied(), Some(1.0));
        assert_eq!(extra.get("tensor_backend_cpu").copied(), Some(1.0));
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(1.0));
        assert_eq!(
            extra.get("tensor_op_coherence_measure_phases").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_coherence_measure_phases_cpu")
                .copied(),
            Some(1.0)
        );
    }

    #[test]
    fn tensor_backend_trace_surfaces_backend_policy_choices() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "wgpu_heuristic_choice",
            data: serde_json::json!({
                "backend": "wgpu",
                "requested_backend": "wgpu",
                "choice_source": "generated",
                "workgroup": 256,
                "lanes": 32,
                "compaction_tile": 2048,
                "fft_radix": 4,
                "fft_segments": 2,
                "override_count": 1,
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "unison_rank_choice",
            data: serde_json::json!({
                "backend": "wgpu",
                "requested_backend": "wgpu",
                "choice_source": "wgpu_generated",
                "candidate_count": 3,
                "best_score": 0.75,
                "baseline_score": 0.25,
                "wgpu_generated_score": 0.75,
                "wgpu_generated_score_delta": 0.5,
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "kdsl_env_bridge",
            data: serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "status": "feature_disabled",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "kv_consensus_soft_rules",
            data: serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "status": "missing_url",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "wasm_tuner_choice",
            data: serde_json::json!({
                "backend": "wgpu",
                "requested_backend": "auto",
                "status": "hit",
            }),
        });

        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(extra.get("backend_policy_events").copied(), Some(5.0));
        assert_eq!(extra.get("backend_policy_wgpu_choices").copied(), Some(1.0));
        assert_eq!(
            extra.get("backend_policy_unison_choices").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra.get("backend_policy_kdsl_env_events").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra.get("backend_policy_kv_soft_events").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra.get("backend_policy_wasm_tuner_events").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_source_wgpu_heuristic_choice_generated")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_source_unison_rank_choice_wgpu_generated")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_status_kdsl_env_bridge_feature_disabled")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_status_kv_consensus_soft_rules_missing_url")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_status_wasm_tuner_choice_hit")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra.get("backend_policy_wgpu_last_workgroup").copied(),
            Some(256.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_wgpu_last_compaction_tile")
                .copied(),
            Some(2048.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_unison_last_candidate_count")
                .copied(),
            Some(3.0)
        );
        assert_eq!(
            extra.get("backend_policy_unison_last_best_score").copied(),
            Some(0.75)
        );
        assert_eq!(
            extra
                .get("backend_policy_unison_last_wgpu_generated_score_delta")
                .copied(),
            Some(0.5)
        );
    }

    #[test]
    fn tensor_backend_trace_records_backendless_tensor_util_routes_as_policy() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "tensor_util_route",
            data: serde_json::json!({
                "requested_backend": "wgpu",
                "selected_backend": "cpu",
                "status": "cpu_threshold",
                "choice_source": "threshold_guard",
                "values": 8,
                "threshold": 1024,
            }),
        });

        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert!(extra.get("tensor_ops_total").is_none());
        assert_eq!(extra.get("backend_policy_events").copied(), Some(1.0));
        assert_eq!(
            extra.get("backend_policy_tensor_util_routes").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_status_tensor_util_route_cpu_threshold")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_source_tensor_util_route_threshold_guard")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra.get("backend_policy_tensor_util_last_values").copied(),
            Some(8.0)
        );
        assert_eq!(
            extra
                .get("backend_policy_tensor_util_last_threshold")
                .copied(),
            Some(1024.0)
        );
    }

    #[test]
    fn gradient_health_counts_non_finite_entries() {
        let mut health = GradientHealth::default();
        health.record_values(&[1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -2.0]);

        let mut extra = HashMap::new();
        health.write_extra(&mut extra);
        assert_eq!(extra.get("grad_values_total").copied(), Some(5.0));
        assert_eq!(extra.get("grad_values_finite").copied(), Some(2.0));
        assert_eq!(extra.get("grad_values_non_finite").copied(), Some(3.0));
        assert_eq!(extra.get("grad_values_nan").copied(), Some(1.0));
        assert_eq!(extra.get("grad_values_infinite").copied(), Some(2.0));
        assert_eq!(extra.get("grad_non_finite_detected").copied(), Some(1.0));
        assert!((extra["grad_values_non_finite_ratio"] - 0.6).abs() < 1e-6);
        assert!((extra["grad_l2_finite"] - 5.0f64.sqrt()).abs() < 1e-6);
        assert_eq!(extra.get("grad_linf_finite").copied(), Some(2.0));
    }

    #[test]
    fn tensor_value_health_counts_non_finite_entries() {
        let tensor = Tensor::from_vec(
            1,
            5,
            vec![1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -2.0],
        )
        .unwrap();
        let health = TensorValueHealth::from_tensor(&tensor);

        let mut extra = HashMap::new();
        health.write_extra(&mut extra, "sample_tensor");
        assert_eq!(extra.get("sample_tensor_values_total").copied(), Some(5.0));
        assert_eq!(extra.get("sample_tensor_finite_values").copied(), Some(2.0));
        assert_eq!(
            extra.get("sample_tensor_non_finite_values").copied(),
            Some(3.0)
        );
        assert_eq!(extra.get("sample_tensor_nan_values").copied(), Some(1.0));
        assert_eq!(
            extra.get("sample_tensor_infinite_values").copied(),
            Some(2.0)
        );
        assert_eq!(
            extra.get("sample_tensor_non_finite_detected").copied(),
            Some(1.0)
        );
        assert!((extra["sample_tensor_non_finite_ratio"] - 0.6).abs() < 1e-6);
        assert!((extra["sample_tensor_l2_finite"] - 5.0f64.sqrt()).abs() < 1e-6);
        assert_eq!(extra.get("sample_tensor_linf_finite").copied(), Some(2.0));
    }

    #[test]
    fn coherence_repair_trace_writes_trainer_extra_metrics() {
        let signal = CoherenceSignal {
            dominant_channel: Some(1),
            preserved_channels: 4,
            mean_coherence: 0.25,
            z_bias: 0.1,
            energy_ratio: 0.5,
            entropy: 0.2,
            label: CoherenceLabel::DiffuseDrift,
            repaired_non_finite_weights: 2,
            repaired_negative_weights: 1,
            pre_discard_repaired_non_finite: 3,
            pre_discard_repaired_negative: 4,
        };

        let mut extra = HashMap::new();
        write_coherence_repair_extra(Some(&signal), &mut extra);

        assert_eq!(
            extra.get("coherence_repaired_non_finite_weights").copied(),
            Some(2.0)
        );
        assert_eq!(
            extra.get("coherence_repaired_negative_weights").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra.get("coherence_repaired_weights_total").copied(),
            Some(3.0)
        );
        assert_eq!(
            extra
                .get("coherence_pre_discard_repaired_non_finite")
                .copied(),
            Some(3.0)
        );
        assert_eq!(
            extra
                .get("coherence_pre_discard_repaired_negative")
                .copied(),
            Some(4.0)
        );
        assert_eq!(
            extra.get("coherence_pre_discard_repairs_total").copied(),
            Some(7.0)
        );
        assert_eq!(extra.get("coherence_repairs_total").copied(), Some(10.0));
        assert_eq!(extra.get("coherence_repaired_detected").copied(), Some(1.0));
    }

    #[test]
    fn tensor_backend_trace_records_f64_precision_cpu_backend() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "concept_diffusion_state_normalise",
            data: serde_json::json!({
                "backend": "f64_cpu",
                "requested_backend": "auto",
                "precision_backend": "f64_cpu",
                "state_sum_backend": "f64_cpu",
            }),
        });

        let mut epoch = EpochTensorBackendStats::default();
        epoch.accumulate_trace(&trace);
        assert_eq!(epoch.ops_total, 1);
        assert_eq!(epoch.backend_f64_cpu, 1);
        assert_eq!(epoch.backend_other, 0);

        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(extra.get("tensor_backend_f64_cpu").copied(), Some(1.0));
        assert_eq!(
            extra
                .get("tensor_op_backend_concept_diffusion_state_normalise_f64_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_concept_diffusion_state_normalise_precision_f64_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_concept_diffusion_state_normalise_state_sum_f64_cpu")
                .copied(),
            Some(1.0)
        );
    }

    #[test]
    fn tensor_backend_trace_normalizes_kernel_backend_labels() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "layer_norm",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "auto",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("wgpu_dense should satisfy strict WGPU expectation");
        let mut epoch = EpochTensorBackendStats::default();
        epoch.accumulate_trace(&trace);
        assert_eq!(epoch.backend_wgpu, 1);
        assert_eq!(epoch.kernel_backend_wgpu_dense, 1);
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(extra.get("tensor_backend_wgpu").copied(), Some(1.0));
        assert_eq!(
            extra.get("tensor_kernel_backend_wgpu_dense").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra.get("tensor_op_backend_layer_norm_wgpu").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_kernel_backend_layer_norm_wgpu_dense")
                .copied(),
            Some(1.0)
        );
    }

    #[test]
    fn strict_backend_trace_ignores_metadata_only_backends_when_kernel_matches() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "matmul",
            data: serde_json::json!({
                "backend": "wgpu",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "graph_readout",
            data: serde_json::json!({
                "backend": "composite",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "reshape",
            data: serde_json::json!({
                "backend": "view",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "tensor_biome_canopy",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("metadata-only composite/view events should not fail strict WGPU");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(extra.get("tensor_backend_wgpu").copied(), Some(1.0));
        assert_eq!(extra.get("tensor_backend_composite").copied(), Some(1.0));
        assert_eq!(extra.get("tensor_backend_hybrid").copied(), Some(1.0));
        assert_eq!(extra.get("tensor_backend_view").copied(), Some(1.0));
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(0.0));
    }

    #[test]
    fn strict_backend_trace_counts_hybrid_normalization_cpu_as_fallback() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "hadamard",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "layer_norm_backward",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "input_gradient_backend": "hybrid",
                "input_gradient_reduction_backend": "wgpu",
                "normalization_backend": "cpu",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("hybrid metadata has WGPU kernel evidence from the affine reducer");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_op_backend_layer_norm_backward_input_gradient_hybrid")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_layer_norm_backward_input_gradient_reduction_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_layer_norm_backward_normalization_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(1.0));
    }

    #[test]
    fn strict_backend_trace_counts_lstm_recurrent_cpu_sub_backends_as_fallbacks() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "matmul",
            data: serde_json::json!({
                "backend": "wgpu",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "lstm_forward",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "input_projection_backend": "wgpu",
                "bias_backend": "wgpu",
                "recurrent_backend": "wgpu",
                "gate_activation_backend": "cpu",
                "estimated_gate_activation_ops": 64,
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "lstm_backward",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "recurrent_backend": "wgpu",
                "gate_activation_backend": "cpu",
                "bptt_backend": "cpu",
                "bptt_scan_backend": "cpu",
                "bptt_gate_derivative_backend": "cpu",
                "bptt_cell_recurrence_backend": "cpu",
                "bptt_state_carry_backend": "cpu",
                "input_gradient_backend": "wgpu",
                "raw_parameter_gradient_backend": "hybrid",
                "parameter_gradient_reduction_backend": "wgpu",
                "bias_gradient_backend": "wgpu",
                "parameter_gradient_scale_backend": "wgpu",
                "estimated_gate_activation_ops": 64,
                "estimated_bptt_ops": 112,
                "estimated_bptt_gate_derivative_ops": 96,
                "estimated_bptt_cell_recurrence_ops": 32,
                "estimated_bptt_state_carry_ops": 16,
                "estimated_bptt_scan_steps": 4,
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("hybrid LSTM metadata has WGPU projection/scaling evidence");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_forward_recurrent_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_forward_gate_activation_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_backward_bptt_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_backward_bptt_scan_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_backward_bptt_gate_derivative_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_backward_bptt_cell_recurrence_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_backward_bptt_state_carry_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_backward_input_gradient_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_backward_parameter_gradient_reduction_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_lstm_backward_bias_gradient_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("lstm_forward_estimated_gate_activation_ops")
                .copied(),
            Some(64.0)
        );
        assert_eq!(
            extra
                .get("lstm_backward_estimated_gate_activation_ops")
                .copied(),
            Some(64.0)
        );
        assert_eq!(
            extra
                .get("lstm_forward_estimated_gate_activation_wgpu_ops")
                .copied(),
            Some(0.0)
        );
        assert_eq!(
            extra
                .get("lstm_backward_estimated_gate_activation_wgpu_ops")
                .copied(),
            Some(0.0)
        );
        assert_eq!(
            extra.get("lstm_backward_estimated_bptt_ops").copied(),
            Some(112.0)
        );
        assert_eq!(
            extra
                .get("lstm_backward_estimated_bptt_gate_derivative_ops")
                .copied(),
            Some(96.0)
        );
        assert_eq!(
            extra
                .get("lstm_backward_estimated_bptt_cell_recurrence_ops")
                .copied(),
            Some(32.0)
        );
        assert_eq!(
            extra
                .get("lstm_backward_estimated_bptt_state_carry_ops")
                .copied(),
            Some(16.0)
        );
        assert_eq!(
            extra
                .get("lstm_backward_estimated_bptt_scan_steps")
                .copied(),
            Some(4.0)
        );
        assert_eq!(
            extra.get("lstm_estimated_gate_activation_ops").copied(),
            Some(128.0)
        );
        assert_eq!(
            extra
                .get("lstm_estimated_gate_activation_cpu_debt_ops")
                .copied(),
            Some(128.0)
        );
        assert_eq!(
            extra
                .get("lstm_estimated_gate_activation_wgpu_ops")
                .copied(),
            Some(0.0)
        );
        assert_eq!(
            extra.get("lstm_estimated_cpu_debt_ops").copied(),
            Some(240.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(6.0));
    }

    #[test]
    fn strict_backend_trace_counts_lstm_wgpu_gate_activation_ops_as_resolved() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "lstm_forward",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "input_projection_backend": "wgpu",
                "bias_backend": "wgpu",
                "recurrent_backend": "wgpu",
                "gate_activation_backend": "wgpu",
                "estimated_gate_activation_ops": 64,
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "lstm_backward",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "recurrent_backend": "wgpu",
                "gate_activation_backend": "wgpu",
                "bptt_backend": "wgpu",
                "bptt_scan_backend": "wgpu",
                "bptt_gate_derivative_backend": "wgpu",
                "bptt_cell_recurrence_backend": "wgpu",
                "bptt_state_carry_backend": "wgpu",
                "input_gradient_backend": "wgpu",
                "raw_parameter_gradient_backend": "hybrid",
                "parameter_gradient_reduction_backend": "wgpu",
                "bias_gradient_backend": "wgpu",
                "parameter_gradient_scale_backend": "wgpu",
                "estimated_gate_activation_ops": 64,
                "estimated_bptt_ops": 112,
                "estimated_bptt_wgpu_ops": 112,
            }),
        });

        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("lstm_forward_estimated_gate_activation_wgpu_ops")
                .copied(),
            Some(64.0)
        );
        assert_eq!(
            extra
                .get("lstm_backward_estimated_gate_activation_wgpu_ops")
                .copied(),
            Some(64.0)
        );
        assert_eq!(
            extra
                .get("lstm_estimated_gate_activation_wgpu_ops")
                .copied(),
            Some(128.0)
        );
        assert_eq!(
            extra
                .get("lstm_estimated_gate_activation_cpu_debt_ops")
                .copied(),
            Some(0.0)
        );
        assert_eq!(
            extra.get("lstm_estimated_bptt_wgpu_ops").copied(),
            Some(112.0)
        );
        assert_eq!(extra.get("lstm_estimated_cpu_debt_ops").copied(), Some(0.0));
    }

    #[test]
    fn strict_backend_trace_counts_zrba_cpu_eigen_sub_backends_as_fallbacks() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "sum_axis0_scaled",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "zrba_cov_head_forward",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "reduction_backend": "wgpu",
                "covariance_centering_backend": "cpu",
                "covariance_accumulation_backend": "wgpu",
                "low_rank_projection_backend": "cpu_eigen",
                "psd_projection_backend": "cpu_eigen",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("Z-RBA covariance reductions provide WGPU evidence");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_op_backend_zrba_cov_head_forward_covariance_centering_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zrba_cov_head_forward_covariance_accumulation_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zrba_cov_head_forward_low_rank_projection_cpu_eigen")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zrba_cov_head_forward_psd_projection_cpu_eigen")
                .copied(),
            Some(1.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(3.0));
    }

    #[test]
    fn strict_backend_trace_counts_topos_rewrite_sub_backend_as_fallback() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "add_scaled",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "tensor_biome_canopy",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "accumulation_backend": "wgpu",
                "normalise_backend": "wgpu",
                "rewrite_backend": "topos_cpu",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("biome canopy tensor utility work provides WGPU evidence");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_op_backend_tensor_biome_canopy_rewrite_topos_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(1.0));
    }

    #[test]
    fn strict_backend_trace_counts_desire_probability_sub_backends_as_wgpu() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "scale",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "desire_softmax",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "softmax_backend": "wgpu",
                "exp_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "desire_normalise",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "sanitize_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("Desire probability normalisation provides WGPU scale evidence");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_op_backend_desire_softmax_softmax_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_desire_softmax_exp_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_desire_softmax_distribution_scale_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_desire_normalise_sanitize_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_desire_normalise_distribution_scale_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(0.0));
    }

    #[test]
    fn strict_backend_trace_counts_semantic_tensor_util_sub_backends_as_wgpu() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "scale",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "zspace_semantic_window",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "window_energy_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "zspace_semantic_distribution_fusion",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "fusion_accumulation_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("semantic distribution scaling provides WGPU scale evidence");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_semantic_window_window_energy_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_semantic_window_distribution_scale_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_semantic_distribution_fusion_fusion_accumulation_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get(
                    "tensor_op_backend_zspace_semantic_distribution_fusion_distribution_scale_wgpu"
                )
                .copied(),
            Some(1.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(0.0));
    }

    #[test]
    fn strict_backend_trace_counts_zspace_layer_component_backends() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "mul_row",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "zspace_mixer_forward",
            data: serde_json::json!({
                "backend": "composite",
                "requested_backend": "wgpu",
                "broadcast_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "zspace_mixer_backward",
            data: serde_json::json!({
                "backend": "composite",
                "requested_backend": "wgpu",
                "broadcast_backend": "wgpu",
                "gradient_reduction_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "zspace_projector_forward",
            data: serde_json::json!({
                "backend": "composite",
                "requested_backend": "wgpu",
                "rewrite_backend": "cpu",
                "projection_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "zspace_projector_backward",
            data: serde_json::json!({
                "backend": "cpu",
                "requested_backend": "wgpu",
                "projection_backend": "cpu",
                "projection_gradient_backend": "cpu",
                "saturation_gradient_backend": "cpu",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect_err("CPU projector backward is still strict-GPU debt");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_mixer_forward_broadcast_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_mixer_backward_broadcast_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_mixer_backward_gradient_reduction_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_projector_forward_rewrite_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_projector_forward_projection_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_projector_backward_projection_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_projector_backward_projection_gradient_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_zspace_projector_backward_saturation_gradient_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(5.0));
    }

    #[test]
    fn strict_backend_trace_counts_activation_runtime_component_backends() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "scale",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "non_liner_forward",
            data: serde_json::json!({
                "backend": "composite",
                "requested_backend": "wgpu",
                "preactivation_backend": "wgpu",
                "activation_backend": "cpu",
                "geometry_backend": "cpu",
                "broadcast_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "dropout_forward",
            data: serde_json::json!({
                "backend": "composite",
                "requested_backend": "wgpu",
                "mask_backend": "wgpu",
                "rng_backend": "cpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "dynamic_field_stochastic_schrodinger_forward",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "deterministic_backend": "wgpu",
                "rng_backend": "cpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "dynamic_field_stochastic_schrodinger_backward",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "input_gradient_backend": "wgpu",
                "gradient_reduction_backend": "wgpu",
                "gradient_scale_backend": "cpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "wave_scan_stack_forward",
            data: serde_json::json!({
                "backend": "composite",
                "requested_backend": "wgpu",
                "merge_backend": "cpu",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("runtime component metadata has WGPU kernel evidence");
        let mut epoch = EpochTensorBackendStats::default();
        epoch.accumulate_trace(&trace);
        assert_eq!(epoch.requested_wgpu_component_hits, 6);
        assert_eq!(epoch.requested_wgpu_component_fallbacks, 4);
        let mut epoch_extra = HashMap::new();
        epoch.write_extra(&mut epoch_extra);
        assert_eq!(
            epoch_extra
                .get("epoch_tensor_backend_requested_wgpu_component_hits")
                .copied(),
            Some(6.0)
        );
        assert_eq!(
            epoch_extra
                .get("epoch_tensor_backend_requested_wgpu_component_fallbacks")
                .copied(),
            Some(4.0)
        );

        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_backend_requested_wgpu_component_hits")
                .copied(),
            Some(6.0)
        );
        assert_eq!(
            extra
                .get("tensor_backend_requested_wgpu_component_fallbacks")
                .copied(),
            Some(4.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_non_liner_forward_preactivation_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get(
                    "tensor_op_backend_requested_wgpu_component_hit_non_liner_forward_preactivation_wgpu"
                )
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_non_liner_forward_activation_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get(
                    "tensor_op_backend_requested_wgpu_component_fallback_non_liner_forward_activation_cpu"
                )
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_non_liner_forward_geometry_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get(
                    "tensor_op_backend_requested_wgpu_component_fallback_non_liner_forward_geometry_cpu"
                )
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_dropout_forward_mask_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_requested_wgpu_component_hit_dropout_forward_mask_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_dropout_forward_rng_cpu")
                .copied(),
            Some(1.0)
        );
        assert!(extra
            .get("tensor_op_backend_requested_wgpu_component_fallback_dropout_forward_rng_cpu")
            .is_none());
        assert_eq!(
            extra
                .get("tensor_op_backend_dynamic_field_stochastic_schrodinger_forward_deterministic_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get(
                    "tensor_op_backend_requested_wgpu_component_hit_dynamic_field_stochastic_schrodinger_forward_deterministic_wgpu"
                )
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_dynamic_field_stochastic_schrodinger_forward_rng_cpu")
                .copied(),
            Some(1.0)
        );
        assert!(
            extra
                .get(
                    "tensor_op_backend_requested_wgpu_component_fallback_dynamic_field_stochastic_schrodinger_forward_rng_cpu"
                )
                .is_none()
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_dynamic_field_stochastic_schrodinger_backward_gradient_scale_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get(
                    "tensor_op_backend_requested_wgpu_component_fallback_dynamic_field_stochastic_schrodinger_backward_gradient_scale_cpu"
                )
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_wave_scan_stack_forward_merge_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get(
                    "tensor_op_backend_requested_wgpu_component_fallback_wave_scan_stack_forward_merge_cpu"
                )
                .copied(),
            Some(1.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(4.0));
    }

    #[test]
    fn tensor_backend_trace_counts_requested_wgpu_runtime_fallbacks() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "matmul",
            data: serde_json::json!({
                "backend": "naive",
                "requested_backend": "wgpu",
                "fallback": {
                    "from": "wgpu",
                    "reason": "runtime_unavailable",
                },
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "matmul_prepacked_bias",
            data: serde_json::json!({
                "backend": "wgpu",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "sum_abs",
            data: serde_json::json!({
                "backend": "cpu",
                "requested_backend": "wgpu",
            }),
        });

        let mut epoch = EpochTensorBackendStats::default();
        epoch.accumulate_trace(&trace);
        assert_eq!(epoch.requested_wgpu_hits, 1);
        assert_eq!(epoch.requested_wgpu_runtime_fallbacks, 1);
        let mut epoch_extra = HashMap::new();
        epoch.write_extra(&mut epoch_extra);
        assert_eq!(
            epoch_extra
                .get("epoch_tensor_backend_requested_wgpu_hits")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            epoch_extra
                .get("epoch_tensor_backend_requested_wgpu_runtime_fallbacks")
                .copied(),
            Some(1.0)
        );

        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra.get("tensor_backend_requested_wgpu_hits").copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_backend_requested_wgpu_runtime_fallbacks")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_requested_wgpu_hit_matmul_prepacked_bias_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_wgpu_runtime_fallback_matmul_naive")
                .copied(),
            Some(1.0)
        );
        assert!(extra
            .get("tensor_op_backend_wgpu_runtime_fallback_sum_abs_cpu")
            .is_none());
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(2.0));
    }

    #[test]
    fn strict_backend_trace_counts_language_semantic_mixed_sub_backends() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "scale",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "semantic_bridge_window_distribution",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "semantic_sparse_scan_backend": "semantic_cpu",
                "semantic_accumulation_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "concept_hint_distribution",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "semantic_inference_backend": "semantic_bridge_window_distribution",
                "semantic_sparse_scan_backend": "semantic_cpu",
                "semantic_sanitize_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });
        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("language semantic distribution scaling provides WGPU evidence");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_op_backend_semantic_bridge_window_distribution_semantic_sparse_scan_semantic_cpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_semantic_bridge_window_distribution_semantic_accumulation_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get(
                    "tensor_op_backend_semantic_bridge_window_distribution_distribution_scale_wgpu"
                )
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get(
                    "tensor_op_backend_concept_hint_distribution_semantic_sparse_scan_semantic_cpu"
                )
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_concept_hint_distribution_semantic_inference_semantic_bridge_window_distribution")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_concept_hint_distribution_semantic_sanitize_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(2.0));
    }

    #[test]
    fn strict_backend_trace_counts_language_probability_sum_sub_backends_as_wgpu() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "scale",
            data: serde_json::json!({
                "backend": "wgpu_dense",
                "requested_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "desire_automation_vector_normalise",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "sanitize_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "gw_marginal_normalise",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "marginal_sum_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "gw_marginal_normalise_in_place",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "marginal_sum_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });
        trace.record(&TensorOpMetaEvent {
            op_name: "sparse_kernel_probability_row",
            data: serde_json::json!({
                "backend": "hybrid",
                "requested_backend": "wgpu",
                "row_sum_backend": "wgpu",
                "distribution_scale_backend": "wgpu",
            }),
        });

        trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect("language probability distribution scaling provides WGPU evidence");
        let mut extra = HashMap::new();
        trace.write_extra(&mut extra);
        assert_eq!(
            extra
                .get("tensor_op_backend_desire_automation_vector_normalise_sanitize_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_desire_automation_vector_normalise_distribution_scale_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_gw_marginal_normalise_marginal_sum_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_gw_marginal_normalise_in_place_marginal_sum_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(
            extra
                .get("tensor_op_backend_sparse_kernel_probability_row_row_sum_wgpu")
                .copied(),
            Some(1.0)
        );
        assert_eq!(extra.get("tensor_backend_fallbacks").copied(), Some(0.0));
    }

    #[test]
    fn strict_backend_trace_rejects_metadata_only_without_kernel_evidence() {
        let mut trace = TensorBackendStepTrace::default();
        trace.record(&TensorOpMetaEvent {
            op_name: "coherence_wave_forward",
            data: serde_json::json!({
                "backend": "composite",
                "requested_backend": "wgpu",
            }),
        });

        let err = trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect_err("metadata-only events are not tensor kernel evidence");
        match err {
            TensorError::BackendFailure { backend, message } => {
                assert_eq!(backend, "wgpu");
                assert!(
                    message.contains("no tensor kernel backend metadata"),
                    "unexpected strict GPU message: {message}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn tensor_backend_trace_rejects_missing_metadata_under_strict_gpu() {
        let trace = TensorBackendStepTrace::default();
        let err = trace
            .validate_expected_backend(BackendKind::Wgpu)
            .expect_err("strict GPU needs tensor backend evidence");

        match err {
            TensorError::BackendFailure { backend, message } => {
                assert_eq!(backend, "wgpu");
                assert!(message.contains("no tensor kernel backend metadata"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn strict_gpu_trainer_rejects_cpu_tensor_backend_fallback() {
        with_strict_gpu_env(Some("1"), || {
            let caps = DeviceCaps::wgpu(32, true, 256);
            let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
            let mut model = Linear::new("strict_linear", 2, 1).unwrap();
            trainer.prepare(&mut model).unwrap();
            let before = model.weight().value().clone();
            let schedule = trainer.roundtable(2, 1, RoundtableConfig::default());
            let dataset = vec![(
                Tensor::from_vec(2, 2, vec![0.5, -1.0, 1.5, 0.25]).unwrap(),
                Tensor::from_vec(2, 1, vec![0.1, -0.2]).unwrap(),
            )];
            let mut loss = MeanSquaredError::new();
            let err = trainer
                .train_epoch(&mut model, &mut loss, dataset, &schedule)
                .expect_err("strict GPU trainer should reject non-GPU tensor backend");

            match err {
                TensorError::BackendFailure { backend, message } => {
                    assert_eq!(backend, "wgpu");
                    assert!(
                        message.contains("SPIRALTORCH_STRICT_GPU")
                            || message.contains("wgpu")
                            || message.contains("WGPU"),
                        "unexpected strict GPU message: {message}"
                    );
                }
                other => panic!("unexpected error: {other:?}"),
            }
            assert_eq!(*model.weight().value(), before);
        });
    }

    #[test]
    fn curvature_scheduler_adjusts_curvature_and_records_metrics() {
        let caps = DeviceCaps::wgpu(8, true, 64);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.1, 0.01);
        let scheduler = CurvatureScheduler::new(-1.0, -2.0, -0.2, 0.01)
            .with_step(0.2)
            .with_tolerance(0.0)
            .with_smoothing(1.0);
        trainer.enable_curvature_scheduler(scheduler);
        let mut module = FixedGradientModule::new(4.0);
        trainer.prepare(&mut module).unwrap();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let input = Tensor::from_vec(1, 1, vec![1.0]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![0.0]).unwrap();
        let dataset = vec![
            (input.clone(), target.clone()),
            (input.clone(), target.clone()),
            (input, target),
        ];
        let mut loss = MeanSquaredError::new();
        trainer
            .train_epoch(&mut module, &mut loss, dataset, &schedule)
            .unwrap();
        assert!(trainer.curvature() != -1.0);
        let metrics = trainer
            .curvature_metrics()
            .expect("curvature metrics recorded");
        assert!(metrics.raw_pressure > 0.0);
        assert_eq!(metrics.curvature, trainer.curvature());
    }

    #[test]
    fn curvature_scheduler_caps_huge_pressure_without_poisoning_state() {
        let mut scheduler = CurvatureScheduler::new(-1.0, -2.0, -0.2, 0.01)
            .with_step(0.2)
            .with_smoothing(1.0);

        let decision = scheduler.observe_pressure(f32::MAX);

        assert!(decision.raw_pressure.is_finite());
        assert!(decision.raw_pressure <= CURVATURE_PRESSURE_MAX);
        assert!(decision.smoothed_pressure.is_finite());
        assert!(scheduler.current().is_finite());
        assert!(scheduler.last_pressure().unwrap().is_finite());
        assert!(scheduler.last_pressure_variance().unwrap().is_finite());
        assert!(scheduler.last_pressure_rel_variance().unwrap().is_finite());
    }

    #[test]
    fn trainer_consumes_desire_bridge_summary() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = Sequential::new();
        model.push(Linear::new("lin", 2, 1).unwrap());
        trainer.prepare(&mut model).unwrap();

        let automation = build_language_automation();
        let bridge = DesireTrainerBridge::new();
        let mut pipeline = DesirePipeline::builder(automation)
            .with_trainer_bridge(&bridge)
            .with_sink(DesireTriggerBuffer::new())
            .build();

        let logits = vec![2.0, 0.4];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let start = Instant::now();
        let anchor = SystemTime::now();
        for step in 0..6 {
            let now = start + Duration::from_millis((step * 150) as u64);
            let timestamp = anchor + Duration::from_millis((step * 150) as u64);
            pipeline
                .step_at(&logits, step % 2, &concept, now, timestamp)
                .unwrap();
        }

        assert!(bridge.len() >= 6);
        trainer.enable_desire_pipeline(bridge.clone());

        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
        ];
        let mut loss = MeanSquaredError::new();
        trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();

        assert!(bridge.is_empty());
    }

    #[test]
    fn trainer_consumes_roundtable_bridge_summary() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = Sequential::new();
        model.push(Linear::new("lin", 2, 1).unwrap());
        trainer.prepare(&mut model).unwrap();

        let automation = build_language_automation();
        let bridge = DesireRoundtableBridge::new();
        let mut pipeline = DesirePipeline::builder(automation)
            .with_roundtable_bridge(&bridge)
            .build();

        let logits = vec![2.0, 0.6];
        let concept = ConceptHint::Distribution(vec![0.55, 0.45]);
        let start = Instant::now();
        let anchor = SystemTime::now();
        for step in 0..6 {
            let now = start + Duration::from_millis((step * 90) as u64);
            let timestamp = anchor + Duration::from_millis((step * 90) as u64);
            pipeline
                .step_at(&logits, step % 2, &concept, now, timestamp)
                .unwrap();
        }

        trainer.enable_desire_roundtable_bridge(bridge.clone());

        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
        ];
        let mut loss = MeanSquaredError::new();
        trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();

        assert!(bridge.drain_summary().unwrap().is_none());
        let summary = trainer.desire_roundtable_summary();
        assert!(summary.is_some());
    }

    #[test]
    fn trainer_enables_desire_bundle() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut model = Sequential::new();
        model.push(Linear::new("lin", 2, 1).unwrap());
        trainer.prepare(&mut model).unwrap();

        let automation = build_language_automation();
        let trainer_bridge = DesireTrainerBridge::new();
        let roundtable_bridge = DesireRoundtableBridge::new();
        let bundle = DesireTelemetryBundle::new()
            .with_trainer_bridge(&trainer_bridge)
            .with_roundtable_bridge(&roundtable_bridge);

        let mut pipeline = DesirePipeline::builder(automation)
            .with_telemetry_bundle(&bundle)
            .with_sink(DesireTriggerBuffer::new())
            .build();

        let logits = vec![2.2, 0.5];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let start = Instant::now();
        let anchor = SystemTime::now();
        for step in 0..6 {
            let now = start + Duration::from_millis((step * 120) as u64);
            let timestamp = anchor + Duration::from_millis((step * 120) as u64);
            pipeline
                .step_at(&logits, step % 2, &concept, now, timestamp)
                .unwrap();
        }

        trainer.enable_desire_telemetry(&bundle);

        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let dataset = vec![
            (
                Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
            ),
            (
                Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
            ),
        ];
        let mut loss = MeanSquaredError::new();
        trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();

        assert!(trainer_bridge.is_empty());
        assert!(roundtable_bridge.is_empty());
        assert!(trainer.desire_roundtable_summary().is_some());
    }

    #[test]
    fn trainer_emits_gnn_roundtable_signal() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let adjacency = Tensor::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let context = GraphContext::from_adjacency(adjacency).unwrap();
        let mut builder =
            ZSpaceGraphNetworkBuilder::new(context, NonZeroUsize::new(2).unwrap(), -1.0, 0.05);
        builder.push_layer(
            GraphLayerSpec::new(NonZeroUsize::new(2).unwrap())
                .with_activation(GraphActivation::Relu),
        );
        let mut model = builder.build("gnn_trainer").unwrap();
        trainer.prepare(&mut model).unwrap();

        let bridge = RoundtableGnnBridge::new();
        trainer.enable_gnn_roundtable_bridge(bridge.clone());

        let schedule = trainer.roundtable(2, 2, RoundtableConfig::default());
        let dataset = vec![
            (
                Tensor::from_vec(2, 2, vec![1.0, 0.0, 0.5, -0.5]).unwrap(),
                Tensor::from_vec(2, 2, vec![0.2, -0.1, 0.3, 0.4]).unwrap(),
            ),
            (
                Tensor::from_vec(2, 2, vec![0.0, 1.0, -0.3, 0.8]).unwrap(),
                Tensor::from_vec(2, 2, vec![0.1, 0.0, -0.2, 0.6]).unwrap(),
            ),
        ];
        let mut loss = MeanSquaredError::new();
        trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();

        assert!(trainer.gnn_roundtable_signal().is_some());
        let latest = bridge.latest().unwrap();
        assert!(latest.is_some());
        assert!(!bridge.is_empty());
    }

    #[test]
    fn trainer_traces_gnn_band_replays_with_labels() {
        let tracer = Arc::new(Mutex::new(GraphFlowTracer::new()));
        let caps = DeviceCaps::cpu();
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let adjacency = Tensor::from_vec(
            4,
            4,
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();
        let context = GraphContext::from_adjacency(adjacency).unwrap();
        let mut builder =
            ZSpaceGraphNetworkBuilder::new(context, NonZeroUsize::new(2).unwrap(), -1.0, 0.05)
                .with_tracer(tracer.clone());
        builder.push_layer(
            GraphLayerSpec::new(NonZeroUsize::new(3).unwrap()).with_aggregation(
                crate::NeighborhoodAggregation::multi_hop_sum(NonZeroUsize::new(2).unwrap())
                    .with_include_self(true)
                    .with_attenuation(0.6),
            ),
        );
        builder.push_layer(
            GraphLayerSpec::new(NonZeroUsize::new(2).unwrap())
                .with_activation(GraphActivation::Relu),
        );
        let mut model = builder.build("gnn_band_trace").unwrap();
        trainer.prepare(&mut model).unwrap();

        let schedule = trainer.roundtable(
            4,
            2,
            RoundtableConfig::default()
                .with_top_k(1)
                .with_mid_k(1)
                .with_bottom_k(1)
                .with_here_tolerance(1e-5),
        );
        let dataset = vec![(
            Tensor::from_vec(4, 2, vec![1.0, 0.25, 0.5, -0.75, -0.5, 1.0, 0.75, -0.25]).unwrap(),
            Tensor::from_vec(4, 2, vec![0.2, -0.1, 0.3, 0.15, -0.15, 0.25, 0.1, -0.2]).unwrap(),
        )];
        let mut loss = MeanSquaredError::new();
        trainer
            .train_epoch(&mut model, &mut loss, dataset, &schedule)
            .unwrap();

        let reports = tracer
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .drain();
        let band_reports: Vec<_> = reports
            .iter()
            .filter_map(|report| {
                report
                    .roundtable
                    .as_ref()
                    .and_then(|trace| trace.band_pass.as_ref().map(|pass| (report, trace, pass)))
            })
            .collect();
        assert_eq!(band_reports.len(), 6);
        let labels: Vec<_> = band_reports
            .iter()
            .map(|(_, _, pass)| pass.band.as_str())
            .collect();
        assert_eq!(labels.iter().filter(|label| **label == "above").count(), 2);
        assert_eq!(labels.iter().filter(|label| **label == "here").count(), 2);
        assert_eq!(
            labels.iter().filter(|label| **label == "beneath").count(),
            2
        );
        assert!(band_reports
            .iter()
            .all(|(_, _, pass)| pass.gradient_l1 > 0.0));
        assert!(band_reports
            .iter()
            .all(|(_, trace, _)| !trace.aggregation.effective_coefficients.is_empty()));
        let mut layer0_coeffs = HashMap::new();
        for (report, trace, pass) in &band_reports {
            if trace.aggregation.effective_coefficients.len() == 3
                && !layer0_coeffs.contains_key(pass.band.as_str())
            {
                layer0_coeffs.insert(
                    pass.band.as_str(),
                    (
                        report.layer.clone(),
                        trace.aggregation.effective_coefficients.clone(),
                    ),
                );
            }
        }
        assert_eq!(layer0_coeffs.len(), 3);
        let above = &layer0_coeffs["above"].1;
        let here = &layer0_coeffs["here"].1;
        let beneath = &layer0_coeffs["beneath"].1;
        assert_ne!(above, here);
        assert_ne!(here, beneath);
        assert_ne!(above, beneath);
    }

    #[cfg(feature = "golden")]
    #[test]
    fn trainer_records_golden_council_snapshot() {
        let caps = DeviceCaps::wgpu(16, true, 128);
        let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let mut pulse = GoldenBlackcatPulse::idle();
        pulse.exploration_drive = 0.7;
        let winner = HeurOp {
            origin: "roundtable-1".to_string(),
            kind: HeurOpKind::AppendSoft {
                script: "k:topk(2)".to_string(),
                weight: 1.2,
            },
            issued_at: SystemTime::now(),
        };
        let snapshot = GoldenCouncilSnapshot {
            epoch: 4,
            high_watermark: 9,
            missing_ranges: Vec::new(),
            winners: vec![winner.clone()],
            evidence: CouncilEvidence {
                band_energy: (1.0, 0.8, 0.6),
                graph_flow: 0.4,
                psi: 0.2,
                geometry: (0.5, 0.3, 0.1),
            },
            exploration_bias: 1.2,
            optimization_bias: 0.95,
            synergy_bias: 1.1,
            reinforcement_bias: 1.05,
            resonance: 0.4,
            stability: 0.86,
            momentum: 0.3,
            divergence: 0.2,
            schedule_hint: (1.0, 0.8, 1.1, 0.9),
            pulse_recap: pulse,
        };
        trainer.record_golden_council(&snapshot);
        let stored = trainer
            .last_golden_council_snapshot()
            .expect("snapshot stored");
        assert!((stored.exploration_bias - 1.2).abs() < 1e-4);
        assert_eq!(stored.schedule_hint.2, 1.1);
        assert_eq!(stored.pulse_recap.exploration_drive, 0.7);
        let winners = trainer.last_council().expect("cloneable snapshot").winners;
        assert_eq!(winners.len(), 1);
        match &winners[0].kind {
            HeurOpKind::AppendSoft { weight, .. } => assert!((*weight - 1.2).abs() < 1e-4),
            other => panic!("unexpected winner {:?}", other),
        }
    }
}
