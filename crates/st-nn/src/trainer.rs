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

use crate::cloud::CloudTargetSummary;
use crate::gnn::spiralk::{GraphConsensusBridge, GraphConsensusDigest};
#[cfg(feature = "golden")]
use crate::golden::{GoldenBlackcatPulse, GoldenCooperativeDirective, GoldenCouncilSnapshot};
#[cfg(feature = "psi")]
use crate::language::{DesirePsiBridge, DesirePsiSummary};
use crate::language::{
    DesireRoundtableBridge, DesireRoundtableSummary, DesireTrainerBridge, DesireTrainerSummary,
};
use crate::loss::Loss;
use crate::module::Module;
use crate::plan::RankPlanner;
use crate::roundtable::{
    simulate_proposal_locally, BlackcatModerator, BlackcatScore, DistConfig, GlobalProposal,
    HeurOpKind, HeurOpLog, MetaConductor, ModeratorMinutes, OutcomeBand, RoundtableNode,
};
use crate::schedule::{BandEnergy, GradientBands, RoundtableConfig, RoundtableSchedule};
use crate::{PureResult, Tensor};
use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::unison_heuristics::RankKind;
use st_core::ecosystem::{
    CloudConnector, ConnectorEvent, DistributionSummary, EcosystemRegistry, MetricSample,
    RankPlanSummary, RoundtableConfigSummary, RoundtableSummary,
};
#[cfg(feature = "collapse")]
use st_core::engine::collapse_drive::{CollapseConfig, CollapseDrive, DriveCmd};
use st_core::ops::rank_entry::RankPlan;
use st_core::runtime::autopilot::Autopilot;
use st_core::runtime::blackcat::{BlackCatRuntime, BlackcatRuntimeStats, StepMetrics};
#[cfg(feature = "collapse")]
use st_core::telemetry::hub::CollapsePulse;
use st_core::telemetry::hub::{self, LoopbackEnvelope, SoftlogicZFeedback};
#[cfg(feature = "psi")]
use st_core::telemetry::psi::{PsiComponent, PsiConfig, PsiInput, PsiMeter, PsiReading};
#[cfg(feature = "psychoid")]
use st_core::telemetry::psychoid::{PsychoidConfig, PsychoidEvent, PsychoidMeter, PsychoidReading};
use st_core::theory::zpulse::ZScale;
use st_tensor::{topos::OpenCartesianTopos, GradientSummary};
use std::collections::HashMap;
use std::env;
use std::time::{Duration, Instant, SystemTime};

/// Adaptive curvature controller that nudges the trainer's hyperbolic geometry
/// towards the pressure window observed in recent gradients.
#[derive(Debug, Clone)]
pub struct CurvatureScheduler {
    min_curvature: f32,
    max_curvature: f32,
    target_pressure: f32,
    tolerance: f32,
    step: f32,
    alpha: f32,
    current: f32,
    ema_pressure: Option<f32>,
}

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
            alpha: 0.2,
            current,
            ema_pressure: None,
        }
    }

    /// Adjusts the maximum step a single observation may move the curvature by.
    pub fn with_step(mut self, step: f32) -> Self {
        if step.is_finite() && step > 0.0 {
            self.step = step;
        }
        self
    }

    /// Adjusts the tolerated pressure band before the curvature is nudged.
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        if tolerance.is_finite() && tolerance >= 0.0 {
            self.tolerance = tolerance;
        }
        self
    }

    /// Adjusts the smoothing factor applied to the pressure EMA.
    pub fn with_smoothing(mut self, alpha: f32) -> Self {
        if alpha.is_finite() && alpha > 0.0 {
            self.alpha = alpha.clamp(0.01, 1.0);
        }
        self
    }

    /// Returns the curvature currently enforced by the scheduler.
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Returns the configured target pressure.
    pub fn target_pressure(&self) -> f32 {
        self.target_pressure
    }

    /// Returns the last smoothed pressure observation, if available.
    pub fn last_pressure(&self) -> Option<f32> {
        self.ema_pressure
    }

    /// Synchronises the scheduler with an externally adjusted curvature and
    /// clears the pressure history so subsequent observations start fresh.
    pub fn sync(&mut self, curvature: f32) {
        self.current = curvature
            .clamp(self.min_curvature, self.max_curvature)
            .min(-1e-6);
        self.ema_pressure = None;
    }

    /// Records a gradient summary returning the raw/smoothed pressure alongside
    /// the curvature chosen for the next step.
    pub fn observe(&mut self, summary: GradientSummary) -> CurvatureDecision {
        let raw_pressure = summary.mean_abs();
        let smoothed = match self.ema_pressure {
            Some(prev) => prev + self.alpha * (raw_pressure - prev),
            None => raw_pressure,
        };
        self.ema_pressure = Some(smoothed);

        let mut curvature = self.current;
        let mut changed = false;
        if smoothed > self.target_pressure + self.tolerance {
            let next = (self.current + self.step).min(self.max_curvature);
            if (next - self.current).abs() > f32::EPSILON {
                curvature = next;
                changed = true;
            }
        } else if smoothed < self.target_pressure - self.tolerance {
            let next = (self.current - self.step).max(self.min_curvature);
            if (next - self.current).abs() > f32::EPSILON {
                curvature = next;
                changed = true;
            }
        }

        if changed {
            self.current = curvature;
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

/// High-level orchestrator that keeps hypergrad, SpiralK, and module updates aligned.
pub struct ModuleTrainer {
    planner: RankPlanner,
    curvature: f32,
    hyper_learning_rate: f32,
    fallback_learning_rate: f32,
    real_learning_rate: Option<f32>,
    blackcat: Option<BlackCatRuntime>,
    blackcat_moderator: Option<BlackcatModerator>,
    autopilot: Option<Autopilot>,
    band_weight_fn: Option<BandWeightFn>,
    injector_enabled: bool,
    distribution: Option<RoundtableNode>,
    meta_conductor: Option<MetaConductor>,
    heur_log: HeurOpLog,
    rewrite_budget: Option<RewriteBudget>,
    softlogic: SoftLogicFlex,
    desire_bridge: Option<DesireTrainerBridge>,
    desire_roundtable_bridge: Option<DesireRoundtableBridge>,
    last_desire_roundtable_summary: Option<DesireRoundtableSummary>,
    #[cfg(feature = "psi")]
    desire_psi_bridge: Option<DesirePsiBridge>,
    graph_bridge: Option<GraphConsensusBridge>,
    graph_pending: Option<GraphConsensusDigest>,
    graph_last_hint: Option<String>,
    curvature_scheduler: Option<CurvatureScheduler>,
    last_curvature_metrics: Option<CurvatureMetrics>,
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

/// Function pointer used to convert band energy into Above/Here/Beneath weights.
pub type BandWeightFn = fn(BandEnergy) -> (f32, f32, f32);

fn append_cloud_targets(metadata: &mut HashMap<String, String>, targets: &[CloudConnector]) {
    CloudTargetSummary::from_targets(targets).extend_map(metadata);
}

#[derive(Debug, Clone)]
struct SoftLogicFlex {
    inertia: f32,
    drift_gain: f32,
    psi_gain: f32,
    loss_gain: f32,
    floor: f32,
    scale_gain: f32,
    last_weights: (f32, f32, f32),
    last_z: f32,
    last_feedback: Option<SoftlogicZFeedback>,
}

impl SoftLogicFlex {
    fn new() -> Self {
        let mut flex = Self {
            inertia: env::var("SPIRAL_SOFTLOGIC_INERTIA")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .map(|v| v.clamp(0.0, 0.95))
                .unwrap_or(0.65),
            drift_gain: env::var("SPIRAL_SOFTLOGIC_DRIFT_GAIN")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .map(|v| v.clamp(0.0, 1.0))
                .unwrap_or(0.25),
            psi_gain: env::var("SPIRAL_SOFTLOGIC_PSI_GAIN")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .map(|v| v.clamp(0.0, 2.0))
                .unwrap_or(0.5),
            loss_gain: env::var("SPIRAL_SOFTLOGIC_LOSS_GAIN")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .map(|v| v.clamp(0.0, 1.5))
                .unwrap_or(0.35),
            floor: env::var("SPIRAL_SOFTLOGIC_FLOOR")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .map(|v| v.clamp(0.05, 1.0))
                .unwrap_or(0.25),
            scale_gain: env::var("SPIRAL_SOFTLOGIC_SCALE_GAIN")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .map(|v| v.clamp(0.0, 1.5))
                .unwrap_or(0.2),
            last_weights: (1.0, 1.0, 1.0),
            last_z: 0.0,
            last_feedback: None,
        };
        if flex.inertia >= 0.95 {
            flex.inertia = 0.95;
        }
        flex
    }

    fn prepare_weights(&mut self, band_energy: &BandEnergy) -> (f32, f32, f32) {
        let norm = (band_energy.above.abs() + band_energy.here.abs() + band_energy.beneath.abs())
            .max(1e-4);
        let asymmetry = (band_energy.above - band_energy.beneath) / norm;
        let drift_term = band_energy.drift.tanh();
        let z_bias = self
            .last_feedback
            .as_ref()
            .map(|feedback| feedback.z_signal)
            .unwrap_or(self.last_z);
        let mut target_above = 1.0 + (asymmetry * self.drift_gain) + (z_bias * self.psi_gain);
        let mut target_here =
            1.0 + ((band_energy.here / norm) - (band_energy.drift.abs() / norm)) * self.loss_gain;
        let mut target_beneath = 1.0 - (asymmetry * self.drift_gain) - (z_bias * self.psi_gain)
            + (-drift_term * self.drift_gain * 0.5);

        if self.scale_gain > 0.0 {
            if let Some(scale) = self.last_feedback.and_then(|feedback| feedback.scale) {
                let bias = (-scale.log_radius).tanh();
                let explore = bias.max(0.0);
                let settle = (-bias).max(0.0);
                target_above *= 1.0 + self.scale_gain * explore;
                target_here *= 1.0 + self.scale_gain * 0.5 * (explore - settle);
                target_beneath *= 1.0 + self.scale_gain * settle;
            }
        }

        let target = (
            target_above.clamp(self.floor, 3.0),
            target_here.clamp(self.floor, 2.5),
            target_beneath.clamp(self.floor, 3.0),
        );
        self.last_weights = (
            Self::lerp(self.last_weights.0, target.0, 1.0 - self.inertia),
            Self::lerp(self.last_weights.1, target.1, 1.0 - self.inertia),
            Self::lerp(self.last_weights.2, target.2, 1.0 - self.inertia),
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
        let psi_total = psi_total.unwrap_or(0.0);
        let total = (band_energy.above + band_energy.here + band_energy.beneath).max(1e-4);
        let asym = (band_energy.above - band_energy.beneath) / total;
        let drift = band_energy.drift;
        let raw_signal = 0.6 * (psi_total - weighted_loss) + 0.3 * asym + 0.1 * drift;
        let z_signal = raw_signal.tanh();
        self.last_z = Self::lerp(self.last_z, z_signal, 1.0 - self.inertia);
        let feedback = SoftlogicZFeedback {
            psi_total,
            weighted_loss,
            band_energy: (band_energy.above, band_energy.here, band_energy.beneath),
            drift,
            z_signal: self.last_z,
            scale: scale_hint,
        };
        self.last_feedback = Some(feedback);
        feedback
    }

    fn lerp(current: f32, target: f32, factor: f32) -> f32 {
        current + (target - current) * factor
    }
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

        Self {
            planner: RankPlanner::new(caps),
            curvature,
            hyper_learning_rate,
            fallback_learning_rate,
            real_learning_rate: None,
            blackcat: None,
            blackcat_moderator: None,
            autopilot: None,
            band_weight_fn: None,
            injector_enabled: false,
            distribution: None,
            meta_conductor: None,
            heur_log: HeurOpLog::default(),
            rewrite_budget: None,
            softlogic: SoftLogicFlex::new(),
            desire_bridge: None,
            desire_roundtable_bridge: None,
            last_desire_roundtable_summary: None,
            #[cfg(feature = "psi")]
            desire_psi_bridge: None,
            graph_bridge: None,
            graph_pending: None,
            graph_last_hint: None,
            curvature_scheduler: None,
            last_curvature_metrics: None,
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

    /// Clears any attached roundtable desire bridge.
    pub fn disable_desire_roundtable_bridge(&mut self) {
        self.desire_roundtable_bridge = None;
        self.last_desire_roundtable_summary = None;
    }

    /// Installs a curvature scheduler so the trainer can adapt its hyperbolic
    /// geometry based on recent gradient pressure observations.
    pub fn enable_curvature_scheduler(&mut self, mut scheduler: CurvatureScheduler) {
        scheduler.sync(self.curvature);
        self.curvature_scheduler = Some(scheduler);
        self.last_curvature_metrics = None;
    }

    /// Disables any configured curvature scheduler and clears cached metrics.
    pub fn disable_curvature_scheduler(&mut self) {
        self.curvature_scheduler = None;
        self.last_curvature_metrics = None;
    }

    /// Returns the most recently recorded curvature metrics emitted by the
    /// scheduler, when available.
    pub fn curvature_metrics(&self) -> Option<CurvatureMetrics> {
        self.last_curvature_metrics
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

        let mut cfg = PsiConfig::default();
        cfg.enabled = true;
        cfg.components = PsiComponent::defaults();

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
    pub fn roundtable(&self, rows: u32, cols: u32, config: RoundtableConfig) -> RoundtableSchedule {
        let schedule = RoundtableSchedule::new(&self.planner, rows, cols, config);
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

    /// Returns the curvature used for hypergrad preparation.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the learning rate used for hypergrad updates.
    pub fn hyper_learning_rate(&self) -> f32 {
        self.hyper_learning_rate
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

    /// Attaches hypergrad tapes to all parameters of the provided module.
    pub fn prepare<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.attach_hypergrad(self.curvature, self.hyper_learning_rate)?;
        if let Some(rate) = self.real_learning_rate {
            module.attach_realgrad(rate)?;
        }
        Ok(())
    }

    /// Attaches hypergrad tapes with an explicit topos shared across parameters.
    pub fn prepare_with_topos<M: Module>(
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
    pub fn zero<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.zero_accumulators()
    }

    /// Applies the parameter updates using either the hypergrad tape or the fallback rate.
    pub fn step<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.apply_step(self.fallback_learning_rate)
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
        M: Module,
        L: Loss,
        I: IntoIterator,
        I::Item: IntoBatch,
    {
        if let Some(budget) = self.rewrite_budget.as_mut() {
            budget.begin_epoch();
        }
        self.zero(module)?;
        #[cfg(feature = "psi")]
        self.bootstrap_psi(schedule);
        #[cfg(feature = "psychoid")]
        self.bootstrap_psychoid(schedule);
        #[cfg(feature = "collapse")]
        self.bootstrap_collapse(schedule);
        let mut total_loss = 0.0f32;
        let mut steps = 0usize;
        for batch in batches.into_iter() {
            let (input, target) = batch.into_batch()?;
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
            let prediction = module.forward(&input)?;
            let loss_value = loss.forward(&prediction, &target)?;
            let step_loss = loss_value.data().iter().copied().sum::<f32>();
            let grad_output = loss.backward(&prediction, &target)?;
            let mut band_energy = schedule.band_energy(&grad_output)?;
            let baseline_band_energy = BandEnergy {
                above: band_energy.above,
                here: band_energy.here,
                beneath: band_energy.beneath,
                drift: band_energy.drift,
            };
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
            bands.scale_inplace(weights.0, weights.1, weights.2);
            let weight_mean = (weights.0 + weights.1 + weights.2) / 3.0;
            let weighted_loss = step_loss * weight_mean.max(0.0);
            total_loss += weighted_loss;
            let mut extra = HashMap::new();
            extra.insert("softlogic_w_above".to_string(), weights.0 as f64);
            extra.insert("softlogic_w_here".to_string(), weights.1 as f64);
            extra.insert("softlogic_w_beneath".to_string(), weights.2 as f64);
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
            let _ = module.backward_bands(&input, &bands)?;
            if let Some(bridge) = self.graph_bridge.as_ref() {
                self.graph_pending = bridge.digest(&baseline_band_energy)?;
            }
            #[cfg(feature = "psychoid")]
            let mut psychoid_events = 0usize;
            #[cfg(feature = "psi")]
            let mut psi_snapshot: Option<PsiReading> = None;
            #[cfg(feature = "psi")]
            {
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
            let z_feedback =
                self.softlogic
                    .observe(&band_energy, weighted_loss, psi_total_opt, scale_hint);
            hub::set_softlogic_z(z_feedback);
            extra.insert("softlogic_z".to_string(), z_feedback.z_signal as f64);
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
                        let collapse_hint =
                            psi_total_opt.filter(|value| *value > 0.0).or_else(|| {
                                if summary.mean_psi > 0.0 {
                                    Some(summary.mean_psi)
                                } else {
                                    None
                                }
                            });
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
            let curvature_summary = if self.curvature_scheduler.is_some() {
                Some(Self::collect_gradient_summary(module)?)
            } else {
                None
            };
            self.step(module)?;
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
                }
            }
            self.zero(module)?;
            steps += 1;

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
        Ok(EpochStats {
            batches: steps,
            total_loss,
            average_loss: if steps == 0 {
                0.0
            } else {
                total_loss / steps as f32
            },
        })
    }

    fn apply_curvature_scheduler<M: Module>(
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
        }
        Ok(Some(decision))
    }

    fn estimate_device_load(&self) -> f64 {
        let caps = self.planner.device_caps();
        caps.occupancy_score(caps.max_workgroup) as f64
    }

    fn collect_gradient_summary<M: Module>(module: &M) -> PureResult<GradientSummary> {
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

    fn retune_hypergrads<M: Module>(
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

    #[cfg(feature = "collapse")]
    fn bootstrap_collapse(&mut self, schedule: &RoundtableSchedule) {
        if self.collapse.is_some() || !schedule.collapse_enabled() {
            return;
        }
        let cfg = CollapseConfig::automated(schedule.psi_hint());
        self.collapse = Some(CollapseDrive::new(cfg));
    }

    #[cfg(feature = "collapse")]
    fn apply_grad_scale<M: Module>(&self, module: &mut M, scale: f32) -> PureResult<()> {
        if (scale - 1.0).abs() <= f32::EPSILON {
            return Ok(());
        }
        module.visit_parameters_mut(&mut |param| {
            param.scale_accumulators(scale);
            Ok(())
        })
    }

    #[cfg(feature = "collapse")]
    fn clip_grad_global_norm<M: Module>(&self, module: &mut M, max_norm: f32) -> PureResult<()> {
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
    fn optimizer_mul_lr<M: Module>(&mut self, module: &mut M, factor: f32) -> PureResult<()> {
        if !factor.is_finite() || factor <= 0.0 {
            return Ok(());
        }
        self.fallback_learning_rate *= factor;
        self.hyper_learning_rate *= factor;
        if let Some(rate) = self.real_learning_rate.as_mut() {
            *rate *= factor;
        }
        module.visit_parameters_mut(&mut |param| {
            param.scale_learning_rate(factor);
            Ok(())
        })
    }

    #[cfg(feature = "psi")]
    fn collect_grad_l2<M: Module>(module: &M) -> PureResult<f32> {
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
            self.heur_log.append(op.clone());
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

/// Metrics captured while running [`ModuleTrainer::train_epoch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochStats {
    pub batches: usize,
    pub total_loss: f32,
    pub average_loss: f32,
}

#[derive(Default)]
struct CurvatureGradientAccumulator {
    l1: f64,
    sum_squares: f64,
    linf: f32,
    count: usize,
}

impl CurvatureGradientAccumulator {
    fn accumulate(&mut self, summary: GradientSummary) {
        self.l1 += summary.l1() as f64;
        self.sum_squares += summary.sum_squares() as f64;
        self.linf = self.linf.max(summary.linf());
        self.count += summary.count();
    }

    fn finish(self) -> GradientSummary {
        if self.count == 0 {
            GradientSummary::default()
        } else {
            GradientSummary::from_moments(
                self.l1 as f32,
                self.sum_squares as f32,
                self.linf,
                self.count,
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
    use crate::roundtable::{HeurOp, HeurOpKind};
    use crate::schedule::RoundtableConfig;
    #[cfg(feature = "golden")]
    use crate::CouncilEvidence;
    use st_core::runtime::blackcat::{bandit::SoftBanditMode, zmeta::ZMetaParams, ChoiceGroups};
    use st_tensor::topos::OpenCartesianTopos;
    use std::collections::HashMap;
    use std::time::{Duration, Instant, SystemTime};

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

    #[test]
    fn trainer_attaches_and_steps() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
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
        let trainer = ModuleTrainer::new(caps, -0.9, 0.06, 0.02);
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
            let mut metrics = StepMetrics::default();
            metrics.step_time_ms = 10.0;
            metrics.mem_peak_mb = 256.0;
            metrics.retry_rate = 0.05;
            metrics.extra.insert("grad_norm".into(), 0.4);
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
