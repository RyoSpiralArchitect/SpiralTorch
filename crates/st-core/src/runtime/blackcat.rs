// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use bandit::{SoftBandit, SoftBanditMode};
use rewrite::HeurStore;
use spiral_config::determinism;
use st_frac::FracBackend;
use thiserror::Error;
use tracing::{debug, instrument, warn};
use wilson::wilson_lower;
use zmeta::{ZMetaES, ZMetaParams};

use crate::heur::free_energy::{
    evaluate_free_energy, BandEnergy, FreeEnergyConfig, FreeEnergyError, FreeEnergyObservation,
    FreeEnergyReport, FreeEnergyRequest,
};
use crate::plugin::{global_registry, PluginEvent};
use crate::telemetry::{monitoring::MonitoringHub, trace_init};

#[derive(Clone, Debug, Error, PartialEq)]
pub enum BlackCatError {
    #[error(transparent)]
    FreeEnergy(#[from] FreeEnergyError),
    #[error("BlackCat field '{field}' must be finite, got {value}")]
    NonFinite { field: &'static str, value: f64 },
    #[error("BlackCat field '{field}' must be non-negative, got {value}")]
    Negative { field: &'static str, value: f64 },
    #[error("BlackCat context dimension mismatch: expected {expected}, got {actual}")]
    ContextDimension { expected: usize, actual: usize },
    #[error("BlackCat adaptation state is invalid at '{field}'")]
    InvalidAdaptationState { field: &'static str },
}

/// Metrics reported by a training loop back into the runtime.
#[derive(Clone, Debug, Default)]
pub struct StepMetrics {
    pub step_time_ms: f64,
    pub mem_peak_mb: f64,
    pub retry_rate: f64,
    pub extra: HashMap<String, f64>,
}

impl StepMetrics {
    fn extra_first(&self, keys: &[&str]) -> Option<f64> {
        keys.iter().find_map(|key| self.extra.get(*key).copied())
    }

    /// Translate runtime metrics into the canonical free-energy observation.
    /// Missing optional signals are neutral; supplied values remain strict.
    pub fn free_energy_observation(&self, external_penalty: f64) -> FreeEnergyObservation {
        let reference_loss = self
            .extra_first(&[
                "reference_loss",
                "loss_before",
                "step_loss",
                "loss_weighted",
            ])
            .unwrap_or(0.0);
        let candidate_loss = self
            .extra_first(&["candidate_loss", "loss_after", "loss_weighted"])
            .unwrap_or(reference_loss);
        FreeEnergyObservation {
            reference_loss,
            candidate_loss,
            step_time_ms: self.step_time_ms,
            memory_mb: self.mem_peak_mb,
            retry_rate: self.retry_rate,
            observation_entropy: self
                .extra_first(&["observation_entropy", "attn_entropy", "entropy"])
                .unwrap_or(0.0),
            external_penalty,
            band: BandEnergy {
                above: self.extra.get("band_above").copied().unwrap_or(0.0) as f32,
                here: self.extra.get("band_here").copied().unwrap_or(0.0) as f32,
                beneath: self.extra.get("band_beneath").copied().unwrap_or(0.0) as f32,
            },
        }
    }
}

/// Reward shaping configuration that supplies scales and weights to the
/// canonical variational free-energy evaluator.
#[derive(Clone, Debug)]
pub struct RewardCfg {
    pub lam_speed: f64,
    pub lam_mem: f64,
    pub lam_stab: f64,
    pub scale_speed: f64,
    pub scale_mem: f64,
    pub scale_stab: f64,
}

impl Default for RewardCfg {
    fn default() -> Self {
        Self {
            lam_speed: 0.5,
            lam_mem: 0.3,
            lam_stab: 0.2,
            scale_speed: 10.0,
            scale_mem: 1024.0,
            scale_stab: 1.0,
        }
    }
}

impl RewardCfg {
    pub fn free_energy_config(&self) -> Result<FreeEnergyConfig, FreeEnergyError> {
        FreeEnergyConfig::default()
            .with_resource_scales(self.scale_speed, self.scale_mem, self.scale_stab)?
            .with_component_weights(
                1.0,
                self.lam_speed,
                self.lam_mem,
                self.lam_stab,
                self.lam_stab,
                1.0,
            )
    }

    pub fn report(
        &self,
        metrics: &StepMetrics,
        frac_penalty: f64,
    ) -> Result<FreeEnergyReport, FreeEnergyError> {
        evaluate_free_energy(FreeEnergyRequest {
            observation: metrics.free_energy_observation(frac_penalty),
            config: self.free_energy_config()?,
        })
    }

    pub fn try_score(
        &self,
        metrics: &StepMetrics,
        frac_penalty: f64,
    ) -> Result<f64, FreeEnergyError> {
        self.report(metrics, frac_penalty)
            .map(|report| report.utility)
    }

    /// Compatibility scalar. Guarded runtime paths should use [`Self::report`]
    /// or [`Self::try_score`] so invalid observations cannot be mistaken for a
    /// legitimate reward.
    pub fn score(&self, metrics: &StepMetrics, frac_penalty: f64) -> f64 {
        self.try_score(metrics, frac_penalty)
            .unwrap_or(f64::NEG_INFINITY)
    }
}

/// Named groups of candidate choices (tile, merge strategy, etc.).
#[derive(Clone, Debug)]
pub struct ChoiceGroups {
    pub groups: HashMap<String, Vec<String>>,
}

/// Multi-armed contextual bandit that operates over named groups.
#[derive(Clone)]
pub struct MultiBandit {
    arms: HashMap<String, SoftBandit>,
}

impl MultiBandit {
    pub fn new(groups: &ChoiceGroups, feat_dim: usize, mode: SoftBanditMode) -> Self {
        let mut arms = HashMap::new();
        let feat_dim = feat_dim.max(1);
        for (name, opts) in &groups.groups {
            if !opts.is_empty() {
                arms.insert(name.clone(), SoftBandit::new(opts.clone(), feat_dim, mode));
            }
        }
        Self { arms }
    }

    pub fn select_all(&mut self, context: &[f64]) -> HashMap<String, String> {
        let mut picks = HashMap::new();
        for (name, bandit) in self.arms.iter_mut() {
            let choice = bandit.select(context);
            picks.insert(name.clone(), choice);
        }
        picks
    }

    pub fn update_all(&mut self, context: &[f64], reward: f64) {
        for bandit in self.arms.values_mut() {
            bandit.update_last(context, reward);
        }
    }

    fn validate_state(&self, context_dim: usize) -> Result<(), &'static str> {
        self.arms
            .values()
            .try_for_each(|bandit| bandit.validate_state(context_dim))
    }

    fn validate_selection_scores(&self, context: &[f64]) -> Result<(), &'static str> {
        self.arms
            .values()
            .try_for_each(|bandit| bandit.validate_selection_scores(context))
    }
}

/// BlackCat orchestrator that joins ES search with contextual bandits.
pub struct BlackCatRuntime {
    pub z: ZMetaES,
    pub bandits: MultiBandit,
    pub heur: HeurStore,
    pub reward: RewardCfg,
    context_dim: usize,
    last_context: Vec<f64>,
    last_picks: HashMap<String, String>,
    last_step_start: Option<Instant>,
    stats_alpha: f64,
    stats_steps: u64,
    reward_mean: f64,
    reward_m2: f64,
    last_reward: f64,
    last_free_energy_report: Option<FreeEnergyReport>,
    metrics_ema: MetricsEma,
    frac_penalty_ema: RollingEma,
    extra_ema: HashMap<String, RollingEma>,
    monitoring: MonitoringHub,
}

impl BlackCatRuntime {
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(groups, heur_path), fields(feat_dim = feat_dim, bandit_mode = ?mode))]
    pub fn new(
        z_params: ZMetaParams,
        groups: ChoiceGroups,
        feat_dim: usize,
        mode: SoftBanditMode,
        heur_path: Option<String>,
    ) -> Self {
        trace_init::init_tracing();
        let mut params = z_params;
        if determinism::config().enabled {
            params.seed = determinism::config().seed_for("st-core/blackcat/zmeta");
        }
        let bandits = MultiBandit::new(&groups, feat_dim, mode);
        let heur = HeurStore::new(heur_path);
        let stats_alpha = 0.2;
        let runtime = Self {
            z: ZMetaES::new(params),
            bandits,
            heur,
            reward: RewardCfg::default(),
            context_dim: feat_dim.max(1),
            last_context: vec![0.0; feat_dim.max(1)],
            last_picks: HashMap::new(),
            last_step_start: None,
            stats_alpha,
            stats_steps: 0,
            reward_mean: 0.0,
            reward_m2: 0.0,
            last_reward: 0.0,
            last_free_energy_report: None,
            metrics_ema: MetricsEma::new(stats_alpha),
            frac_penalty_ema: RollingEma::new(stats_alpha),
            extra_ema: HashMap::new(),
            monitoring: MonitoringHub::default(),
        };
        debug!("blackcat runtime initialised");
        runtime
    }

    /// Call at the beginning of a training step.
    #[instrument(skip(self))]
    pub fn begin_step(&mut self) {
        self.last_step_start = Some(Instant::now());
    }

    /// Build a contextual feature vector from runtime metrics.
    #[allow(
        clippy::too_many_arguments,
        reason = "Context vector requires these inputs for bandit feature parity"
    )]
    pub fn make_context(
        &self,
        batches: u32,
        tiles: u32,
        depth: u32,
        device_code: u32,
        load: f64,
        extras: &[(String, f64)],
        feat_dim: usize,
    ) -> Vec<f64> {
        let target_dim = if feat_dim == 0 {
            self.context_dim
        } else {
            feat_dim
        };
        let mut ctx = vec![
            1.0,
            batches as f64,
            tiles as f64,
            depth as f64,
            (device_code % 1024) as f64 / 1024.0,
            load,
        ];
        for (_, value) in extras.iter() {
            ctx.push(*value);
        }
        ctx.resize(target_dim, 0.0);
        ctx
    }

    /// Choose all groups at once, storing the picks and context internally.
    #[instrument(skip(self, context), fields(context_len = context.len()))]
    pub fn choose(&mut self, context: Vec<f64>) -> HashMap<String, String> {
        match self.try_choose(context) {
            Ok(picks) => picks,
            Err(error) => {
                warn!(error = %error, "blackcat context rejected");
                let bus = global_registry().event_bus();
                if bus.has_listeners("BlackCatChooseRejected") {
                    bus.publish(&PluginEvent::custom(
                        "BlackCatChooseRejected",
                        serde_json::json!({"error": error.to_string()}),
                    ));
                }
                HashMap::new()
            }
        }
    }

    /// Guarded choice path. Context shape and finiteness are checked before
    /// any bandit or runtime state is changed.
    pub fn try_choose(
        &mut self,
        context: Vec<f64>,
    ) -> Result<HashMap<String, String>, BlackCatError> {
        self.validate_context(&context)?;
        self.bandits
            .validate_state(self.context_dim)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        let mut next_bandits = self.bandits.clone();
        next_bandits
            .validate_selection_scores(&context)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        let picks = next_bandits.select_all(&context);
        next_bandits
            .validate_state(self.context_dim)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        self.bandits = next_bandits;
        self.last_context = context;
        self.last_picks = picks.clone();
        debug!(picks = ?self.last_picks, "blackcat bandit picks");
        let bus = global_registry().event_bus();
        if bus.has_listeners("BlackCatChoose") {
            bus.publish(&PluginEvent::custom(
                "BlackCatChoose",
                serde_json::json!({
                    "context": &self.last_context,
                    "picks": &self.last_picks,
                }),
            ));
        }
        Ok(picks)
    }

    /// Compatibility update path. Invalid observations are rejected before any
    /// ES, bandit, or telemetry state is mutated.
    #[instrument(skip(self, metrics), fields(step_time = metrics.step_time_ms, retry_rate = metrics.retry_rate))]
    pub fn post_step(&mut self, metrics: &StepMetrics) -> f64 {
        match self.try_post_step_report(metrics) {
            Ok(report) => report.utility,
            Err(error) => {
                warn!(error = %error, "blackcat free-energy reward rejected");
                let bus = global_registry().event_bus();
                if bus.has_listeners("BlackCatRewardRejected") {
                    bus.publish(&PluginEvent::custom(
                        "BlackCatRewardRejected",
                        serde_json::json!({
                            "error": error.to_string(),
                            "semantic_owner": crate::heur::free_energy::FREE_ENERGY_SEMANTIC_OWNER,
                            "contract_version": crate::heur::free_energy::FREE_ENERGY_CONTRACT_VERSION,
                        }),
                    ));
                }
                f64::NEG_INFINITY
            }
        }
    }

    /// Guarded scalar update for callers that need explicit failure handling.
    pub fn try_post_step(&mut self, metrics: &StepMetrics) -> Result<f64, BlackCatError> {
        self.try_post_step_report(metrics)
            .map(|report| report.utility)
    }

    /// Update both the ES search and contextual bandits from one canonical
    /// free-energy report, committing state only after validation succeeds.
    #[instrument(skip(self, metrics), fields(step_time = metrics.step_time_ms, retry_rate = metrics.retry_rate))]
    pub fn try_post_step_report(
        &mut self,
        metrics: &StepMetrics,
    ) -> Result<FreeEnergyReport, BlackCatError> {
        let curr_penalty = self.z.frac_penalty();
        let report = self.reward.report(metrics, curr_penalty)?;
        let reward_current = report.utility;
        let proposed_penalty = self.z.frac_penalty_proposed();
        validate_blackcat_non_negative("zmeta.proposed_frac_penalty", proposed_penalty)?;
        self.z
            .validate_adaptation_state()
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        self.bandits
            .validate_state(self.context_dim)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        self.validate_statistics_state()?;
        self.validate_context(&self.last_context)?;
        let grad_norm = metrics.extra.get("grad_norm").copied().unwrap_or(0.0);
        validate_blackcat_non_negative("grad_norm", grad_norm)?;
        let loss_var = metrics
            .extra
            .get("loss_var")
            .or_else(|| metrics.extra.get("loss_variance"))
            .copied()
            .unwrap_or(1.0);
        validate_blackcat_non_negative("loss_variance", loss_var)?;
        let proposed_reward = reward_current - (proposed_penalty - curr_penalty);
        validate_blackcat_finite("zmeta.proposed_reward", proposed_reward)?;

        let mut next_z = self.z.clone();
        next_z.update(reward_current, proposed_reward, Some(&self.last_context));
        next_z.temp_schedule(metrics.retry_rate, grad_norm, loss_var);
        next_z
            .validate_adaptation_state()
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        validate_blackcat_non_negative("zmeta.next_frac_penalty", next_z.frac_penalty())?;
        validate_blackcat_non_negative(
            "zmeta.next_proposed_frac_penalty",
            next_z.frac_penalty_proposed(),
        )?;

        let mut next_bandits = self.bandits.clone();
        next_bandits.update_all(&self.last_context, reward_current);
        next_bandits
            .validate_state(self.context_dim)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;

        let (next_stats_steps, next_reward_mean, next_reward_m2) =
            self.next_reward_statistics(reward_current)?;
        let mut next_metrics_ema = self.metrics_ema.clone();
        next_metrics_ema.try_update(metrics)?;
        let mut next_frac_penalty_ema = self.frac_penalty_ema.clone();
        next_frac_penalty_ema.try_update("statistics.frac_penalty_ema", curr_penalty)?;
        let mut next_extra_ema = self.extra_ema.clone();
        for (key, value) in &metrics.extra {
            next_extra_ema
                .entry(key.clone())
                .or_insert_with(|| RollingEma::new(self.stats_alpha))
                .try_update("statistics.extra_ema", *value)?;
        }

        self.z = next_z;
        self.bandits = next_bandits;
        self.stats_steps = next_stats_steps;
        self.reward_mean = next_reward_mean;
        self.reward_m2 = next_reward_m2;
        self.metrics_ema = next_metrics_ema;
        self.frac_penalty_ema = next_frac_penalty_ema;
        self.extra_ema = next_extra_ema;
        self.last_reward = reward_current;
        self.last_free_energy_report = Some(report.clone());
        debug!(
            reward = reward_current,
            frac_penalty = curr_penalty,
            "blackcat step complete"
        );
        let _ = self.monitoring.observe(metrics, reward_current);
        let bus = global_registry().event_bus();
        if bus.has_listeners("BlackCatPostStep") {
            bus.publish(&PluginEvent::custom(
                "BlackCatPostStep",
                serde_json::json!({
                    "reward": reward_current,
                    "frac_penalty": curr_penalty,
                    "free_energy": &report,
                    "picks": &self.last_picks,
                    "metrics": {
                        "step_time_ms": metrics.step_time_ms,
                        "mem_peak_mb": metrics.mem_peak_mb,
                        "retry_rate": metrics.retry_rate,
                        "extra": &metrics.extra,
                    }
                }),
            ));
        }
        Ok(report)
    }

    fn validate_statistics_state(&self) -> Result<(), BlackCatError> {
        for (field, value) in [
            ("statistics.alpha", self.stats_alpha),
            ("statistics.reward_mean", self.reward_mean),
            ("statistics.reward_m2", self.reward_m2),
            ("statistics.last_reward", self.last_reward),
        ] {
            validate_blackcat_finite(field, value)?;
        }
        if !(0.0..=1.0).contains(&self.stats_alpha) {
            return Err(BlackCatError::InvalidAdaptationState {
                field: "statistics.alpha",
            });
        }
        if self.reward_m2 < 0.0 {
            return Err(BlackCatError::InvalidAdaptationState {
                field: "statistics.reward_m2",
            });
        }
        self.metrics_ema.validate_state()?;
        self.frac_penalty_ema
            .validate_state("statistics.frac_penalty_ema")?;
        self.extra_ema
            .values()
            .try_for_each(|ema| ema.validate_state("statistics.extra_ema"))
    }

    fn next_reward_statistics(&self, reward: f64) -> Result<(u64, f64, f64), BlackCatError> {
        let steps =
            self.stats_steps
                .checked_add(1)
                .ok_or(BlackCatError::InvalidAdaptationState {
                    field: "statistics.steps",
                })?;
        if steps == 1 {
            return Ok((steps, reward, 0.0));
        }

        let delta = validate_blackcat_finite("statistics.reward_delta", reward - self.reward_mean)?;
        let mean = validate_blackcat_finite(
            "statistics.reward_mean",
            self.reward_mean + delta / steps as f64,
        )?;
        let delta_after_mean =
            validate_blackcat_finite("statistics.reward_delta_after_mean", reward - mean)?;
        let m2 = validate_blackcat_finite(
            "statistics.reward_m2",
            self.reward_m2 + delta * delta_after_mean,
        )?;
        if m2 < 0.0 {
            return Err(BlackCatError::InvalidAdaptationState {
                field: "statistics.reward_m2",
            });
        }
        Ok((steps, mean, m2))
    }

    /// Preview the current reward without mutating runtime state.
    pub fn preview_reward_report(
        &self,
        metrics: &StepMetrics,
    ) -> Result<FreeEnergyReport, FreeEnergyError> {
        self.reward.report(metrics, self.z.frac_penalty())
    }

    /// Most recent report that was committed to the runtime.
    pub fn last_free_energy_report(&self) -> Option<&FreeEnergyReport> {
        self.last_free_energy_report.as_ref()
    }

    fn validate_context(&self, context: &[f64]) -> Result<(), BlackCatError> {
        if context.len() != self.context_dim {
            return Err(BlackCatError::ContextDimension {
                expected: self.context_dim,
                actual: context.len(),
            });
        }
        for value in context {
            validate_blackcat_finite("context", *value)?;
        }
        Ok(())
    }

    /// Access the embedded monitoring hub for instrumentation.
    pub fn monitoring(&self) -> &MonitoringHub {
        &self.monitoring
    }

    /// Mutable access to attach exporters or tweak configuration.
    pub fn monitoring_mut(&mut self) -> &mut MonitoringHub {
        &mut self.monitoring
    }

    /// Returns the dimensionality expected by the contextual bandits.
    pub fn context_dim(&self) -> usize {
        self.context_dim
    }

    /// Returns the current fractional regularisation penalty tracked by ZMeta.
    pub fn frac_penalty(&self) -> f64 {
        self.z.frac_penalty()
    }

    /// Overrides the fractional regulariser backend.
    pub fn set_frac_backend(&mut self, backend: FracBackend) {
        self.z.set_frac_backend(backend);
    }

    /// Returns the current exploration temperature tracked by the runtime.
    pub fn temperature(&self) -> f64 {
        self.z.temperature()
    }

    /// Try to adopt a new soft heuristic guarded by the Wilson lower bound.
    pub fn try_adopt_soft(
        &mut self,
        rule_text: &str,
        wins: u32,
        trials: u32,
        baseline_p: f64,
    ) -> bool {
        let lb = wilson_lower(wins as i32, trials as i32, 1.96);
        if lb > baseline_p {
            let mut info = HashMap::new();
            info.insert("wins".to_string(), wins as f64);
            info.insert("trials".to_string(), trials as f64);
            self.heur
                .append(&format!("{}  # blackcat", rule_text.trim()), &info);
            return true;
        }
        false
    }

    /// Returns the duration since the last [`Self::begin_step`] call.
    pub fn elapsed_since_begin(&self) -> Option<Duration> {
        self.last_step_start.map(|start| start.elapsed())
    }

    /// Returns the last contextual feature vector used for bandit updates.
    pub fn last_context(&self) -> &[f64] {
        &self.last_context
    }

    /// Returns the picks that were selected during the last [`Self::choose`] call.
    pub fn last_picks(&self) -> &HashMap<String, String> {
        &self.last_picks
    }

    /// Returns aggregated runtime statistics derived from recent updates.
    pub fn stats(&self) -> BlackcatRuntimeStats {
        let reward_std = if self.stats_steps > 1 {
            (self.reward_m2 / (self.stats_steps - 1) as f64)
                .abs()
                .sqrt()
        } else {
            0.0
        };
        let extras = self
            .extra_ema
            .iter()
            .filter_map(|(key, ema)| ema.value().map(|value| (key.clone(), value)))
            .collect();
        BlackcatRuntimeStats {
            steps: self.stats_steps,
            reward_mean: self.reward_mean,
            reward_std,
            last_reward: self.last_reward,
            step_time_ms_ema: self.metrics_ema.step_time(),
            mem_peak_mb_ema: self.metrics_ema.mem_peak(),
            retry_rate_ema: self.metrics_ema.retry_rate(),
            frac_penalty_ema: self
                .frac_penalty_ema
                .value()
                .unwrap_or_else(|| self.z.frac_penalty()),
            extras,
        }
    }
}

fn validate_blackcat_finite(field: &'static str, value: f64) -> Result<f64, BlackCatError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(BlackCatError::NonFinite { field, value })
    }
}

fn validate_blackcat_non_negative(field: &'static str, value: f64) -> Result<f64, BlackCatError> {
    validate_blackcat_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(BlackCatError::Negative { field, value })
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlackcatRuntimeStats {
    pub steps: u64,
    pub reward_mean: f64,
    pub reward_std: f64,
    pub last_reward: f64,
    pub step_time_ms_ema: f64,
    pub mem_peak_mb_ema: f64,
    pub retry_rate_ema: f64,
    pub frac_penalty_ema: f64,
    pub extras: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
struct MetricsEma {
    step_time: RollingEma,
    mem_peak: RollingEma,
    retry_rate: RollingEma,
}

impl MetricsEma {
    fn new(alpha: f64) -> Self {
        Self {
            step_time: RollingEma::new(alpha),
            mem_peak: RollingEma::new(alpha),
            retry_rate: RollingEma::new(alpha),
        }
    }

    fn try_update(&mut self, metrics: &StepMetrics) -> Result<(), BlackCatError> {
        self.step_time
            .try_update("statistics.step_time_ema", metrics.step_time_ms)?;
        self.mem_peak
            .try_update("statistics.mem_peak_ema", metrics.mem_peak_mb)?;
        self.retry_rate
            .try_update("statistics.retry_rate_ema", metrics.retry_rate)
    }

    fn validate_state(&self) -> Result<(), BlackCatError> {
        self.step_time.validate_state("statistics.step_time_ema")?;
        self.mem_peak.validate_state("statistics.mem_peak_ema")?;
        self.retry_rate.validate_state("statistics.retry_rate_ema")
    }

    fn step_time(&self) -> f64 {
        self.step_time.value().unwrap_or(0.0)
    }

    fn mem_peak(&self) -> f64 {
        self.mem_peak.value().unwrap_or(0.0)
    }

    fn retry_rate(&self) -> f64 {
        self.retry_rate.value().unwrap_or(0.0)
    }
}

#[derive(Clone, Debug)]
struct RollingEma {
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl RollingEma {
    fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1.0e-3, 0.999),
            value: 0.0,
            initialized: false,
        }
    }

    fn try_update(&mut self, field: &'static str, sample: f64) -> Result<(), BlackCatError> {
        if !sample.is_finite() {
            return Ok(());
        }
        let value = if self.initialized {
            validate_blackcat_finite(field, self.alpha * sample + (1.0 - self.alpha) * self.value)?
        } else {
            sample
        };
        self.value = value;
        self.initialized = true;
        Ok(())
    }

    fn validate_state(&self, field: &'static str) -> Result<(), BlackCatError> {
        validate_blackcat_finite(field, self.alpha)?;
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(BlackCatError::InvalidAdaptationState { field });
        }
        if self.initialized {
            validate_blackcat_finite(field, self.value)?;
        }
        Ok(())
    }

    fn value(&self) -> Option<f64> {
        if self.initialized {
            Some(self.value)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_runtime() -> BlackCatRuntime {
        let mut groups = HashMap::new();
        groups.insert("tile".to_string(), vec!["a".to_string(), "b".to_string()]);
        let groups = ChoiceGroups { groups };
        BlackCatRuntime::new(ZMetaParams::default(), groups, 4, SoftBanditMode::TS, None)
    }

    #[test]
    fn runtime_accumulates_stats() {
        let mut runtime = sample_runtime();
        runtime.begin_step();
        let mut metrics = StepMetrics {
            step_time_ms: 12.5,
            mem_peak_mb: 512.0,
            retry_rate: 0.1,
            ..Default::default()
        };
        metrics.extra.insert("grad_norm".into(), 0.5);
        let reward1 = runtime.post_step(&metrics);
        let stats1 = runtime.stats();
        assert_eq!(stats1.steps, 1);
        assert!((stats1.reward_mean - reward1).abs() < 1e-9);
        assert_eq!(stats1.reward_std, 0.0);
        assert!(stats1.step_time_ms_ema > 0.0);
        assert!(stats1.mem_peak_mb_ema > 0.0);
        assert_eq!(stats1.extras.get("grad_norm").cloned().unwrap(), 0.5);

        runtime.begin_step();
        let mut metrics2 = StepMetrics {
            step_time_ms: 6.0,
            mem_peak_mb: 256.0,
            retry_rate: 0.05,
            ..Default::default()
        };
        metrics2.extra.insert("grad_norm".into(), 0.25);
        let _ = runtime.post_step(&metrics2);
        let stats2 = runtime.stats();
        assert_eq!(stats2.steps, 2);
        assert!(stats2.reward_std >= 0.0);
        assert!(stats2.step_time_ms_ema <= stats1.step_time_ms_ema);
        assert!(stats2.extras.get("grad_norm").cloned().unwrap() <= 0.5);
        assert!(stats2.frac_penalty_ema >= 0.0);
        let report = runtime
            .last_free_energy_report()
            .expect("committed report is retained");
        assert_eq!(
            report.contract_version,
            crate::heur::free_energy::FREE_ENERGY_CONTRACT_VERSION
        );
        assert_eq!(report.utility, stats2.last_reward);
    }

    #[test]
    fn runtime_reward_uses_loss_and_band_observations() {
        let reward = RewardCfg::default();
        let mut above = StepMetrics::default();
        above.extra.insert("reference_loss".into(), 0.8);
        above.extra.insert("candidate_loss".into(), 0.5);
        above.extra.insert("band_above".into(), 1.0);
        let mut beneath = above.clone();
        beneath.extra.remove("band_above");
        beneath.extra.insert("band_beneath".into(), 1.0);

        let above_report = reward.report(&above, 0.0).expect("above report");
        let beneath_report = reward.report(&beneath, 0.0).expect("beneath report");
        assert!((above_report.components.loss + 0.3).abs() < 1e-12);
        assert_eq!(above_report.distribution.dominant_band, "above");
        assert_eq!(beneath_report.distribution.dominant_band, "beneath");
        assert!(above_report.utility > beneath_report.utility);
    }

    #[test]
    fn rejected_reward_does_not_mutate_runtime() {
        let mut runtime = sample_runtime();
        let metrics = StepMetrics {
            step_time_ms: f64::NAN,
            ..StepMetrics::default()
        };

        let error = runtime
            .try_post_step_report(&metrics)
            .expect_err("non-finite metrics must fail closed");
        assert!(matches!(
            error,
            BlackCatError::FreeEnergy(FreeEnergyError::NonFinite { .. })
        ));
        assert_eq!(runtime.stats().steps, 0);
        assert!(runtime.last_free_energy_report().is_none());

        assert_eq!(runtime.post_step(&metrics), f64::NEG_INFINITY);
        assert_eq!(runtime.stats().steps, 0);
        assert!(runtime.last_free_energy_report().is_none());
    }

    #[test]
    fn rejected_adaptation_signal_does_not_mutate_runtime() {
        let mut runtime = sample_runtime();
        let initial_temperature = runtime.temperature();
        let mut metrics = StepMetrics::default();
        metrics.extra.insert("grad_norm".into(), f64::NAN);

        let error = runtime
            .try_post_step_report(&metrics)
            .expect_err("non-finite adaptation input must fail closed");
        assert!(matches!(
            error,
            BlackCatError::NonFinite {
                field: "grad_norm",
                ..
            }
        ));
        assert_eq!(runtime.stats().steps, 0);
        assert_eq!(runtime.temperature(), initial_temperature);
        assert!(runtime.last_free_energy_report().is_none());
    }

    #[test]
    fn guarded_choice_rejects_context_drift_without_state_changes() {
        let mut runtime = sample_runtime();
        let initial_context = runtime.last_context().to_vec();

        let dimension_error = runtime
            .try_choose(vec![1.0, 2.0])
            .expect_err("context dimension drift must fail closed");
        assert_eq!(
            dimension_error,
            BlackCatError::ContextDimension {
                expected: 4,
                actual: 2,
            }
        );
        let finite_error = runtime
            .try_choose(vec![1.0, 0.0, f64::NAN, 0.0])
            .expect_err("non-finite context must fail closed");
        assert!(matches!(
            finite_error,
            BlackCatError::NonFinite {
                field: "context",
                ..
            }
        ));
        assert_eq!(runtime.last_context(), initial_context);
        assert!(runtime.last_picks().is_empty());
    }

    #[test]
    fn bandit_overflow_rejects_the_whole_runtime_update() {
        let mut runtime = sample_runtime();
        runtime
            .try_choose(vec![f64::MAX, 0.0, 0.0, 0.0])
            .expect("finite context can be selected before posterior update");
        let z_before = runtime.z.z().to_vec();
        let temperature_before = runtime.temperature();

        let error = runtime
            .try_post_step_report(&StepMetrics::default())
            .expect_err("overflowing posterior update must fail closed");
        assert_eq!(
            error,
            BlackCatError::InvalidAdaptationState {
                field: "bandit.posterior_state",
            }
        );
        assert_eq!(runtime.z.z(), z_before);
        assert_eq!(runtime.temperature(), temperature_before);
        assert_eq!(runtime.stats().steps, 0);
        assert!(runtime.last_free_energy_report().is_none());
    }

    #[test]
    fn guarded_choice_rejects_non_finite_derived_scores() {
        let mut groups = HashMap::new();
        groups.insert("tile".to_string(), vec!["a".to_string(), "b".to_string()]);
        let mut runtime = BlackCatRuntime::new(
            ZMetaParams::default(),
            ChoiceGroups { groups },
            4,
            SoftBanditMode::UCB,
            None,
        );
        let initial_context = runtime.last_context().to_vec();

        let error = runtime
            .try_choose(vec![f64::MAX, 0.0, 0.0, 0.0])
            .expect_err("non-finite UCB score must fail closed");
        assert_eq!(
            error,
            BlackCatError::InvalidAdaptationState {
                field: "bandit.selection_score",
            }
        );
        assert_eq!(runtime.last_context(), initial_context);
        assert!(runtime.last_picks().is_empty());
    }

    #[test]
    fn reward_statistics_overflow_rejects_the_whole_runtime_update() {
        let mut runtime = sample_runtime();
        let magnitude = f64::MAX * 0.75;
        let mut negative = StepMetrics::default();
        negative.extra.insert("candidate_loss".into(), magnitude);
        let committed = runtime
            .try_post_step_report(&negative)
            .expect("first extreme but finite reward is representable");
        let stats_before = runtime.stats();
        let z_before = runtime.z.z().to_vec();
        let temperature_before = runtime.temperature();

        let mut positive = StepMetrics::default();
        positive.extra.insert("reference_loss".into(), magnitude);
        positive.extra.insert("candidate_loss".into(), 0.0);
        let error = runtime
            .try_post_step_report(&positive)
            .expect_err("overflowing reward delta must fail closed");
        assert!(matches!(
            error,
            BlackCatError::NonFinite {
                field: "statistics.reward_delta",
                ..
            }
        ));

        let stats_after = runtime.stats();
        assert_eq!(stats_after.steps, stats_before.steps);
        assert_eq!(stats_after.reward_mean, stats_before.reward_mean);
        assert_eq!(stats_after.reward_std, stats_before.reward_std);
        assert_eq!(stats_after.last_reward, stats_before.last_reward);
        assert_eq!(runtime.z.z(), z_before);
        assert_eq!(runtime.temperature(), temperature_before);
        assert_eq!(runtime.last_free_energy_report(), Some(&committed));
    }
}

// =================== zmeta.rs ===================
pub mod zmeta {
    use super::FracBackend;
    use randless::{Rng, StdRng};

    #[derive(Clone, Debug)]
    pub struct ZMetaParams {
        pub dim: usize,
        pub sigma: f64,
        pub lr: f64,
        pub alpha_frac: f64,
        pub lam_frac: f64,
        pub orientation_eta: f64,
        pub orientation_eps: f64,
        pub seed: u64,
    }

    impl Default for ZMetaParams {
        fn default() -> Self {
            Self {
                dim: 6,
                sigma: 0.15,
                lr: 0.1,
                alpha_frac: 0.35,
                lam_frac: 0.1,
                orientation_eta: 0.15,
                orientation_eps: 1e-3,
                seed: 42,
            }
        }
    }

    #[derive(Clone)]
    pub struct ZMetaES {
        z: Vec<f64>,
        dir: Vec<f64>,
        params: ZMetaParams,
        rng: StdRng,
        frac_backend: FracBackend,
        temp: f64,
        temp_min: f64,
        temp_max: f64,
        structural: Vec<f64>,
    }

    impl ZMetaES {
        pub fn new(params: ZMetaParams) -> Self {
            let rng = StdRng::seed_from_u64(params.seed);
            let mut dir = vec![0.0; params.dim];
            for value in dir.iter_mut() {
                *value = rng.gauss(0.0, 1.0);
            }
            normalize(&mut dir);
            let dim = params.dim;
            Self {
                z: vec![0.0; dim],
                dir,
                params,
                rng,
                frac_backend: FracBackend::CpuRadix2,
                temp: 1.0,
                temp_min: 0.1,
                temp_max: 4.0,
                structural: vec![0.0; dim],
            }
        }

        pub fn z(&self) -> &[f64] {
            &self.z
        }

        /// Returns the current exploration temperature.
        pub fn temperature(&self) -> f64 {
            self.temp
        }

        /// Overrides the exploration temperature bounds.
        pub fn set_temp_bounds(&mut self, t_min: f64, t_max: f64) {
            let mut lower = t_min.max(0.0);
            let mut upper = t_max.max(lower + f64::EPSILON);
            if lower > upper {
                std::mem::swap(&mut lower, &mut upper);
            }
            self.temp_min = lower;
            self.temp_max = upper;
            self.temp = self.temp.clamp(self.temp_min, self.temp_max);
        }

        /// Adjusts the exploration temperature using retry/gradient signals.
        pub fn temp_schedule(&mut self, retry: f64, grad_norm: f64, loss_var: f64) {
            let stagnation = (1.0 - (loss_var / (1.0 + loss_var))).clamp(0.0, 1.0);
            let grad_term = (grad_norm / (1.0 + grad_norm)).clamp(0.0, 1.0);
            let instability = retry + 0.5 * grad_term;
            let delta = 0.2 * stagnation - 0.3 * instability;
            self.temp = (self.temp + delta).clamp(self.temp_min, self.temp_max);
        }

        /// Sets the backend used for fractional regularisation.
        pub fn set_frac_backend(&mut self, backend: FracBackend) {
            self.frac_backend = backend;
        }

        pub fn frac_penalty(&self) -> f64 {
            frac_penalty_backend(
                &self.z,
                self.params.alpha_frac,
                self.params.lam_frac,
                &self.frac_backend,
            )
        }

        pub fn frac_penalty_proposed(&self) -> f64 {
            let proposed: Vec<f64> = self
                .z
                .iter()
                .zip(self.dir.iter())
                .map(|(z, d)| z + self.params.sigma * d)
                .collect();
            frac_penalty_backend(
                &proposed,
                self.params.alpha_frac,
                self.params.lam_frac,
                &self.frac_backend,
            )
        }

        pub(super) fn validate_adaptation_state(&self) -> Result<(), &'static str> {
            if self.z.len() != self.params.dim
                || self.dir.len() != self.params.dim
                || self.structural.len() != self.params.dim
            {
                return Err("zmeta.vector_shape");
            }
            for (field, value) in [
                ("zmeta.params.sigma", self.params.sigma),
                ("zmeta.params.lr", self.params.lr),
                ("zmeta.params.alpha_frac", self.params.alpha_frac),
                ("zmeta.params.lam_frac", self.params.lam_frac),
                ("zmeta.params.orientation_eta", self.params.orientation_eta),
                ("zmeta.params.orientation_eps", self.params.orientation_eps),
                ("zmeta.temperature", self.temp),
                ("zmeta.temperature_min", self.temp_min),
                ("zmeta.temperature_max", self.temp_max),
            ] {
                if !value.is_finite() {
                    return Err(field);
                }
            }
            for (field, value) in [
                ("zmeta.params.sigma", self.params.sigma),
                ("zmeta.params.lr", self.params.lr),
                ("zmeta.params.alpha_frac", self.params.alpha_frac),
                ("zmeta.params.lam_frac", self.params.lam_frac),
                ("zmeta.params.orientation_eta", self.params.orientation_eta),
            ] {
                if value < 0.0 {
                    return Err(field);
                }
            }
            if self.params.orientation_eps <= 0.0 {
                return Err("zmeta.params.orientation_eps");
            }
            if self.temp_min < 0.0
                || self.temp_max < self.temp_min
                || self.temp < self.temp_min
                || self.temp > self.temp_max
            {
                return Err("zmeta.temperature_bounds");
            }
            if self
                .z
                .iter()
                .chain(self.dir.iter())
                .chain(self.structural.iter())
                .any(|value| !value.is_finite())
            {
                return Err("zmeta.vector_state");
            }
            let direction_norm = stable_norm(&self.dir);
            if !direction_norm.is_finite()
                || (!self.dir.is_empty() && (direction_norm - 1.0).abs() > 1.0e-6)
            {
                return Err("zmeta.direction_norm");
            }
            let structural_norm = stable_norm(&self.structural);
            if !structural_norm.is_finite() || structural_norm > 1.0 + 1.0e-6 {
                return Err("zmeta.structural_norm");
            }
            Ok(())
        }

        pub fn update(
            &mut self,
            reward_current: f64,
            reward_proposed: f64,
            structural: Option<&[f64]>,
        ) {
            let delta_reward = reward_proposed - reward_current;
            let structural_delta = self.ingest_structural(structural);
            let improved = delta_reward > 0.0;

            if improved {
                for (z, d) in self.z.iter_mut().zip(self.dir.iter()) {
                    *z += self.params.lr * (self.params.sigma * d);
                }
                for dir in self.dir.iter_mut() {
                    *dir = 0.7 * (*dir) + 0.3 * self.rng.gauss(0.0, 1.0);
                }
            } else {
                let mut kick: Vec<f64> = (0..self.params.dim)
                    .map(|_| self.rng.gauss(0.0, 0.5))
                    .collect();
                let projection = dot(&kick, &self.dir);
                for (k, d) in kick.iter_mut().zip(self.dir.iter()) {
                    *k -= projection * d;
                }
                for (dir, kick) in self.dir.iter_mut().zip(kick.iter()) {
                    *dir = 0.9 * (*dir) + 0.1 * (*kick);
                }
            }

            normalize(&mut self.dir);

            if let Some(delta) = structural_delta {
                self.apply_structural_drive(delta, delta_reward);
            }
        }

        #[allow(dead_code)]
        fn ingest_structural_legacy(&mut self, structural: Option<&[f64]>) -> Option<Vec<f64>> {
            let raw = structural?;
            if self.params.dim == 0 {
                return None;
            }

            let mut new_vec = vec![0.0f64; self.params.dim];
            let mut any = false;
            for (idx, slot) in new_vec.iter_mut().enumerate() {
                if let Some(value) = raw.get(idx).copied() {
                    if value.is_finite() {
                        *slot = value;
                        any |= value.abs() > 1e-12;
                    }
                }
            }
            if !any {
                return None;
            }

            if !normalize_above(&mut new_vec, 1.0e-9) {
                return None;
            }

            let prev = self.structural.clone();
            self.structural = new_vec;
            Some(
                self.structural
                    .iter()
                    .zip(prev.iter())
                    .map(|(new, old)| new - old)
                    .collect(),
            )
        }

        fn ingest_structural(&mut self, structural: Option<&[f64]>) -> Option<Vec<f64>> {
            let raw = structural?;
            if self.params.dim == 0 {
                return None;
            }

            let mut new_vec = vec![0.0f64; self.params.dim];
            let mut any = false;
            for (idx, slot) in new_vec.iter_mut().enumerate() {
                if let Some(value) = raw.get(idx).copied() {
                    if value.is_finite() {
                        *slot = value;
                        any |= value.abs() > 1e-12;
                    }
                }
            }
            if !any {
                return None;
            }

            if !normalize_above(&mut new_vec, 1.0e-9) {
                return None;
            }

            let prev = self.structural.clone();
            self.structural = new_vec;
            Some(
                self.structural
                    .iter()
                    .zip(prev.iter())
                    .map(|(new, old)| new - old)
                    .collect(),
            )
        }

        #[allow(dead_code)]
        fn apply_structural_drive_legacy(&mut self, mut delta: Vec<f64>, delta_reward: f64) {
            if delta_reward.abs() <= 1e-9 {
                return;
            }
            let gain = delta_reward.tanh();
            if !gain.is_finite() || gain.abs() <= 1e-6 {
                return;
            }

            let delta_norm = stable_norm(&delta);
            if delta_norm <= 1e-9 {
                return;
            }

            for value in delta.iter_mut() {
                *value *= gain;
            }

            self.logistic_project_step(&delta);
        }

        fn apply_structural_drive(&mut self, mut delta: Vec<f64>, delta_reward: f64) {
            if delta_reward.abs() <= 1e-9 {
                return;
            }
            let gain = delta_reward.tanh();
            if !gain.is_finite() || gain.abs() <= 1e-6 {
                return;
            }

            let delta_norm = stable_norm(&delta);
            if delta_norm <= 1e-9 {
                return;
            }

            for value in delta.iter_mut() {
                *value *= gain;
            }

            self.logistic_project_step(&delta);
        }

        #[allow(dead_code)]
        fn logistic_project_step_legacy(&mut self, drive: &[f64]) {
            if self.dir.is_empty() {
                return;
            }

            let structural_norm_sq = self.structural.iter().map(|v| v * v).sum::<f64>();
            if structural_norm_sq <= 1e-12 {
                return;
            }

            let mut projected = Vec::with_capacity(self.dir.len());
            let dot_nd = dot(&self.dir, drive);
            for (n, d) in self.dir.iter().zip(drive.iter()) {
                projected.push(d - dot_nd * n);
            }

            let proj_norm_sq = projected.iter().map(|v| v * v).sum::<f64>();
            if proj_norm_sq <= 1e-12 {
                return;
            }

            // unused 警告を抑制（意味は変えない）
            let _proj_norm = proj_norm_sq.sqrt();

            let dot_nr = self
                .dir
                .iter()
                .zip(self.structural.iter())
                .map(|(n, r)| n * r)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            let p_t = 0.5 * (1.0 + dot_nr);
            let eps = self.params.orientation_eps.max(1e-6);
            let denom = 2.0 * p_t * (1.0 - p_t) + eps;
            let eta = self.params.orientation_eta.max(0.0);
            if eta <= 0.0 {
                return;
            }

            for (n, proj) in self.dir.iter_mut().zip(projected.iter()) {
                *n += eta * (*proj) / denom;
            }
            normalize(&mut self.dir);
        }

        fn logistic_project_step(&mut self, drive: &[f64]) {
            if self.dir.is_empty() {
                return;
            }

            let structural_norm = stable_norm(&self.structural);
            if structural_norm <= 1.0e-6 {
                return;
            }

            let mut projected = Vec::with_capacity(self.dir.len());
            let dot_nd = dot(&self.dir, drive);
            for (n, d) in self.dir.iter().zip(drive.iter()) {
                projected.push(d - dot_nd * n);
            }

            let proj_norm = stable_norm(&projected);
            if proj_norm <= 1.0e-12 {
                return;
            }

            let scale = 1.0 / proj_norm.max(1.0);
            for value in projected.iter_mut() {
                *value *= scale;
            }

            let dot_nr = self
                .dir
                .iter()
                .zip(self.structural.iter())
                .map(|(n, r)| n * r)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            let p_t = 0.5 * (1.0 + dot_nr);
            let eps = self.params.orientation_eps.max(1e-6);
            let denom = 2.0 * p_t * (1.0 - p_t) + eps;
            let eta = self.params.orientation_eta.max(0.0);
            if eta <= 0.0 {
                return;
            }

            for (n, proj) in self.dir.iter_mut().zip(projected.iter()) {
                *n += eta * (*proj) / denom;
            }
            normalize(&mut self.dir);
        }
    }

    fn normalize(vec: &mut [f64]) {
        let _ = normalize_above(vec, 0.0);
    }

    fn normalize_above(vec: &mut [f64], minimum_norm: f64) -> bool {
        let scale = vec.iter().map(|value| value.abs()).fold(0.0f64, f64::max);
        if !scale.is_finite() || scale == 0.0 {
            return false;
        }
        let scaled_norm = vec
            .iter()
            .map(|value| value / scale)
            .fold(0.0f64, |norm, value| norm.hypot(value));
        if !scaled_norm.is_finite() || scaled_norm == 0.0 || scale <= minimum_norm / scaled_norm {
            return false;
        }
        for value in vec {
            *value = (*value / scale) / scaled_norm;
        }
        true
    }

    fn stable_norm(vec: &[f64]) -> f64 {
        vec.iter().fold(0.0, |norm, value| norm.hypot(*value))
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn frac_penalty(z: &[f64], alpha: f64, lam: f64) -> f64 {
        if z.len() < 3 {
            return 0.0;
        }
        let mut acc = 0.0;
        for idx in 1..(z.len() - 1) {
            let d2 = z[idx - 1] - 2.0 * z[idx] + z[idx + 1];
            acc += d2.abs().powf(1.0 + alpha);
        }
        lam * acc
    }

    fn frac_penalty_backend(z: &[f64], alpha: f64, lam: f64, backend: &FracBackend) -> f64 {
        let base = frac_penalty(z, alpha, lam);
        match backend {
            FracBackend::CpuRadix2 => base,
            FracBackend::Wgpu { radix } => {
                let radix_factor = (*radix as f64).max(2.0) / 2.0;
                base * (1.0 + 0.15 * (radix_factor - 1.0))
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn norm(vec: &[f64]) -> f64 {
            vec.iter().map(|v| v * v).sum::<f64>().sqrt()
        }

        #[test]
        fn logistic_projection_keeps_unit_norm() {
            let params = ZMetaParams {
                dim: 3,
                seed: 7,
                orientation_eta: 0.2,
                orientation_eps: 5e-3,
                ..Default::default()
            };
            let mut es = ZMetaES::new(params);
            assert!((norm(&es.dir) - 1.0).abs() < 1e-6);

            let context1 = [0.2, 0.5, -0.3];
            let delta1 = es.ingest_structural(Some(&context1)).unwrap();
            es.apply_structural_drive(delta1, 0.15);
            assert!((norm(&es.dir) - 1.0).abs() < 1e-6);

            let context2 = [0.8, -0.1, 0.3];
            let delta2 = es.ingest_structural(Some(&context2)).unwrap();
            let dir_before = es.dir.clone();
            es.apply_structural_drive(delta2, 0.25);
            assert!((norm(&es.dir) - 1.0).abs() < 1e-6);
            let delta_dir = es
                .dir
                .iter()
                .zip(dir_before.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            assert!(delta_dir > 1e-6);
        }

        #[test]
        fn structural_normalisation_handles_large_finite_contexts() {
            let mut es = ZMetaES::new(ZMetaParams {
                dim: 3,
                seed: 11,
                ..Default::default()
            });
            let delta = es
                .ingest_structural(Some(&[f64::MAX, f64::MAX, 0.0]))
                .expect("large finite context remains representable");

            assert!(delta.iter().all(|value| value.is_finite()));
            assert!((norm(&es.structural) - 1.0).abs() < 1.0e-12);
            es.apply_structural_drive(delta, 0.25);
            es.validate_adaptation_state()
                .expect("stable normalisation preserves valid state");
        }
    }

    mod randless {
        use std::cell::Cell;

        pub trait Rng {
            fn gauss(&self, mu: f64, sigma: f64) -> f64;
        }

        #[derive(Clone)]
        pub struct StdRng {
            state: Cell<u64>,
        }

        impl StdRng {
            pub fn seed_from_u64(seed: u64) -> Self {
                Self {
                    state: Cell::new(seed | 1),
                }
            }

            fn next_u64(&self) -> u64 {
                let mut x = self.state.get();
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.state.set(x);
                x
            }

            fn uniform01(&self) -> f64 {
                (self.next_u64() as f64) / (u64::MAX as f64)
            }
        }

        impl Rng for StdRng {
            fn gauss(&self, mu: f64, sigma: f64) -> f64 {
                let u1 = self.uniform01().max(1e-12);
                let u2 = self.uniform01();
                let radius = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                mu + sigma * radius * theta.cos()
            }
        }
    }
}

// =================== bandit.rs ===================
pub mod bandit {

    #[derive(Clone, Copy, Debug)]
    pub enum SoftBanditMode {
        TS,
        UCB,
    }

    #[derive(Clone, Debug)]
    pub struct LinTSArm {
        dim: usize,
        a: Vec<f64>,
        b: Vec<f64>,
    }

    impl LinTSArm {
        pub fn new(dim: usize, lambda: f64) -> Self {
            Self {
                dim,
                a: eye_flat(dim, lambda),
                b: vec![0.0; dim],
            }
        }

        pub fn sample_score(&self, x: &[f64]) -> f64 {
            let mean = solve_spd_diag(&self.a, &self.b, self.dim);
            dot(&mean, x)
        }

        pub fn ucb_score(&self, x: &[f64], c: f64) -> f64 {
            let ainv = inv_spd_diag(&self.a, self.dim);
            let mean = matvec_flat(&ainv, &self.b, self.dim);
            let variance = quad_form(&ainv, x);
            dot(&mean, x) + c * variance.max(0.0).sqrt()
        }

        pub fn update(&mut self, x: &[f64], reward: f64) {
            rank1_add(&mut self.a, x, self.dim);
            for (bi, xi) in self.b.iter_mut().zip(x) {
                *bi += reward * xi;
            }
        }

        fn validate_state(&self, context_dim: usize) -> Result<(), &'static str> {
            let matrix_len = self
                .dim
                .checked_mul(self.dim)
                .ok_or("bandit.posterior_shape")?;
            if self.dim != context_dim || self.a.len() != matrix_len || self.b.len() != self.dim {
                return Err("bandit.posterior_shape");
            }
            if self
                .a
                .iter()
                .chain(self.b.iter())
                .any(|value| !value.is_finite())
            {
                return Err("bandit.posterior_state");
            }
            for index in 0..self.dim {
                if self.a[index * self.dim + index] <= 0.0 {
                    return Err("bandit.posterior_diagonal");
                }
            }
            Ok(())
        }
    }

    #[derive(Clone)]
    pub struct SoftBandit {
        choices: Vec<String>,
        arms: Vec<LinTSArm>,
        last_index: usize,
        mode: SoftBanditMode,
    }

    impl SoftBandit {
        pub fn new(choices: Vec<String>, feat_dim: usize, mode: SoftBanditMode) -> Self {
            let arms = (0..choices.len())
                .map(|_| LinTSArm::new(feat_dim, 1.0))
                .collect();
            Self {
                choices,
                arms,
                last_index: 0,
                mode,
            }
        }

        pub fn select(&mut self, x: &[f64]) -> String {
            let mut best = f64::MIN;
            let mut idx = 0usize;
            for (i, arm) in self.arms.iter().enumerate() {
                let score = match self.mode {
                    SoftBanditMode::TS => arm.sample_score(x),
                    SoftBanditMode::UCB => arm.ucb_score(x, 1.0),
                };
                if score > best {
                    best = score;
                    idx = i;
                }
            }
            self.last_index = idx;
            self.choices[idx].clone()
        }

        pub fn update_last(&mut self, x: &[f64], reward: f64) {
            if let Some(arm) = self.arms.get_mut(self.last_index) {
                arm.update(x, reward);
            }
        }

        pub(super) fn validate_state(&self, context_dim: usize) -> Result<(), &'static str> {
            if self.choices.is_empty()
                || self.choices.len() != self.arms.len()
                || self.last_index >= self.choices.len()
            {
                return Err("bandit.choice_state");
            }
            self.arms
                .iter()
                .try_for_each(|arm| arm.validate_state(context_dim))
        }

        pub(super) fn validate_selection_scores(
            &self,
            context: &[f64],
        ) -> Result<(), &'static str> {
            for arm in &self.arms {
                let score = match self.mode {
                    SoftBanditMode::TS => arm.sample_score(context),
                    SoftBanditMode::UCB => arm.ucb_score(context, 1.0),
                };
                if !score.is_finite() {
                    return Err("bandit.selection_score");
                }
            }
            Ok(())
        }
    }

    fn eye_flat(dim: usize, lambda: f64) -> Vec<f64> {
        let mut matrix = vec![0.0; dim * dim];
        for i in 0..dim {
            matrix[i * dim + i] = lambda;
        }
        matrix
    }

    fn matvec_flat(a: &[f64], x: &[f64], dim: usize) -> Vec<f64> {
        let mut y = vec![0.0; dim];
        for i in 0..dim {
            let mut sum = 0.0;
            for j in 0..dim {
                sum += a[i * dim + j] * x[j];
            }
            y[i] = sum;
        }
        y
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn rank1_add(a: &mut [f64], x: &[f64], dim: usize) {
        for i in 0..dim {
            for j in 0..dim {
                a[i * dim + j] += x[i] * x[j];
            }
        }
    }

    fn inv_spd_diag(a: &[f64], dim: usize) -> Vec<f64> {
        let mut inv = vec![0.0; dim * dim];
        for i in 0..dim {
            let value = a[i * dim + i];
            inv[i * dim + i] = if value.abs() > 1e-12 {
                1.0 / value
            } else {
                1.0
            };
        }
        inv
    }

    fn solve_spd_diag(a: &[f64], b: &[f64], dim: usize) -> Vec<f64> {
        let mut x = vec![0.0; dim];
        for i in 0..dim {
            let value = a[i * dim + i];
            x[i] = if value.abs() > 1e-12 {
                b[i] / value
            } else {
                b[i]
            };
        }
        x
    }

    fn quad_form(a: &[f64], x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() {
            for j in 0..x.len() {
                sum += x[i] * a[i * x.len() + j] * x[j];
            }
        }
        sum
    }
}

// =================== wilson.rs ===================
pub mod wilson {
    pub fn wilson_lower(successes: i32, trials: i32, z: f64) -> f64 {
        if trials <= 0 {
            return 0.0;
        }
        let n = trials as f64;
        let s = successes as f64;
        let p = (s / n).clamp(0.0, 1.0);
        let denom = 1.0 + z * z / n;
        let center = (p + z * z / (2.0 * n)) / denom;
        let radius = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).max(0.0).sqrt() / denom;
        (center - radius).max(0.0)
    }
}

// =================== rewrite.rs ===================
pub mod rewrite {
    use std::collections::HashMap;
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::path::PathBuf;

    pub struct HeurStore {
        path: PathBuf,
    }

    impl HeurStore {
        pub fn new(custom: Option<String>) -> Self {
            let path = custom.map(PathBuf::from).unwrap_or(default_path());
            if let Some(dir) = path.parent() {
                let _ = fs::create_dir_all(dir);
            }
            Self { path }
        }

        pub fn append(&self, rule_text: &str, info: &HashMap<String, f64>) {
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
            {
                let meta = serde_like_json(info);
                let line = format!("{}  # {}\n", rule_text.trim(), meta);
                let _ = file.write_all(line.as_bytes());
            }
        }

        pub fn path(&self) -> &PathBuf {
            &self.path
        }
    }

    fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        let mut path = PathBuf::from(home);
        path.push(".spiraltorch/heur/heur.kdsl");
        path
    }

    fn serde_like_json(info: &HashMap<String, f64>) -> String {
        let mut out = String::from("{");
        let mut first = true;
        for (key, value) in info.iter() {
            if !first {
                out.push_str(", ");
            }
            first = false;
            out.push_str(&format!("\"{}\":{:.6}", key, value));
        }
        out.push('}');
        out
    }
}

// =================== ab.rs ===================
pub mod ab {
    use super::{RewardCfg, StepMetrics};

    pub struct ABRunner {
        reward: RewardCfg,
    }

    impl ABRunner {
        pub fn new(reward: RewardCfg) -> Self {
            Self { reward }
        }

        pub fn run<F, G>(&self, mut a: F, mut b: G, trials: usize) -> (usize, usize)
        where
            F: FnMut() -> StepMetrics,
            G: FnMut() -> StepMetrics,
        {
            let mut wins_a = 0usize;
            let mut wins_b = 0usize;
            for _ in 0..trials {
                let ma = a();
                let mb = b();
                let ra = self.reward.score(&ma, 0.0);
                let rb = self.reward.score(&mb, 0.0);
                if ra > rb {
                    wins_a += 1;
                } else {
                    wins_b += 1;
                }
            }
            (wins_a, wins_b)
        }
    }
}
