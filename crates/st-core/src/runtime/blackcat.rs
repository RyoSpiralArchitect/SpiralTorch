// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

use bandit::{BanditDecisionWitness, SoftBandit, SoftBanditMode};
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

/// Stable semantic contract emitted with every contextual bandit decision.
pub const BLACKCAT_BANDIT_CONTRACT: &str = "spiraltorch.blackcat.contextual-bandit";
/// Schema version for [`BlackCatSelectionWitness`].
pub const BLACKCAT_BANDIT_CONTRACT_VERSION: u32 = 1;
/// Maximum accepted contextual feature width for guarded construction.
pub const BLACKCAT_MAX_FEATURE_DIM: usize = 256;
/// Maximum accepted named choice groups for guarded construction.
pub const BLACKCAT_MAX_CHOICE_GROUPS: usize = 128;
/// Maximum accepted choices in one group for guarded construction.
pub const BLACKCAT_MAX_CHOICES_PER_GROUP: usize = 256;
/// Aggregate precision-matrix cells permitted across all arms.
pub const BLACKCAT_MAX_POSTERIOR_CELLS: usize = 4_194_304;
/// Maximum accepted ZMeta search dimension for guarded construction.
pub const BLACKCAT_MAX_ZMETA_DIM: usize = 4096;

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
    #[error("BlackCat received a hint for unknown choice group '{id}'")]
    UnknownChoiceHint { id: String },
    #[error("BlackCat choice group '{id}' does not admit hinted value '{choice}'")]
    InvalidChoiceHint { id: String, choice: String },
    #[error("BlackCat selection {selection_id} is still awaiting reward or explicit abandonment")]
    PendingSelection { selection_id: u64 },
    #[error("BlackCat cannot choose because no non-empty choice groups are configured")]
    NoChoiceGroups,
    #[error("invalid BlackCat configuration at '{field}': {detail}")]
    InvalidConfig { field: &'static str, detail: String },
}

/// Complete, replay-oriented witness for one contextual bandit selection.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BlackCatSelectionWitness {
    pub contract: &'static str,
    pub contract_version: u32,
    pub selection_id: u64,
    pub context: Vec<f64>,
    pub hints: BTreeMap<String, String>,
    pub picks: BTreeMap<String, String>,
    pub decisions: BTreeMap<String, BanditDecisionWitness>,
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
    arms: BTreeMap<String, SoftBandit>,
}

/// Decision evidence and executable picks for one multi-group choice.
pub type MultiBanditSelection = (
    HashMap<String, String>,
    BTreeMap<String, BanditDecisionWitness>,
);

impl MultiBandit {
    /// Compatibility constructor. Invalid aggregate domains produce an empty,
    /// fail-closed sentinel; prefer [`Self::try_new_seeded`] for direct use.
    pub fn new(groups: &ChoiceGroups, feat_dim: usize, mode: SoftBanditMode) -> Self {
        Self::new_seeded(
            groups,
            feat_dim,
            mode,
            determinism::config().seed_for("st-core/blackcat/bandits"),
        )
    }

    /// Builds bandits whose per-group random streams are derived from one
    /// explicit seed. Invalid aggregate domains produce an empty fail-closed
    /// sentinel; group insertion order cannot change valid streams.
    pub fn new_seeded(
        groups: &ChoiceGroups,
        feat_dim: usize,
        mode: SoftBanditMode,
        seed: u64,
    ) -> Self {
        Self::try_new_seeded(groups, feat_dim, mode, seed).unwrap_or_else(|_| Self {
            arms: BTreeMap::new(),
        })
    }

    /// Guarded constructor that bounds the aggregate posterior allocation,
    /// validates every finite choice domain, and derives stable group streams.
    pub fn try_new_seeded(
        groups: &ChoiceGroups,
        feat_dim: usize,
        mode: SoftBanditMode,
        seed: u64,
    ) -> Result<Self, BlackCatError> {
        validate_choice_groups(groups, feat_dim)?;
        let mut arms = BTreeMap::new();
        for (name, opts) in &groups.groups {
            let bandit = SoftBandit::try_new_seeded(
                opts.clone(),
                feat_dim,
                mode,
                bandit::derive_group_seed(seed, name),
            )
            .map_err(|field| BlackCatError::InvalidConfig {
                field,
                detail: format!("choice group {name:?} could not initialize its posterior"),
            })?;
            arms.insert(name.clone(), bandit);
        }
        Ok(Self { arms })
    }

    pub fn select_all(&mut self, context: &[f64]) -> HashMap<String, String> {
        match self.try_select_all(context) {
            Ok((picks, _)) => picks,
            Err(_) => HashMap::new(),
        }
    }

    /// Guarded selection for every group without an external hint prior.
    pub fn try_select_all(
        &mut self,
        context: &[f64],
    ) -> Result<MultiBanditSelection, BlackCatError> {
        self.try_select_all_with_hints(context, &BTreeMap::new())
    }

    /// Selects every group. A validated hint resolves only an equivalent
    /// posterior projection; otherwise the sampled TS or deterministic UCB
    /// score remains authoritative. Selection is atomic across all groups.
    pub fn try_select_all_with_hints(
        &mut self,
        context: &[f64],
        hints: &BTreeMap<String, String>,
    ) -> Result<MultiBanditSelection, BlackCatError> {
        if self.is_empty() {
            return Err(BlackCatError::NoChoiceGroups);
        }
        self.validate_hints(hints)?;
        let mut next = self.clone();
        next.validate_state(context.len())
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        next.validate_selection_scores(context)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        let selection = next
            .select_all_with_hints_in_place(context, hints)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        next.validate_state(context.len())
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        *self = next;
        Ok(selection)
    }

    fn select_all_with_hints_in_place(
        &mut self,
        context: &[f64],
        hints: &BTreeMap<String, String>,
    ) -> Result<MultiBanditSelection, &'static str> {
        let mut picks = HashMap::new();
        let mut decisions = BTreeMap::new();
        for (name, bandit) in self.arms.iter_mut() {
            let decision =
                bandit.try_select_with_hint(context, hints.get(name).map(String::as_str))?;
            picks.insert(name.clone(), decision.chosen.clone());
            decisions.insert(name.clone(), decision);
        }
        Ok((picks, decisions))
    }

    pub fn try_update_all(&mut self, context: &[f64], reward: f64) -> Result<(), &'static str> {
        let mut next = self.clone();
        for bandit in next.arms.values_mut() {
            bandit.try_update_last(context, reward)?;
        }
        *self = next;
        Ok(())
    }

    /// Releases all pending group decisions without posterior credit.
    pub fn try_abandon_all(&mut self) -> Result<(), &'static str> {
        let mut next = self.clone();
        for bandit in next.arms.values_mut() {
            bandit.try_abandon_last_selection()?;
        }
        *self = next;
        Ok(())
    }

    /// Reports the common pending state and rejects impossible cross-group
    /// disagreement.
    pub fn selection_pending(&self) -> Result<bool, &'static str> {
        let mut states = self.arms.values().map(SoftBandit::selection_pending);
        let Some(expected) = states.next() else {
            return Ok(false);
        };
        if states.any(|pending| pending != expected) {
            return Err("bandit.pending_state");
        }
        Ok(expected)
    }

    /// Compatibility update path. Guarded runtime code uses
    /// [`Self::try_update_all`] and commits a cloned state atomically.
    pub fn update_all(&mut self, context: &[f64], reward: f64) {
        let _ = self.try_update_all(context, reward);
    }

    /// Stable snapshot of every named choice domain owned by the bandits.
    pub fn choice_domains(&self) -> BTreeMap<String, Vec<String>> {
        self.arms
            .iter()
            .map(|(name, bandit)| (name.clone(), bandit.choices().to_vec()))
            .collect()
    }

    fn validate_state(&self, context_dim: usize) -> Result<(), &'static str> {
        self.arms.iter().try_for_each(|(name, bandit)| {
            if name.trim().is_empty() {
                return Err("bandit.group_id");
            }
            bandit.validate_state(context_dim)
        })
    }

    fn validate_selection_scores(&self, context: &[f64]) -> Result<(), &'static str> {
        self.arms
            .values()
            .try_for_each(|bandit| bandit.validate_selection_scores(context))
    }

    fn is_empty(&self) -> bool {
        self.arms.is_empty()
    }

    /// Stable observation counts for every choice in every group.
    pub fn observation_counts(&self) -> BTreeMap<String, BTreeMap<String, u64>> {
        self.arms
            .iter()
            .map(|(name, bandit)| (name.clone(), bandit.observation_counts()))
            .collect()
    }

    fn validate_hints(&self, hints: &BTreeMap<String, String>) -> Result<(), BlackCatError> {
        for (id, choice) in hints {
            let Some(bandit) = self.arms.get(id) else {
                return Err(BlackCatError::UnknownChoiceHint { id: id.clone() });
            };
            if !bandit.choices().contains(choice) {
                return Err(BlackCatError::InvalidChoiceHint {
                    id: id.clone(),
                    choice: choice.clone(),
                });
            }
        }
        Ok(())
    }
}

/// BlackCat orchestrator that joins ES search with contextual bandits.
pub struct BlackCatRuntime {
    z: ZMetaES,
    bandits: MultiBandit,
    heur: HeurStore,
    reward: RewardCfg,
    configuration_error: Option<BlackCatError>,
    context_dim: usize,
    last_context: Vec<f64>,
    last_picks: HashMap<String, String>,
    last_selection_witness: Option<BlackCatSelectionWitness>,
    selection_pending: bool,
    selection_counter: u64,
    last_credited_selection_id: Option<u64>,
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
    /// Compatibility constructor. Invalid configuration is retained as a
    /// fail-closed sentinel without allocating attacker-sized posterior state;
    /// prefer [`Self::try_new`] for an explicit construction error.
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
        let mut configuration_error =
            validate_blackcat_configuration(&z_params, &groups, feat_dim).err();
        let valid_configuration = configuration_error.is_none();
        let effective_feat_dim = if valid_configuration { feat_dim } else { 1 };
        let mut params = if valid_configuration {
            z_params
        } else {
            ZMetaParams::default()
        };
        if determinism::config().enabled {
            params.seed = determinism::config().seed_for("st-core/blackcat/zmeta");
        }
        let bandits = if valid_configuration {
            match MultiBandit::try_new_seeded(&groups, effective_feat_dim, mode, params.seed) {
                Ok(bandits) => bandits,
                Err(error) => {
                    configuration_error = Some(error);
                    MultiBandit {
                        arms: BTreeMap::new(),
                    }
                }
            }
        } else {
            MultiBandit {
                arms: BTreeMap::new(),
            }
        };
        let heur = HeurStore::new(heur_path);
        let stats_alpha = 0.2;
        let runtime = Self {
            z: ZMetaES::new(params),
            bandits,
            heur,
            reward: RewardCfg::default(),
            configuration_error,
            context_dim: effective_feat_dim,
            last_context: vec![0.0; effective_feat_dim],
            last_picks: HashMap::new(),
            last_selection_witness: None,
            selection_pending: false,
            selection_counter: 0,
            last_credited_selection_id: None,
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

    /// Guarded constructor for a directly usable Rust runtime.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        z_params: ZMetaParams,
        groups: ChoiceGroups,
        feat_dim: usize,
        mode: SoftBanditMode,
        heur_path: Option<String>,
    ) -> Result<Self, BlackCatError> {
        validate_blackcat_configuration(&z_params, &groups, feat_dim)?;
        let runtime = Self::new(z_params, groups, feat_dim, mode, heur_path);
        match runtime.configuration_error.clone() {
            Some(error) => Err(error),
            None => Ok(runtime),
        }
    }

    /// Call at the beginning of a training step.
    #[instrument(skip(self))]
    pub fn begin_step(&mut self) {
        self.last_step_start = Some(Instant::now());
    }

    /// Compatibility context builder. Invalid dimensions or observations
    /// produce an empty context that guarded selection rejects.
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
        self.try_make_context(batches, tiles, depth, device_code, load, extras, feat_dim)
            .unwrap_or_default()
    }

    /// Builds a bounded contextual feature vector owned by this runtime's
    /// declared feature contract. `feat_dim == 0` selects that declared width;
    /// any other value must match it exactly.
    #[allow(
        clippy::too_many_arguments,
        reason = "Context vector requires these inputs for bandit feature parity"
    )]
    pub fn try_make_context(
        &self,
        batches: u32,
        tiles: u32,
        depth: u32,
        device_code: u32,
        load: f64,
        extras: &[(String, f64)],
        feat_dim: usize,
    ) -> Result<Vec<f64>, BlackCatError> {
        self.validate_configuration()?;
        let target_dim = if feat_dim == 0 {
            self.context_dim
        } else {
            feat_dim
        };
        if target_dim != self.context_dim {
            return Err(BlackCatError::ContextDimension {
                expected: self.context_dim,
                actual: target_dim,
            });
        }
        let available_extra_features = target_dim.saturating_sub(6);
        if extras.len() > available_extra_features {
            return Err(BlackCatError::InvalidConfig {
                field: "context.extras",
                detail: format!(
                    "at most {available_extra_features} extra values fit feature dimension {target_dim}, got {}",
                    extras.len()
                ),
            });
        }
        validate_blackcat_finite("context.load", load)?;
        for (_, value) in extras {
            validate_blackcat_finite("context.extra", *value)?;
        }
        let base = [
            1.0,
            batches as f64,
            tiles as f64,
            depth as f64,
            (device_code % 1024) as f64 / 1024.0,
            load,
        ];
        let mut ctx = Vec::with_capacity(target_dim);
        ctx.extend(base.into_iter().take(target_dim));
        ctx.extend(extras.iter().map(|(_, value)| *value));
        ctx.resize(target_dim, 0.0);
        Ok(ctx)
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
        self.try_choose_with_hints(context, &BTreeMap::new())
    }

    /// Guarded choice path with deterministic equivalent-posterior priors.
    /// Hints must name existing groups and values. Once projected posterior
    /// mean or uncertainty differs, the sampled TS or UCB score decides.
    pub fn try_choose_with_hints(
        &mut self,
        context: Vec<f64>,
        hints: &BTreeMap<String, String>,
    ) -> Result<HashMap<String, String>, BlackCatError> {
        self.validate_configuration()?;
        if self.selection_pending {
            return Err(BlackCatError::PendingSelection {
                selection_id: self.selection_counter,
            });
        }
        if self.bandits.is_empty() {
            return Err(BlackCatError::NoChoiceGroups);
        }
        self.validate_context(&context)?;
        self.validate_selection_state()?;
        self.bandits
            .validate_state(self.context_dim)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        self.bandits.validate_hints(hints)?;
        let mut next_bandits = self.bandits.clone();
        next_bandits
            .validate_selection_scores(&context)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        let (picks, decisions) = next_bandits
            .select_all_with_hints_in_place(&context, hints)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        next_bandits
            .validate_state(self.context_dim)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        let selection_id =
            self.selection_counter
                .checked_add(1)
                .ok_or(BlackCatError::InvalidAdaptationState {
                    field: "bandit.selection_count",
                })?;
        let witness = BlackCatSelectionWitness {
            contract: BLACKCAT_BANDIT_CONTRACT,
            contract_version: BLACKCAT_BANDIT_CONTRACT_VERSION,
            selection_id,
            context: context.clone(),
            hints: hints.clone(),
            picks: picks
                .iter()
                .map(|(id, choice)| (id.clone(), choice.clone()))
                .collect(),
            decisions,
        };
        self.bandits = next_bandits;
        self.last_context = context;
        self.last_picks = picks.clone();
        self.last_selection_witness = Some(witness);
        self.selection_pending = true;
        self.selection_counter = selection_id;
        debug!(picks = ?self.last_picks, "blackcat bandit picks");
        let bus = global_registry().event_bus();
        if bus.has_listeners("BlackCatChoose") {
            bus.publish(&PluginEvent::custom(
                "BlackCatChoose",
                serde_json::json!({
                    "selection": &self.last_selection_witness,
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

    /// Updates ES, statistics, and telemetry from one canonical free-energy
    /// report. A contextual posterior is updated only when a pending selection
    /// exists, and that selection is credited at most once. All state commits
    /// only after validation succeeds.
    #[instrument(skip(self, metrics), fields(step_time = metrics.step_time_ms, retry_rate = metrics.retry_rate))]
    pub fn try_post_step_report(
        &mut self,
        metrics: &StepMetrics,
    ) -> Result<FreeEnergyReport, BlackCatError> {
        self.validate_configuration()?;
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
        self.validate_selection_state()?;
        let credited_selection_id = self.selection_pending.then_some(self.selection_counter);
        if credited_selection_id.is_some() {
            self.validate_context(&self.last_context)?;
        }
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
        next_z.update(
            reward_current,
            proposed_reward,
            credited_selection_id.map(|_| self.last_context.as_slice()),
        );
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
        if credited_selection_id.is_some() {
            next_bandits
                .try_update_all(&self.last_context, reward_current)
                .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        }
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
        if let Some(selection_id) = credited_selection_id {
            self.selection_pending = false;
            self.last_credited_selection_id = Some(selection_id);
        }
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
                    "bandit_contract": BLACKCAT_BANDIT_CONTRACT,
                    "bandit_contract_version": BLACKCAT_BANDIT_CONTRACT_VERSION,
                    "credited_selection_id": credited_selection_id,
                    "bandit_observations": self.bandits.observation_counts(),
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

    fn validate_selection_state(&self) -> Result<(), BlackCatError> {
        let bandits_pending = self
            .bandits
            .selection_pending()
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        if bandits_pending != self.selection_pending {
            return Err(BlackCatError::InvalidAdaptationState {
                field: "bandit.pending_state",
            });
        }
        if self.selection_pending {
            let witness = self.last_selection_witness.as_ref().ok_or(
                BlackCatError::InvalidAdaptationState {
                    field: "bandit.pending_witness",
                },
            )?;
            if witness.selection_id != self.selection_counter
                || witness.context != self.last_context
                || witness.picks.len() != self.last_picks.len()
                || witness
                    .picks
                    .iter()
                    .any(|(id, choice)| self.last_picks.get(id) != Some(choice))
            {
                return Err(BlackCatError::InvalidAdaptationState {
                    field: "bandit.pending_witness",
                });
            }
        }
        if self
            .last_credited_selection_id
            .is_some_and(|selection_id| selection_id > self.selection_counter)
        {
            return Err(BlackCatError::InvalidAdaptationState {
                field: "bandit.credited_selection",
            });
        }
        Ok(())
    }

    fn validate_configuration(&self) -> Result<(), BlackCatError> {
        match &self.configuration_error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
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

    /// Returns the retained construction error for compatibility-created
    /// fail-closed runtimes.
    pub fn configuration_error(&self) -> Option<&BlackCatError> {
        self.configuration_error.as_ref()
    }

    /// Returns a deterministic snapshot of all contextual control domains.
    pub fn choice_domains(&self) -> BTreeMap<String, Vec<String>> {
        self.bandits.choice_domains()
    }

    /// Returns the current fractional regularisation penalty tracked by ZMeta.
    pub fn frac_penalty(&self) -> f64 {
        self.z.frac_penalty()
    }

    /// Returns the current latent ZMeta coordinates without granting mutation
    /// access to the adaptation state.
    pub fn z_state(&self) -> &[f64] {
        self.z.z()
    }

    /// Overrides the fractional regulariser backend.
    pub fn set_frac_backend(&mut self, backend: FracBackend) {
        self.z.set_frac_backend(backend);
    }

    /// Returns the current exploration temperature tracked by the runtime.
    pub fn temperature(&self) -> f64 {
        self.z.temperature()
    }

    /// Returns the canonical reward configuration used by guarded reports.
    pub fn reward_config(&self) -> &RewardCfg {
        &self.reward
    }

    /// Replaces reward shaping only after it can build a valid canonical
    /// free-energy configuration.
    pub fn try_set_reward_config(&mut self, reward: RewardCfg) -> Result<(), BlackCatError> {
        reward.free_energy_config()?;
        self.reward = reward;
        Ok(())
    }

    /// Returns the append-only heuristic store path without exposing writes
    /// that bypass the Wilson adoption guard.
    pub fn heuristics_path(&self) -> &std::path::Path {
        self.heur.path()
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

    /// Returns the complete witness for the most recent bandit selection.
    pub fn last_selection_witness(&self) -> Option<&BlackCatSelectionWitness> {
        self.last_selection_witness.as_ref()
    }

    /// Returns the selection currently eligible for exactly one bandit reward.
    pub fn pending_selection_id(&self) -> Option<u64> {
        self.selection_pending.then_some(self.selection_counter)
    }

    /// Returns the most recent selection that actually updated the posterior.
    pub fn last_credited_selection_id(&self) -> Option<u64> {
        self.last_credited_selection_id
    }

    /// Explicitly abandons an unexecuted selection without updating any arm.
    pub fn abandon_pending_selection(&mut self) -> Option<u64> {
        self.try_abandon_pending_selection().unwrap_or(None)
    }

    /// Guarded abandonment path that validates the pending witness before
    /// releasing its one-shot reward credit.
    pub fn try_abandon_pending_selection(&mut self) -> Result<Option<u64>, BlackCatError> {
        self.validate_selection_state()?;
        let Some(selection_id) = self.pending_selection_id() else {
            return Ok(None);
        };
        let mut next_bandits = self.bandits.clone();
        next_bandits
            .try_abandon_all()
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        self.bandits = next_bandits;
        self.selection_pending = false;
        let bus = global_registry().event_bus();
        if bus.has_listeners("BlackCatSelectionAbandoned") {
            bus.publish(&PluginEvent::custom(
                "BlackCatSelectionAbandoned",
                serde_json::json!({
                    "selection": &self.last_selection_witness,
                    "bandit_contract": BLACKCAT_BANDIT_CONTRACT,
                    "bandit_contract_version": BLACKCAT_BANDIT_CONTRACT_VERSION,
                }),
            ));
        }
        Ok(Some(selection_id))
    }

    /// Stable posterior observation counts for audit and replay assertions.
    pub fn bandit_observation_counts(&self) -> BTreeMap<String, BTreeMap<String, u64>> {
        self.bandits.observation_counts()
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

fn validate_blackcat_configuration(
    z_params: &ZMetaParams,
    groups: &ChoiceGroups,
    feat_dim: usize,
) -> Result<(), BlackCatError> {
    let invalid =
        |field: &'static str, detail: String| BlackCatError::InvalidConfig { field, detail };
    if feat_dim == 0 || feat_dim > BLACKCAT_MAX_FEATURE_DIM {
        return Err(invalid(
            "feat_dim",
            format!("expected 1..={BLACKCAT_MAX_FEATURE_DIM}, got {feat_dim}"),
        ));
    }
    if z_params.dim == 0 || z_params.dim > BLACKCAT_MAX_ZMETA_DIM {
        return Err(invalid(
            "zmeta.dim",
            format!(
                "expected 1..={BLACKCAT_MAX_ZMETA_DIM}, got {}",
                z_params.dim
            ),
        ));
    }
    for (field, value, strictly_positive) in [
        ("zmeta.sigma", z_params.sigma, false),
        ("zmeta.lr", z_params.lr, false),
        ("zmeta.alpha_frac", z_params.alpha_frac, false),
        ("zmeta.lam_frac", z_params.lam_frac, false),
        ("zmeta.orientation_eta", z_params.orientation_eta, false),
        ("zmeta.orientation_eps", z_params.orientation_eps, true),
    ] {
        if !value.is_finite() || value < 0.0 || (strictly_positive && value == 0.0) {
            return Err(invalid(
                field,
                format!(
                    "expected a finite {} value, got {value}",
                    if strictly_positive {
                        "positive"
                    } else {
                        "non-negative"
                    }
                ),
            ));
        }
    }
    validate_choice_groups(groups, feat_dim)
}

fn validate_choice_groups(groups: &ChoiceGroups, feat_dim: usize) -> Result<(), BlackCatError> {
    let invalid =
        |field: &'static str, detail: String| BlackCatError::InvalidConfig { field, detail };
    if feat_dim == 0 || feat_dim > BLACKCAT_MAX_FEATURE_DIM {
        return Err(invalid(
            "feat_dim",
            format!("expected 1..={BLACKCAT_MAX_FEATURE_DIM}, got {feat_dim}"),
        ));
    }
    if groups.groups.len() > BLACKCAT_MAX_CHOICE_GROUPS {
        return Err(invalid(
            "choice_groups",
            format!(
                "at most {BLACKCAT_MAX_CHOICE_GROUPS} groups are allowed, got {}",
                groups.groups.len()
            ),
        ));
    }
    let mut group_ids = groups.groups.keys().collect::<Vec<_>>();
    group_ids.sort();
    let mut total_choices = 0usize;
    for id in group_ids {
        if id.is_empty() || id.trim() != id {
            return Err(invalid(
                "choice_group.id",
                format!("group identifier {id:?} must be non-empty and trimmed"),
            ));
        }
        let choices = &groups.groups[id];
        if choices.is_empty() || choices.len() > BLACKCAT_MAX_CHOICES_PER_GROUP {
            return Err(invalid(
                "choice_group.domain",
                format!(
                    "group {id:?} requires 1..={BLACKCAT_MAX_CHOICES_PER_GROUP} choices, got {}",
                    choices.len()
                ),
            ));
        }
        let mut unique = HashSet::with_capacity(choices.len());
        for choice in choices {
            if choice.is_empty() || choice.trim() != choice {
                return Err(invalid(
                    "choice_group.choice",
                    format!("choice {choice:?} in group {id:?} must be non-empty and trimmed"),
                ));
            }
            if !unique.insert(choice.as_str()) {
                return Err(invalid(
                    "choice_group.choice",
                    format!("choice {choice:?} is duplicated in group {id:?}"),
                ));
            }
        }
        total_choices = total_choices
            .checked_add(choices.len())
            .ok_or_else(|| invalid("choice_groups", "choice count overflowed".to_string()))?;
    }
    let posterior_cells = feat_dim
        .checked_mul(feat_dim)
        .and_then(|cells| cells.checked_mul(total_choices))
        .ok_or_else(|| invalid("posterior_cells", "posterior size overflowed".to_string()))?;
    if posterior_cells > BLACKCAT_MAX_POSTERIOR_CELLS {
        return Err(invalid(
            "posterior_cells",
            format!(
                "at most {BLACKCAT_MAX_POSTERIOR_CELLS} cells are allowed, got {posterior_cells}"
            ),
        ));
    }
    Ok(())
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
    fn reward_configuration_replacement_is_guarded_and_transactional() {
        let mut runtime = sample_runtime();
        let original_speed_scale = runtime.reward_config().scale_speed;
        let invalid = RewardCfg {
            scale_speed: 0.0,
            ..RewardCfg::default()
        };
        assert!(matches!(
            runtime.try_set_reward_config(invalid),
            Err(BlackCatError::FreeEnergy(_))
        ));
        assert_eq!(runtime.reward_config().scale_speed, original_speed_scale);

        let valid = RewardCfg {
            scale_speed: 5.0,
            ..RewardCfg::default()
        };
        runtime
            .try_set_reward_config(valid)
            .expect("valid reward configuration");
        assert_eq!(runtime.reward_config().scale_speed, 5.0);
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
    fn choice_hints_break_ties_without_overriding_learned_scores() {
        let mut runtime = sample_runtime();
        let context = vec![1.0, 0.0, 0.0, 0.0];
        let hinted_b = BTreeMap::from([("tile".to_string(), "b".to_string())]);
        assert_eq!(
            runtime
                .try_choose_with_hints(context.clone(), &hinted_b)
                .expect("valid tie-breaking hint")
                .get("tile")
                .map(String::as_str),
            Some("b")
        );
        let first_witness = runtime.last_selection_witness().expect("selection witness");
        assert_eq!(first_witness.contract, BLACKCAT_BANDIT_CONTRACT);
        assert!(!first_witness.decisions["tile"].sampling_applied);
        assert!(matches!(
            runtime.try_choose_with_hints(context.clone(), &hinted_b),
            Err(BlackCatError::PendingSelection { selection_id: 1 })
        ));

        let mut metrics = StepMetrics::default();
        metrics.extra.insert("reference_loss".to_string(), 100.0);
        metrics.extra.insert("candidate_loss".to_string(), 0.0);
        runtime
            .try_post_step_report(&metrics)
            .expect("hinted selection receives one guarded reward");
        assert_eq!(runtime.last_credited_selection_id(), Some(1));
        let hinted_a = BTreeMap::from([("tile".to_string(), "a".to_string())]);
        assert_eq!(
            runtime
                .try_choose_with_hints(context, &hinted_a)
                .expect("learned score remains authoritative")
                .get("tile")
                .map(String::as_str),
            Some("b")
        );
        assert!(runtime.last_selection_witness().unwrap().decisions["tile"].sampling_applied);
    }

    #[test]
    fn choice_hints_reject_unknown_groups_and_values() {
        let mut runtime = sample_runtime();
        let context = vec![1.0, 0.0, 0.0, 0.0];
        let unknown = BTreeMap::from([("missing".to_string(), "a".to_string())]);
        assert!(matches!(
            runtime.try_choose_with_hints(context.clone(), &unknown),
            Err(BlackCatError::UnknownChoiceHint { .. })
        ));
        let invalid = BTreeMap::from([("tile".to_string(), "missing".to_string())]);
        assert!(matches!(
            runtime.try_choose_with_hints(context, &invalid),
            Err(BlackCatError::InvalidChoiceHint { .. })
        ));
        assert!(runtime.last_picks().is_empty());
    }

    #[test]
    fn thompson_selection_rejects_non_finite_predictive_variance_atomically() {
        let mut runtime = sample_runtime();
        let initial_context = runtime.last_context().to_vec();
        let initial_counts = runtime.bandit_observation_counts();
        let error = runtime
            .try_choose(vec![f64::MAX, 0.0, 0.0, 0.0])
            .expect_err("non-finite predictive variance must fail closed");
        assert_eq!(
            error,
            BlackCatError::InvalidAdaptationState {
                field: "bandit.selection_score",
            }
        );
        assert_eq!(runtime.last_context(), initial_context);
        assert_eq!(runtime.bandit_observation_counts(), initial_counts);
        assert_eq!(runtime.pending_selection_id(), None);
        assert!(runtime.last_selection_witness().is_none());
    }

    #[test]
    fn each_selection_receives_at_most_one_bandit_reward() {
        let mut runtime = sample_runtime();
        let context = vec![1.0, 0.0, 0.0, 0.0];
        let initial_counts = runtime.bandit_observation_counts();

        runtime
            .try_post_step_report(&StepMetrics::default())
            .expect("observation-only report remains supported");
        assert_eq!(runtime.bandit_observation_counts(), initial_counts);
        assert_eq!(runtime.last_credited_selection_id(), None);

        runtime.try_choose(context).expect("bandit selection");
        assert_eq!(runtime.pending_selection_id(), Some(1));
        runtime
            .try_post_step_report(&StepMetrics::default())
            .expect("first reward credits the selection");
        let credited_counts = runtime.bandit_observation_counts();
        assert_eq!(runtime.pending_selection_id(), None);
        assert_eq!(runtime.last_credited_selection_id(), Some(1));
        assert_eq!(credited_counts["tile"].values().copied().sum::<u64>(), 1);

        runtime
            .try_post_step_report(&StepMetrics::default())
            .expect("later observation has no pending bandit credit");
        assert_eq!(runtime.bandit_observation_counts(), credited_counts);
    }

    #[test]
    fn group_insertion_order_cannot_change_seeded_decisions() {
        let mut left_groups = HashMap::new();
        left_groups.insert(
            "tile".to_string(),
            vec!["small".to_string(), "large".to_string()],
        );
        left_groups.insert("wg".to_string(), vec!["128".to_string(), "256".to_string()]);
        let mut right_groups = HashMap::new();
        right_groups.insert("wg".to_string(), vec!["128".to_string(), "256".to_string()]);
        right_groups.insert(
            "tile".to_string(),
            vec!["small".to_string(), "large".to_string()],
        );
        let mut left = BlackCatRuntime::new(
            ZMetaParams::default(),
            ChoiceGroups {
                groups: left_groups,
            },
            2,
            SoftBanditMode::TS,
            None,
        );
        let mut right = BlackCatRuntime::new(
            ZMetaParams::default(),
            ChoiceGroups {
                groups: right_groups,
            },
            2,
            SoftBanditMode::TS,
            None,
        );

        assert_eq!(
            left.try_choose(vec![1.0, 0.25]).expect("left picks"),
            right.try_choose(vec![1.0, 0.25]).expect("right picks")
        );
        assert_eq!(
            left.last_selection_witness().unwrap().decisions,
            right.last_selection_witness().unwrap().decisions
        );
    }

    #[test]
    fn duplicate_choice_domains_fail_before_rng_or_runtime_mutation() {
        let groups = ChoiceGroups {
            groups: HashMap::from([(
                "tile".to_string(),
                vec!["same".to_string(), "same".to_string()],
            )]),
        };
        let mut runtime =
            BlackCatRuntime::new(ZMetaParams::default(), groups, 2, SoftBanditMode::TS, None);
        let error = runtime
            .try_choose(vec![1.0, 0.0])
            .expect_err("duplicate domain must fail closed");
        assert!(matches!(
            error,
            BlackCatError::InvalidConfig {
                field: "choice_group.choice",
                ..
            }
        ));
        assert_eq!(runtime.pending_selection_id(), None);
        assert!(runtime.last_selection_witness().is_none());
    }

    #[test]
    fn empty_named_choice_group_is_not_silently_discarded() {
        let groups = ChoiceGroups {
            groups: HashMap::from([("tile".to_string(), Vec::new())]),
        };
        let mut runtime =
            BlackCatRuntime::new(ZMetaParams::default(), groups, 2, SoftBanditMode::TS, None);
        let error = runtime
            .try_choose(vec![1.0, 0.0])
            .expect_err("empty named group must remain visible and invalid");
        assert!(matches!(
            error,
            BlackCatError::InvalidConfig {
                field: "choice_group.domain",
                ..
            }
        ));
        assert!(runtime.choice_domains().is_empty());
    }

    #[test]
    fn guarded_constructor_bounds_posterior_allocation() {
        let groups = ChoiceGroups {
            groups: HashMap::from([("tile".to_string(), vec!["a".to_string()])]),
        };
        let result = BlackCatRuntime::try_new(
            ZMetaParams::default(),
            groups.clone(),
            BLACKCAT_MAX_FEATURE_DIM + 1,
            SoftBanditMode::TS,
            None,
        );
        assert!(matches!(
            result,
            Err(BlackCatError::InvalidConfig {
                field: "feat_dim",
                ..
            })
        ));

        let mut compatibility = BlackCatRuntime::new(
            ZMetaParams::default(),
            groups,
            usize::MAX,
            SoftBanditMode::TS,
            None,
        );
        assert!(matches!(
            compatibility.configuration_error(),
            Some(BlackCatError::InvalidConfig {
                field: "feat_dim",
                ..
            })
        ));
        assert!(matches!(
            compatibility.try_choose(vec![0.0]),
            Err(BlackCatError::InvalidConfig {
                field: "feat_dim",
                ..
            })
        ));
    }

    #[test]
    fn guarded_multi_bandit_bounds_aggregate_posterior_allocation() {
        let choices = |prefix: &str| {
            (0..33)
                .map(|index| format!("{prefix}-{index}"))
                .collect::<Vec<_>>()
        };
        let groups = ChoiceGroups {
            groups: HashMap::from([
                ("left".to_string(), choices("left")),
                ("right".to_string(), choices("right")),
            ]),
        };
        assert!(matches!(
            MultiBandit::try_new_seeded(&groups, BLACKCAT_MAX_FEATURE_DIM, SoftBanditMode::TS, 7,),
            Err(BlackCatError::InvalidConfig {
                field: "posterior_cells",
                ..
            })
        ));
        assert!(
            MultiBandit::new_seeded(&groups, BLACKCAT_MAX_FEATURE_DIM, SoftBanditMode::TS, 7,)
                .choice_domains()
                .is_empty()
        );
    }

    #[test]
    fn guarded_multi_bandit_selection_has_an_explicit_reward_slot() {
        let groups = ChoiceGroups {
            groups: HashMap::from([
                ("tile".to_string(), vec!["a".to_string(), "b".to_string()]),
                ("radix".to_string(), vec!["2".to_string(), "4".to_string()]),
            ]),
        };
        let mut bandits = MultiBandit::try_new_seeded(&groups, 2, SoftBanditMode::TS, 11)
            .expect("valid aggregate bandit");
        let (picks, decisions) = bandits
            .try_select_all(&[1.0, 0.25])
            .expect("guarded decision");
        assert_eq!(picks.len(), 2);
        assert_eq!(decisions.len(), 2);
        assert_eq!(bandits.selection_pending(), Ok(true));
        assert!(matches!(
            bandits.try_select_all(&[1.0, 0.25]),
            Err(BlackCatError::InvalidAdaptationState {
                field: "bandit.pending_selection",
            })
        ));
        bandits
            .try_abandon_all()
            .expect("explicit aggregate abandonment");
        assert_eq!(bandits.selection_pending(), Ok(false));

        bandits
            .try_select_all(&[1.0, 0.25])
            .expect("fresh decision after abandonment");
        bandits
            .try_update_all(&[1.0, 0.25], 0.75)
            .expect("one aggregate reward");
        assert_eq!(bandits.selection_pending(), Ok(false));
        assert_eq!(
            bandits
                .observation_counts()
                .values()
                .flat_map(|group| group.values())
                .sum::<u64>(),
            2
        );
    }

    #[test]
    fn context_builder_cannot_escape_the_runtime_feature_contract() {
        let runtime = sample_runtime();
        assert_eq!(
            runtime
                .try_make_context(2, 4, 8, 16, 0.5, &[], 0)
                .expect("declared context"),
            vec![1.0, 2.0, 4.0, 8.0]
        );
        assert!(matches!(
            runtime.try_make_context(2, 4, 8, 16, 0.5, &[], usize::MAX),
            Err(BlackCatError::ContextDimension {
                expected: 4,
                actual: usize::MAX,
            })
        ));
        assert!(matches!(
            runtime.try_make_context(2, 4, 8, 16, f64::NAN, &[], 0),
            Err(BlackCatError::NonFinite {
                field: "context.load",
                ..
            })
        ));
        assert!(runtime
            .make_context(2, 4, 8, 16, 0.5, &[], usize::MAX)
            .is_empty());
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
    use std::collections::{BTreeMap, HashSet};

    use nalgebra::{DMatrix, DVector};
    use rand::Rng;

    const POSTERIOR_REGULARIZATION: f64 = 1.0;
    const EXPLORATION_SCALE: f64 = 1.0;
    /// Full-covariance posterior estimator used by TS and UCB decisions.
    pub const POSTERIOR_ESTIMATOR: &str = "bayesian-linear-cholesky-v1";
    /// Stable pseudo-random stream and Gaussian transform used by TS.
    pub const THOMPSON_RNG_ALGORITHM: &str = "splitmix64-box-muller-v1";

    /// Contextual decision policy used by [`SoftBandit`].
    #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum SoftBanditMode {
        TS,
        UCB,
    }

    impl SoftBanditMode {
        pub const fn as_str(self) -> &'static str {
            match self {
                Self::TS => "thompson_sampling",
                Self::UCB => "upper_confidence_bound",
            }
        }
    }

    /// Posterior projection used to explain one candidate decision.
    #[derive(Clone, Debug, PartialEq, serde::Serialize)]
    pub struct BanditArmWitness {
        pub choice: String,
        pub posterior_mean: f64,
        pub predictive_stddev: f64,
        pub decision_score: f64,
        pub observations: u64,
        pub hinted: bool,
    }

    /// Complete decision witness for one named choice group.
    #[derive(Clone, Debug, PartialEq, serde::Serialize)]
    pub struct BanditDecisionWitness {
        pub mode: SoftBanditMode,
        pub chosen: String,
        pub decision_index: u64,
        pub posterior_estimator: &'static str,
        pub posterior_regularization: f64,
        pub exploration_scale: f64,
        pub sampling_applied: bool,
        pub rng_algorithm: Option<&'static str>,
        pub rng_stream_seed: Option<u64>,
        pub arms: Vec<BanditArmWitness>,
    }

    #[derive(Clone, Copy, Debug)]
    struct PosteriorProjection {
        mean: f64,
        stddev: f64,
    }

    #[derive(Clone, Debug)]
    pub struct LinTSArm {
        dim: usize,
        a: Vec<f64>,
        b: Vec<f64>,
        observations: u64,
    }

    impl LinTSArm {
        /// Compatibility constructor. Invalid configuration creates an
        /// unusable sentinel that every guarded operation rejects; prefer
        /// [`Self::try_new`] when configuration is not compile-time fixed.
        pub fn new(dim: usize, lambda: f64) -> Self {
            Self::try_new(dim, lambda).unwrap_or_else(|_| Self {
                dim: 0,
                a: Vec::new(),
                b: Vec::new(),
                observations: 0,
            })
        }

        /// Guarded constructor for a regularized Bayesian linear arm.
        pub fn try_new(dim: usize, lambda: f64) -> Result<Self, &'static str> {
            if dim == 0 || dim > super::BLACKCAT_MAX_FEATURE_DIM {
                return Err("bandit.posterior_shape");
            }
            if !lambda.is_finite() || lambda <= 0.0 {
                return Err("bandit.posterior_regularization");
            }
            dim.checked_mul(dim).ok_or("bandit.posterior_shape")?;
            Ok(Self {
                dim,
                a: eye_flat(dim, lambda),
                b: vec![0.0; dim],
                observations: 0,
            })
        }

        /// Compatibility sampler backed by thread entropy. Runtime decisions
        /// use the owned, replayable stream in [`SoftBandit`] instead.
        pub fn sample_score(&self, x: &[f64]) -> f64 {
            let mut rng = rand::thread_rng();
            match self.project(x) {
                Ok(projection) => {
                    let score = projection.mean
                        + EXPLORATION_SCALE
                            * projection.stddev
                            * standard_normal_from_rand(&mut rng);
                    if score.is_finite() {
                        score
                    } else {
                        f64::NAN
                    }
                }
                Err(_) => f64::NAN,
            }
        }

        pub fn ucb_score(&self, x: &[f64], c: f64) -> f64 {
            self.try_ucb_score(x, c).unwrap_or(f64::NAN)
        }

        /// Guarded UCB projection for direct Rust consumers.
        pub fn try_ucb_score(&self, x: &[f64], c: f64) -> Result<f64, &'static str> {
            if !c.is_finite() || c < 0.0 {
                return Err("bandit.exploration_scale");
            }
            let projection = self.project(x)?;
            let score = projection.mean + c * projection.stddev;
            score
                .is_finite()
                .then_some(score)
                .ok_or("bandit.selection_score")
        }

        pub fn update(&mut self, x: &[f64], reward: f64) {
            let _ = self.try_update(x, reward);
        }

        /// Guarded full-covariance posterior update.
        pub fn try_update(&mut self, x: &[f64], reward: f64) -> Result<(), &'static str> {
            validate_context(x, self.dim)?;
            if !reward.is_finite() {
                return Err("bandit.reward");
            }
            let mut next = self.clone();
            rank1_add(&mut next.a, x, self.dim);
            for (bi, xi) in next.b.iter_mut().zip(x) {
                *bi += reward * xi;
            }
            next.observations = next
                .observations
                .checked_add(1)
                .ok_or("bandit.observation_count")?;
            next.validate_state(self.dim)?;
            *self = next;
            Ok(())
        }

        /// Returns `(posterior_mean, predictive_stddev)` for one context.
        pub fn try_posterior_projection(&self, x: &[f64]) -> Result<(f64, f64), &'static str> {
            let projection = self.project(x)?;
            Ok((projection.mean, projection.stddev))
        }

        /// Number of rewards committed to this arm.
        pub fn observations(&self) -> u64 {
            self.observations
        }

        fn project(&self, x: &[f64]) -> Result<PosteriorProjection, &'static str> {
            validate_context(x, self.dim)?;
            let matrix = self.precision_matrix()?;
            let cholesky = matrix.cholesky().ok_or("bandit.posterior_spd")?;
            let b = DVector::from_column_slice(&self.b);
            let context = DVector::from_column_slice(x);
            let mean_parameters = cholesky.solve(&b);
            let covariance_context = cholesky.solve(&context);
            if mean_parameters.iter().any(|value| !value.is_finite())
                || covariance_context.iter().any(|value| !value.is_finite())
            {
                return Err("bandit.posterior_solution");
            }
            let mean = context.dot(&mean_parameters);
            let variance = context.dot(&covariance_context);
            if !mean.is_finite() || !variance.is_finite() {
                return Err("bandit.selection_score");
            }
            let variance_tolerance = 64.0
                * f64::EPSILON
                * context.norm_squared().max(1.0)
                * covariance_context.norm().max(1.0);
            if variance < -variance_tolerance {
                return Err("bandit.posterior_variance");
            }
            let stddev = variance.max(0.0).sqrt();
            if !stddev.is_finite() {
                return Err("bandit.selection_score");
            }
            Ok(PosteriorProjection { mean, stddev })
        }

        fn precision_matrix(&self) -> Result<DMatrix<f64>, &'static str> {
            let matrix_len = self
                .dim
                .checked_mul(self.dim)
                .ok_or("bandit.posterior_shape")?;
            if self.dim == 0 || self.a.len() != matrix_len || self.b.len() != self.dim {
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
            let scale = self
                .a
                .iter()
                .fold(1.0f64, |acc, value| acc.max(value.abs()));
            let symmetry_tolerance = 64.0 * f64::EPSILON * scale;
            for row in 0..self.dim {
                for col in 0..row {
                    if (self.a[row * self.dim + col] - self.a[col * self.dim + row]).abs()
                        > symmetry_tolerance
                    {
                        return Err("bandit.posterior_symmetry");
                    }
                }
            }
            Ok(DMatrix::from_row_slice(self.dim, self.dim, &self.a))
        }

        fn validate_state(&self, context_dim: usize) -> Result<(), &'static str> {
            if self.dim != context_dim {
                return Err("bandit.posterior_shape");
            }
            self.precision_matrix()?
                .cholesky()
                .ok_or("bandit.posterior_spd")?;
            Ok(())
        }
    }

    #[derive(Clone)]
    pub struct SoftBandit {
        choices: Vec<String>,
        arms: Vec<LinTSArm>,
        last_index: usize,
        mode: SoftBanditMode,
        rng: BanditRng,
        rng_seed: u64,
        decisions: u64,
        selection_pending: bool,
    }

    impl SoftBandit {
        pub fn new(choices: Vec<String>, feat_dim: usize, mode: SoftBanditMode) -> Self {
            Self::new_seeded(choices, feat_dim, mode, 0x424c_4143_4b43_4154)
        }

        pub fn new_seeded(
            choices: Vec<String>,
            feat_dim: usize,
            mode: SoftBanditMode,
            seed: u64,
        ) -> Self {
            Self::try_new_seeded(choices, feat_dim, mode, seed).unwrap_or_else(|_| Self {
                choices: Vec::new(),
                arms: Vec::new(),
                last_index: 0,
                mode,
                rng: BanditRng::new(seed),
                rng_seed: seed,
                decisions: 0,
                selection_pending: false,
            })
        }

        /// Guarded constructor for one finite choice domain and posterior
        /// stream.
        pub fn try_new_seeded(
            choices: Vec<String>,
            feat_dim: usize,
            mode: SoftBanditMode,
            seed: u64,
        ) -> Result<Self, &'static str> {
            if feat_dim == 0 || feat_dim > super::BLACKCAT_MAX_FEATURE_DIM {
                return Err("bandit.posterior_shape");
            }
            if choices.is_empty() || choices.len() > super::BLACKCAT_MAX_CHOICES_PER_GROUP {
                return Err("bandit.choice_state");
            }
            let mut unique = HashSet::with_capacity(choices.len());
            if choices.iter().any(|choice| {
                choice.is_empty() || choice.trim() != choice || !unique.insert(choice)
            }) {
                return Err("bandit.choice_domain");
            }
            feat_dim
                .checked_mul(feat_dim)
                .and_then(|cells| cells.checked_mul(choices.len()))
                .filter(|cells| *cells <= super::BLACKCAT_MAX_POSTERIOR_CELLS)
                .ok_or("bandit.posterior_size")?;
            let arms = (0..choices.len())
                .map(|_| LinTSArm::new(feat_dim, POSTERIOR_REGULARIZATION))
                .collect();
            Ok(Self {
                choices,
                arms,
                last_index: 0,
                mode,
                rng: BanditRng::new(seed),
                rng_seed: seed,
                decisions: 0,
                selection_pending: false,
            })
        }

        pub fn select(&mut self, x: &[f64]) -> String {
            self.try_select(x)
                .map(|decision| decision.chosen)
                .unwrap_or_default()
        }

        /// Guarded selection without an external prior.
        pub fn try_select(&mut self, x: &[f64]) -> Result<BanditDecisionWitness, &'static str> {
            self.try_select_with_hint(x, None)
        }

        /// Guarded selection with an optional equivalent-posterior tie prior.
        pub fn try_select_with_hint(
            &mut self,
            x: &[f64],
            hint: Option<&str>,
        ) -> Result<BanditDecisionWitness, &'static str> {
            if hint.is_some_and(|value| !self.choices.iter().any(|choice| choice == value)) {
                return Err("bandit.choice_hint");
            }
            let mut next = self.clone();
            let decision = next.select_with_hint_in_place(x, hint)?;
            *self = next;
            Ok(decision)
        }

        fn select_with_hint_in_place(
            &mut self,
            x: &[f64],
            hint: Option<&str>,
        ) -> Result<BanditDecisionWitness, &'static str> {
            self.validate_state(x.len())?;
            if self.selection_pending {
                return Err("bandit.pending_selection");
            }
            validate_context(x, self.arms[0].dim)?;
            let hint_index =
                hint.and_then(|value| self.choices.iter().position(|choice| choice == value));
            let projections = self
                .arms
                .iter()
                .map(|arm| arm.project(x))
                .collect::<Result<Vec<_>, _>>()?;
            let equivalent_hint_prior = hint_index.is_some()
                && projections
                    .windows(2)
                    .all(|pair| projections_equivalent(pair[0], pair[1]));
            let sampling_applied = self.mode == SoftBanditMode::TS && !equivalent_hint_prior;
            let mut scores = Vec::with_capacity(self.arms.len());
            for projection in &projections {
                let score = match self.mode {
                    SoftBanditMode::TS if sampling_applied => {
                        projection.mean
                            + EXPLORATION_SCALE * projection.stddev * self.rng.standard_normal()
                    }
                    SoftBanditMode::TS => projection.mean,
                    SoftBanditMode::UCB => projection.mean + EXPLORATION_SCALE * projection.stddev,
                };
                if !score.is_finite() {
                    return Err("bandit.selection_score");
                }
                scores.push(score);
            }
            let mut idx = hint_index.unwrap_or(0);
            let mut best = scores[idx];
            for (index, score) in scores.iter().copied().enumerate() {
                if score > best {
                    best = score;
                    idx = index;
                }
            }
            let decision_index = self
                .decisions
                .checked_add(1)
                .ok_or("bandit.decision_count")?;
            self.last_index = idx;
            self.decisions = decision_index;
            self.selection_pending = true;
            Ok(BanditDecisionWitness {
                mode: self.mode,
                chosen: self.choices[idx].clone(),
                decision_index,
                posterior_estimator: POSTERIOR_ESTIMATOR,
                posterior_regularization: POSTERIOR_REGULARIZATION,
                exploration_scale: EXPLORATION_SCALE,
                sampling_applied,
                rng_algorithm: (self.mode == SoftBanditMode::TS).then_some(THOMPSON_RNG_ALGORITHM),
                rng_stream_seed: (self.mode == SoftBanditMode::TS).then_some(self.rng_seed),
                arms: self
                    .choices
                    .iter()
                    .zip(self.arms.iter())
                    .zip(projections)
                    .zip(scores)
                    .map(
                        |(((choice, arm), projection), decision_score)| BanditArmWitness {
                            choice: choice.clone(),
                            posterior_mean: projection.mean,
                            predictive_stddev: projection.stddev,
                            decision_score,
                            observations: arm.observations,
                            hinted: hint == Some(choice.as_str()),
                        },
                    )
                    .collect(),
            })
        }

        pub fn update_last(&mut self, x: &[f64], reward: f64) {
            let _ = self.try_update_last(x, reward);
        }

        /// Credits the pending decision exactly once.
        pub fn try_update_last(&mut self, x: &[f64], reward: f64) -> Result<(), &'static str> {
            if !self.selection_pending {
                return Err("bandit.missing_selection");
            }
            let arm = self
                .arms
                .get_mut(self.last_index)
                .ok_or("bandit.choice_state")?;
            arm.try_update(x, reward)?;
            self.selection_pending = false;
            Ok(())
        }

        /// Releases a pending decision without posterior credit.
        pub fn try_abandon_last_selection(&mut self) -> Result<(), &'static str> {
            if !self.selection_pending {
                return Err("bandit.missing_selection");
            }
            self.selection_pending = false;
            Ok(())
        }

        /// Returns whether this bandit is awaiting reward or abandonment.
        pub fn selection_pending(&self) -> bool {
            self.selection_pending
        }

        /// Declared finite choice domain in stable decision order.
        pub fn choices(&self) -> &[String] {
            &self.choices
        }

        pub(super) fn validate_state(&self, context_dim: usize) -> Result<(), &'static str> {
            if self.choices.is_empty()
                || self.choices.len() != self.arms.len()
                || self.last_index >= self.choices.len()
            {
                return Err("bandit.choice_state");
            }
            let mut unique = HashSet::with_capacity(self.choices.len());
            if self
                .choices
                .iter()
                .any(|choice| choice.trim().is_empty() || !unique.insert(choice.as_str()))
            {
                return Err("bandit.choice_domain");
            }
            self.rng.validate_state()?;
            if self.selection_pending && self.decisions == 0 {
                return Err("bandit.pending_state");
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
                let projection = arm.project(context)?;
                let score = projection.mean + EXPLORATION_SCALE * projection.stddev;
                if !score.is_finite()
                    || !projection.mean.is_finite()
                    || !projection.stddev.is_finite()
                {
                    return Err("bandit.selection_score");
                }
            }
            Ok(())
        }

        /// Stable observation count for every declared choice.
        pub fn observation_counts(&self) -> BTreeMap<String, u64> {
            self.choices
                .iter()
                .zip(self.arms.iter())
                .map(|(choice, arm)| (choice.clone(), arm.observations))
                .collect()
        }
    }

    fn eye_flat(dim: usize, lambda: f64) -> Vec<f64> {
        let mut matrix = vec![0.0; dim * dim];
        for i in 0..dim {
            matrix[i * dim + i] = lambda;
        }
        matrix
    }

    fn rank1_add(a: &mut [f64], x: &[f64], dim: usize) {
        for i in 0..dim {
            for j in 0..dim {
                a[i * dim + j] += x[i] * x[j];
            }
        }
    }

    fn validate_context(context: &[f64], expected: usize) -> Result<(), &'static str> {
        if context.len() != expected {
            return Err("bandit.context_shape");
        }
        if context.iter().any(|value| !value.is_finite()) {
            return Err("bandit.context_state");
        }
        Ok(())
    }

    fn projections_equivalent(left: PosteriorProjection, right: PosteriorProjection) -> bool {
        close_projection(left.mean, right.mean) && close_projection(left.stddev, right.stddev)
    }

    fn close_projection(left: f64, right: f64) -> bool {
        (left - right).abs() <= 32.0 * f64::EPSILON * left.abs().max(right.abs()).max(1.0)
    }

    fn standard_normal_from_rand(rng: &mut impl Rng) -> f64 {
        let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[derive(Clone)]
    struct BanditRng {
        state: u64,
        spare_normal: Option<f64>,
    }

    impl BanditRng {
        fn new(seed: u64) -> Self {
            Self {
                state: seed,
                spare_normal: None,
            }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = self.state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^ (value >> 31)
        }

        fn uniform_open01(&mut self) -> f64 {
            const DENOMINATOR: f64 = (1_u64 << 53) as f64;
            ((self.next_u64() >> 11) as f64 + 0.5) / DENOMINATOR
        }

        fn standard_normal(&mut self) -> f64 {
            if let Some(value) = self.spare_normal.take() {
                return value;
            }
            let u1 = self.uniform_open01();
            let u2 = self.uniform_open01();
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * u2;
            self.spare_normal = Some(radius * angle.sin());
            radius * angle.cos()
        }

        fn validate_state(&self) -> Result<(), &'static str> {
            if self.spare_normal.is_some_and(|value| !value.is_finite()) {
                return Err("bandit.rng_state");
            }
            Ok(())
        }
    }

    pub(super) fn derive_group_seed(base_seed: u64, group: &str) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ base_seed.rotate_left(17);
        for byte in group.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn posterior_projection_uses_full_covariance() {
            let mut arm = LinTSArm::new(2, 1.0);
            arm.try_update(&[1.0, 1.0], 2.0).expect("valid update");
            let projection = arm.project(&[1.0, 0.0]).expect("valid posterior");

            assert!((projection.mean - 2.0 / 3.0).abs() < 1.0e-12);
            assert!((projection.stddev - (2.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);
        }

        #[test]
        fn seeded_thompson_sampling_is_replayable_and_explores() {
            let choices = vec!["a".to_string(), "b".to_string()];
            let mut left = SoftBandit::new_seeded(choices.clone(), 2, SoftBanditMode::TS, 17);
            let mut right = SoftBandit::new_seeded(choices, 2, SoftBanditMode::TS, 17);
            let mut sequence = Vec::new();
            for _ in 0..32 {
                let left_pick = left
                    .try_select_with_hint(&[1.0, 0.5], None)
                    .expect("left decision")
                    .chosen;
                let right_pick = right
                    .try_select_with_hint(&[1.0, 0.5], None)
                    .expect("right decision")
                    .chosen;
                assert_eq!(left_pick, right_pick);
                sequence.push(left_pick);
                left.try_abandon_last_selection()
                    .expect("release left delayed reward slot");
                right
                    .try_abandon_last_selection()
                    .expect("release right delayed reward slot");
            }
            assert!(sequence.iter().any(|choice| choice == "a"));
            assert!(sequence.iter().any(|choice| choice == "b"));
        }

        #[test]
        fn hint_resolves_only_an_equivalent_thompson_prior() {
            let mut bandit = SoftBandit::new_seeded(
                vec!["a".to_string(), "b".to_string()],
                2,
                SoftBanditMode::TS,
                23,
            );
            let first = bandit
                .try_select_with_hint(&[1.0, 0.0], Some("b"))
                .expect("equivalent prior");
            assert_eq!(first.chosen, "b");
            assert!(!first.sampling_applied);
            bandit
                .try_update_last(&[1.0, 0.0], 100.0)
                .expect("posterior update");

            let second = bandit
                .try_select_with_hint(&[1.0, 0.0], Some("a"))
                .expect("learned posterior");
            assert!(second.sampling_applied);
            assert_eq!(second.chosen, "b");
        }

        #[test]
        fn delayed_reward_slot_cannot_be_overwritten_or_credited_twice() {
            let mut bandit = SoftBandit::new_seeded(
                vec!["a".to_string(), "b".to_string()],
                2,
                SoftBanditMode::TS,
                29,
            );
            assert_eq!(
                bandit.try_update_last(&[1.0, 0.0], 1.0),
                Err("bandit.missing_selection")
            );
            assert!(matches!(
                bandit.try_select_with_hint(&[1.0, 0.0], Some("missing")),
                Err("bandit.choice_hint")
            ));
            bandit
                .try_select_with_hint(&[1.0, 0.0], None)
                .expect("first selection");
            assert!(matches!(
                bandit.try_select_with_hint(&[1.0, 0.0], None),
                Err("bandit.pending_selection")
            ));
            bandit
                .try_update_last(&[1.0, 0.0], 1.0)
                .expect("one reward");
            assert_eq!(
                bandit.try_update_last(&[1.0, 0.0], 1.0),
                Err("bandit.missing_selection")
            );
        }

        #[test]
        fn decision_witness_identifies_the_maximum_sampled_score() {
            let mut bandit = SoftBandit::new_seeded(
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                2,
                SoftBanditMode::TS,
                31,
            );
            let decision = bandit
                .try_select_with_hint(&[1.0, -0.25], None)
                .expect("decision");
            let winner = decision
                .arms
                .iter()
                .max_by(|left, right| left.decision_score.total_cmp(&right.decision_score))
                .expect("candidate");
            assert_eq!(decision.chosen, winner.choice);
            assert!(decision.sampling_applied);
            assert_eq!(decision.posterior_estimator, POSTERIOR_ESTIMATOR);
            assert_eq!(decision.rng_algorithm, Some(THOMPSON_RNG_ALGORITHM));
            assert_eq!(decision.rng_stream_seed, Some(31));
        }

        #[test]
        fn asymmetric_precision_state_is_rejected() {
            let mut arm = LinTSArm::new(2, 1.0);
            arm.a[1] = 0.25;
            assert_eq!(arm.validate_state(2), Err("bandit.posterior_symmetry"));
        }

        #[test]
        fn guarded_arm_constructor_rejects_invalid_configuration_without_panicking() {
            assert!(matches!(
                LinTSArm::try_new(0, 1.0),
                Err("bandit.posterior_shape")
            ));
            assert!(matches!(
                LinTSArm::try_new(2, f64::NAN),
                Err("bandit.posterior_regularization")
            ));
            assert!(LinTSArm::new(0, -1.0).validate_state(1).is_err());
            assert!(matches!(
                SoftBandit::try_new_seeded(
                    vec!["a".to_string()],
                    super::super::BLACKCAT_MAX_FEATURE_DIM + 1,
                    SoftBanditMode::TS,
                    1,
                ),
                Err("bandit.posterior_shape")
            ));
            let mut compatibility =
                SoftBandit::new_seeded(vec!["a".to_string()], usize::MAX, SoftBanditMode::TS, 1);
            assert!(compatibility.select(&[0.0]).is_empty());
        }
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
            Self { path }
        }

        pub fn append(&self, rule_text: &str, info: &HashMap<String, f64>) {
            if let Some(dir) = self.path.parent() {
                let _ = fs::create_dir_all(dir);
            }
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
