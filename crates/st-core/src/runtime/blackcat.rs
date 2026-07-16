// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

use bandit::{BanditDecisionWitness, SoftBandit, SoftBanditMode};
use rewrite::{HeurStore, HeurStoreDisposition, HeurStoreReceipt};
use spiral_config::determinism;
use st_frac::FracBackend;
use thiserror::Error;
use tracing::{debug, instrument, warn};
use zmeta::{ZMetaES, ZMetaParams, ZMetaProposalWitness, ZMetaUpdateWitness};

use crate::heur::evidence::{try_wilson_interval, WilsonInterval};
use crate::heur::free_energy::{
    evaluate_free_energy, BandEnergy, FreeEnergyConfig, FreeEnergyError, FreeEnergyObservation,
    FreeEnergyReport, FreeEnergyRequest,
};
use crate::plugin::{global_registry, PluginEvent};
use crate::telemetry::{monitoring::MonitoringHub, trace_init};

/// Stable semantic contract emitted with every contextual bandit decision.
pub const BLACKCAT_BANDIT_CONTRACT: &str = "spiraltorch.blackcat.contextual-bandit";
/// Schema version for [`BlackCatSelectionWitness`].
pub const BLACKCAT_BANDIT_CONTRACT_VERSION: u32 = 2;
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
/// Stable contract for evidence-backed heuristic adoption.
pub const BLACKCAT_HEURISTIC_ADOPTION_CONTRACT: &str =
    "spiraltorch.blackcat.soft_heuristic_adoption";
pub const BLACKCAT_HEURISTIC_ADOPTION_CONTRACT_VERSION: u32 = 1;
pub const BLACKCAT_MAX_TRACKED_HEURISTICS: usize = 1024;
pub const BLACKCAT_MAX_HEURISTIC_RULE_BYTES: usize = 16 * 1024;

fn canonical_heuristic_rule_text(rule_text: &str) -> Result<&str, &'static str> {
    let canonical = rule_text.trim();
    if canonical.is_empty() {
        return Err("heuristic rule must not be empty");
    }
    if canonical.len() > BLACKCAT_MAX_HEURISTIC_RULE_BYTES {
        return Err("heuristic rule exceeds the guarded byte limit");
    }
    if canonical
        .chars()
        .any(|character| matches!(character, '\n' | '\r' | '\0'))
    {
        return Err("heuristic rule must be one line and contain no NUL byte");
    }
    Ok(canonical)
}

fn canonical_heuristic_rule(rule_text: &str) -> Result<(String, String), BlackCatError> {
    let canonical = canonical_heuristic_rule_text(rule_text).map_err(|detail| {
        BlackCatError::InvalidConfig {
            field: "heuristic_adoption.rule",
            detail: detail.to_string(),
        }
    })?;
    Ok((canonical.to_string(), rewrite::rule_id(canonical)))
}

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
    #[error(transparent)]
    ZMeta(#[from] zmeta::ZMetaError),
    #[error(transparent)]
    Wilson(#[from] crate::heur::evidence::WilsonError),
    #[error(transparent)]
    HeuristicStore(#[from] rewrite::HeurStoreError),
}

/// Complete, replay-oriented witness for one contextual bandit selection.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BlackCatSelectionWitness {
    pub contract: &'static str,
    pub contract_version: u32,
    pub selection_id: u64,
    pub zmeta: ZMetaProposalWitness,
    pub context: Vec<f64>,
    pub hints: BTreeMap<String, String>,
    pub picks: BTreeMap<String, String>,
    pub decisions: BTreeMap<String, BanditDecisionWitness>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct SoftHeuristicAdoptionConfig {
    pub minimum_trials: u64,
    pub baseline_probability: f64,
    pub confidence_z: f64,
}

impl Default for SoftHeuristicAdoptionConfig {
    fn default() -> Self {
        Self {
            minimum_trials: 8,
            baseline_probability: 0.5,
            confidence_z: 1.96,
        }
    }
}

impl SoftHeuristicAdoptionConfig {
    pub fn validate(&self) -> Result<(), BlackCatError> {
        if self.minimum_trials == 0
            || self.minimum_trials > crate::heur::evidence::WILSON_MAX_EXACT_TRIALS
        {
            return Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.minimum_trials",
                detail: format!(
                    "expected 1..={}, got {}",
                    crate::heur::evidence::WILSON_MAX_EXACT_TRIALS,
                    self.minimum_trials
                ),
            });
        }
        if !self.baseline_probability.is_finite()
            || !(0.0..1.0).contains(&self.baseline_probability)
        {
            return Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.baseline_probability",
                detail: format!(
                    "expected a finite probability in [0, 1), got {}",
                    self.baseline_probability
                ),
            });
        }
        if !self.confidence_z.is_finite() || self.confidence_z <= 0.0 {
            return Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.confidence_z",
                detail: format!(
                    "expected a finite positive confidence z, got {}",
                    self.confidence_z
                ),
            });
        }
        let _ = try_wilson_interval(0, 1, self.confidence_z)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SoftHeuristicAdoptionDecision {
    Collecting,
    BelowThreshold,
    Adopted,
    AlreadyPersisted,
    AlreadyAdopted,
    PersistenceFailed,
}

impl SoftHeuristicAdoptionDecision {
    pub fn is_persisted(self) -> bool {
        matches!(
            self,
            Self::Adopted | Self::AlreadyPersisted | Self::AlreadyAdopted
        )
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct SoftHeuristicAdoptionReport {
    pub contract: &'static str,
    pub contract_version: u32,
    pub rule_id: String,
    pub rule: String,
    pub interval: WilsonInterval,
    pub minimum_trials: u64,
    pub baseline_probability: f64,
    pub eligible: bool,
    pub decision: SoftHeuristicAdoptionDecision,
    pub persistence: Option<HeurStoreReceipt>,
    pub persistence_error: Option<rewrite::HeurStoreError>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SoftHeuristicEvidenceMode {
    StreamingStep,
    CumulativeSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SoftHeuristicEvidenceState {
    successes: u64,
    trials: u64,
    config: SoftHeuristicAdoptionConfig,
    mode: SoftHeuristicEvidenceMode,
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
    heuristic_adoption: SoftHeuristicAdoptionConfig,
    heuristic_evidence: BTreeMap<String, SoftHeuristicEvidenceState>,
    adopted_heuristics: HashSet<String>,
    last_heuristic_adoption: Option<SoftHeuristicAdoptionReport>,
    configuration_error: Option<BlackCatError>,
    context_dim: usize,
    last_base_context: Vec<f64>,
    last_context: Vec<f64>,
    last_picks: HashMap<String, String>,
    last_selection_witness: Option<BlackCatSelectionWitness>,
    selection_pending: bool,
    selection_counter: u64,
    last_credited_selection_id: Option<u64>,
    last_zmeta_update: Option<ZMetaUpdateWitness>,
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
        let heur = match HeurStore::try_new(heur_path) {
            Ok(store) => store,
            Err(error) => {
                configuration_error.get_or_insert_with(|| error.into());
                HeurStore::new(None)
            }
        };
        let stats_alpha = 0.2;
        let runtime = Self {
            z: ZMetaES::new(params),
            bandits,
            heur,
            reward: RewardCfg::default(),
            heuristic_adoption: SoftHeuristicAdoptionConfig::default(),
            heuristic_evidence: BTreeMap::new(),
            adopted_heuristics: HashSet::new(),
            last_heuristic_adoption: None,
            configuration_error,
            context_dim: effective_feat_dim,
            last_base_context: vec![0.0; effective_feat_dim],
            last_context: vec![0.0; effective_feat_dim],
            last_picks: HashMap::new(),
            last_selection_witness: None,
            selection_pending: false,
            selection_counter: 0,
            last_credited_selection_id: None,
            last_zmeta_update: None,
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
        self.z.validate_adaptation_state()?;
        self.bandits
            .validate_state(self.context_dim)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        self.bandits.validate_hints(hints)?;
        let selection_id =
            self.selection_counter
                .checked_add(1)
                .ok_or(BlackCatError::InvalidAdaptationState {
                    field: "bandit.selection_count",
                })?;
        let mut next_z = self.z.clone();
        let zmeta = next_z.try_prepare_context(selection_id, &context)?;
        let effective_context = zmeta.effective_context.clone();
        self.validate_context(&effective_context)?;
        let mut next_bandits = self.bandits.clone();
        next_bandits
            .validate_selection_scores(&effective_context)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        let (picks, decisions) = next_bandits
            .select_all_with_hints_in_place(&effective_context, hints)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        next_bandits
            .validate_state(self.context_dim)
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        let witness = BlackCatSelectionWitness {
            contract: BLACKCAT_BANDIT_CONTRACT,
            contract_version: BLACKCAT_BANDIT_CONTRACT_VERSION,
            selection_id,
            zmeta,
            context: effective_context.clone(),
            hints: hints.clone(),
            picks: picks
                .iter()
                .map(|(id, choice)| (id.clone(), choice.clone()))
                .collect(),
            decisions,
        };
        self.z = next_z;
        self.bandits = next_bandits;
        self.last_base_context = context;
        self.last_context = effective_context;
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
        let evaluated_penalty = self.z.try_active_frac_penalty()?;
        validate_blackcat_non_negative("zmeta.evaluated_frac_penalty", evaluated_penalty)?;
        let report = self.reward.report(metrics, evaluated_penalty)?;
        let reward_current = report.utility;
        self.z.validate_adaptation_state()?;
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

        let mut next_z = self.z.clone();
        let zmeta_update = credited_selection_id
            .map(|selection_id| next_z.try_tell(selection_id, reward_current))
            .transpose()?;
        next_z.try_temp_schedule(metrics.retry_rate, grad_norm, loss_var)?;
        next_z.validate_adaptation_state()?;
        validate_blackcat_non_negative("zmeta.next_frac_penalty", next_z.frac_penalty())?;

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
        next_frac_penalty_ema.try_update("statistics.frac_penalty_ema", evaluated_penalty)?;
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
        if let Some(update) = &zmeta_update {
            self.last_zmeta_update = Some(update.clone());
        }
        if let Some(selection_id) = credited_selection_id {
            self.selection_pending = false;
            self.last_credited_selection_id = Some(selection_id);
        }
        debug!(
            reward = reward_current,
            frac_penalty = evaluated_penalty,
            "blackcat step complete"
        );
        let _ = self.monitoring.observe(metrics, reward_current);
        let bus = global_registry().event_bus();
        if bus.has_listeners("BlackCatPostStep") {
            bus.publish(&PluginEvent::custom(
                "BlackCatPostStep",
                serde_json::json!({
                    "reward": reward_current,
                    "frac_penalty": evaluated_penalty,
                    "free_energy": &report,
                    "zmeta_update": &zmeta_update,
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
        self.z.validate_adaptation_state()?;
        let bandits_pending = self
            .bandits
            .selection_pending()
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        if bandits_pending != self.selection_pending {
            return Err(BlackCatError::InvalidAdaptationState {
                field: "bandit.pending_state",
            });
        }
        let zmeta_pending = self.z.pending_evaluation_id();
        if zmeta_pending != self.pending_selection_id() {
            return Err(BlackCatError::InvalidAdaptationState {
                field: "zmeta.pending_state",
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
                || witness.zmeta.evaluation_id != witness.selection_id
                || witness.zmeta.base_context != self.last_base_context
                || witness.zmeta.effective_context != self.last_context
                || self.z.last_proposal() != Some(&witness.zmeta)
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
        self.reward.report(metrics, self.z.active_frac_penalty())
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

    /// Returns the penalty of the latent currently affecting an in-flight
    /// selection, or the incumbent penalty when no selection is pending.
    pub fn active_frac_penalty(&self) -> f64 {
        self.z.active_frac_penalty()
    }

    /// Returns the current latent ZMeta coordinates without granting mutation
    /// access to the adaptation state.
    pub fn z_state(&self) -> &[f64] {
        self.z.z()
    }

    /// Records the preferred fractional execution route. The proposal witness
    /// separately names this request and the canonical Rust semantic backend.
    pub fn set_frac_backend(&mut self, backend: FracBackend) {
        let _ = self.try_set_frac_backend(backend);
    }

    /// Guarded backend request. Routing choices never change the canonical
    /// fractional objective.
    pub fn try_set_frac_backend(&mut self, backend: FracBackend) -> Result<(), BlackCatError> {
        self.z.try_set_frac_backend(backend)?;
        Ok(())
    }

    /// Returns the current exploration temperature tracked by the runtime.
    pub fn temperature(&self) -> f64 {
        self.z.temperature()
    }

    /// Number of rewarded ZMeta evaluations committed by the runtime.
    pub fn zmeta_generation(&self) -> u64 {
        self.z.generation()
    }

    /// Reward attached to the current accepted ZMeta incumbent.
    pub fn zmeta_incumbent_reward(&self) -> Option<f64> {
        self.z.incumbent_reward()
    }

    /// Returns the canonical reward configuration used by guarded reports.
    pub fn reward_config(&self) -> &RewardCfg {
        &self.reward
    }

    /// Replaces reward shaping only before the runtime has committed learning
    /// under another objective. Semantically identical configurations are
    /// harmless no-ops; callers that need a new objective epoch must construct
    /// a fresh runtime so ZMeta, bandit posteriors, and reward statistics cannot
    /// mix evidence from different utilities.
    pub fn try_set_reward_config(&mut self, reward: RewardCfg) -> Result<(), BlackCatError> {
        self.validate_configuration()?;
        let next_config = reward.free_energy_config()?;
        let current_config = self.reward.free_energy_config()?;
        self.validate_selection_state()?;
        self.validate_statistics_state()?;
        if next_config == current_config {
            return Ok(());
        }
        if let Some(selection_id) = self.pending_selection_id() {
            return Err(BlackCatError::PendingSelection { selection_id });
        }
        let posterior_has_observations = self
            .bandits
            .observation_counts()
            .values()
            .flat_map(BTreeMap::values)
            .any(|observations| *observations > 0);
        if self.stats_steps > 0
            || self.z.generation() > 0
            || self.last_credited_selection_id.is_some()
            || posterior_has_observations
        {
            return Err(BlackCatError::InvalidConfig {
                field: "reward_config",
                detail: "the reward objective is locked after learning begins; construct a new BlackCatRuntime for a new objective epoch"
                    .to_string(),
            });
        }
        self.reward = reward;
        Ok(())
    }

    /// Returns the append-only heuristic store path without exposing writes
    /// that bypass the Wilson adoption guard.
    pub fn heuristics_path(&self) -> &std::path::Path {
        self.heur.path()
    }

    /// Returns the startup witness when the former default store was imported.
    pub fn heuristic_store_startup_legacy_import(&self) -> Option<&rewrite::HeurStoreLegacyImport> {
        self.heur.startup_legacy_import()
    }

    pub fn heuristic_adoption_config(&self) -> SoftHeuristicAdoptionConfig {
        self.heuristic_adoption
    }

    /// Reconfigure evidence thresholds before the first heuristic observation.
    pub fn try_set_heuristic_adoption_config(
        &mut self,
        config: SoftHeuristicAdoptionConfig,
    ) -> Result<(), BlackCatError> {
        config.validate()?;
        if config == self.heuristic_adoption {
            return Ok(());
        }
        if !self.heuristic_evidence.is_empty() || !self.adopted_heuristics.is_empty() {
            return Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption",
                detail: "evidence thresholds are locked after the first observation; construct a new BlackCatRuntime for a new evidence epoch"
                    .to_string(),
            });
        }
        self.heuristic_adoption = config;
        Ok(())
    }

    pub fn last_heuristic_adoption_report(&self) -> Option<&SoftHeuristicAdoptionReport> {
        self.last_heuristic_adoption.as_ref()
    }

    pub fn tracked_heuristic_count(&self) -> usize {
        self.heuristic_evidence.len()
    }

    /// Add one Bernoulli observation to a rule's evidence and persist it once
    /// its Wilson lower bound clears the configured baseline.
    pub fn try_observe_soft(
        &mut self,
        rule_text: &str,
        success: bool,
    ) -> Result<SoftHeuristicAdoptionReport, BlackCatError> {
        self.validate_configuration()?;
        self.heuristic_adoption.validate()?;
        let (rule, rule_id) = canonical_heuristic_rule(rule_text)?;
        let already_adopted = self.adopted_heuristics.contains(&rule_id);
        let mut next_evidence = self.heuristic_evidence.clone();
        if !next_evidence.contains_key(&rule_id)
            && next_evidence.len() >= BLACKCAT_MAX_TRACKED_HEURISTICS
        {
            return Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.rules",
                detail: format!(
                    "at most {BLACKCAT_MAX_TRACKED_HEURISTICS} distinct rules may be tracked"
                ),
            });
        }
        let evidence = next_evidence
            .entry(rule_id.clone())
            .or_insert(SoftHeuristicEvidenceState {
                successes: 0,
                trials: 0,
                config: self.heuristic_adoption,
                mode: SoftHeuristicEvidenceMode::StreamingStep,
            });
        if evidence.mode != SoftHeuristicEvidenceMode::StreamingStep
            || evidence.config != self.heuristic_adoption
        {
            return Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.evidence_policy",
                detail: "a rule cannot mix streaming observations with cumulative snapshots or change thresholds within one evidence epoch"
                    .to_string(),
            });
        }
        evidence.trials =
            evidence
                .trials
                .checked_add(1)
                .ok_or(BlackCatError::InvalidAdaptationState {
                    field: "heuristic_adoption.trials",
                })?;
        if success {
            evidence.successes =
                evidence
                    .successes
                    .checked_add(1)
                    .ok_or(BlackCatError::InvalidAdaptationState {
                        field: "heuristic_adoption.successes",
                    })?;
        }
        let interval = try_wilson_interval(
            evidence.successes,
            evidence.trials,
            self.heuristic_adoption.confidence_z,
        )?;
        let enough_trials = interval.trials >= self.heuristic_adoption.minimum_trials;
        let eligible =
            enough_trials && interval.lower > self.heuristic_adoption.baseline_probability;
        let mut next_adopted = self.adopted_heuristics.clone();
        let (decision, persistence, persistence_error) = if already_adopted {
            (SoftHeuristicAdoptionDecision::AlreadyAdopted, None, None)
        } else if !enough_trials {
            (SoftHeuristicAdoptionDecision::Collecting, None, None)
        } else if !eligible {
            (SoftHeuristicAdoptionDecision::BelowThreshold, None, None)
        } else {
            match self.persist_soft_heuristic(&rule, &rule_id, interval, self.heuristic_adoption) {
                Ok(receipt) => {
                    next_adopted.insert(rule_id.clone());
                    let decision = match receipt.disposition {
                        HeurStoreDisposition::Appended => SoftHeuristicAdoptionDecision::Adopted,
                        HeurStoreDisposition::AlreadyPresent => {
                            SoftHeuristicAdoptionDecision::AlreadyPersisted
                        }
                    };
                    (decision, Some(receipt), None)
                }
                Err(error) => (
                    SoftHeuristicAdoptionDecision::PersistenceFailed,
                    None,
                    Some(error),
                ),
            }
        };
        let report = self.soft_adoption_report(
            rule,
            rule_id,
            interval,
            self.heuristic_adoption,
            eligible,
            decision,
            persistence,
            persistence_error,
        );
        self.heuristic_evidence = next_evidence;
        self.adopted_heuristics = next_adopted;
        self.commit_soft_adoption_report(report.clone());
        Ok(report)
    }

    /// Evaluate externally accumulated evidence and return a complete adoption
    /// report. Counts must move monotonically if the same rule was seen before.
    pub fn try_adopt_soft_report(
        &mut self,
        rule_text: &str,
        wins: u32,
        trials: u32,
        baseline_probability: f64,
    ) -> Result<SoftHeuristicAdoptionReport, BlackCatError> {
        self.validate_configuration()?;
        let config = SoftHeuristicAdoptionConfig {
            minimum_trials: 1,
            baseline_probability,
            confidence_z: 1.96,
        };
        config.validate()?;
        let (rule, rule_id) = canonical_heuristic_rule(rule_text)?;
        let interval =
            try_wilson_interval(u64::from(wins), u64::from(trials), config.confidence_z)?;
        let mut next_evidence = self.heuristic_evidence.clone();
        if !next_evidence.contains_key(&rule_id)
            && next_evidence.len() >= BLACKCAT_MAX_TRACKED_HEURISTICS
        {
            return Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.rules",
                detail: format!(
                    "at most {BLACKCAT_MAX_TRACKED_HEURISTICS} distinct rules may be tracked"
                ),
            });
        }
        if let Some(previous) = next_evidence.get(&rule_id) {
            if previous.mode != SoftHeuristicEvidenceMode::CumulativeSnapshot
                || previous.config != config
            {
                return Err(BlackCatError::InvalidConfig {
                    field: "heuristic_adoption.evidence_policy",
                    detail: "a rule cannot mix streaming observations with cumulative snapshots or change thresholds within one evidence epoch"
                        .to_string(),
                });
            }
            let Some(successes_delta) = interval.successes.checked_sub(previous.successes) else {
                return Err(BlackCatError::InvalidConfig {
                    field: "heuristic_adoption.evidence",
                    detail: "cumulative wins/trials must advance monotonically for a tracked rule"
                        .to_string(),
                });
            };
            let Some(trials_delta) = interval.trials.checked_sub(previous.trials) else {
                return Err(BlackCatError::InvalidConfig {
                    field: "heuristic_adoption.evidence",
                    detail: "cumulative wins/trials must advance monotonically for a tracked rule"
                        .to_string(),
                });
            };
            if successes_delta > trials_delta {
                return Err(BlackCatError::InvalidConfig {
                    field: "heuristic_adoption.evidence",
                    detail: "cumulative wins/trials must advance monotonically for a tracked rule"
                        .to_string(),
                });
            }
        }
        next_evidence.insert(
            rule_id.clone(),
            SoftHeuristicEvidenceState {
                successes: interval.successes,
                trials: interval.trials,
                config,
                mode: SoftHeuristicEvidenceMode::CumulativeSnapshot,
            },
        );
        let eligible = interval.lower > config.baseline_probability;
        let mut next_adopted = self.adopted_heuristics.clone();
        let (decision, persistence, persistence_error) = if self
            .adopted_heuristics
            .contains(&rule_id)
        {
            (SoftHeuristicAdoptionDecision::AlreadyAdopted, None, None)
        } else if !eligible {
            (SoftHeuristicAdoptionDecision::BelowThreshold, None, None)
        } else {
            match self.persist_soft_heuristic(&rule, &rule_id, interval, config) {
                Ok(receipt) => {
                    next_adopted.insert(rule_id.clone());
                    let decision = match receipt.disposition {
                        HeurStoreDisposition::Appended => SoftHeuristicAdoptionDecision::Adopted,
                        HeurStoreDisposition::AlreadyPresent => {
                            SoftHeuristicAdoptionDecision::AlreadyPersisted
                        }
                    };
                    (decision, Some(receipt), None)
                }
                Err(error) => (
                    SoftHeuristicAdoptionDecision::PersistenceFailed,
                    None,
                    Some(error),
                ),
            }
        };
        let report = self.soft_adoption_report(
            rule,
            rule_id,
            interval,
            config,
            eligible,
            decision,
            persistence,
            persistence_error,
        );
        self.heuristic_evidence = next_evidence;
        self.adopted_heuristics = next_adopted;
        self.commit_soft_adoption_report(report.clone());
        Ok(report)
    }

    /// Compatibility boolean. It is `false` for invalid evidence, persistence
    /// failures, and candidates that have not crossed the guarded threshold.
    pub fn try_adopt_soft(
        &mut self,
        rule_text: &str,
        wins: u32,
        trials: u32,
        baseline_p: f64,
    ) -> bool {
        self.try_adopt_soft_report(rule_text, wins, trials, baseline_p)
            .map(|report| report.decision.is_persisted())
            .unwrap_or(false)
    }

    fn persist_soft_heuristic(
        &self,
        rule: &str,
        rule_id: &str,
        interval: WilsonInterval,
        config: SoftHeuristicAdoptionConfig,
    ) -> Result<HeurStoreReceipt, rewrite::HeurStoreError> {
        self.heur.append_once(
            rule,
            &rewrite::HeurStoreRecord {
                contract: BLACKCAT_HEURISTIC_ADOPTION_CONTRACT,
                contract_version: BLACKCAT_HEURISTIC_ADOPTION_CONTRACT_VERSION,
                rule_id: rule_id.to_string(),
                interval,
                minimum_trials: config.minimum_trials,
                baseline_probability: config.baseline_probability,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn soft_adoption_report(
        &self,
        rule: String,
        rule_id: String,
        interval: WilsonInterval,
        config: SoftHeuristicAdoptionConfig,
        eligible: bool,
        decision: SoftHeuristicAdoptionDecision,
        persistence: Option<HeurStoreReceipt>,
        persistence_error: Option<rewrite::HeurStoreError>,
    ) -> SoftHeuristicAdoptionReport {
        SoftHeuristicAdoptionReport {
            contract: BLACKCAT_HEURISTIC_ADOPTION_CONTRACT,
            contract_version: BLACKCAT_HEURISTIC_ADOPTION_CONTRACT_VERSION,
            rule_id,
            rule,
            interval,
            minimum_trials: config.minimum_trials,
            baseline_probability: config.baseline_probability,
            eligible,
            decision,
            persistence,
            persistence_error,
        }
    }

    fn commit_soft_adoption_report(&mut self, report: SoftHeuristicAdoptionReport) {
        self.last_heuristic_adoption = Some(report.clone());
        let bus = global_registry().event_bus();
        if bus.has_listeners("BlackCatHeuristicEvidence") {
            bus.publish(&PluginEvent::custom(
                "BlackCatHeuristicEvidence",
                serde_json::json!({"adoption": report}),
            ));
        }
    }

    /// Returns the duration since the last [`Self::begin_step`] call.
    pub fn elapsed_since_begin(&self) -> Option<Duration> {
        self.last_step_start.map(|start| start.elapsed())
    }

    /// Returns the undeformed contextual feature vector supplied by the caller.
    pub fn last_base_context(&self) -> &[f64] {
        &self.last_base_context
    }

    /// Returns the ZMeta-deformed contextual feature vector used for bandit updates.
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

    /// Returns the most recent committed ZMeta reward transition.
    pub fn last_zmeta_update(&self) -> Option<&ZMetaUpdateWitness> {
        self.last_zmeta_update.as_ref()
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
        let mut next_z = self.z.clone();
        let zmeta = next_z.try_abandon(selection_id)?;
        let mut next_bandits = self.bandits.clone();
        next_bandits
            .try_abandon_all()
            .map_err(|field| BlackCatError::InvalidAdaptationState { field })?;
        next_z.validate_adaptation_state()?;
        self.z = next_z;
        self.bandits = next_bandits;
        self.selection_pending = false;
        let bus = global_registry().event_bus();
        if bus.has_listeners("BlackCatSelectionAbandoned") {
            bus.publish(&PluginEvent::custom(
                "BlackCatSelectionAbandoned",
                serde_json::json!({
                    "selection": &self.last_selection_witness,
                    "zmeta": &zmeta,
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
        ("zmeta.sigma", z_params.sigma, true),
        ("zmeta.lr", z_params.lr, true),
        ("zmeta.alpha_frac", z_params.alpha_frac, true),
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
    let step_radius = z_params.lr * z_params.sigma;
    if !step_radius.is_finite() || step_radius <= 0.0 {
        return Err(invalid(
            "zmeta.step_radius",
            format!("learning_rate*sigma must be finite and positive, got {step_radius}"),
        ));
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
    use std::fs;
    use std::path::Path;

    fn sample_runtime() -> BlackCatRuntime {
        let mut groups = HashMap::new();
        groups.insert("tile".to_string(), vec!["a".to_string(), "b".to_string()]);
        let groups = ChoiceGroups { groups };
        BlackCatRuntime::new(ZMetaParams::default(), groups, 4, SoftBanditMode::TS, None)
    }

    fn sample_runtime_with_heur_path(path: &Path) -> BlackCatRuntime {
        let mut groups = HashMap::new();
        groups.insert("tile".to_string(), vec!["a".to_string(), "b".to_string()]);
        BlackCatRuntime::try_new(
            ZMetaParams::default(),
            ChoiceGroups { groups },
            4,
            SoftBanditMode::TS,
            Some(path.to_string_lossy().into_owned()),
        )
        .expect("runtime with a guarded heuristic path")
    }

    fn four_trial_adoption() -> SoftHeuristicAdoptionConfig {
        SoftHeuristicAdoptionConfig {
            minimum_trials: 4,
            baseline_probability: 0.5,
            confidence_z: 1.96,
        }
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

        runtime
            .try_choose(vec![1.0, 0.0, 0.0, 0.0])
            .expect("baseline selection");
        let pending_change = RewardCfg {
            scale_speed: 6.0,
            ..RewardCfg::default()
        };
        assert!(matches!(
            runtime.try_set_reward_config(pending_change),
            Err(BlackCatError::PendingSelection { selection_id: 1 })
        ));
        assert_eq!(runtime.reward_config().scale_speed, 5.0);
        assert_eq!(runtime.pending_selection_id(), Some(1));

        runtime
            .try_post_step_report(&StepMetrics::default())
            .expect("baseline reward remains creditable after rejection");
        let locked_change = RewardCfg {
            scale_speed: 7.0,
            ..RewardCfg::default()
        };
        assert!(matches!(
            runtime.try_set_reward_config(locked_change),
            Err(BlackCatError::InvalidConfig {
                field: "reward_config",
                ..
            })
        ));
        assert_eq!(runtime.reward_config().scale_speed, 5.0);
        assert_eq!(runtime.zmeta_generation(), 1);
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
    fn zmeta_candidate_is_executed_in_context_before_reward_acceptance() {
        let mut runtime = sample_runtime();
        let context = vec![1.0, 0.25, -0.5, 0.75];

        runtime
            .try_choose(context.clone())
            .expect("baseline selection");
        let baseline = runtime
            .last_selection_witness()
            .expect("baseline witness")
            .zmeta
            .clone();
        assert_eq!(baseline.kind, zmeta::ZMetaEvaluationKind::Baseline);
        assert_eq!(baseline.base_context, baseline.effective_context);
        let baseline_report = runtime
            .try_post_step_report(&StepMetrics::default())
            .expect("baseline reward");
        assert_eq!(runtime.zmeta_generation(), 1);
        assert_eq!(runtime.z_state(), &[0.0; 6]);

        runtime
            .try_choose(context.clone())
            .expect("candidate selection");
        let candidate = runtime
            .last_selection_witness()
            .expect("candidate witness")
            .zmeta
            .clone();
        assert_eq!(candidate.kind, zmeta::ZMetaEvaluationKind::Candidate);
        assert_ne!(candidate.effective_context, candidate.base_context);
        assert_eq!(runtime.last_base_context(), context);
        assert_eq!(runtime.last_context(), candidate.effective_context);

        let mut improved = StepMetrics::default();
        improved.extra.insert("reference_loss".into(), 10.0);
        improved.extra.insert("candidate_loss".into(), 0.0);
        let candidate_report = runtime
            .try_post_step_report(&improved)
            .expect("candidate reward");
        assert!(candidate_report.utility > baseline_report.utility);
        let accepted = runtime.last_zmeta_update().expect("ZMeta update");
        assert!(accepted.accepted);
        assert_eq!(accepted.evaluated_z, candidate.evaluated_z);
        assert_eq!(runtime.z_state(), candidate.evaluated_z);

        let incumbent = runtime.z_state().to_vec();
        runtime.try_choose(context).expect("worse candidate");
        let mut degraded = StepMetrics::default();
        degraded.extra.insert("reference_loss".into(), 0.0);
        degraded.extra.insert("candidate_loss".into(), 10.0);
        runtime
            .try_post_step_report(&degraded)
            .expect("degraded reward remains valid");
        assert!(!runtime.last_zmeta_update().unwrap().accepted);
        assert_eq!(runtime.z_state(), incumbent);
    }

    #[test]
    fn observation_only_reports_and_abandonment_do_not_train_zmeta() {
        let mut runtime = sample_runtime();
        runtime
            .try_post_step_report(&StepMetrics::default())
            .expect("observation-only report");
        assert_eq!(runtime.zmeta_generation(), 0);
        assert_eq!(runtime.zmeta_incumbent_reward(), None);
        assert!(runtime.last_zmeta_update().is_none());

        let context = vec![1.0, 0.25, -0.5, 0.75];
        runtime
            .try_choose(context.clone())
            .expect("pending baseline");
        assert_eq!(runtime.abandon_pending_selection(), Some(1));
        assert_eq!(runtime.zmeta_generation(), 0);
        assert_eq!(runtime.zmeta_incumbent_reward(), None);
        assert_eq!(runtime.pending_selection_id(), None);

        runtime.try_choose(context).expect("replacement baseline");
        assert_eq!(
            runtime.last_selection_witness().unwrap().zmeta.kind,
            zmeta::ZMetaEvaluationKind::Baseline
        );
        assert_eq!(runtime.pending_selection_id(), Some(2));
    }

    #[test]
    fn zmeta_and_bandit_replay_multiple_seeded_generations() {
        let mut left = sample_runtime();
        let mut right = sample_runtime();
        let context = vec![1.0, 0.25, -0.5, 0.75];
        let rewards = [(0.0, 0.0), (8.0, 0.0), (0.0, 4.0), (12.0, 0.0)];

        for (reference_loss, candidate_loss) in rewards {
            assert_eq!(
                left.try_choose(context.clone()).expect("left choice"),
                right.try_choose(context.clone()).expect("right choice")
            );
            assert_eq!(
                left.last_selection_witness(),
                right.last_selection_witness()
            );
            let mut metrics = StepMetrics::default();
            metrics
                .extra
                .insert("reference_loss".into(), reference_loss);
            metrics
                .extra
                .insert("candidate_loss".into(), candidate_loss);
            assert_eq!(
                left.try_post_step_report(&metrics).expect("left reward"),
                right.try_post_step_report(&metrics).expect("right reward")
            );
            assert_eq!(left.z_state(), right.z_state());
            assert_eq!(left.last_zmeta_update(), right.last_zmeta_update());
            assert_eq!(
                left.bandit_observation_counts(),
                right.bandit_observation_counts()
            );
        }
    }

    #[test]
    fn rejected_pending_reward_preserves_a_retryable_joint_transaction() {
        let mut runtime = sample_runtime();
        runtime
            .try_choose(vec![1.0, 0.25, -0.5, 0.75])
            .expect("pending baseline");
        let proposal = runtime
            .last_selection_witness()
            .expect("selection witness")
            .clone();
        let counts = runtime.bandit_observation_counts();
        let temperature = runtime.temperature();

        let mut invalid = StepMetrics::default();
        invalid.extra.insert("grad_norm".into(), f64::NAN);
        assert!(matches!(
            runtime.try_post_step_report(&invalid),
            Err(BlackCatError::NonFinite {
                field: "grad_norm",
                ..
            })
        ));
        assert_eq!(runtime.pending_selection_id(), Some(1));
        assert_eq!(runtime.zmeta_generation(), 0);
        assert_eq!(runtime.last_selection_witness(), Some(&proposal));
        assert_eq!(runtime.bandit_observation_counts(), counts);
        assert_eq!(runtime.temperature(), temperature);

        runtime
            .try_post_step_report(&StepMetrics::default())
            .expect("same pending evaluation remains creditable");
        assert_eq!(runtime.pending_selection_id(), None);
        assert_eq!(runtime.zmeta_generation(), 1);
        assert_eq!(
            runtime
                .bandit_observation_counts()
                .values()
                .flat_map(|group| group.values())
                .sum::<u64>(),
            1
        );
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

    #[test]
    fn soft_heuristic_adoption_requires_cumulative_evidence_and_appends_once() {
        let temp = tempfile::tempdir().expect("temporary directory");
        let path = temp.path().join("heur.kdsl");
        let mut runtime = sample_runtime_with_heur_path(&path);
        runtime
            .try_set_heuristic_adoption_config(four_trial_adoption())
            .expect("adoption configuration");
        let rule = "  when grad_norm > 1.0 => clip  ";

        for expected_trials in 1..4 {
            let report = runtime
                .try_observe_soft(rule, true)
                .expect("collecting evidence");
            assert_eq!(report.decision, SoftHeuristicAdoptionDecision::Collecting);
            assert_eq!(report.interval.trials, expected_trials);
            assert!(!path.exists());
        }

        let adopted = runtime
            .try_observe_soft(rule, true)
            .expect("four successes clear the guarded threshold");
        assert_eq!(adopted.decision, SoftHeuristicAdoptionDecision::Adopted);
        assert!(adopted.eligible);
        assert_eq!(adopted.rule, "when grad_norm > 1.0 => clip");
        assert_eq!(adopted.interval.successes, 4);
        assert_eq!(adopted.interval.trials, 4);
        assert!(adopted.interval.lower > 0.5);
        let receipt = adopted.persistence.as_ref().expect("durable receipt");
        assert_eq!(receipt.disposition, HeurStoreDisposition::Appended);
        assert!(receipt.durable_sync);
        let persisted = fs::read(&path).expect("persisted rule");
        assert!(String::from_utf8_lossy(&persisted).contains(&adopted.rule_id));

        let repeated = runtime
            .try_observe_soft(rule, false)
            .expect("an adopted rule keeps accumulating evidence");
        assert_eq!(
            repeated.decision,
            SoftHeuristicAdoptionDecision::AlreadyAdopted
        );
        assert_eq!(repeated.interval.successes, 4);
        assert_eq!(repeated.interval.trials, 5);
        assert!(!repeated.eligible);
        assert_eq!(fs::read(&path).expect("unchanged rule file"), persisted);
        assert_eq!(runtime.tracked_heuristic_count(), 1);
        assert_eq!(runtime.last_heuristic_adoption_report(), Some(&repeated));
    }

    #[test]
    fn heuristic_store_identity_prevents_duplicate_writes_across_runtimes() {
        let temp = tempfile::tempdir().expect("temporary directory");
        let path = temp.path().join("heur.kdsl");
        let rule = "when entropy > 2.0 => cool";

        let mut first = sample_runtime_with_heur_path(&path);
        first
            .try_set_heuristic_adoption_config(four_trial_adoption())
            .expect("first configuration");
        for _ in 0..4 {
            first
                .try_observe_soft(rule, true)
                .expect("first evidence epoch");
        }
        let persisted = fs::read(&path).expect("first persisted snapshot");

        let mut second = sample_runtime_with_heur_path(&path);
        second
            .try_set_heuristic_adoption_config(four_trial_adoption())
            .expect("second configuration");
        let mut report = None;
        for _ in 0..4 {
            report = Some(
                second
                    .try_observe_soft(rule, true)
                    .expect("replayed evidence epoch"),
            );
        }
        let report = report.expect("fourth report");
        assert_eq!(
            report.decision,
            SoftHeuristicAdoptionDecision::AlreadyPersisted
        );
        assert_eq!(
            report
                .persistence
                .as_ref()
                .map(|receipt| receipt.disposition),
            Some(HeurStoreDisposition::AlreadyPresent)
        );
        assert_eq!(fs::read(&path).expect("deduplicated snapshot"), persisted);
    }

    #[test]
    fn persistence_failure_is_reported_and_retryable_without_fake_adoption() {
        let temp = tempfile::tempdir().expect("temporary directory");
        let blocker = temp.path().join("blocked");
        fs::write(&blocker, b"not a directory").expect("blocker file");
        let path = blocker.join("heur.kdsl");
        let mut runtime = sample_runtime_with_heur_path(&path);
        runtime
            .try_set_heuristic_adoption_config(four_trial_adoption())
            .expect("adoption configuration");

        let mut report = None;
        for _ in 0..4 {
            report = Some(
                runtime
                    .try_observe_soft("when loss falls => retain", true)
                    .expect("evidence remains observable despite store failure"),
            );
        }
        let failed = report.expect("failure report");
        assert_eq!(
            failed.decision,
            SoftHeuristicAdoptionDecision::PersistenceFailed
        );
        assert!(failed.eligible);
        assert!(failed.persistence.is_none());
        assert_eq!(
            failed
                .persistence_error
                .as_ref()
                .map(|error| error.operation),
            Some("create_parent")
        );
        assert!(!failed.decision.is_persisted());

        fs::remove_file(&blocker).expect("remove blocker");
        let recovered = runtime
            .try_observe_soft("when loss falls => retain", true)
            .expect("next observation retries persistence");
        assert_eq!(recovered.decision, SoftHeuristicAdoptionDecision::Adopted);
        assert_eq!(recovered.interval.trials, 5);
        assert!(path.exists());
    }

    #[test]
    fn external_evidence_is_guarded_monotonic_and_configuration_locks() {
        let temp = tempfile::tempdir().expect("temporary directory");
        let path = temp.path().join("heur.kdsl");
        let mut runtime = sample_runtime_with_heur_path(&path);
        let first = runtime
            .try_adopt_soft_report("when stable => retain", 1, 2, 0.5)
            .expect("valid initial snapshot");
        assert_eq!(
            first.decision,
            SoftHeuristicAdoptionDecision::BelowThreshold
        );
        assert!(!path.exists());

        assert!(matches!(
            runtime.try_adopt_soft_report("when stable => retain", 2, 3, 0.4),
            Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.evidence_policy",
                ..
            })
        ));
        assert!(matches!(
            runtime.try_observe_soft("when stable => retain", true),
            Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.evidence_policy",
                ..
            })
        ));
        assert!(matches!(
            runtime.try_adopt_soft_report("when stable => retain", 0, 2, 0.5),
            Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.evidence",
                ..
            })
        ));
        assert!(matches!(
            runtime.try_adopt_soft_report("when stable => retain", 3, 3, 0.5),
            Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.evidence",
                ..
            })
        ));
        assert!(matches!(
            runtime.try_adopt_soft_report("when impossible => reject", 2, 1, 0.5),
            Err(BlackCatError::Wilson(_))
        ));
        assert!(matches!(
            runtime.try_adopt_soft_report("line one\nline two", 1, 1, 0.5),
            Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption.rule",
                ..
            })
        ));
        assert!(matches!(
            runtime.try_set_heuristic_adoption_config(four_trial_adoption()),
            Err(BlackCatError::InvalidConfig {
                field: "heuristic_adoption",
                ..
            })
        ));
        assert_eq!(runtime.tracked_heuristic_count(), 1);
    }
}

// =================== zmeta.rs ===================
pub mod zmeta {
    use super::{FracBackend, BLACKCAT_MAX_ZMETA_DIM};
    use crate::runtime::zspace_optimizer::{
        evaluate_zspace_fractional_regularizer, ZSpaceMetaOptimizerError,
        ZSPACE_FRACTIONAL_REGULARIZER_FORMULA, ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND,
        ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER,
    };
    use randless::{Rng, StdRng};
    use thiserror::Error;

    /// Stable ask/tell contract for BlackCat's latent context search.
    pub const ZMETA_ES_CONTRACT: &str = "spiraltorch.blackcat.zmeta-es";
    /// Schema version for [`ZMetaProposalWitness`] and [`ZMetaUpdateWitness`].
    pub const ZMETA_ES_CONTRACT_VERSION: u32 = 1;
    /// Bounded deformation applied to a contextual feature.
    pub const ZMETA_CONTEXT_DEFORMATION: &str =
        "x_effective[i]=x[i]+mean_j(tanh(z_eval[j])) where j mod context_dim=i; tile if empty";
    /// Candidate rule for the temperature-scaled `(1+1)` evolution strategy.
    pub const ZMETA_PROPOSAL_RULE: &str =
        "z_eval=z_incumbent+learning_rate*sigma*temperature*direction";

    #[derive(Clone, Debug, Error, PartialEq)]
    pub enum ZMetaError {
        #[error("invalid ZMeta configuration at '{field}': {detail}")]
        InvalidConfig { field: &'static str, detail: String },
        #[error("invalid ZMeta state at '{field}'")]
        InvalidState { field: &'static str },
        #[error("ZMeta evaluation {evaluation_id} is still awaiting reward or abandonment")]
        PendingEvaluation { evaluation_id: u64 },
        #[error("ZMeta has no evaluation awaiting reward")]
        MissingEvaluation,
        #[error("ZMeta expected evaluation {expected}, received {actual}")]
        EvaluationMismatch { expected: u64, actual: u64 },
        #[error("ZMeta field '{field}' must be finite, got {value}")]
        NonFinite { field: &'static str, value: f64 },
        #[error(transparent)]
        Fractional(#[from] ZSpaceMetaOptimizerError),
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ZMetaEvaluationKind {
        Baseline,
        Candidate,
    }

    /// Replay-oriented witness emitted before a latent state is evaluated.
    #[derive(Clone, Debug, PartialEq, serde::Serialize)]
    pub struct ZMetaProposalWitness {
        pub contract: &'static str,
        pub contract_version: u32,
        pub evaluation_id: u64,
        pub kind: ZMetaEvaluationKind,
        pub proposal_rule: &'static str,
        pub context_deformation: &'static str,
        pub fractional_formula: &'static str,
        pub fractional_semantic_owner: &'static str,
        pub fractional_semantic_backend: &'static str,
        pub requested_fractional_backend: String,
        pub incumbent_z: Vec<f64>,
        pub evaluated_z: Vec<f64>,
        pub direction: Vec<f64>,
        pub base_step_radius: f64,
        pub temperature: f64,
        pub step_radius: f64,
        pub incumbent_reward: Option<f64>,
        pub base_context: Vec<f64>,
        pub effective_context: Vec<f64>,
        pub structural_projection: Vec<f64>,
        pub fractional_energy: f64,
        pub fractional_penalty: f64,
    }

    /// Atomic state transition produced when an evaluated latent receives a reward.
    #[derive(Clone, Debug, PartialEq, serde::Serialize)]
    pub struct ZMetaUpdateWitness {
        pub contract: &'static str,
        pub contract_version: u32,
        pub evaluation_id: u64,
        pub kind: ZMetaEvaluationKind,
        pub reward: f64,
        pub incumbent_reward_before: Option<f64>,
        pub incumbent_reward_after: f64,
        pub reward_delta: Option<f64>,
        pub accepted: bool,
        pub generation: u64,
        pub z_before: Vec<f64>,
        pub evaluated_z: Vec<f64>,
        pub z_after: Vec<f64>,
        pub direction_after: Vec<f64>,
        pub fractional_penalty_before: f64,
        pub fractional_penalty_evaluated: f64,
        pub fractional_penalty_after: f64,
    }

    #[derive(Clone, Debug, PartialEq, serde::Serialize)]
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

    impl ZMetaParams {
        pub fn validate(&self) -> Result<(), ZMetaError> {
            if self.dim == 0 || self.dim > BLACKCAT_MAX_ZMETA_DIM {
                return Err(ZMetaError::InvalidConfig {
                    field: "dim",
                    detail: format!("expected 1..={BLACKCAT_MAX_ZMETA_DIM}, got {}", self.dim),
                });
            }
            for (field, value, strictly_positive) in [
                ("sigma", self.sigma, true),
                ("lr", self.lr, true),
                ("alpha_frac", self.alpha_frac, true),
                ("lam_frac", self.lam_frac, false),
                ("orientation_eta", self.orientation_eta, false),
                ("orientation_eps", self.orientation_eps, true),
            ] {
                if !value.is_finite() || value < 0.0 || (strictly_positive && value == 0.0) {
                    return Err(ZMetaError::InvalidConfig {
                        field,
                        detail: format!(
                            "expected a finite {} value, got {value}",
                            if strictly_positive {
                                "positive"
                            } else {
                                "non-negative"
                            }
                        ),
                    });
                }
            }
            let step_radius = self.lr * self.sigma;
            if !step_radius.is_finite() || step_radius <= 0.0 {
                return Err(ZMetaError::InvalidConfig {
                    field: "step_radius",
                    detail: format!(
                        "learning_rate*sigma must be finite and positive, got {step_radius}"
                    ),
                });
            }
            Ok(())
        }
    }

    #[derive(Clone)]
    struct PendingEvaluation {
        witness: ZMetaProposalWitness,
        structural_before: Vec<f64>,
        structural_delta: Option<Vec<f64>>,
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
        configuration_error: Option<ZMetaError>,
        pending: Option<PendingEvaluation>,
        evaluation_counter: u64,
        generation: u64,
        incumbent_reward: Option<f64>,
        last_proposal: Option<ZMetaProposalWitness>,
        last_update: Option<ZMetaUpdateWitness>,
    }

    impl ZMetaES {
        /// Compatibility constructor. Invalid parameters produce a bounded,
        /// fail-closed sentinel; direct Rust callers should prefer
        /// [`Self::try_new`].
        pub fn new(params: ZMetaParams) -> Self {
            match Self::try_new(params) {
                Ok(value) => value,
                Err(error) => Self::build(ZMetaParams::default(), Some(error)),
            }
        }

        /// Construct a directly usable ask/tell evolution strategy.
        pub fn try_new(params: ZMetaParams) -> Result<Self, ZMetaError> {
            params.validate()?;
            Ok(Self::build(params, None))
        }

        fn build(params: ZMetaParams, configuration_error: Option<ZMetaError>) -> Self {
            let rng = StdRng::seed_from_u64(params.seed);
            let mut dir = vec![0.0; params.dim];
            for value in dir.iter_mut() {
                *value = rng.gauss(0.0, 1.0);
            }
            if !normalize_above(&mut dir, 0.0) {
                dir.fill(0.0);
                dir[0] = 1.0;
            }
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
                configuration_error,
                pending: None,
                evaluation_counter: 0,
                generation: 0,
                incumbent_reward: None,
                last_proposal: None,
                last_update: None,
            }
        }

        pub fn z(&self) -> &[f64] {
            &self.z
        }

        /// Returns the validated search and regularization parameters.
        pub fn params(&self) -> &ZMetaParams {
            &self.params
        }

        /// Returns the unit search direction used by the next candidate.
        pub fn direction(&self) -> &[f64] {
            &self.dir
        }

        /// Returns the retained error of a compatibility-created sentinel.
        pub fn configuration_error(&self) -> Option<&ZMetaError> {
            self.configuration_error.as_ref()
        }

        /// Returns the latent that is currently affecting an in-flight choice.
        pub fn active_z(&self) -> &[f64] {
            self.pending
                .as_ref()
                .map(|pending| pending.witness.evaluated_z.as_slice())
                .unwrap_or(&self.z)
        }

        pub fn pending_evaluation_id(&self) -> Option<u64> {
            self.pending
                .as_ref()
                .map(|pending| pending.witness.evaluation_id)
        }

        pub fn generation(&self) -> u64 {
            self.generation
        }

        pub fn incumbent_reward(&self) -> Option<f64> {
            self.incumbent_reward
        }

        pub fn last_proposal(&self) -> Option<&ZMetaProposalWitness> {
            self.last_proposal.as_ref()
        }

        pub fn last_update(&self) -> Option<&ZMetaUpdateWitness> {
            self.last_update.as_ref()
        }

        /// Returns the current exploration temperature.
        pub fn temperature(&self) -> f64 {
            self.temp
        }

        /// Compatibility temperature-bound setter. Invalid bounds leave the
        /// state unchanged.
        pub fn set_temp_bounds(&mut self, t_min: f64, t_max: f64) {
            let _ = self.try_set_temp_bounds(t_min, t_max);
        }

        pub fn try_set_temp_bounds(&mut self, t_min: f64, t_max: f64) -> Result<(), ZMetaError> {
            self.validate_adaptation_state()?;
            if let Some(pending) = &self.pending {
                return Err(ZMetaError::PendingEvaluation {
                    evaluation_id: pending.witness.evaluation_id,
                });
            }
            require_finite("temperature_min", t_min)?;
            require_finite("temperature_max", t_max)?;
            if t_min <= 0.0 || t_max <= t_min {
                return Err(ZMetaError::InvalidConfig {
                    field: "temperature_bounds",
                    detail: format!("expected 0 < min < max, got [{t_min}, {t_max}]"),
                });
            }
            self.temp_min = t_min;
            self.temp_max = t_max;
            self.temp = self.temp.clamp(t_min, t_max);
            Ok(())
        }

        /// Adjusts the exploration temperature using retry/gradient signals.
        pub fn temp_schedule(&mut self, retry: f64, grad_norm: f64, loss_var: f64) {
            let _ = self.try_temp_schedule(retry, grad_norm, loss_var);
        }

        pub fn try_temp_schedule(
            &mut self,
            retry: f64,
            grad_norm: f64,
            loss_var: f64,
        ) -> Result<f64, ZMetaError> {
            self.validate_adaptation_state()?;
            if let Some(pending) = &self.pending {
                return Err(ZMetaError::PendingEvaluation {
                    evaluation_id: pending.witness.evaluation_id,
                });
            }
            for (field, value) in [
                ("temperature.retry", retry),
                ("temperature.grad_norm", grad_norm),
                ("temperature.loss_variance", loss_var),
            ] {
                require_finite(field, value)?;
                if value < 0.0 {
                    return Err(ZMetaError::InvalidConfig {
                        field,
                        detail: format!("expected a non-negative value, got {value}"),
                    });
                }
            }
            let stagnation = (1.0 - (loss_var / (1.0 + loss_var))).clamp(0.0, 1.0);
            let grad_term = (grad_norm / (1.0 + grad_norm)).clamp(0.0, 1.0);
            let instability = retry + 0.5 * grad_term;
            let delta = 0.2 * stagnation - 0.3 * instability;
            let next = self.temp + delta;
            require_finite("temperature.next", next)?;
            self.temp = next.clamp(self.temp_min, self.temp_max);
            Ok(self.temp)
        }

        /// Records the preferred fractional execution route without changing
        /// the canonical Rust regularizer semantics.
        pub fn set_frac_backend(&mut self, backend: FracBackend) {
            let _ = self.try_set_frac_backend(backend);
        }

        pub fn try_set_frac_backend(&mut self, backend: FracBackend) -> Result<(), ZMetaError> {
            self.validate_adaptation_state()?;
            if let Some(pending) = &self.pending {
                return Err(ZMetaError::PendingEvaluation {
                    evaluation_id: pending.witness.evaluation_id,
                });
            }
            validate_fractional_backend(&backend)?;
            self.frac_backend = backend;
            Ok(())
        }

        pub fn frac_penalty(&self) -> f64 {
            self.try_frac_penalty().unwrap_or(f64::NAN)
        }

        pub fn try_frac_penalty(&self) -> Result<f64, ZMetaError> {
            fractional_penalty(&self.z, &self.params, &self.frac_backend)
                .map(|(_, penalty)| penalty)
        }

        pub fn active_frac_penalty(&self) -> f64 {
            self.try_active_frac_penalty().unwrap_or(f64::NAN)
        }

        pub fn try_active_frac_penalty(&self) -> Result<f64, ZMetaError> {
            if let Some(pending) = &self.pending {
                Ok(pending.witness.fractional_penalty)
            } else {
                self.try_frac_penalty()
            }
        }

        pub fn frac_penalty_proposed(&self) -> f64 {
            let radius = self.params.lr * self.params.sigma * self.temp;
            let proposed = self
                .z
                .iter()
                .zip(&self.dir)
                .map(|(value, direction)| value + radius * direction)
                .collect::<Vec<_>>();
            fractional_penalty(&proposed, &self.params, &self.frac_backend)
                .map(|(_, penalty)| penalty)
                .unwrap_or(f64::NAN)
        }

        pub fn validate_adaptation_state(&self) -> Result<(), ZMetaError> {
            if let Some(error) = &self.configuration_error {
                return Err(error.clone());
            }
            self.params.validate()?;
            validate_fractional_backend(&self.frac_backend)?;
            if self.z.len() != self.params.dim
                || self.dir.len() != self.params.dim
                || self.structural.len() != self.params.dim
            {
                return Err(ZMetaError::InvalidState {
                    field: "vector_shape",
                });
            }
            for (field, value) in [
                ("temperature", self.temp),
                ("temperature_min", self.temp_min),
                ("temperature_max", self.temp_max),
            ] {
                require_finite(field, value)?;
            }
            if self.temp_min <= 0.0
                || self.temp_max <= self.temp_min
                || self.temp < self.temp_min
                || self.temp > self.temp_max
            {
                return Err(ZMetaError::InvalidState {
                    field: "temperature_bounds",
                });
            }
            if self
                .z
                .iter()
                .chain(self.dir.iter())
                .chain(self.structural.iter())
                .any(|value| !value.is_finite())
            {
                return Err(ZMetaError::InvalidState {
                    field: "vector_state",
                });
            }
            let direction_norm = stable_norm(&self.dir);
            if !direction_norm.is_finite()
                || (!self.dir.is_empty() && (direction_norm - 1.0).abs() > 1.0e-6)
            {
                return Err(ZMetaError::InvalidState {
                    field: "direction_norm",
                });
            }
            let structural_norm = stable_norm(&self.structural);
            if !structural_norm.is_finite() || structural_norm > 1.0 + 1.0e-6 {
                return Err(ZMetaError::InvalidState {
                    field: "structural_norm",
                });
            }
            if self
                .incumbent_reward
                .is_some_and(|reward| !reward.is_finite())
            {
                return Err(ZMetaError::InvalidState {
                    field: "incumbent_reward",
                });
            }
            if self.generation > self.evaluation_counter {
                return Err(ZMetaError::InvalidState {
                    field: "generation",
                });
            }
            if let Some(pending) = &self.pending {
                let witness = &pending.witness;
                let base_step_radius = self.params.lr * self.params.sigma;
                let candidate_step_radius = base_step_radius * self.temp;
                let step_radius = match witness.kind {
                    ZMetaEvaluationKind::Baseline => 0.0,
                    ZMetaEvaluationKind::Candidate => candidate_step_radius,
                };
                if witness.evaluation_id != self.evaluation_counter
                    || self.last_proposal.as_ref() != Some(witness)
                    || witness.contract != ZMETA_ES_CONTRACT
                    || witness.contract_version != ZMETA_ES_CONTRACT_VERSION
                    || witness.proposal_rule != ZMETA_PROPOSAL_RULE
                    || witness.context_deformation != ZMETA_CONTEXT_DEFORMATION
                    || witness.fractional_formula != ZSPACE_FRACTIONAL_REGULARIZER_FORMULA
                    || witness.fractional_semantic_owner != ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER
                    || witness.fractional_semantic_backend != ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND
                    || witness.requested_fractional_backend != backend_label(&self.frac_backend)
                    || witness.incumbent_z != self.z
                    || witness.incumbent_reward != self.incumbent_reward
                    || witness.direction != self.dir
                    || !approximately_equal(witness.base_step_radius, base_step_radius)
                    || !approximately_equal(witness.temperature, self.temp)
                    || !approximately_equal(witness.step_radius, step_radius)
                    || witness.evaluated_z.len() != self.params.dim
                    || witness.structural_projection != self.structural
                    || pending.structural_before.len() != self.params.dim
                    || pending
                        .structural_before
                        .iter()
                        .any(|value| !value.is_finite())
                    || pending.structural_delta.as_ref().is_some_and(|delta| {
                        delta.len() != self.params.dim
                            || delta.iter().any(|value| !value.is_finite())
                    })
                    || witness.base_context.is_empty()
                    || witness.effective_context.len() != witness.base_context.len()
                    || witness
                        .base_context
                        .iter()
                        .chain(witness.direction.iter())
                        .chain(witness.effective_context.iter())
                        .chain(witness.evaluated_z.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(ZMetaError::InvalidState {
                        field: "pending_witness",
                    });
                }
                let expected_z = match witness.kind {
                    ZMetaEvaluationKind::Baseline => self.z.clone(),
                    ZMetaEvaluationKind::Candidate => self
                        .z
                        .iter()
                        .zip(&self.dir)
                        .map(|(value, direction)| value + candidate_step_radius * direction)
                        .collect(),
                };
                let effective_context = deform_context(&witness.base_context, &expected_z)?;
                let (energy, penalty) =
                    fractional_penalty(&witness.evaluated_z, &self.params, &self.frac_backend)?;
                if expected_z != witness.evaluated_z
                    || effective_context != witness.effective_context
                    || !approximately_equal(energy, witness.fractional_energy)
                    || !approximately_equal(penalty, witness.fractional_penalty)
                {
                    return Err(ZMetaError::InvalidState {
                        field: "pending_derivation",
                    });
                }
            }
            self.try_frac_penalty()?;
            Ok(())
        }

        /// Prepare one baseline or candidate latent and apply it to a context.
        /// The state is committed only after every derived value validates.
        pub fn try_ask(&mut self, context: &[f64]) -> Result<ZMetaProposalWitness, ZMetaError> {
            let evaluation_id =
                self.evaluation_counter
                    .checked_add(1)
                    .ok_or(ZMetaError::InvalidState {
                        field: "evaluation_counter",
                    })?;
            self.try_prepare_context(evaluation_id, context)
        }

        /// Prepare an evaluation with an externally coordinated identifier.
        pub fn try_prepare_context(
            &mut self,
            evaluation_id: u64,
            context: &[f64],
        ) -> Result<ZMetaProposalWitness, ZMetaError> {
            self.validate_adaptation_state()?;
            if let Some(pending) = &self.pending {
                return Err(ZMetaError::PendingEvaluation {
                    evaluation_id: pending.witness.evaluation_id,
                });
            }
            let expected =
                self.evaluation_counter
                    .checked_add(1)
                    .ok_or(ZMetaError::InvalidState {
                        field: "evaluation_counter",
                    })?;
            if evaluation_id != expected {
                return Err(ZMetaError::EvaluationMismatch {
                    expected,
                    actual: evaluation_id,
                });
            }
            if context.is_empty() || context.iter().any(|value| !value.is_finite()) {
                return Err(ZMetaError::InvalidConfig {
                    field: "context",
                    detail: "expected one or more finite contextual features".to_string(),
                });
            }

            let mut next = self.clone();
            let structural_before = next.structural.clone();
            let structural_delta = next.ingest_structural(Some(context));
            let kind = if next.incumbent_reward.is_some() {
                ZMetaEvaluationKind::Candidate
            } else {
                ZMetaEvaluationKind::Baseline
            };
            let base_step_radius = next.params.lr * next.params.sigma;
            let candidate_step_radius = base_step_radius * next.temp;
            if !candidate_step_radius.is_finite() || candidate_step_radius <= 0.0 {
                return Err(ZMetaError::InvalidState {
                    field: "proposal_step_radius",
                });
            }
            let step_radius = match kind {
                ZMetaEvaluationKind::Baseline => 0.0,
                ZMetaEvaluationKind::Candidate => candidate_step_radius,
            };
            let evaluated_z = match kind {
                ZMetaEvaluationKind::Baseline => next.z.clone(),
                ZMetaEvaluationKind::Candidate => next
                    .z
                    .iter()
                    .zip(&next.dir)
                    .map(|(value, direction)| value + candidate_step_radius * direction)
                    .collect(),
            };
            if evaluated_z.iter().any(|value| !value.is_finite()) {
                return Err(ZMetaError::InvalidState { field: "proposal" });
            }
            let effective_context = deform_context(context, &evaluated_z)?;
            let (fractional_energy, fractional_penalty) =
                fractional_penalty(&evaluated_z, &next.params, &next.frac_backend)?;
            let witness = ZMetaProposalWitness {
                contract: ZMETA_ES_CONTRACT,
                contract_version: ZMETA_ES_CONTRACT_VERSION,
                evaluation_id,
                kind,
                proposal_rule: ZMETA_PROPOSAL_RULE,
                context_deformation: ZMETA_CONTEXT_DEFORMATION,
                fractional_formula: ZSPACE_FRACTIONAL_REGULARIZER_FORMULA,
                fractional_semantic_owner: ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER,
                fractional_semantic_backend: ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND,
                requested_fractional_backend: backend_label(&next.frac_backend),
                incumbent_z: next.z.clone(),
                evaluated_z,
                direction: next.dir.clone(),
                base_step_radius,
                temperature: next.temp,
                step_radius,
                incumbent_reward: next.incumbent_reward,
                base_context: context.to_vec(),
                effective_context,
                structural_projection: next.structural.clone(),
                fractional_energy,
                fractional_penalty,
            };
            next.pending = Some(PendingEvaluation {
                witness: witness.clone(),
                structural_before,
                structural_delta,
            });
            next.evaluation_counter = evaluation_id;
            next.last_proposal = Some(witness.clone());
            next.validate_adaptation_state()?;
            *self = next;
            Ok(witness)
        }

        /// Commit exactly one observed reward to the pending latent evaluation.
        pub fn try_tell_pending(&mut self, reward: f64) -> Result<ZMetaUpdateWitness, ZMetaError> {
            let evaluation_id = self
                .pending_evaluation_id()
                .ok_or(ZMetaError::MissingEvaluation)?;
            self.try_tell(evaluation_id, reward)
        }

        /// Commit a reward with an externally coordinated identifier.
        pub fn try_tell(
            &mut self,
            evaluation_id: u64,
            reward: f64,
        ) -> Result<ZMetaUpdateWitness, ZMetaError> {
            self.validate_adaptation_state()?;
            require_finite("reward", reward)?;
            let mut next = self.clone();
            let pending = next.pending.take().ok_or(ZMetaError::MissingEvaluation)?;
            if pending.witness.evaluation_id != evaluation_id {
                return Err(ZMetaError::EvaluationMismatch {
                    expected: pending.witness.evaluation_id,
                    actual: evaluation_id,
                });
            }
            let z_before = next.z.clone();
            let incumbent_reward_before = next.incumbent_reward;
            let reward_delta = incumbent_reward_before.map(|baseline| reward - baseline);
            if reward_delta.is_some_and(|delta| !delta.is_finite()) {
                return Err(ZMetaError::InvalidState {
                    field: "reward_delta",
                });
            }
            let accepted = match pending.witness.kind {
                ZMetaEvaluationKind::Baseline => true,
                ZMetaEvaluationKind::Candidate => reward_delta.is_some_and(|delta| delta > 0.0),
            };
            if pending.witness.kind == ZMetaEvaluationKind::Candidate && accepted {
                next.z = pending.witness.evaluated_z.clone();
            }
            if accepted {
                next.incumbent_reward = Some(reward);
            }
            if let Some(delta) = reward_delta {
                next.adapt_direction(accepted)?;
                if let Some(structural_delta) = pending.structural_delta {
                    next.apply_structural_drive(structural_delta, delta);
                    if !normalize_above(&mut next.dir, 0.0) {
                        return Err(ZMetaError::InvalidState {
                            field: "direction_update",
                        });
                    }
                }
            }
            next.generation = next
                .generation
                .checked_add(1)
                .ok_or(ZMetaError::InvalidState {
                    field: "generation",
                })?;
            let incumbent_reward_after = next.incumbent_reward.ok_or(ZMetaError::InvalidState {
                field: "incumbent_reward",
            })?;
            let penalty_before = fractional_penalty(&z_before, &next.params, &next.frac_backend)?.1;
            let penalty_after = next.try_frac_penalty()?;
            let update = ZMetaUpdateWitness {
                contract: ZMETA_ES_CONTRACT,
                contract_version: ZMETA_ES_CONTRACT_VERSION,
                evaluation_id,
                kind: pending.witness.kind,
                reward,
                incumbent_reward_before,
                incumbent_reward_after,
                reward_delta,
                accepted,
                generation: next.generation,
                z_before,
                evaluated_z: pending.witness.evaluated_z,
                z_after: next.z.clone(),
                direction_after: next.dir.clone(),
                fractional_penalty_before: penalty_before,
                fractional_penalty_evaluated: pending.witness.fractional_penalty,
                fractional_penalty_after: penalty_after,
            };
            next.last_update = Some(update.clone());
            next.validate_adaptation_state()?;
            *self = next;
            Ok(update)
        }

        /// Release an unevaluated candidate without changing the incumbent or
        /// consuming a direction update.
        pub fn try_abandon_pending(&mut self) -> Result<ZMetaProposalWitness, ZMetaError> {
            let evaluation_id = self
                .pending_evaluation_id()
                .ok_or(ZMetaError::MissingEvaluation)?;
            self.try_abandon(evaluation_id)
        }

        /// Release an evaluation with an externally coordinated identifier.
        pub fn try_abandon(
            &mut self,
            evaluation_id: u64,
        ) -> Result<ZMetaProposalWitness, ZMetaError> {
            self.validate_adaptation_state()?;
            let mut next = self.clone();
            let pending = next.pending.take().ok_or(ZMetaError::MissingEvaluation)?;
            if pending.witness.evaluation_id != evaluation_id {
                return Err(ZMetaError::EvaluationMismatch {
                    expected: pending.witness.evaluation_id,
                    actual: evaluation_id,
                });
            }
            next.structural = pending.structural_before;
            next.validate_adaptation_state()?;
            let witness = pending.witness;
            *self = next;
            Ok(witness)
        }

        fn adapt_direction(&mut self, accepted: bool) -> Result<(), ZMetaError> {
            if accepted {
                for direction in &mut self.dir {
                    *direction = 0.7 * *direction + 0.3 * self.rng.gauss(0.0, 1.0);
                }
            } else {
                let mut kick = (0..self.params.dim)
                    .map(|_| self.rng.gauss(0.0, 0.5))
                    .collect::<Vec<_>>();
                let projection = dot(&kick, &self.dir);
                for (value, direction) in kick.iter_mut().zip(&self.dir) {
                    *value -= projection * direction;
                }
                for (direction, value) in self.dir.iter_mut().zip(kick) {
                    *direction = 0.9 * *direction + 0.1 * value;
                }
            }
            if !normalize_above(&mut self.dir, 0.0) {
                return Err(ZMetaError::InvalidState {
                    field: "direction_update",
                });
            }
            Ok(())
        }

        fn ingest_structural(&mut self, structural: Option<&[f64]>) -> Option<Vec<f64>> {
            let raw = structural?;
            if self.params.dim == 0 || raw.is_empty() {
                return None;
            }

            let raw_scale = raw.iter().map(|value| value.abs()).fold(0.0f64, f64::max);
            if !raw_scale.is_finite() || raw_scale == 0.0 {
                return None;
            }
            let mut new_vec = vec![0.0f64; self.params.dim];
            let mut counts = vec![0usize; self.params.dim];
            for (index, value) in raw.iter().copied().enumerate() {
                let coordinate = index % self.params.dim;
                new_vec[coordinate] += value / raw_scale;
                counts[coordinate] += 1;
            }
            for (index, value) in new_vec.iter_mut().enumerate() {
                *value = if counts[index] == 0 {
                    raw[index % raw.len()] / raw_scale
                } else {
                    *value / counts[index] as f64
                };
            }

            if !normalize_above(&mut new_vec, 0.0) {
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

    fn require_finite(field: &'static str, value: f64) -> Result<f64, ZMetaError> {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(ZMetaError::NonFinite { field, value })
        }
    }

    fn validate_fractional_backend(backend: &FracBackend) -> Result<(), ZMetaError> {
        match backend {
            FracBackend::CpuRadix2 => Ok(()),
            FracBackend::Wgpu { radix } if *radix >= 2 && radix.is_power_of_two() => Ok(()),
            FracBackend::Wgpu { radix } => Err(ZMetaError::InvalidConfig {
                field: "fractional_backend.radix",
                detail: format!("expected a power of two in 2..=128, got {radix}"),
            }),
        }
    }

    fn backend_label(backend: &FracBackend) -> String {
        match backend {
            FracBackend::CpuRadix2 => "cpu_radix2".to_string(),
            FracBackend::Wgpu { radix } => format!("wgpu_radix{radix}"),
        }
    }

    fn fractional_penalty(
        z: &[f64],
        params: &ZMetaParams,
        backend: &FracBackend,
    ) -> Result<(f64, f64), ZMetaError> {
        validate_fractional_backend(backend)?;
        let report = evaluate_zspace_fractional_regularizer(z, params.alpha_frac)?;
        let penalty = report.energy * params.lam_frac;
        require_finite("fractional_penalty", penalty)?;
        Ok((report.energy, penalty))
    }

    fn deform_context(context: &[f64], z: &[f64]) -> Result<Vec<f64>, ZMetaError> {
        if context.is_empty() || z.is_empty() {
            return Err(ZMetaError::InvalidState {
                field: "context_deformation_dimension",
            });
        }
        let mut deformation = vec![0.0; context.len()];
        let mut counts = vec![0usize; context.len()];
        for (index, value) in z.iter().copied().enumerate() {
            let feature = index % context.len();
            deformation[feature] += value.tanh();
            counts[feature] += 1;
        }
        for (index, value) in deformation.iter_mut().enumerate() {
            *value = if counts[index] == 0 {
                z[index % z.len()].tanh()
            } else {
                *value / counts[index] as f64
            };
        }
        context
            .iter()
            .zip(deformation)
            .map(|(value, offset)| {
                let deformed = *value + offset;
                if deformed.is_finite() {
                    Ok(deformed)
                } else {
                    Err(ZMetaError::InvalidState {
                        field: "effective_context",
                    })
                }
            })
            .collect()
    }

    fn approximately_equal(left: f64, right: f64) -> bool {
        if left == right {
            return true;
        }
        let scale = left.abs().max(right.abs()).max(1.0);
        (left - right).abs() <= 128.0 * f64::EPSILON * scale
    }

    #[cfg(test)]
    fn assert_backend_invariant(z: &[f64], params: &ZMetaParams) {
        let cpu =
            fractional_penalty(z, params, &FracBackend::CpuRadix2).expect("CPU semantic penalty");
        for radix in [2, 4, 8, 16] {
            let routed = fractional_penalty(z, params, &FracBackend::Wgpu { radix })
                .expect("WGPU route shares semantics");
            assert_eq!(cpu, routed);
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

        #[test]
        fn structural_projection_uses_context_features_beyond_the_latent_width() {
            let mut es = ZMetaES::try_new(ZMetaParams {
                dim: 2,
                seed: 13,
                ..Default::default()
            })
            .expect("valid ES");

            let proposal = es
                .try_ask(&[0.0, 0.0, 0.0, 1.0])
                .expect("tail feature remains visible");
            assert_eq!(proposal.structural_projection[0], 0.0);
            assert!(proposal.structural_projection[1] > 0.0);
            assert!(approximately_equal(
                stable_norm(&proposal.structural_projection),
                1.0
            ));
        }

        #[test]
        fn ask_tell_establishes_baseline_then_accepts_and_rejects_candidates() {
            let mut es = ZMetaES::try_new(ZMetaParams {
                dim: 3,
                seed: 19,
                ..Default::default()
            })
            .expect("valid ES");
            let context = [1.0, 0.25, -0.5, 0.75];

            let baseline = es.try_ask(&context).expect("baseline proposal");
            assert_eq!(baseline.kind, ZMetaEvaluationKind::Baseline);
            assert_eq!(baseline.incumbent_z, baseline.evaluated_z);
            assert_eq!(baseline.base_context, baseline.effective_context);
            let baseline_update = es.try_tell_pending(0.25).expect("baseline reward");
            assert!(baseline_update.accepted);
            assert_eq!(es.generation(), 1);
            assert_eq!(es.z(), &[0.0, 0.0, 0.0]);

            let candidate = es
                .try_prepare_context(2, &context)
                .expect("candidate proposal");
            assert_eq!(candidate.kind, ZMetaEvaluationKind::Candidate);
            assert_ne!(candidate.evaluated_z, candidate.incumbent_z);
            assert_ne!(candidate.effective_context, candidate.base_context);
            assert!(candidate
                .effective_context
                .iter()
                .zip(&candidate.base_context)
                .all(|(effective, base)| (effective - base).abs() <= 1.0));
            let accepted_z = candidate.evaluated_z.clone();
            let accepted = es.try_tell(2, 0.75).expect("improved candidate");
            assert!(accepted.accepted);
            assert_eq!(es.z(), accepted_z);

            let incumbent = es.z().to_vec();
            es.try_prepare_context(3, &context).expect("next candidate");
            let rejected = es.try_tell(3, -0.5).expect("worse candidate");
            assert!(!rejected.accepted);
            assert_eq!(es.z(), incumbent);
            assert_eq!(es.incumbent_reward(), Some(0.75));
            assert_eq!(es.generation(), 3);
        }

        #[test]
        fn telemetry_temperature_controls_the_next_candidate_radius() {
            let mut es = ZMetaES::try_new(ZMetaParams {
                dim: 3,
                seed: 21,
                ..Default::default()
            })
            .expect("valid ES");
            let context = [1.0, 0.25, -0.5];
            es.try_ask(&context).expect("baseline proposal");
            es.try_tell_pending(0.25).expect("baseline reward");

            let temperature = es
                .try_temp_schedule(0.0, 0.0, 0.0)
                .expect("stable telemetry raises exploration temperature");
            let candidate = es.try_ask(&context).expect("candidate proposal");
            assert!(temperature > 1.0);
            assert_eq!(candidate.temperature, temperature);
            assert!(approximately_equal(
                candidate.base_step_radius,
                es.params().lr * es.params().sigma
            ));
            assert!(approximately_equal(
                candidate.step_radius,
                candidate.base_step_radius * candidate.temperature
            ));
            let actual_radius = candidate
                .evaluated_z
                .iter()
                .zip(&candidate.incumbent_z)
                .fold(0.0f64, |norm, (candidate, incumbent)| {
                    norm.hypot(candidate - incumbent)
                });
            assert!(approximately_equal(actual_radius, candidate.step_radius));

            let before = es.clone();
            assert!(matches!(
                es.try_temp_schedule(0.0, 0.0, 0.0),
                Err(ZMetaError::PendingEvaluation { evaluation_id: 2 })
            ));
            assert!(matches!(
                es.try_set_temp_bounds(0.5, 2.0),
                Err(ZMetaError::PendingEvaluation { evaluation_id: 2 })
            ));
            assert_eq!(es.temperature(), before.temperature());
            assert_eq!(es.pending_evaluation_id(), before.pending_evaluation_id());
            assert_eq!(es.last_proposal(), before.last_proposal());

            es.try_abandon_pending().expect("release candidate");
            assert!(matches!(
                es.try_set_temp_bounds(0.0, 2.0),
                Err(ZMetaError::InvalidConfig {
                    field: "temperature_bounds",
                    ..
                })
            ));
        }

        #[test]
        fn failed_tell_and_abandonment_are_atomic() {
            let mut es = ZMetaES::try_new(ZMetaParams {
                dim: 3,
                seed: 23,
                ..Default::default()
            })
            .expect("valid ES");
            let context = [1.0, 0.5, -0.25];
            let structural_before = es.structural.clone();
            let proposal = es
                .try_prepare_context(1, &context)
                .expect("pending baseline");

            assert!(matches!(
                es.try_tell(2, 0.0),
                Err(ZMetaError::EvaluationMismatch {
                    expected: 1,
                    actual: 2
                })
            ));
            assert_eq!(es.pending_evaluation_id(), Some(1));
            assert!(matches!(
                es.try_tell(1, f64::NAN),
                Err(ZMetaError::NonFinite {
                    field: "reward",
                    ..
                })
            ));
            assert_eq!(es.pending_evaluation_id(), Some(1));

            let abandoned = es.try_abandon(1).expect("abandon proposal");
            assert_eq!(abandoned, proposal);
            assert_eq!(es.pending_evaluation_id(), None);
            assert_eq!(es.structural, structural_before);
            assert_eq!(es.generation(), 0);
            assert_eq!(es.incumbent_reward(), None);
        }

        #[test]
        fn fractional_semantics_are_backend_invariant() {
            let params = ZMetaParams {
                dim: 4,
                alpha_frac: 0.45,
                lam_frac: 0.3,
                ..Default::default()
            };
            assert_backend_invariant(&[0.25, -0.75, 0.5, 0.125], &params);
            assert!(matches!(
                validate_fractional_backend(&FracBackend::Wgpu { radix: 3 }),
                Err(ZMetaError::InvalidConfig {
                    field: "fractional_backend.radix",
                    ..
                })
            ));
        }

        #[test]
        fn context_projection_keeps_every_latent_coordinate_causal() {
            let context = [1.0, 0.0, 0.0, 0.0];
            let mut z = vec![0.0; 6];
            z[4] = 1.0;
            let effective = deform_context(&context, &z).expect("folded deformation");
            assert!(effective[0] > context[0]);
            assert_eq!(effective[1], context[1]);

            let tiled = deform_context(&[0.0; 8], &[0.5, -0.5])
                .expect("short latents tile across features");
            assert!(tiled
                .iter()
                .enumerate()
                .all(|(index, value)| value.signum() == if index % 2 == 0 { 1.0 } else { -1.0 }));
        }

        #[test]
        fn guarded_constructor_rejects_degenerate_search_parameters() {
            assert!(matches!(
                ZMetaES::try_new(ZMetaParams {
                    sigma: 0.0,
                    ..Default::default()
                }),
                Err(ZMetaError::InvalidConfig { field: "sigma", .. })
            ));
            let compatibility = ZMetaES::new(ZMetaParams {
                dim: usize::MAX,
                ..Default::default()
            });
            assert!(compatibility.validate_adaptation_state().is_err());
            assert_eq!(compatibility.z().len(), ZMetaParams::default().dim);
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
            let mut idx = if equivalent_hint_prior {
                hint_index.unwrap_or(0)
            } else {
                0
            };
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
        fn non_equivalent_ucb_score_ties_ignore_the_hint() {
            let mut bandit = SoftBandit::new_seeded(
                vec!["canonical".to_string(), "hinted".to_string()],
                1,
                SoftBanditMode::UCB,
                23,
            );
            bandit.arms[1].a[0] = 4.0;
            bandit.arms[1].b[0] = 2.0;
            bandit.arms[1].observations = 3;

            let decision = bandit
                .try_select_with_hint(&[1.0], Some("hinted"))
                .expect("valid tied UCB decision");
            assert_ne!(
                decision.arms[0].posterior_mean,
                decision.arms[1].posterior_mean
            );
            assert_ne!(
                decision.arms[0].predictive_stddev,
                decision.arms[1].predictive_stddev
            );
            assert!(
                (decision.arms[0].decision_score - decision.arms[1].decision_score).abs() < 1.0e-12
            );
            assert_eq!(decision.chosen, "canonical");
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
    pub use crate::heur::evidence::{
        try_wilson_interval, try_wilson_lower, WilsonError, WilsonInterval,
        WILSON_INTERVAL_CONTRACT, WILSON_INTERVAL_CONTRACT_VERSION, WILSON_INTERVAL_FORMULA,
        WILSON_MAX_EXACT_TRIALS,
    };

    /// Compatibility scalar for legacy callers. Guarded code should retain the
    /// typed interval or error returned by [`try_wilson_interval`].
    pub fn wilson_lower(successes: i32, trials: i32, z: f64) -> f64 {
        let Ok(successes) = u64::try_from(successes) else {
            return 0.0;
        };
        let Ok(trials) = u64::try_from(trials) else {
            return 0.0;
        };
        try_wilson_lower(successes, trials, z).unwrap_or(0.0)
    }
}

// =================== rewrite.rs ===================
pub mod rewrite {
    use std::fmt;
    use std::fs::{self, File, OpenOptions};
    use std::io::{self, Read};
    use std::path::{Path, PathBuf};

    use serde::Serialize;
    use sha2::{Digest, Sha256};

    use crate::heur::evidence::WilsonInterval;
    use crate::runtime::persistence::atomic_write;

    use super::canonical_heuristic_rule_text;

    pub const HEUR_STORE_CONTRACT: &str = "spiraltorch.blackcat.heur_store";
    pub const HEUR_STORE_CONTRACT_VERSION: u32 = 1;
    pub const HEUR_STORE_LEGACY_IMPORT_CONTRACT: &str =
        "spiraltorch.blackcat.heur_store.legacy_import";
    pub const HEUR_STORE_LEGACY_IMPORT_CONTRACT_VERSION: u32 = 1;
    pub const HEUR_STORE_MAX_BYTES: u64 = 64 * 1024 * 1024;

    #[derive(Clone, Debug, PartialEq, Serialize)]
    pub struct HeurStoreError {
        pub operation: &'static str,
        pub path: PathBuf,
        pub kind: String,
        pub detail: String,
    }

    impl fmt::Display for HeurStoreError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "heuristic store {} failed for '{}': {} ({})",
                self.operation,
                self.path.display(),
                self.detail,
                self.kind
            )
        }
    }

    impl std::error::Error for HeurStoreError {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum HeurStoreDisposition {
        Appended,
        AlreadyPresent,
    }

    #[derive(Clone, Debug, PartialEq, Serialize)]
    pub struct HeurStoreLegacyImport {
        pub contract: &'static str,
        pub contract_version: u32,
        pub source_path: PathBuf,
        pub source_bytes: u64,
        pub source_sha256: String,
    }

    #[derive(Clone, Debug, PartialEq, Serialize)]
    pub struct HeurStoreReceipt {
        pub contract: &'static str,
        pub contract_version: u32,
        pub path: PathBuf,
        pub rule_id: String,
        pub disposition: HeurStoreDisposition,
        pub bytes_before: u64,
        pub bytes_after: u64,
        pub durable_sync: bool,
        pub legacy_import: Option<HeurStoreLegacyImport>,
    }

    #[derive(Clone, Debug, PartialEq, Serialize)]
    pub struct HeurStoreRecord {
        pub contract: &'static str,
        pub contract_version: u32,
        pub rule_id: String,
        pub interval: WilsonInterval,
        pub minimum_trials: u64,
        pub baseline_probability: f64,
    }

    #[derive(Clone, Debug)]
    pub struct HeurStore {
        path: PathBuf,
        legacy_path: Option<PathBuf>,
        startup_legacy_import: Option<HeurStoreLegacyImport>,
    }

    impl HeurStore {
        pub fn new(custom: Option<String>) -> Self {
            Self::try_new(custom).unwrap_or_else(|_| {
                let (path, legacy_path) = default_paths();
                Self {
                    path,
                    legacy_path,
                    startup_legacy_import: None,
                }
            })
        }

        pub fn try_new(custom: Option<String>) -> Result<Self, HeurStoreError> {
            let (path, legacy_path) = match custom {
                Some(path) if path.trim().is_empty() => {
                    return Err(store_error(
                        "configure",
                        PathBuf::from(path),
                        "invalid_input",
                        "custom heuristic path must not be empty",
                    ));
                }
                Some(path) => (PathBuf::from(path), None),
                None => default_paths(),
            };
            Self::try_new_paths(path, legacy_path)
        }

        fn try_new_paths(
            path: PathBuf,
            legacy_path: Option<PathBuf>,
        ) -> Result<Self, HeurStoreError> {
            if path.as_os_str().is_empty() {
                return Err(store_error(
                    "configure",
                    path,
                    "invalid_input",
                    "heuristic path must not be empty",
                ));
            }
            let mut store = Self {
                path,
                legacy_path,
                startup_legacy_import: None,
            };
            store.startup_legacy_import = store.migrate_legacy_snapshot()?;
            Ok(store)
        }

        pub fn append_once(
            &self,
            rule_text: &str,
            record: &HeurStoreRecord,
        ) -> Result<HeurStoreReceipt, HeurStoreError> {
            let canonical = canonical_heuristic_rule_text(rule_text).map_err(|detail| {
                store_error("validate", self.path.clone(), "invalid_input", detail)
            })?;
            let expected_id = rule_id(canonical);
            if record.rule_id != expected_id {
                return Err(store_error(
                    "validate",
                    self.path.clone(),
                    "invalid_input",
                    "record rule_id does not match the canonical rule text",
                ));
            }
            if let Some(dir) = self.path.parent() {
                fs::create_dir_all(dir)
                    .map_err(|source| io_error("create_parent", &self.path, source))?;
            }
            let _store_lock = HeurStoreLock::acquire(&self.path)?;
            let bytes = read_bounded(&self.path)?;
            let bytes_before = u64::try_from(bytes.len()).map_err(|_| {
                store_error(
                    "read",
                    self.path.clone(),
                    "capacity",
                    "existing heuristic file length is not representable",
                )
            })?;
            let (mut bytes, legacy_import) = self.import_legacy(bytes)?;
            let existing = std::str::from_utf8(&bytes).map_err(|source| {
                store_error(
                    "decode",
                    self.path.clone(),
                    "invalid_data",
                    source.to_string(),
                )
            })?;
            let already_present = existing.lines().any(|line| {
                let line = line.trim();
                if line == canonical {
                    return true;
                }
                if line
                    .split_once("  # ")
                    .is_some_and(|(legacy_rule, _)| legacy_rule.trim() == canonical)
                {
                    return true;
                }
                line.strip_prefix("# blackcat ")
                    .and_then(|metadata| serde_json::from_str::<serde_json::Value>(metadata).ok())
                    .and_then(|metadata| {
                        metadata
                            .get("rule_id")
                            .and_then(serde_json::Value::as_str)
                            .map(|rule_id| rule_id == record.rule_id)
                    })
                    .unwrap_or(false)
            });
            if already_present {
                let durable_sync = legacy_import.is_some();
                if durable_sync {
                    atomic_write(&self.path, &bytes)
                        .map_err(|source| io_error("atomic_write", &self.path, source))?;
                }
                let bytes_after = u64::try_from(bytes.len()).map_err(|_| {
                    store_error(
                        "migrate",
                        self.path.clone(),
                        "capacity",
                        "migrated heuristic file length is not representable",
                    )
                })?;
                return Ok(HeurStoreReceipt {
                    contract: HEUR_STORE_CONTRACT,
                    contract_version: HEUR_STORE_CONTRACT_VERSION,
                    path: self.path.clone(),
                    rule_id: record.rule_id.clone(),
                    disposition: HeurStoreDisposition::AlreadyPresent,
                    bytes_before,
                    bytes_after,
                    durable_sync,
                    legacy_import,
                });
            }

            let metadata = serde_json::to_string(record).map_err(|source| {
                store_error(
                    "encode",
                    self.path.clone(),
                    "invalid_data",
                    source.to_string(),
                )
            })?;
            if !bytes.is_empty() && !bytes.ends_with(b"\n") {
                bytes.push(b'\n');
            }
            let entry = format!("# blackcat {metadata}\n{canonical}\n");
            let next_len = bytes.len().checked_add(entry.len()).ok_or_else(|| {
                store_error(
                    "append",
                    self.path.clone(),
                    "capacity",
                    "heuristic file length overflowed",
                )
            })?;
            let bytes_after = u64::try_from(next_len).map_err(|_| {
                store_error(
                    "append",
                    self.path.clone(),
                    "capacity",
                    "heuristic file length is not representable",
                )
            })?;
            if bytes_after > HEUR_STORE_MAX_BYTES {
                return Err(store_error(
                    "append",
                    self.path.clone(),
                    "capacity",
                    format!(
                        "heuristic file would have {bytes_after} bytes, maximum is {HEUR_STORE_MAX_BYTES}"
                    ),
                ));
            }
            bytes.extend_from_slice(entry.as_bytes());
            atomic_write(&self.path, &bytes)
                .map_err(|source| io_error("atomic_write", &self.path, source))?;
            Ok(HeurStoreReceipt {
                contract: HEUR_STORE_CONTRACT,
                contract_version: HEUR_STORE_CONTRACT_VERSION,
                path: self.path.clone(),
                rule_id: record.rule_id.clone(),
                disposition: HeurStoreDisposition::Appended,
                bytes_before,
                bytes_after,
                durable_sync: true,
                legacy_import,
            })
        }

        pub fn path(&self) -> &Path {
            &self.path
        }

        pub fn startup_legacy_import(&self) -> Option<&HeurStoreLegacyImport> {
            self.startup_legacy_import.as_ref()
        }

        #[cfg(target_arch = "wasm32")]
        fn migrate_legacy_snapshot(&self) -> Result<Option<HeurStoreLegacyImport>, HeurStoreError> {
            Ok(None)
        }

        #[cfg(not(target_arch = "wasm32"))]
        fn migrate_legacy_snapshot(&self) -> Result<Option<HeurStoreLegacyImport>, HeurStoreError> {
            let Some(source_path) = self.legacy_path.as_ref() else {
                return Ok(None);
            };
            match fs::metadata(source_path) {
                Ok(metadata) if metadata.len() == 0 => return Ok(None),
                Ok(_) => {}
                Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
                Err(source) => return Err(io_error("metadata_legacy", source_path, source)),
            }
            if let Some(dir) = self.path.parent() {
                fs::create_dir_all(dir)
                    .map_err(|source| io_error("create_parent", &self.path, source))?;
            }
            let _store_lock = HeurStoreLock::acquire(&self.path)?;
            let current = read_bounded(&self.path)?;
            let (merged, legacy_import) = self.import_legacy(current)?;
            if legacy_import.is_some() {
                atomic_write(&self.path, &merged)
                    .map_err(|source| io_error("atomic_write", &self.path, source))?;
            }
            Ok(legacy_import)
        }

        fn import_legacy(
            &self,
            current: Vec<u8>,
        ) -> Result<(Vec<u8>, Option<HeurStoreLegacyImport>), HeurStoreError> {
            let Some(source_path) = self.legacy_path.as_ref() else {
                return Ok((current, None));
            };
            if source_path == &self.path {
                return Ok((current, None));
            }
            let remaining = HEUR_STORE_MAX_BYTES
                .checked_sub(current.len() as u64)
                .ok_or_else(|| {
                    store_error(
                        "migrate",
                        self.path.clone(),
                        "capacity",
                        "canonical heuristic file exceeds the guarded byte limit",
                    )
                })?;
            let legacy = read_bounded_with_limit(source_path, remaining)?;
            if legacy.is_empty() {
                return Ok((current, None));
            }
            std::str::from_utf8(&legacy).map_err(|source| {
                store_error(
                    "decode_legacy",
                    source_path.clone(),
                    "invalid_data",
                    source.to_string(),
                )
            })?;
            let current_text = std::str::from_utf8(&current).map_err(|source| {
                store_error(
                    "decode",
                    self.path.clone(),
                    "invalid_data",
                    source.to_string(),
                )
            })?;
            let source_bytes = u64::try_from(legacy.len()).map_err(|_| {
                store_error(
                    "migrate",
                    source_path.clone(),
                    "capacity",
                    "legacy heuristic file length is not representable",
                )
            })?;
            let legacy_import = HeurStoreLegacyImport {
                contract: HEUR_STORE_LEGACY_IMPORT_CONTRACT,
                contract_version: HEUR_STORE_LEGACY_IMPORT_CONTRACT_VERSION,
                source_path: source_path.clone(),
                source_bytes,
                source_sha256: sha256_hex(&legacy),
            };
            let metadata = serde_json::to_string(&legacy_import).map_err(|source| {
                store_error(
                    "encode_migration",
                    self.path.clone(),
                    "invalid_data",
                    source.to_string(),
                )
            })?;
            let marker = format!("# blackcat-migration {metadata}");
            if current_text.lines().any(|line| line == marker) {
                return Ok((current, None));
            }

            let separator_bytes = usize::from(!legacy.ends_with(b"\n"));
            let marker_bytes = marker.len().checked_add(1).ok_or_else(|| {
                store_error(
                    "migrate",
                    self.path.clone(),
                    "capacity",
                    "legacy migration marker length overflowed",
                )
            })?;
            let next_len = legacy
                .len()
                .checked_add(separator_bytes)
                .and_then(|length| length.checked_add(marker_bytes))
                .and_then(|length| length.checked_add(current.len()))
                .ok_or_else(|| {
                    store_error(
                        "migrate",
                        self.path.clone(),
                        "capacity",
                        "merged heuristic file length overflowed",
                    )
                })?;
            if next_len as u64 > HEUR_STORE_MAX_BYTES {
                return Err(store_error(
                    "migrate",
                    self.path.clone(),
                    "capacity",
                    format!(
                        "merged heuristic file would have {next_len} bytes, maximum is {HEUR_STORE_MAX_BYTES}"
                    ),
                ));
            }
            let mut merged = Vec::with_capacity(next_len);
            merged.extend_from_slice(&legacy);
            if !legacy.ends_with(b"\n") {
                merged.push(b'\n');
            }
            merged.extend_from_slice(marker.as_bytes());
            merged.push(b'\n');
            merged.extend_from_slice(&current);
            Ok((merged, Some(legacy_import)))
        }
    }

    struct HeurStoreLock {
        file: File,
    }

    impl HeurStoreLock {
        fn acquire(path: &Path) -> Result<Self, HeurStoreError> {
            let file_name = path
                .file_name()
                .map(|name| name.to_string_lossy())
                .unwrap_or_else(|| "heur.kdsl".into());
            let lock_path = path.with_file_name(format!(".{file_name}.lock"));
            let file = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(&lock_path)
                .map_err(|source| io_error("open_lock", &lock_path, source))?;
            file.lock()
                .map_err(|source| io_error("lock", &lock_path, source))?;
            Ok(Self { file })
        }
    }

    impl Drop for HeurStoreLock {
        fn drop(&mut self) {
            let _ = self.file.unlock();
        }
    }

    fn default_paths() -> (PathBuf, Option<PathBuf>) {
        if let Ok(path) = std::env::var("SPIRAL_HEUR_FILE") {
            if !path.trim().is_empty() {
                return (PathBuf::from(path), None);
            }
        }
        let root = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".spiraltorch");
        (root.join("heur.kdsl"), Some(root.join("heur/heur.kdsl")))
    }

    pub fn rule_id(rule_text: &str) -> String {
        sha256_hex(rule_text.as_bytes())
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(&mut encoded, "{byte:02x}");
        }
        encoded
    }

    fn read_bounded(path: &Path) -> Result<Vec<u8>, HeurStoreError> {
        read_bounded_with_limit(path, HEUR_STORE_MAX_BYTES)
    }

    fn read_bounded_with_limit(path: &Path, limit: u64) -> Result<Vec<u8>, HeurStoreError> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => return Err(io_error("open", path, source)),
        };
        let metadata = file
            .metadata()
            .map_err(|source| io_error("metadata", path, source))?;
        if metadata.len() > limit {
            return Err(store_error(
                "read",
                path.to_path_buf(),
                "capacity",
                format!(
                    "existing heuristic file has {} bytes, remaining maximum is {limit}",
                    metadata.len()
                ),
            ));
        }
        let capacity = usize::try_from(metadata.len()).map_err(|_| {
            store_error(
                "read",
                path.to_path_buf(),
                "capacity",
                "existing heuristic file length is not representable",
            )
        })?;
        let mut bytes = Vec::with_capacity(capacity);
        file.take(limit.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|source| io_error("read", path, source))?;
        if bytes.len() as u64 > limit {
            return Err(store_error(
                "read",
                path.to_path_buf(),
                "capacity",
                format!("heuristic file grew beyond {limit} bytes while reading"),
            ));
        }
        Ok(bytes)
    }

    fn io_error(operation: &'static str, path: &Path, source: io::Error) -> HeurStoreError {
        store_error(
            operation,
            path.to_path_buf(),
            format!("{:?}", source.kind()).to_lowercase(),
            source.to_string(),
        )
    }

    fn store_error(
        operation: &'static str,
        path: PathBuf,
        kind: impl Into<String>,
        detail: impl Into<String>,
    ) -> HeurStoreError {
        HeurStoreError {
            operation,
            path,
            kind: kind.into(),
            detail: detail.into(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::{Arc, Barrier};

        fn record(rule: &str) -> HeurStoreRecord {
            HeurStoreRecord {
                contract: "test.contract",
                contract_version: 1,
                rule_id: rule_id(rule),
                interval: crate::heur::evidence::try_wilson_interval(4, 4, 1.96)
                    .expect("test interval"),
                minimum_trials: 4,
                baseline_probability: 0.5,
            }
        }

        #[test]
        fn append_once_writes_a_deterministic_durable_record() {
            let temp = tempfile::tempdir().expect("temporary directory");
            let path = temp.path().join("nested/heur.kdsl");
            let store = HeurStore::try_new(Some(path.to_string_lossy().into_owned()))
                .expect("guarded store");
            let record = record("abc");
            assert_eq!(
                record.rule_id,
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );

            let first = store
                .append_once("  abc  ", &record)
                .expect("first durable append");
            assert_eq!(first.disposition, HeurStoreDisposition::Appended);
            assert_eq!(first.bytes_before, 0);
            assert!(first.bytes_after > 0);
            assert!(first.durable_sync);
            let metadata = serde_json::to_string(&record).expect("deterministic metadata");
            let expected = format!("# blackcat {metadata}\nabc\n").into_bytes();
            assert_eq!(fs::read(&path).expect("stored bytes"), expected);

            let second = store
                .append_once("abc", &record)
                .expect("duplicate receipt");
            assert_eq!(second.disposition, HeurStoreDisposition::AlreadyPresent);
            assert_eq!(second.bytes_before, first.bytes_after);
            assert_eq!(second.bytes_after, first.bytes_after);
            assert!(!second.durable_sync);
            assert_eq!(fs::read(&path).expect("unchanged bytes"), expected);
        }

        #[test]
        fn default_store_imports_legacy_history_once_without_deleting_it() {
            let temp = tempfile::tempdir().expect("temporary directory");
            let path = temp.path().join(".spiraltorch/heur.kdsl");
            let legacy_path = temp.path().join(".spiraltorch/heur/heur.kdsl");
            fs::create_dir_all(legacy_path.parent().expect("legacy parent"))
                .expect("legacy directory");
            let legacy_bytes = b"legacy-rule  # {\"wins\":4,\"trials\":4}\n";
            let canonical_bytes = b"canonical-rule\n";
            fs::write(&legacy_path, legacy_bytes).expect("legacy snapshot");
            fs::write(&path, canonical_bytes).expect("canonical snapshot");
            let store = HeurStore::try_new_paths(path.clone(), Some(legacy_path.clone()))
                .expect("default-path store imports legacy history at startup");
            let import = store
                .startup_legacy_import()
                .expect("typed startup legacy import witness");
            assert_eq!(import.contract, HEUR_STORE_LEGACY_IMPORT_CONTRACT);
            assert_eq!(import.source_path, legacy_path);
            assert_eq!(import.source_bytes, legacy_bytes.len() as u64);
            assert_eq!(import.source_sha256, sha256_hex(legacy_bytes));

            let migrated = fs::read(&path).expect("migrated canonical snapshot");
            let migrated_text = std::str::from_utf8(&migrated).expect("UTF-8 snapshot");
            let legacy_position = migrated_text.find("legacy-rule").expect("legacy rule");
            let marker_position = migrated_text
                .find("# blackcat-migration ")
                .expect("migration marker");
            let canonical_position = migrated_text
                .find("canonical-rule")
                .expect("canonical rule");
            assert!(legacy_position < marker_position);
            assert!(marker_position < canonical_position);
            assert_eq!(
                fs::read(&legacy_path).expect("legacy retained"),
                legacy_bytes
            );

            let receipt = store
                .append_once("legacy-rule", &record("legacy-rule"))
                .expect("legacy inline rule is recognized after migration");
            assert_eq!(receipt.disposition, HeurStoreDisposition::AlreadyPresent);
            assert_eq!(receipt.bytes_before, migrated.len() as u64);
            assert_eq!(receipt.bytes_after, migrated.len() as u64);
            assert!(!receipt.durable_sync);
            assert!(receipt.legacy_import.is_none());
            assert_eq!(fs::read(&path).expect("stable canonical bytes"), migrated);

            let extended_legacy =
                b"legacy-rule  # {\"wins\":4,\"trials\":4}\nlate-rule  # {\"wins\":8,\"trials\":8}\n";
            fs::write(&legacy_path, extended_legacy).expect("late legacy update");
            let late = store
                .append_once("late-rule", &record("late-rule"))
                .expect("a late legacy update is imported under the same lock");
            assert_eq!(late.disposition, HeurStoreDisposition::AlreadyPresent);
            assert!(late.durable_sync);
            assert_eq!(
                late.legacy_import
                    .as_ref()
                    .expect("late import witness")
                    .source_sha256,
                sha256_hex(extended_legacy)
            );
            assert!(fs::read_to_string(&path)
                .expect("late canonical snapshot")
                .contains("late-rule"));
        }

        #[test]
        fn store_rejects_ambiguous_rules_and_mismatched_identity_before_writing() {
            let temp = tempfile::tempdir().expect("temporary directory");
            let path = temp.path().join("heur.kdsl");
            let store = HeurStore::try_new(Some(path.to_string_lossy().into_owned()))
                .expect("guarded store");

            let mismatch = record("different");
            assert_eq!(
                store
                    .append_once("abc", &mismatch)
                    .expect_err("identity mismatch")
                    .operation,
                "validate"
            );
            let newline = record("line one\nline two");
            assert_eq!(
                store
                    .append_once("line one\nline two", &newline)
                    .expect_err("multiline rule")
                    .operation,
                "validate"
            );
            let oversized = "a".repeat(super::super::BLACKCAT_MAX_HEURISTIC_RULE_BYTES + 1);
            let oversized_record = record(&oversized);
            assert_eq!(
                store
                    .append_once(&oversized, &oversized_record)
                    .expect_err("oversized rule")
                    .kind,
                "invalid_input"
            );
            assert!(!path.exists());
            assert!(HeurStore::try_new(Some("  ".to_string())).is_err());
        }

        #[test]
        fn store_rejects_an_oversized_existing_snapshot_before_reading_it() {
            let temp = tempfile::tempdir().expect("temporary directory");
            let path = temp.path().join("heur.kdsl");
            File::create(&path)
                .expect("sparse snapshot")
                .set_len(HEUR_STORE_MAX_BYTES + 1)
                .expect("oversized logical length");
            let store = HeurStore::try_new(Some(path.to_string_lossy().into_owned()))
                .expect("guarded store");
            let error = store
                .append_once("abc", &record("abc"))
                .expect_err("oversized input is rejected before allocation");
            assert_eq!(error.operation, "read");
            assert_eq!(error.kind, "capacity");
            assert_eq!(
                fs::metadata(&path).expect("snapshot metadata").len(),
                HEUR_STORE_MAX_BYTES + 1
            );
        }

        #[test]
        fn store_serializes_concurrent_read_modify_write_transactions() {
            let temp = tempfile::tempdir().expect("temporary directory");
            let path = temp.path().join("heur.kdsl");
            let store = Arc::new(
                HeurStore::try_new(Some(path.to_string_lossy().into_owned()))
                    .expect("guarded store"),
            );
            let barrier = Arc::new(Barrier::new(2));
            let left_record = record("left-rule");
            let right_record = record("right-rule");

            let (left, right) = std::thread::scope(|scope| {
                let left_store = Arc::clone(&store);
                let left_barrier = Arc::clone(&barrier);
                let left = scope.spawn(move || {
                    left_barrier.wait();
                    left_store.append_once("left-rule", &left_record)
                });
                let right_store = Arc::clone(&store);
                let right_barrier = Arc::clone(&barrier);
                let right = scope.spawn(move || {
                    right_barrier.wait();
                    right_store.append_once("right-rule", &right_record)
                });
                (
                    left.join().expect("left writer"),
                    right.join().expect("right writer"),
                )
            });
            assert_eq!(
                left.expect("left append").disposition,
                HeurStoreDisposition::Appended
            );
            assert_eq!(
                right.expect("right append").disposition,
                HeurStoreDisposition::Appended
            );
            let contents = fs::read_to_string(&path).expect("serialized snapshot");
            assert!(contents.lines().any(|line| line == "left-rule"));
            assert!(contents.lines().any(|line| line == "right-rule"));
        }
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
