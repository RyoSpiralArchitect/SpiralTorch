// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Rust-owned adaptive runtime contract.
//!
//! [`Autopilot`] turns a fixed, validated context schema into BlackCat choices,
//! constrains those choices to declared finite domains, and credits each choice
//! at most once. Its device profile is a replayable witness: exact bounded
//! samples back the advertised p50, p95, and mean statistics. Python and WASM
//! clients should expose this contract rather than reconstruct its semantics.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backend::device_caps::DeviceCaps;
use crate::plugin::{global_registry, PluginEvent};
use crate::runtime::blackcat::{BlackCatError, BlackCatRuntime, StepMetrics};
use crate::runtime::persistence::atomic_write;
use crate::telemetry::trace_init;
use spiral_config::determinism;
use tracing::{debug, info, instrument, warn};

/// Stable owner for the persisted Autopilot profile contract.
pub const AUTOPILOT_PROFILE_CONTRACT: &str = "st-core.runtime.autopilot.profile";
/// Current persisted Autopilot profile schema.
pub const AUTOPILOT_PROFILE_SCHEMA_VERSION: u32 = 2;
/// Stable owner for contextual feature ordering.
pub const AUTOPILOT_CONTEXT_CONTRACT: &str = "st-core.runtime.autopilot.context";
/// Current contextual feature schema.
pub const AUTOPILOT_CONTEXT_SCHEMA_VERSION: u32 = 2;
/// Canonical names and order of the fixed contextual features.
pub const AUTOPILOT_BASE_CONTEXT_FEATURES: [&str; 8] = [
    "intercept",
    "batch",
    "tiles",
    "depth",
    "device_lane_width_over_1024",
    "device_max_workgroup_over_4096",
    "device_load",
    "device_subgroup",
];
/// Number of recent observations retained by the exact profile estimators.
pub const AUTOPILOT_PROFILE_WINDOW: usize = 128;
/// Estimator used for the latency and memory profile quantiles.
pub const AUTOPILOT_QUANTILE_ESTIMATOR: &str = "exact_hyndman_fan_type_7";
/// Estimator used for the retry-rate profile statistic.
pub const AUTOPILOT_RETRY_ESTIMATOR: &str = "window_neumaier_arithmetic_mean";

const BASE_CONTEXT_DIM: usize = AUTOPILOT_BASE_CONTEXT_FEATURES.len();

/// Autopilot operating modes controlling how aggressively tuning overrides
/// user supplied hints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoMode {
    /// Fully automatic (default). ENV/CLI are considered hints only.
    Auto,
    /// Prefer persisted choices only across equivalent posterior projections;
    /// otherwise the runtime's TS or UCB decision remains authoritative.
    Hint,
    /// Disabled. No dynamic tuning.
    Off,
}

/// A named, finite control domain owned by the Rust Autopilot contract.
#[derive(Clone, Debug)]
pub struct KnobSpec {
    /// Stable control identifier shared with the BlackCat choice group.
    pub id: String,
    /// Complete set of values the Autopilot may emit for this control.
    pub domain: Vec<String>,
    /// Fallback value used when no BlackCat arm or persisted hint supplies one.
    pub default_idx: usize,
}

/// Configuration consumed by [`Autopilot::new`] and [`Autopilot::try_new`].
#[derive(Clone, Debug)]
pub struct AutoConfig {
    /// Degree of automatic control.
    pub mode: AutoMode,
    /// Optional closed-world control contract. Empty derives it from BlackCat.
    pub knobs: Vec<KnobSpec>,
    /// Context width expected by BlackCat. Values below eight use the canonical prefix.
    pub feat_dim: usize,
    /// Ordered names for dimensions after the fixed eight-feature base schema.
    pub extra_features: Vec<String>,
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            mode: AutoMode::Auto,
            knobs: Vec::new(),
            feat_dim: BASE_CONTEXT_DIM,
            extra_features: Vec::new(),
        }
    }
}

/// A statistically named profile whose fields are derived from the bounded
/// observation window, rather than from similarly named moving averages.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceProfile {
    /// Most recently credited control values.
    pub chosen: HashMap<String, String>,
    /// Type-7 median step latency over the retained observation window.
    pub step_ms_p50: f32,
    /// Type-7 p95 peak memory over the retained observation window.
    pub mem_mb_p95: f32,
    /// Compensated arithmetic mean retry rate over the retained window.
    pub retry_rate: f32,
}

/// A successful, durable Autopilot update.
#[derive(Clone, Debug, PartialEq)]
pub struct AutopilotReport {
    /// Canonical BlackCat free-energy utility credited to this suggestion.
    pub reward: f64,
    /// Profile snapshot durably committed by this report.
    pub profile: DeviceProfile,
    /// Number of observations backing the profile snapshot.
    pub sample_count: usize,
}

/// Explicit failures at the Autopilot semantic boundary.
#[derive(Debug, Error)]
pub enum AutopilotError {
    #[error(transparent)]
    BlackCat(#[from] BlackCatError),
    #[error("invalid Autopilot configuration at '{field}': {detail}")]
    InvalidConfig { field: &'static str, detail: String },
    #[error("Autopilot field '{field}' must be finite, got {value}")]
    NonFinite { field: String, value: f64 },
    #[error("Autopilot field '{field}' must be non-negative, got {value}")]
    Negative { field: String, value: f64 },
    #[error("Autopilot field '{field}' must not exceed {maximum}, got {value}")]
    OutOfRange {
        field: String,
        value: f64,
        maximum: f64,
    },
    #[error("Autopilot workload field '{field}' must be positive")]
    NonPositiveWorkload { field: &'static str },
    #[error("Autopilot extra feature '{feature}' is duplicated")]
    DuplicateFeature { feature: String },
    #[error("Autopilot context is missing configured feature '{feature}'")]
    MissingFeature { feature: String },
    #[error("Autopilot context supplied undeclared feature '{feature}'")]
    UnexpectedFeature { feature: String },
    #[error("Autopilot runtime produced undeclared knob '{id}'")]
    UndeclaredKnob { id: String },
    #[error("Autopilot knob '{id}' does not admit choice '{choice}'")]
    InvalidKnobChoice { id: String, choice: String },
    #[error("Autopilot report requires one successful suggestion")]
    MissingSuggestion,
    #[error("Autopilot already has a suggestion awaiting reward or abandonment")]
    PendingSuggestion,
    #[error("could not read or write Autopilot profile '{}': {source}", path.display())]
    ProfileIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Autopilot reward {reward} committed, but profile durability failed: {source}")]
    ProfileCommit {
        reward: f64,
        #[source]
        source: Box<AutopilotError>,
    },
    #[error("invalid Autopilot profile '{}': {detail}", path.display())]
    ProfileFormat { path: PathBuf, detail: String },
}

#[derive(Clone, Debug)]
struct KnobContract {
    domain: Vec<String>,
    default: String,
}

#[derive(Clone, Debug, Default)]
struct ProfileHistory {
    step_ms: VecDeque<f64>,
    mem_mb: VecDeque<f64>,
    retry_rate: VecDeque<f64>,
}

impl ProfileHistory {
    fn sample_count(&self) -> usize {
        self.step_ms.len()
    }

    fn push(&mut self, metrics: &StepMetrics) {
        push_bounded(&mut self.step_ms, metrics.step_time_ms);
        push_bounded(&mut self.mem_mb, metrics.mem_peak_mb);
        push_bounded(&mut self.retry_rate, metrics.retry_rate);
    }

    fn profile(&self, chosen: HashMap<String, String>) -> DeviceProfile {
        DeviceProfile {
            chosen,
            step_ms_p50: quantile(&self.step_ms, 0.50) as f32,
            mem_mb_p95: quantile(&self.mem_mb, 0.95) as f32,
            retry_rate: arithmetic_mean(&self.retry_rate) as f32,
        }
    }

    fn validate(&self, path: &Path) -> Result<(), AutopilotError> {
        let lengths = [self.step_ms.len(), self.mem_mb.len(), self.retry_rate.len()];
        if lengths
            .iter()
            .any(|length| *length > AUTOPILOT_PROFILE_WINDOW)
        {
            return Err(profile_format(
                path,
                format!("sample window exceeds capacity {AUTOPILOT_PROFILE_WINDOW}: {lengths:?}"),
            ));
        }
        if lengths[1..].iter().any(|length| *length != lengths[0]) {
            return Err(profile_format(
                path,
                format!("metric sample counts disagree: {lengths:?}"),
            ));
        }
        for (field, values) in [
            ("samples.step_ms", &self.step_ms),
            ("samples.mem_mb", &self.mem_mb),
            ("samples.retry_rate", &self.retry_rate),
        ] {
            for value in values {
                validate_profile_value(path, field, *value)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedProfile {
    contract: String,
    schema_version: u32,
    sample_capacity: usize,
    estimators: PersistedEstimators,
    chosen: BTreeMap<String, String>,
    samples: PersistedSamples,
    summary: PersistedSummary,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedEstimators {
    step_ms_p50: String,
    mem_mb_p95: String,
    retry_rate: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedSamples {
    step_ms: Vec<f64>,
    mem_mb: Vec<f64>,
    retry_rate: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedSummary {
    step_ms_p50: f32,
    mem_mb_p95: f32,
    retry_rate: f32,
}

struct LoadedProfile {
    profile: DeviceProfile,
    history: ProfileHistory,
}

struct PreparedAutopilot {
    profile_key: String,
    profile_dir: PathBuf,
    profile: DeviceProfile,
    history: ProfileHistory,
    knobs: BTreeMap<String, KnobContract>,
    mode: AutoMode,
    feat_dim: usize,
    extra_features: Vec<String>,
}

/// Front-end around [`BlackCatRuntime`] that owns contextual control,
/// statistically sound device profiles, and durable per-device hints.
pub struct Autopilot {
    caps: DeviceCaps,
    runtime: BlackCatRuntime,
    profile_key: String,
    profile_dir: PathBuf,
    profile: DeviceProfile,
    history: ProfileHistory,
    knobs: BTreeMap<String, KnobContract>,
    picks: HashMap<String, String>,
    mode: AutoMode,
    feat_dim: usize,
    extra_features: Vec<String>,
    suggestion_ready: bool,
}

impl Autopilot {
    /// Compatibility constructor. Invalid configuration contracts fail loudly;
    /// corrupt persisted hints are rejected and replaced by an empty profile.
    #[instrument(skip(runtime, cfg), fields(device = ?caps.backend, feat_dim = cfg.feat_dim))]
    pub fn new(caps: DeviceCaps, cfg: AutoConfig, runtime: BlackCatRuntime) -> Self {
        Self::new_with_profile_dir(caps, cfg, runtime, default_profile_dir())
    }

    /// Compatibility constructor with an explicit profile directory.
    pub fn new_with_profile_dir(
        caps: DeviceCaps,
        cfg: AutoConfig,
        runtime: BlackCatRuntime,
        profile_dir: impl Into<PathBuf>,
    ) -> Self {
        trace_init::init_tracing();
        let profile_dir = profile_dir.into();
        let prepared = match Self::prepare(&caps, &cfg, &runtime, profile_dir.clone(), true) {
            Ok(prepared) => prepared,
            Err(error @ AutopilotError::ProfileIo { .. })
            | Err(error @ AutopilotError::ProfileFormat { .. }) => {
                warn!(error = %error, "autopilot profile rejected; starting without persisted hints");
                publish_rejection("profile_load", &error);
                Self::prepare(&caps, &cfg, &runtime, profile_dir, false).unwrap_or_else(|error| {
                    panic!("could not initialise Autopilot after rejecting profile: {error}")
                })
            }
            Err(error) => panic!("could not initialise Autopilot: {error}"),
        };
        Self::from_prepared(caps, runtime, prepared)
    }

    /// Guarded constructor using the standard per-user profile directory.
    pub fn try_new(
        caps: DeviceCaps,
        cfg: AutoConfig,
        runtime: BlackCatRuntime,
    ) -> Result<Self, AutopilotError> {
        let profile_dir = default_profile_dir();
        Self::try_new_with_profile_dir(caps, cfg, runtime, profile_dir)
    }

    /// Guarded constructor with an explicit profile directory. This is useful
    /// for isolated workers and makes profile ownership testable.
    pub fn try_new_with_profile_dir(
        caps: DeviceCaps,
        cfg: AutoConfig,
        runtime: BlackCatRuntime,
        profile_dir: impl Into<PathBuf>,
    ) -> Result<Self, AutopilotError> {
        trace_init::init_tracing();
        let prepared = Self::prepare(&caps, &cfg, &runtime, profile_dir.into(), true)?;
        Ok(Self::from_prepared(caps, runtime, prepared))
    }

    fn prepare(
        caps: &DeviceCaps,
        cfg: &AutoConfig,
        runtime: &BlackCatRuntime,
        profile_dir: PathBuf,
        load_persisted_profile: bool,
    ) -> Result<PreparedAutopilot, AutopilotError> {
        caps.validate()
            .map_err(|error| AutopilotError::InvalidConfig {
                field: "device_caps",
                detail: error.to_string(),
            })?;
        if let Some(error) = runtime.configuration_error() {
            return Err(AutopilotError::InvalidConfig {
                field: "blackcat",
                detail: error.to_string(),
            });
        }
        if cfg.feat_dim == 0 {
            return Err(AutopilotError::InvalidConfig {
                field: "feat_dim",
                detail: "must be positive".to_string(),
            });
        }
        if cfg.feat_dim != runtime.context_dim() {
            return Err(AutopilotError::InvalidConfig {
                field: "feat_dim",
                detail: format!(
                    "configured {}, but BlackCat expects {}",
                    cfg.feat_dim,
                    runtime.context_dim()
                ),
            });
        }
        let extra_features = validate_extra_feature_schema(&cfg.extra_features, cfg.feat_dim)?;
        let mode = if determinism::lock_scheduler() && cfg.mode == AutoMode::Auto {
            AutoMode::Hint
        } else {
            cfg.mode
        };
        let mut knobs = validate_knob_specs(&cfg.knobs)?;
        if knobs.is_empty() {
            knobs = derive_runtime_knob_contract(runtime)?;
        } else {
            validate_runtime_knob_contract(runtime, &knobs)?;
        }
        let key = make_key(caps);
        let loaded = if mode == AutoMode::Off || !load_persisted_profile {
            LoadedProfile {
                profile: DeviceProfile::default(),
                history: ProfileHistory::default(),
            }
        } else {
            load_profile(&profile_dir, &key)?
        };
        let mut profile = loaded.profile;
        let history = loaded.history;
        reconcile_profile_choices(&mut profile, &knobs);
        profile = history.profile(profile.chosen);
        Ok(PreparedAutopilot {
            profile_key: key,
            profile_dir,
            profile,
            history,
            knobs,
            mode,
            feat_dim: cfg.feat_dim,
            extra_features,
        })
    }

    fn from_prepared(
        caps: DeviceCaps,
        runtime: BlackCatRuntime,
        prepared: PreparedAutopilot,
    ) -> Self {
        let autopilot = Self {
            caps,
            runtime,
            profile_key: prepared.profile_key,
            profile_dir: prepared.profile_dir,
            profile: prepared.profile,
            history: prepared.history,
            knobs: prepared.knobs,
            picks: HashMap::new(),
            mode: prepared.mode,
            feat_dim: prepared.feat_dim,
            extra_features: prepared.extra_features,
            suggestion_ready: false,
        };
        info!(
            profile_key = %autopilot.profile_key,
            mode = ?autopilot.mode,
            profile_samples = autopilot.history.sample_count(),
            profile_contract = AUTOPILOT_PROFILE_CONTRACT,
            profile_schema_version = AUTOPILOT_PROFILE_SCHEMA_VERSION,
            "autopilot initialised"
        );
        autopilot
    }

    /// Guarded contextual feature builder. Short dimensions select a canonical
    /// base-feature prefix; named extras follow the configured fixed schema.
    #[allow(clippy::too_many_arguments)]
    pub fn try_build_context(
        &self,
        batch: u32,
        tiles: u32,
        depth: u32,
        device_load: f64,
        extras: &[(String, f64)],
    ) -> Result<Vec<f64>, AutopilotError> {
        let result = self.build_context_checked(batch, tiles, depth, device_load, extras);
        if let Err(error) = &result {
            publish_rejection("context", error);
        }
        result
    }

    fn build_context_checked(
        &self,
        batch: u32,
        tiles: u32,
        depth: u32,
        device_load: f64,
        extras: &[(String, f64)],
    ) -> Result<Vec<f64>, AutopilotError> {
        for (field, value) in [("batch", batch), ("tiles", tiles), ("depth", depth)] {
            if value == 0 {
                return Err(AutopilotError::NonPositiveWorkload { field });
            }
        }
        validate_non_negative("device_load", device_load)?;
        let mut provided = BTreeMap::new();
        for (name, value) in extras {
            validate_extra_feature_name(name, "extras.name")?;
            if provided.insert(name.as_str(), *value).is_some() {
                return Err(AutopilotError::DuplicateFeature {
                    feature: name.clone(),
                });
            }
            validate_finite(format!("extras.{name}"), *value)?;
        }
        if let Some(name) = provided.keys().find(|name| {
            !self
                .extra_features
                .iter()
                .any(|expected| expected == **name)
        }) {
            return Err(AutopilotError::UnexpectedFeature {
                feature: (*name).to_string(),
            });
        }
        if let Some(name) = self
            .extra_features
            .iter()
            .find(|name| !provided.contains_key(name.as_str()))
        {
            return Err(AutopilotError::MissingFeature {
                feature: name.clone(),
            });
        }

        let mut context = vec![
            1.0,
            batch as f64,
            tiles as f64,
            depth as f64,
            (self.caps.lane_width as f64) / 1024.0,
            (self.caps.max_workgroup as f64) / 4096.0,
            device_load,
            if self.caps.subgroup { 1.0 } else { 0.0 },
        ];
        context.truncate(self.feat_dim.min(BASE_CONTEXT_DIM));
        context.extend(
            self.extra_features
                .iter()
                .map(|name| provided[name.as_str()]),
        );
        debug_assert_eq!(context.len(), self.feat_dim);
        Ok(context)
    }

    /// Compatibility context builder. Invalid input produces an unusable empty
    /// context, which the guarded suggestion path rejects without controls.
    #[instrument(skip(self, extras), fields(batch = batch, tiles = tiles, depth = depth))]
    pub fn build_context(
        &self,
        batch: u32,
        tiles: u32,
        depth: u32,
        device_load: f64,
        extras: &[(String, f64)],
    ) -> Vec<f64> {
        self.try_build_context(batch, tiles, depth, device_load, extras)
            .unwrap_or_default()
    }

    /// Guarded suggestion path. BlackCat choices are resolved against the
    /// declared finite knob contract before they become executable controls.
    pub fn try_suggest(
        &mut self,
        context: Vec<f64>,
    ) -> Result<&HashMap<String, String>, AutopilotError> {
        if self.mode == AutoMode::Off {
            self.suggestion_ready = false;
            self.picks.clear();
            return Ok(&self.picks);
        }
        if self.suggestion_ready {
            let error = AutopilotError::PendingSuggestion;
            publish_rejection("suggest", &error);
            return Err(error);
        }
        let context_snapshot = global_registry()
            .event_bus()
            .has_listeners("AutopilotSuggest")
            .then(|| context.clone());
        let runtime_domains = self.runtime.choice_domains();
        let hints = if self.mode == AutoMode::Hint {
            self.profile
                .chosen
                .iter()
                .filter(|(id, choice)| {
                    runtime_domains
                        .get(*id)
                        .is_some_and(|domain| domain.contains(*choice))
                })
                .map(|(id, choice)| (id.clone(), choice.clone()))
                .collect::<BTreeMap<_, _>>()
        } else {
            BTreeMap::new()
        };
        let runtime_picks = self
            .runtime
            .try_choose_with_hints(context, &hints)
            .map_err(AutopilotError::from)
            .inspect_err(|error| publish_rejection("suggest", error))?;
        let picks = match self.resolve_picks(runtime_picks) {
            Ok(picks) => picks,
            Err(error) => {
                let _ = self.runtime.abandon_pending_selection();
                publish_rejection("suggest", &error);
                return Err(error);
            }
        };
        self.picks = picks;
        self.suggestion_ready = true;
        debug!(picks = ?self.picks, "autopilot suggestions updated");
        let bus = global_registry().event_bus();
        if let Some(context_snapshot) = context_snapshot {
            bus.publish(&PluginEvent::custom(
                "AutopilotSuggest",
                serde_json::json!({
                    "profile_key": self.profile_key.as_str(),
                    "mode": format!("{:?}", self.mode),
                    "context": context_snapshot,
                    "context_contract": AUTOPILOT_CONTEXT_CONTRACT,
                    "context_schema_version": AUTOPILOT_CONTEXT_SCHEMA_VERSION,
                    "base_context_features": AUTOPILOT_BASE_CONTEXT_FEATURES,
                    "context_features": self.context_features(),
                    "hints": hints,
                    "picks": ordered_map(&self.picks),
                    "bandit_selection": self.runtime.last_selection_witness(),
                    "profile_contract": AUTOPILOT_PROFILE_CONTRACT,
                    "profile_schema_version": AUTOPILOT_PROFILE_SCHEMA_VERSION,
                }),
            ));
        }
        Ok(&self.picks)
    }

    /// Compatibility suggestion path. Rejections clear all controls and cannot
    /// subsequently receive reward credit.
    #[instrument(skip(self, context))]
    pub fn suggest(&mut self, context: Vec<f64>) -> &HashMap<String, String> {
        if let Err(error) = self.try_suggest(context) {
            warn!(error = %error, "autopilot suggestion rejected");
            if !self.suggestion_ready {
                self.picks.clear();
            }
        }
        &self.picks
    }

    /// Abandons a suggestion that was not executed, preserving posterior
    /// integrity while allowing the next step to request another decision.
    pub fn abandon_suggestion(&mut self) -> Option<u64> {
        if !self.suggestion_ready {
            return None;
        }
        let selection_id = self.runtime.abandon_pending_selection()?;
        self.suggestion_ready = false;
        self.picks.clear();
        Some(selection_id)
    }

    /// Guarded report path. One successful suggestion receives at most one
    /// reward, and the profile is committed with an atomic replacement.
    #[instrument(skip(self, metrics), fields(step_time = metrics.step_time_ms, mem_peak = metrics.mem_peak_mb, retry_rate = metrics.retry_rate))]
    pub fn try_report(
        &mut self,
        metrics: &StepMetrics,
    ) -> Result<Option<AutopilotReport>, AutopilotError> {
        if self.mode == AutoMode::Off {
            return Ok(None);
        }
        if !self.suggestion_ready {
            let error = AutopilotError::MissingSuggestion;
            publish_rejection("report", &error);
            return Err(error);
        }
        validate_metrics(metrics).inspect_err(|error| publish_rejection("report", error))?;

        let mut next_history = self.history.clone();
        next_history.push(metrics);
        let next_profile = next_history.profile(self.picks.clone());
        let reward = self
            .runtime
            .try_post_step(metrics)
            .map_err(AutopilotError::from)
            .inspect_err(|error| publish_rejection("report", error))?;
        self.suggestion_ready = false;
        if let Err(source) = save_profile(
            &self.profile_dir,
            &self.profile_key,
            &next_profile,
            &next_history,
        ) {
            let error = AutopilotError::ProfileCommit {
                reward,
                source: Box::new(source),
            };
            publish_rejection("profile_commit", &error);
            return Err(error);
        }
        self.history = next_history;
        self.profile = next_profile;
        let report = AutopilotReport {
            reward,
            profile: self.profile.clone(),
            sample_count: self.history.sample_count(),
        };
        debug!(profile_key = %self.profile_key, sample_count = report.sample_count, "autopilot metrics reported");
        let bus = global_registry().event_bus();
        if bus.has_listeners("AutopilotReport") {
            bus.publish(&PluginEvent::custom(
                "AutopilotReport",
                serde_json::json!({
                    "profile_key": self.profile_key.as_str(),
                    "mode": format!("{:?}", self.mode),
                    "reward": report.reward,
                    "picks": ordered_map(&self.picks),
                    "metrics": {
                        "step_time_ms": metrics.step_time_ms,
                        "mem_peak_mb": metrics.mem_peak_mb,
                        "retry_rate": metrics.retry_rate,
                        "extra": &metrics.extra,
                    },
                    "profile": {
                        "step_ms_p50": report.profile.step_ms_p50,
                        "mem_mb_p95": report.profile.mem_mb_p95,
                        "retry_rate": report.profile.retry_rate,
                        "sample_count": report.sample_count,
                        "sample_capacity": AUTOPILOT_PROFILE_WINDOW,
                        "estimators": {
                            "step_ms_p50": AUTOPILOT_QUANTILE_ESTIMATOR,
                            "mem_mb_p95": AUTOPILOT_QUANTILE_ESTIMATOR,
                            "retry_rate": AUTOPILOT_RETRY_ESTIMATOR,
                        },
                    },
                    "profile_contract": AUTOPILOT_PROFILE_CONTRACT,
                    "profile_schema_version": AUTOPILOT_PROFILE_SCHEMA_VERSION,
                }),
            ));
        }
        Ok(Some(report))
    }

    /// Compatibility report path. Invalid metrics or persistence failures are
    /// visible in telemetry and never mutate the in-memory profile.
    pub fn report(&mut self, metrics: &StepMetrics) {
        if let Err(error) = self.try_report(metrics) {
            warn!(error = %error, "autopilot report rejected");
        }
    }

    fn resolve_picks(
        &self,
        mut runtime_picks: HashMap<String, String>,
    ) -> Result<HashMap<String, String>, AutopilotError> {
        if self.knobs.is_empty() {
            return Ok(runtime_picks);
        }
        if let Some(id) = runtime_picks
            .keys()
            .find(|id| !self.knobs.contains_key(*id))
        {
            return Err(AutopilotError::UndeclaredKnob { id: id.clone() });
        }
        let mut resolved = HashMap::with_capacity(self.knobs.len());
        for (id, contract) in &self.knobs {
            let choice = runtime_picks
                .remove(id)
                .or_else(|| {
                    (self.mode == AutoMode::Hint)
                        .then(|| self.profile.chosen.get(id).cloned())
                        .flatten()
                })
                .unwrap_or_else(|| contract.default.clone());
            if !contract.domain.contains(&choice) {
                return Err(AutopilotError::InvalidKnobChoice {
                    id: id.clone(),
                    choice,
                });
            }
            resolved.insert(id.clone(), choice);
        }
        Ok(resolved)
    }

    /// Read-only access to BlackCat diagnostics. Mutable access would allow a
    /// caller to replace the arm selected by Autopilot before reward credit.
    pub fn runtime(&self) -> &BlackCatRuntime {
        &self.runtime
    }

    /// Returns the currently configured operating mode.
    pub fn mode(&self) -> AutoMode {
        self.mode
    }

    /// Returns the current profile snapshot.
    pub fn profile(&self) -> &DeviceProfile {
        &self.profile
    }

    /// Returns the number of observations backing profile statistics.
    pub fn profile_sample_count(&self) -> usize {
        self.history.sample_count()
    }

    /// Returns the profile file owned by this controller.
    pub fn profile_path(&self) -> PathBuf {
        profile_path(&self.profile_dir, &self.profile_key)
    }

    /// Returns the complete ordered feature schema consumed by BlackCat.
    pub fn context_features(&self) -> Vec<&str> {
        let mut features =
            AUTOPILOT_BASE_CONTEXT_FEATURES[..self.feat_dim.min(BASE_CONTEXT_DIM)].to_vec();
        features.extend(self.extra_features.iter().map(String::as_str));
        features
    }
}

fn validate_extra_feature_schema(
    features: &[String],
    feat_dim: usize,
) -> Result<Vec<String>, AutopilotError> {
    let expected = feat_dim.saturating_sub(BASE_CONTEXT_DIM);
    if features.len() != expected {
        return Err(AutopilotError::InvalidConfig {
            field: "extra_features",
            detail: format!(
                "context dimension {feat_dim} requires {expected} names, got {}",
                features.len()
            ),
        });
    }
    let mut seen = HashSet::with_capacity(features.len());
    for name in features {
        validate_extra_feature_name(name, "extra_features")?;
        if !seen.insert(name.as_str()) {
            return Err(AutopilotError::DuplicateFeature {
                feature: name.clone(),
            });
        }
    }
    Ok(features.to_vec())
}

fn validate_extra_feature_name(name: &str, field: &'static str) -> Result<(), AutopilotError> {
    if name.trim().is_empty()
        || name != name.trim()
        || name.chars().any(char::is_control)
        || AUTOPILOT_BASE_CONTEXT_FEATURES.contains(&name)
    {
        return Err(AutopilotError::InvalidConfig {
            field,
            detail: format!("invalid or reserved feature name {name:?}"),
        });
    }
    Ok(())
}

fn validate_knob_specs(
    specs: &[KnobSpec],
) -> Result<BTreeMap<String, KnobContract>, AutopilotError> {
    let mut knobs = BTreeMap::new();
    for spec in specs {
        let id = spec.id.trim();
        if id.is_empty() || id != spec.id.as_str() || id.chars().any(char::is_control) {
            return Err(AutopilotError::InvalidConfig {
                field: "knobs.id",
                detail: format!("invalid knob identifier {:?}", spec.id),
            });
        }
        if knobs.contains_key(id) {
            return Err(AutopilotError::InvalidConfig {
                field: "knobs.id",
                detail: format!("duplicate knob '{id}'"),
            });
        }
        if spec.domain.is_empty() {
            return Err(AutopilotError::InvalidConfig {
                field: "knobs.domain",
                detail: format!("knob '{id}' has an empty domain"),
            });
        }
        if spec.default_idx >= spec.domain.len() {
            return Err(AutopilotError::InvalidConfig {
                field: "knobs.default_idx",
                detail: format!(
                    "knob '{id}' default {} is outside domain length {}",
                    spec.default_idx,
                    spec.domain.len()
                ),
            });
        }
        let mut choices = HashSet::with_capacity(spec.domain.len());
        for choice in &spec.domain {
            if choice.is_empty() || choice.chars().any(char::is_control) {
                return Err(AutopilotError::InvalidConfig {
                    field: "knobs.domain",
                    detail: format!("knob '{id}' contains an invalid choice"),
                });
            }
            if !choices.insert(choice.as_str()) {
                return Err(AutopilotError::InvalidConfig {
                    field: "knobs.domain",
                    detail: format!("knob '{id}' repeats choice '{choice}'"),
                });
            }
        }
        let default = spec.domain[spec.default_idx].clone();
        knobs.insert(
            id.to_string(),
            KnobContract {
                domain: spec.domain.clone(),
                default,
            },
        );
    }
    Ok(knobs)
}

fn derive_runtime_knob_contract(
    runtime: &BlackCatRuntime,
) -> Result<BTreeMap<String, KnobContract>, AutopilotError> {
    let mut knobs = BTreeMap::new();
    for (id, domain) in runtime.choice_domains() {
        if id.trim().is_empty() || id != id.trim() || id.chars().any(char::is_control) {
            return Err(AutopilotError::InvalidConfig {
                field: "blackcat.choice_groups.id",
                detail: format!("invalid runtime knob identifier {id:?}"),
            });
        }
        let mut seen = HashSet::with_capacity(domain.len());
        for choice in &domain {
            if choice.is_empty() || choice.chars().any(char::is_control) {
                return Err(AutopilotError::InvalidConfig {
                    field: "blackcat.choice_groups.domain",
                    detail: format!("runtime knob '{id}' contains an invalid choice"),
                });
            }
            if !seen.insert(choice.as_str()) {
                return Err(AutopilotError::InvalidConfig {
                    field: "blackcat.choice_groups.domain",
                    detail: format!("runtime knob '{id}' repeats choice '{choice}'"),
                });
            }
        }
        let Some(default) = domain.first().cloned() else {
            return Err(AutopilotError::InvalidConfig {
                field: "blackcat.choice_groups.domain",
                detail: format!("runtime knob '{id}' has an empty domain"),
            });
        };
        knobs.insert(id, KnobContract { domain, default });
    }
    Ok(knobs)
}

fn validate_runtime_knob_contract(
    runtime: &BlackCatRuntime,
    knobs: &BTreeMap<String, KnobContract>,
) -> Result<(), AutopilotError> {
    if knobs.is_empty() {
        return Ok(());
    }
    for (id, runtime_domain) in runtime.choice_domains() {
        let Some(contract) = knobs.get(&id) else {
            return Err(AutopilotError::InvalidConfig {
                field: "knobs",
                detail: format!("BlackCat exposes undeclared knob '{id}'"),
            });
        };
        let mut seen = HashSet::with_capacity(runtime_domain.len());
        for choice in runtime_domain {
            if !seen.insert(choice.clone()) {
                return Err(AutopilotError::InvalidConfig {
                    field: "knobs.domain",
                    detail: format!("BlackCat knob '{id}' repeats choice '{choice}'"),
                });
            }
            if !contract.domain.contains(&choice) {
                return Err(AutopilotError::InvalidConfig {
                    field: "knobs.domain",
                    detail: format!(
                        "BlackCat knob '{id}' exposes choice '{choice}' outside its contract"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn reconcile_profile_choices(profile: &mut DeviceProfile, knobs: &BTreeMap<String, KnobContract>) {
    if knobs.is_empty() {
        profile.chosen.clear();
        return;
    }
    profile.chosen.retain(|id, choice| {
        knobs
            .get(id)
            .is_some_and(|contract| contract.domain.contains(choice))
    });
}

fn validate_metrics(metrics: &StepMetrics) -> Result<(), AutopilotError> {
    for (field, value) in [
        ("metrics.step_time_ms", metrics.step_time_ms),
        ("metrics.mem_peak_mb", metrics.mem_peak_mb),
        ("metrics.retry_rate", metrics.retry_rate),
    ] {
        validate_non_negative(field, value)?;
        if value > f32::MAX as f64 {
            return Err(AutopilotError::OutOfRange {
                field: field.to_string(),
                value,
                maximum: f32::MAX as f64,
            });
        }
    }
    for (name, value) in &metrics.extra {
        validate_finite(format!("metrics.extra.{name}"), *value)?;
    }
    Ok(())
}

fn validate_finite(field: impl Into<String>, value: f64) -> Result<f64, AutopilotError> {
    let field = field.into();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AutopilotError::NonFinite { field, value })
    }
}

fn validate_non_negative(field: impl Into<String>, value: f64) -> Result<f64, AutopilotError> {
    let field = field.into();
    validate_finite(field.clone(), value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(AutopilotError::Negative { field, value })
    }
}

fn push_bounded(window: &mut VecDeque<f64>, value: f64) {
    if window.len() == AUTOPILOT_PROFILE_WINDOW {
        window.pop_front();
    }
    window.push_back(value);
}

fn quantile(window: &VecDeque<f64>, probability: f64) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    let mut sorted = window.iter().copied().collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }
}

fn arithmetic_mean(window: &VecDeque<f64>) -> f64 {
    if window.is_empty() {
        0.0
    } else {
        let mut sum = 0.0;
        let mut correction = 0.0;
        for value in window {
            let next = sum + value;
            if sum.abs() >= value.abs() {
                correction += (sum - next) + value;
            } else {
                correction += (value - next) + sum;
            }
            sum = next;
        }
        (sum + correction) / window.len() as f64
    }
}

fn make_key(caps: &DeviceCaps) -> String {
    let backend = caps.backend.as_str();
    format!(
        "backend={backend};lane={};wg={};subgroup={}",
        caps.lane_width, caps.max_workgroup, caps.subgroup as u8
    )
}

fn default_profile_dir() -> PathBuf {
    let mut dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.push(".spiraltorch");
    dir.push("profile.d");
    dir
}

fn sanitize(component: &str) -> String {
    component
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn profile_path(profile_dir: &Path, key: &str) -> PathBuf {
    let mut path = profile_dir.join(sanitize(key));
    path.set_extension("prof");
    path
}

fn load_profile(profile_dir: &Path, key: &str) -> Result<LoadedProfile, AutopilotError> {
    let path = profile_path(profile_dir, key);
    let mut file = match File::open(&path) {
        Ok(file) => file,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Ok(LoadedProfile {
                profile: DeviceProfile::default(),
                history: ProfileHistory::default(),
            });
        }
        Err(source) => return Err(profile_io(path, source)),
    };
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|source| profile_io(path.clone(), source))?;
    if contents.trim_start().starts_with('{') {
        parse_json_profile(&path, &contents)
    } else {
        parse_legacy_profile(&path, &contents)
    }
}

fn parse_json_profile(path: &Path, contents: &str) -> Result<LoadedProfile, AutopilotError> {
    let persisted: PersistedProfile = serde_json::from_str(contents)
        .map_err(|error| profile_format(path, format!("JSON decode failed: {error}")))?;
    if persisted.contract != AUTOPILOT_PROFILE_CONTRACT {
        return Err(profile_format(
            path,
            format!("unknown contract {:?}", persisted.contract),
        ));
    }
    if persisted.schema_version != AUTOPILOT_PROFILE_SCHEMA_VERSION {
        return Err(profile_format(
            path,
            format!("unsupported schema version {}", persisted.schema_version),
        ));
    }
    if persisted.sample_capacity != AUTOPILOT_PROFILE_WINDOW {
        return Err(profile_format(
            path,
            format!(
                "sample capacity {} does not match {AUTOPILOT_PROFILE_WINDOW}",
                persisted.sample_capacity
            ),
        ));
    }
    for (field, actual, expected) in [
        (
            "estimators.step_ms_p50",
            persisted.estimators.step_ms_p50.as_str(),
            AUTOPILOT_QUANTILE_ESTIMATOR,
        ),
        (
            "estimators.mem_mb_p95",
            persisted.estimators.mem_mb_p95.as_str(),
            AUTOPILOT_QUANTILE_ESTIMATOR,
        ),
        (
            "estimators.retry_rate",
            persisted.estimators.retry_rate.as_str(),
            AUTOPILOT_RETRY_ESTIMATOR,
        ),
    ] {
        if actual != expected {
            return Err(profile_format(
                path,
                format!("{field} is {actual:?}, expected {expected:?}"),
            ));
        }
    }
    validate_persisted_choices(path, persisted.chosen.iter())?;
    let history = ProfileHistory {
        step_ms: persisted.samples.step_ms.into(),
        mem_mb: persisted.samples.mem_mb.into(),
        retry_rate: persisted.samples.retry_rate.into(),
    };
    history.validate(path)?;
    let profile = history.profile(persisted.chosen.into_iter().collect());
    for (field, expected, actual) in [
        (
            "summary.step_ms_p50",
            persisted.summary.step_ms_p50,
            profile.step_ms_p50,
        ),
        (
            "summary.mem_mb_p95",
            persisted.summary.mem_mb_p95,
            profile.mem_mb_p95,
        ),
        (
            "summary.retry_rate",
            persisted.summary.retry_rate,
            profile.retry_rate,
        ),
    ] {
        if !summary_matches(expected, actual) {
            return Err(profile_format(
                path,
                format!("{field} is {expected}, recomputed value is {actual}"),
            ));
        }
    }
    Ok(LoadedProfile { profile, history })
}

fn parse_legacy_profile(path: &Path, contents: &str) -> Result<LoadedProfile, AutopilotError> {
    let mut chosen = HashMap::new();
    let mut step_ms = None;
    let mut mem_mb = None;
    let mut retry_rate = None;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(profile_format(
                path,
                format!("malformed legacy line {line:?}"),
            ));
        };
        match key {
            "step_ms_p50" => {
                set_legacy_metric(
                    path,
                    key,
                    &mut step_ms,
                    parse_legacy_number(path, key, value)?,
                )?;
            }
            "mem_mb_p95" => {
                set_legacy_metric(
                    path,
                    key,
                    &mut mem_mb,
                    parse_legacy_number(path, key, value)?,
                )?;
            }
            "retry_rate" => {
                set_legacy_metric(
                    path,
                    key,
                    &mut retry_rate,
                    parse_legacy_number(path, key, value)?,
                )?;
            }
            _ => {
                if let Some(id) = key.strip_prefix("chosen.") {
                    validate_persisted_choice(path, id, value)?;
                    if chosen.insert(id.to_string(), value.to_string()).is_some() {
                        return Err(profile_format(
                            path,
                            format!("duplicate legacy knob '{id}'"),
                        ));
                    }
                } else {
                    return Err(profile_format(
                        path,
                        format!("unknown legacy field {key:?}"),
                    ));
                }
            }
        }
    }
    let metric_count = [step_ms, mem_mb, retry_rate]
        .iter()
        .filter(|value| value.is_some())
        .count();
    if metric_count != 0 && metric_count != 3 {
        return Err(profile_format(
            path,
            format!("legacy metric triple is incomplete ({metric_count}/3 fields)"),
        ));
    }
    // Aggregate legacy p50/p95 values are not raw observations. Preserve only
    // compatible control hints rather than promoting summaries to fake samples.
    let history = ProfileHistory::default();
    history.validate(path)?;
    let profile = history.profile(chosen);
    Ok(LoadedProfile { profile, history })
}

fn parse_legacy_number(path: &Path, field: &str, value: &str) -> Result<f64, AutopilotError> {
    let value = value
        .parse::<f64>()
        .map_err(|error| profile_format(path, format!("{field} is invalid: {error}")))?;
    validate_profile_value(path, field, value)?;
    Ok(value)
}

fn set_legacy_metric(
    path: &Path,
    field: &str,
    slot: &mut Option<f64>,
    value: f64,
) -> Result<(), AutopilotError> {
    if slot.replace(value).is_some() {
        return Err(profile_format(
            path,
            format!("duplicate legacy field {field:?}"),
        ));
    }
    Ok(())
}

fn save_profile(
    profile_dir: &Path,
    key: &str,
    profile: &DeviceProfile,
    history: &ProfileHistory,
) -> Result<(), AutopilotError> {
    let path = profile_path(profile_dir, key);
    history.validate(&path)?;
    fs::create_dir_all(profile_dir)
        .map_err(|source| profile_io(profile_dir.to_path_buf(), source))?;
    let persisted = PersistedProfile {
        contract: AUTOPILOT_PROFILE_CONTRACT.to_string(),
        schema_version: AUTOPILOT_PROFILE_SCHEMA_VERSION,
        sample_capacity: AUTOPILOT_PROFILE_WINDOW,
        estimators: PersistedEstimators {
            step_ms_p50: AUTOPILOT_QUANTILE_ESTIMATOR.to_string(),
            mem_mb_p95: AUTOPILOT_QUANTILE_ESTIMATOR.to_string(),
            retry_rate: AUTOPILOT_RETRY_ESTIMATOR.to_string(),
        },
        chosen: ordered_map(&profile.chosen),
        samples: PersistedSamples {
            step_ms: history.step_ms.iter().copied().collect(),
            mem_mb: history.mem_mb.iter().copied().collect(),
            retry_rate: history.retry_rate.iter().copied().collect(),
        },
        summary: PersistedSummary {
            step_ms_p50: profile.step_ms_p50,
            mem_mb_p95: profile.mem_mb_p95,
            retry_rate: profile.retry_rate,
        },
    };
    let mut bytes = serde_json::to_vec_pretty(&persisted)
        .map_err(|error| profile_format(&path, format!("JSON encode failed: {error}")))?;
    bytes.push(b'\n');
    atomic_write(&path, &bytes).map_err(|source| profile_io(path, source))
}

fn validate_profile_value(path: &Path, field: &str, value: f64) -> Result<(), AutopilotError> {
    if !value.is_finite() {
        return Err(profile_format(path, format!("{field} must be finite")));
    }
    if value < 0.0 {
        return Err(profile_format(
            path,
            format!("{field} must be non-negative"),
        ));
    }
    if value > f32::MAX as f64 {
        return Err(profile_format(
            path,
            format!("{field} exceeds the f32 profile range"),
        ));
    }
    Ok(())
}

fn validate_persisted_choices<'a>(
    path: &Path,
    choices: impl IntoIterator<Item = (&'a String, &'a String)>,
) -> Result<(), AutopilotError> {
    for (id, choice) in choices {
        validate_persisted_choice(path, id, choice)?;
    }
    Ok(())
}

fn validate_persisted_choice(path: &Path, id: &str, choice: &str) -> Result<(), AutopilotError> {
    if id.trim().is_empty() || id != id.trim() || id.chars().any(char::is_control) {
        return Err(profile_format(
            path,
            format!("invalid persisted knob id {id:?}"),
        ));
    }
    if choice.is_empty() || choice.chars().any(char::is_control) {
        return Err(profile_format(
            path,
            format!("invalid persisted choice for knob '{id}'"),
        ));
    }
    Ok(())
}

fn summary_matches(expected: f32, actual: f32) -> bool {
    let scale = expected.abs().max(actual.abs()).max(1.0);
    (expected - actual).abs() <= 8.0 * f32::EPSILON * scale
}

fn ordered_map(values: &HashMap<String, String>) -> BTreeMap<String, String> {
    values
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn profile_io(path: PathBuf, source: std::io::Error) -> AutopilotError {
    AutopilotError::ProfileIo { path, source }
}

fn profile_format(path: &Path, detail: impl Into<String>) -> AutopilotError {
    AutopilotError::ProfileFormat {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

fn publish_rejection(stage: &'static str, error: &AutopilotError) {
    let bus = global_registry().event_bus();
    if bus.has_listeners("AutopilotRejected") {
        bus.publish(&PluginEvent::custom(
            "AutopilotRejected",
            serde_json::json!({
                "stage": stage,
                "error": error.to_string(),
                "profile_contract": AUTOPILOT_PROFILE_CONTRACT,
                "profile_schema_version": AUTOPILOT_PROFILE_SCHEMA_VERSION,
            }),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::blackcat::{bandit::SoftBanditMode, zmeta::ZMetaParams, ChoiceGroups};
    use std::fs;
    use tempfile::tempdir;

    fn demo_runtime(feat_dim: usize, groups: HashMap<String, Vec<String>>) -> BlackCatRuntime {
        BlackCatRuntime::new(
            ZMetaParams::default(),
            ChoiceGroups { groups },
            feat_dim,
            SoftBanditMode::TS,
            None,
        )
    }

    fn default_groups() -> HashMap<String, Vec<String>> {
        HashMap::from([
            ("wg".to_string(), vec!["128".to_string(), "256".to_string()]),
            (
                "tile".to_string(),
                vec!["512".to_string(), "1024".to_string(), "2048".to_string()],
            ),
        ])
    }

    fn configured_autopilot(profile_dir: &Path) -> Autopilot {
        Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                mode: AutoMode::Auto,
                knobs: vec![
                    KnobSpec {
                        id: "wg".to_string(),
                        domain: vec!["128".to_string(), "256".to_string()],
                        default_idx: 0,
                    },
                    KnobSpec {
                        id: "tile".to_string(),
                        domain: vec!["512".to_string(), "1024".to_string(), "2048".to_string()],
                        default_idx: 1,
                    },
                ],
                feat_dim: BASE_CONTEXT_DIM,
                extra_features: Vec::new(),
            },
            demo_runtime(BASE_CONTEXT_DIM, default_groups()),
            profile_dir,
        )
        .expect("valid Autopilot")
    }

    fn metrics(step_time_ms: f64, mem_peak_mb: f64, retry_rate: f64) -> StepMetrics {
        StepMetrics {
            step_time_ms,
            mem_peak_mb,
            retry_rate,
            extra: HashMap::new(),
        }
    }

    fn suggest_once(autopilot: &mut Autopilot) {
        let context = autopilot
            .try_build_context(8, 128, 64, 0.5, &[])
            .expect("valid context");
        assert!(!autopilot
            .try_suggest(context)
            .expect("suggestion")
            .is_empty());
    }

    #[test]
    fn profile_uses_exact_window_quantiles_and_round_trips() {
        let temp = tempdir().expect("temporary profile directory");
        let mut autopilot = configured_autopilot(temp.path());
        for index in 1..=20 {
            suggest_once(&mut autopilot);
            autopilot
                .try_report(&metrics(
                    index as f64,
                    (index * 10) as f64,
                    index as f64 / 100.0,
                ))
                .expect("profile report");
        }
        assert_eq!(autopilot.profile().step_ms_p50, 10.5);
        assert_eq!(autopilot.profile().mem_mb_p95, 190.5);
        assert!((autopilot.profile().retry_rate - 0.105).abs() < 1.0e-6);
        assert_eq!(autopilot.profile_sample_count(), 20);

        let reloaded = configured_autopilot(temp.path());
        assert_eq!(reloaded.profile(), autopilot.profile());
        assert_eq!(reloaded.profile_sample_count(), 20);
        let persisted = fs::read_to_string(reloaded.profile_path()).expect("profile JSON");
        assert!(persisted.contains("\"schema_version\": 2"));
        let tile_pos = persisted.find("\"tile\"").expect("tile choice");
        let wg_pos = persisted.find("\"wg\"").expect("wg choice");
        assert!(
            tile_pos < wg_pos,
            "chosen controls are persisted deterministically"
        );
    }

    #[test]
    fn bounded_window_drops_old_samples() {
        let temp = tempdir().expect("temporary profile directory");
        let mut autopilot = configured_autopilot(temp.path());
        for index in 0..(AUTOPILOT_PROFILE_WINDOW + 4) {
            suggest_once(&mut autopilot);
            autopilot
                .try_report(&metrics(index as f64, index as f64, 0.0))
                .expect("profile report");
        }
        assert_eq!(autopilot.profile_sample_count(), AUTOPILOT_PROFILE_WINDOW);
        assert_eq!(autopilot.profile().step_ms_p50, 67.5);
    }

    #[test]
    fn invalid_report_is_rejected_without_profile_mutation() {
        let temp = tempdir().expect("temporary profile directory");
        let mut autopilot = configured_autopilot(temp.path());
        suggest_once(&mut autopilot);
        let before = autopilot.profile().clone();
        let error = autopilot
            .try_report(&metrics(f64::NAN, 10.0, 0.0))
            .expect_err("non-finite metric must fail");
        assert!(matches!(error, AutopilotError::NonFinite { .. }));
        assert_eq!(autopilot.profile(), &before);
        assert_eq!(autopilot.profile_sample_count(), 0);
        assert!(!autopilot.profile_path().exists());
    }

    #[test]
    fn profile_io_failure_discloses_committed_reward_and_prevents_double_credit() {
        let temp = tempdir().expect("temporary profile directory");
        let profile_dir = temp.path().to_path_buf();
        let mut autopilot = configured_autopilot(&profile_dir);
        suggest_once(&mut autopilot);
        fs::remove_dir(&profile_dir).expect("remove empty profile directory");
        fs::write(&profile_dir, b"not a directory").expect("block profile directory creation");

        let error = autopilot
            .try_report(&metrics(4.0, 8.0, 0.0))
            .expect_err("profile persistence must fail");
        assert!(matches!(error, AutopilotError::ProfileCommit { .. }));
        assert_eq!(autopilot.runtime().stats().steps, 1);
        assert_eq!(autopilot.profile_sample_count(), 0);
        assert!(matches!(
            autopilot.try_report(&metrics(4.0, 8.0, 0.0)),
            Err(AutopilotError::MissingSuggestion)
        ));
        fs::remove_file(&profile_dir).expect("remove profile blocker");
    }

    #[test]
    fn report_requires_fresh_suggestion() {
        let temp = tempdir().expect("temporary profile directory");
        let mut autopilot = configured_autopilot(temp.path());
        assert!(matches!(
            autopilot.try_report(&metrics(1.0, 1.0, 0.0)),
            Err(AutopilotError::MissingSuggestion)
        ));
        suggest_once(&mut autopilot);
        autopilot
            .try_report(&metrics(1.0, 1.0, 0.0))
            .expect("first report");
        assert!(matches!(
            autopilot.try_report(&metrics(1.0, 1.0, 0.0)),
            Err(AutopilotError::MissingSuggestion)
        ));
    }

    #[test]
    fn pending_suggestion_requires_report_or_explicit_abandonment() {
        let temp = tempdir().expect("temporary profile directory");
        let mut autopilot = configured_autopilot(temp.path());
        let context = autopilot
            .try_build_context(8, 128, 64, 0.5, &[])
            .expect("valid context");
        let first_picks = autopilot
            .try_suggest(context.clone())
            .expect("first suggestion")
            .clone();
        let first_selection = autopilot
            .runtime()
            .pending_selection_id()
            .expect("pending BlackCat selection");

        assert!(matches!(
            autopilot.try_suggest(context.clone()),
            Err(AutopilotError::PendingSuggestion)
        ));
        assert_eq!(&autopilot.picks, &first_picks);
        assert_eq!(
            autopilot.runtime().pending_selection_id(),
            Some(first_selection)
        );

        assert_eq!(autopilot.abandon_suggestion(), Some(first_selection));
        assert!(autopilot.picks.is_empty());
        autopilot
            .try_suggest(context)
            .expect("new suggestion after explicit abandonment");
        assert_eq!(
            autopilot.runtime().pending_selection_id(),
            Some(first_selection + 1)
        );
    }

    #[test]
    fn context_is_named_stable_and_strict() {
        let temp = tempdir().expect("temporary profile directory");
        let autopilot = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                feat_dim: 10,
                extra_features: vec!["alpha".to_string(), "zeta".to_string()],
                ..AutoConfig::default()
            },
            demo_runtime(10, default_groups()),
            temp.path(),
        )
        .expect("valid Autopilot");
        let context = autopilot
            .try_build_context(
                8,
                128,
                64,
                0.5,
                &[("zeta".to_string(), 2.0), ("alpha".to_string(), 1.0)],
            )
            .expect("valid context");
        assert_eq!(&context[8..], &[1.0, 2.0]);
        assert_eq!(
            autopilot.context_features(),
            vec![
                "intercept",
                "batch",
                "tiles",
                "depth",
                "device_lane_width_over_1024",
                "device_max_workgroup_over_4096",
                "device_load",
                "device_subgroup",
                "alpha",
                "zeta",
            ]
        );
        assert!(matches!(
            autopilot.try_build_context(8, 128, 64, f64::NAN, &[]),
            Err(AutopilotError::NonFinite { .. })
        ));
        assert!(matches!(
            autopilot.try_build_context(
                8,
                128,
                64,
                0.5,
                &[("same".to_string(), 1.0), ("same".to_string(), 2.0)]
            ),
            Err(AutopilotError::DuplicateFeature { .. })
        ));
        assert!(matches!(
            autopilot.try_build_context(8, 128, 64, 0.5, &[("alpha".to_string(), 1.0)]),
            Err(AutopilotError::MissingFeature { .. })
        ));
        assert!(matches!(
            autopilot.try_build_context(
                8,
                128,
                64,
                0.5,
                &[
                    ("one".to_string(), 1.0),
                    ("two".to_string(), 2.0),
                    ("three".to_string(), 3.0),
                ]
            ),
            Err(AutopilotError::UnexpectedFeature { .. })
        ));

        let missing_schema = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                feat_dim: 9,
                ..AutoConfig::default()
            },
            demo_runtime(9, default_groups()),
            temp.path(),
        );
        assert!(matches!(
            missing_schema,
            Err(AutopilotError::InvalidConfig {
                field: "extra_features",
                ..
            })
        ));
    }

    #[test]
    fn short_context_dimension_is_a_canonical_prefix() {
        let temp = tempdir().expect("temporary profile directory");
        let autopilot = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                feat_dim: 4,
                ..AutoConfig::default()
            },
            demo_runtime(4, default_groups()),
            temp.path(),
        )
        .expect("four-feature Autopilot");
        let context = autopilot
            .try_build_context(8, 128, 64, 0.5, &[])
            .expect("canonical prefix");
        assert_eq!(context, vec![1.0, 8.0, 128.0, 64.0]);
        assert!(matches!(
            autopilot.try_build_context(8, 128, 64, 0.5, &[("extra".to_string(), 1.0)]),
            Err(AutopilotError::UnexpectedFeature { .. })
        ));
    }

    #[test]
    fn declared_default_fills_a_control_without_a_bandit() {
        let temp = tempdir().expect("temporary profile directory");
        let mut autopilot = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                knobs: vec![
                    KnobSpec {
                        id: "wg".to_string(),
                        domain: vec!["128".to_string()],
                        default_idx: 0,
                    },
                    KnobSpec {
                        id: "tile".to_string(),
                        domain: vec!["512".to_string(), "1024".to_string()],
                        default_idx: 1,
                    },
                ],
                ..AutoConfig::default()
            },
            demo_runtime(
                BASE_CONTEXT_DIM,
                HashMap::from([("wg".to_string(), vec!["128".to_string()])]),
            ),
            temp.path(),
        )
        .expect("valid knob contract");
        let context = autopilot
            .try_build_context(8, 128, 64, 0.5, &[])
            .expect("context");
        let picks = autopilot.try_suggest(context).expect("suggestion");
        assert_eq!(picks.get("wg").map(String::as_str), Some("128"));
        assert_eq!(picks.get("tile").map(String::as_str), Some("1024"));
    }

    #[test]
    fn knob_contract_rejects_undeclared_runtime_controls() {
        let temp = tempdir().expect("temporary profile directory");
        let result = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                knobs: vec![KnobSpec {
                    id: "wg".to_string(),
                    domain: vec!["128".to_string()],
                    default_idx: 0,
                }],
                ..AutoConfig::default()
            },
            demo_runtime(
                BASE_CONTEXT_DIM,
                HashMap::from([("tile".to_string(), vec!["512".to_string()])]),
            ),
            temp.path(),
        );
        let error = match result {
            Ok(_) => panic!("undeclared BlackCat controls must fail construction"),
            Err(error) => error,
        };
        assert!(matches!(error, AutopilotError::InvalidConfig { .. }));
    }

    #[test]
    fn constructor_rejects_a_fail_closed_blackcat_runtime() {
        let temp = tempdir().expect("temporary profile directory");
        let runtime = demo_runtime(
            BASE_CONTEXT_DIM,
            HashMap::from([("wg".to_string(), vec!["128".to_string(), "128".to_string()])]),
        );
        let result = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig::default(),
            runtime,
            temp.path(),
        );
        let error = match result {
            Ok(_) => panic!("invalid BlackCat configuration must not be hidden"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            AutopilotError::InvalidConfig {
                field: "blackcat",
                ..
            }
        ));
    }

    #[test]
    fn knob_contract_rejects_runtime_choices_outside_domain() {
        let temp = tempdir().expect("temporary profile directory");
        let result = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                knobs: vec![KnobSpec {
                    id: "wg".to_string(),
                    domain: vec!["128".to_string()],
                    default_idx: 0,
                }],
                ..AutoConfig::default()
            },
            demo_runtime(
                BASE_CONTEXT_DIM,
                HashMap::from([("wg".to_string(), vec!["128".to_string(), "256".to_string()])]),
            ),
            temp.path(),
        );
        let error = match result {
            Ok(_) => panic!("out-of-contract BlackCat choices must fail construction"),
            Err(error) => error,
        };
        assert!(matches!(error, AutopilotError::InvalidConfig { .. }));
    }

    #[test]
    fn knob_specs_validate_domains_and_defaults() {
        let temp = tempdir().expect("temporary profile directory");
        let result = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                knobs: vec![KnobSpec {
                    id: "wg".to_string(),
                    domain: vec!["128".to_string()],
                    default_idx: 1,
                }],
                ..AutoConfig::default()
            },
            demo_runtime(BASE_CONTEXT_DIM, default_groups()),
            temp.path(),
        );
        let error = match result {
            Ok(_) => panic!("out-of-range default must fail"),
            Err(error) => error,
        };
        assert!(matches!(error, AutopilotError::InvalidConfig { .. }));
    }

    #[test]
    fn legacy_profile_is_loaded_and_migrated_on_report() {
        let temp = tempdir().expect("temporary profile directory");
        let probe = configured_autopilot(temp.path());
        let path = probe.profile_path();
        drop(probe);
        fs::write(
            &path,
            "step_ms_p50=12.5\nmem_mb_p95=256\nretry_rate=0.1\nchosen.wg=128\nchosen.stale=legacy\n",
        )
        .expect("legacy profile");
        let mut autopilot = configured_autopilot(temp.path());
        assert_eq!(autopilot.profile_sample_count(), 0);
        assert_eq!(autopilot.profile().step_ms_p50, 0.0);
        assert_eq!(
            autopilot.profile().chosen.get("wg").map(String::as_str),
            Some("128")
        );
        assert!(!autopilot.profile().chosen.contains_key("stale"));
        suggest_once(&mut autopilot);
        autopilot
            .try_report(&metrics(7.5, 128.0, 0.0))
            .expect("migrating report");
        assert_eq!(autopilot.profile_sample_count(), 1);
        assert_eq!(autopilot.profile().step_ms_p50, 7.5);
        let persisted = fs::read_to_string(path).expect("migrated profile");
        assert!(persisted.trim_start().starts_with('{'));
    }

    #[test]
    fn corrupt_profile_is_not_silently_accepted() {
        let temp = tempdir().expect("temporary profile directory");
        let probe = configured_autopilot(temp.path());
        let path = probe.profile_path();
        drop(probe);
        fs::write(&path, "step_ms_p50=not-a-number\n").expect("corrupt profile");
        let result = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig::default(),
            demo_runtime(BASE_CONTEXT_DIM, default_groups()),
            temp.path(),
        );
        let error = match result {
            Ok(_) => panic!("corrupt profile must fail"),
            Err(error) => error,
        };
        assert!(matches!(error, AutopilotError::ProfileFormat { .. }));

        let compatibility = Autopilot::new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig::default(),
            demo_runtime(BASE_CONTEXT_DIM, default_groups()),
            temp.path(),
        );
        assert_eq!(compatibility.profile_sample_count(), 0);
        assert!(compatibility.profile().chosen.is_empty());
    }

    #[test]
    fn incomplete_legacy_profile_is_not_promoted_to_a_witness() {
        let temp = tempdir().expect("temporary profile directory");
        let probe = configured_autopilot(temp.path());
        let path = probe.profile_path();
        drop(probe);
        fs::write(&path, "step_ms_p50=12.5\nmem_mb_p95=256\n").expect("incomplete profile");
        let result = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig::default(),
            demo_runtime(BASE_CONTEXT_DIM, default_groups()),
            temp.path(),
        );
        let error = match result {
            Ok(_) => panic!("incomplete legacy metrics must fail"),
            Err(error) => error,
        };
        assert!(matches!(error, AutopilotError::ProfileFormat { .. }));
    }

    #[test]
    fn profile_estimator_contract_is_verified_on_load() {
        let temp = tempdir().expect("temporary profile directory");
        let mut autopilot = configured_autopilot(temp.path());
        suggest_once(&mut autopilot);
        autopilot
            .try_report(&metrics(4.0, 8.0, 0.0))
            .expect("profile report");
        let path = autopilot.profile_path();
        drop(autopilot);
        let mut persisted: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("persisted profile"))
                .expect("profile JSON");
        persisted["estimators"]["retry_rate"] = serde_json::json!("ema");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&persisted).expect("tampered JSON"),
        )
        .expect("tampered profile");
        let result = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                mode: AutoMode::Auto,
                knobs: vec![
                    KnobSpec {
                        id: "wg".to_string(),
                        domain: vec!["128".to_string(), "256".to_string()],
                        default_idx: 0,
                    },
                    KnobSpec {
                        id: "tile".to_string(),
                        domain: vec!["512".to_string(), "1024".to_string(), "2048".to_string()],
                        default_idx: 1,
                    },
                ],
                feat_dim: BASE_CONTEXT_DIM,
                extra_features: Vec::new(),
            },
            demo_runtime(BASE_CONTEXT_DIM, default_groups()),
            temp.path(),
        );
        let error = match result {
            Ok(_) => panic!("tampered estimator contract must fail"),
            Err(error) => error,
        };
        assert!(matches!(error, AutopilotError::ProfileFormat { .. }));
    }

    #[test]
    fn hint_mode_uses_persisted_choice_as_a_bandit_tie_breaker() {
        let temp = tempdir().expect("temporary profile directory");
        let caps = DeviceCaps::wgpu(32, true, 256);
        fs::create_dir_all(temp.path()).expect("profile directory");
        fs::write(
            profile_path(temp.path(), &make_key(&caps)),
            "step_ms_p50=1\nmem_mb_p95=1\nretry_rate=0\nchosen.wg=128\n",
        )
        .expect("legacy profile");
        let mut groups = HashMap::new();
        groups.insert("wg".to_string(), vec!["64".to_string(), "128".to_string()]);
        let mut autopilot = Autopilot::try_new_with_profile_dir(
            caps,
            AutoConfig {
                mode: AutoMode::Hint,
                feat_dim: BASE_CONTEXT_DIM,
                knobs: Vec::new(),
                extra_features: Vec::new(),
            },
            demo_runtime(BASE_CONTEXT_DIM, groups),
            temp.path(),
        )
        .expect("hinted Autopilot");
        let context = autopilot
            .try_build_context(8, 128, 64, 0.5, &[])
            .expect("context");
        assert_eq!(
            autopilot
                .try_suggest(context)
                .expect("hinted suggestion")
                .get("wg")
                .map(String::as_str),
            Some("128")
        );
    }

    #[test]
    fn off_mode_does_not_depend_on_profile_io() {
        let temp = tempdir().expect("temporary profile directory");
        let probe = configured_autopilot(temp.path());
        let path = probe.profile_path();
        drop(probe);
        fs::write(&path, "corrupt profile").expect("corrupt profile");
        let autopilot = Autopilot::try_new_with_profile_dir(
            DeviceCaps::wgpu(32, true, 256),
            AutoConfig {
                mode: AutoMode::Off,
                ..AutoConfig::default()
            },
            demo_runtime(BASE_CONTEXT_DIM, default_groups()),
            temp.path(),
        )
        .expect("disabled Autopilot must not load profiles");
        assert_eq!(autopilot.mode(), AutoMode::Off);
        assert_eq!(autopilot.profile_sample_count(), 0);
    }
}
