//! Canonical Topos route-policy scoring and selection semantics.
//!
//! Rust owns profile scoring, reward projection, deterministic tie-breaking,
//! and selected-route resolution. Bindings should only gather route evidence
//! and translate these typed payloads for their host language.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;
use thiserror::Error;

/// Stable contract identifier shared by Rust, Python, and WASM clients.
pub const TOPOS_ROUTE_POLICY_CONTRACT_VERSION: &str = "spiraltorch.topos_route_policy.v2";
/// Payload kind for route-profile evaluation.
pub const TOPOS_ROUTE_POLICY_KIND: &str = "spiraltorch.topos_route_policy";
/// Payload kind for projected route rewards.
pub const TOPOS_ROUTE_REWARDS_KIND: &str = "spiraltorch.topos_route_rewards";
/// Payload kind for selected-route resolution.
pub const TOPOS_ROUTE_POLICY_RESOLUTION_KIND: &str = "spiraltorch.topos_route_policy_resolution";
/// Crate/module that owns Topos route-policy semantics.
pub const TOPOS_ROUTE_POLICY_SEMANTIC_OWNER: &str = "st-core::runtime::topos_route_policy";
/// Backend label attached to payloads produced by the canonical implementation.
pub const TOPOS_ROUTE_POLICY_SEMANTIC_BACKEND: &str = "rust";
/// Stable identity for the evidence-aware score calculation.
pub const TOPOS_ROUTE_POLICY_SCORE_FORMULA_VERSION: &str =
    "spiraltorch.topos_route_policy.score.v2";
/// Human-readable form of the score calculation carried in audit payloads.
pub const TOPOS_ROUTE_POLICY_SCORE_FORMULA: &str =
    "raw=clamp(weighted_mean(profile_metric_or_0.5)+0.08*(0.5-incomplete_or_0.5),0,1);confidence=n/(n+1);score=0.5+confidence*(raw-0.5);count=0_inactive";
pub const TOPOS_ROUTE_POLICY_PRIOR_MEAN: f64 = 0.5;
pub const TOPOS_ROUTE_POLICY_PRIOR_STRENGTH: f64 = 1.0;
pub const TOPOS_ROUTE_POLICY_INCOMPLETE_ADJUSTMENT_WEIGHT: f64 = 0.08;
pub const TOPOS_ROUTE_POLICY_CONTEXT_WEIGHT_MAX: f64 = 1.25;

pub const TOPOS_ROUTE_POLICY_MAX_ROWS: usize = 4_096;
pub const TOPOS_ROUTE_POLICY_MAX_LABEL_BYTES: usize = 256;
pub const TOPOS_ROUTE_POLICY_MAX_MODE_BYTES: usize = 256;

fn discard_client_projection<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default,
{
    let _ = serde::de::IgnoredAny::deserialize(deserializer)?;
    Ok(T::default())
}

#[derive(Debug, Error, PartialEq)]
pub enum ToposRoutePolicyError {
    #[error("route row count {actual} exceeds limit {max}")]
    TooManyRows { actual: usize, max: usize },
    #[error("route row {index} must have a non-empty label")]
    EmptyLabel { index: usize },
    #[error("route label at row {index} has {actual} bytes, exceeding limit {max}")]
    LabelTooLong {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error("route mode at row {index} has {actual} bytes, exceeding limit {max}")]
    ModeTooLong {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error("route label '{label}' appears more than once")]
    DuplicateLabel { label: String },
    #[error("metric '{field}' at route row {index} must be finite")]
    NonFiniteMetric { index: usize, field: &'static str },
    #[error("metric '{field}' at route row {index} must be in the inclusive range [0, 1]")]
    MetricOutOfRange { index: usize, field: &'static str },
    #[error("metric '{field}' at route row {index} must be non-negative")]
    NegativeMetric { index: usize, field: &'static str },
    #[error(
        "metric 'context_weight' at route row {index} must be in the inclusive range [0, 1.25]"
    )]
    ContextWeightOutOfRange { index: usize },
    #[error("unknown route-policy profile '{profile}'")]
    UnknownProfile { profile: String },
    #[error("route reward count {actual} exceeds limit {max}")]
    TooManyRewards { actual: usize, max: usize },
    #[error("route reward {index} must have a non-empty label")]
    EmptyRewardLabel { index: usize },
    #[error("route reward label at row {index} has {actual} bytes, exceeding limit {max}")]
    RewardLabelTooLong {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error("route reward field '{field}' at row {index} must be finite")]
    NonFiniteReward { index: usize, field: &'static str },
    #[error("route reward at row {index} must be in the inclusive range [0, 1]")]
    RewardOutOfRange { index: usize },
    #[error("route reward at row {index} has zero observations and is not an active route")]
    InactiveReward { index: usize },
    #[error(
        "route reward at row {index} is missing the v2 source-row/score-evidence witness; rebuild rewards from the original route rows"
    )]
    MissingRewardWitness { index: usize },
    #[error("route reward at position {position} declares index {actual}; expected {expected}")]
    RewardIndexMismatch {
        position: usize,
        actual: u32,
        expected: u32,
    },
    #[error(
        "route reward at row {index} uses profile '{actual}', but the reward set uses '{expected}'"
    )]
    MixedRewardProfiles {
        index: usize,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("selected label has {actual} bytes, exceeding limit {max}")]
    SelectedLabelTooLong { actual: usize, max: usize },
    #[error("selected route label '{label}' is not present in the reward set")]
    SelectedLabelNotFound { label: String },
    #[error("selected route index {index} is outside reward set length {len}")]
    SelectedIndexOutOfRange { index: u32, len: usize },
    #[error(
        "route reward evidence at row {index} does not match the Rust score contract: {field}"
    )]
    RewardEvidenceMismatch { index: usize, field: &'static str },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToposRoutePolicyProfile {
    #[default]
    Balanced,
    Quality,
    Grounded,
    Efficiency,
    Latency,
}

impl ToposRoutePolicyProfile {
    pub const ALL: [Self; 5] = [
        Self::Balanced,
        Self::Quality,
        Self::Grounded,
        Self::Efficiency,
        Self::Latency,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Quality => "quality",
            Self::Grounded => "grounded",
            Self::Efficiency => "efficiency",
            Self::Latency => "latency",
        }
    }
}

impl FromStr for ToposRoutePolicyProfile {
    type Err = ToposRoutePolicyError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "balanced" => Ok(Self::Balanced),
            "quality" => Ok(Self::Quality),
            "grounded" => Ok(Self::Grounded),
            "efficiency" => Ok(Self::Efficiency),
            "latency" => Ok(Self::Latency),
            _ => Err(ToposRoutePolicyError::UnknownProfile {
                profile: value.to_owned(),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for ToposRoutePolicyProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let profile = String::deserialize(deserializer)?;
        Self::from_str(&profile).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToposRoutePolicyRow {
    pub label: String,
    pub count: u32,
    pub trace_route_score: Option<f64>,
    pub trace_quality_score: Option<f64>,
    pub trace_efficiency_score: Option<f64>,
    pub trace_text_quality_score: Option<f64>,
    pub response_text_quality_score: Option<f64>,
    pub response_prompt_coverage: Option<f64>,
    pub response_completion_rate: Option<f64>,
    pub response_incomplete_rate: Option<f64>,
    pub response_confidence: Option<f64>,
    pub latency_ms_mean: Option<f64>,
    pub total_tokens: Option<f64>,
    pub adapter_runtime_route_score: Option<f64>,
    pub adapter_guard_score: Option<f64>,
    pub adapter_exploration_score: Option<f64>,
    pub adapter_context_score: Option<f64>,
    pub closure_pressure: Option<f64>,
    pub openness: Option<f64>,
    pub context_weight: Option<f64>,
    pub request_temperature: Option<f64>,
    pub mode: Option<String>,
    /// Canonical scores emitted by Rust. Client-provided values are accepted as
    /// transport compatibility fields but are never deserialized or trusted.
    #[serde(
        default,
        deserialize_with = "discard_client_projection",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub selection_scores: BTreeMap<String, f64>,
    /// Evidence breakdown for each canonical score. This is output-only for
    /// the same reason as `selection_scores`.
    #[serde(
        default,
        deserialize_with = "discard_client_projection",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub selection_evidence: BTreeMap<String, ToposRoutePolicyScoreEvidence>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToposRoutePolicyScoreEvidence {
    pub profile: ToposRoutePolicyProfile,
    pub formula_version: String,
    pub raw_score: f64,
    pub score: f64,
    pub evidence_coverage: f64,
    pub sample_count: u32,
    pub sample_confidence: f64,
    pub observed_metrics: Vec<String>,
    pub missing_metrics: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToposRoutePolicyEvaluationRequest {
    pub rows: Vec<ToposRoutePolicyRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToposRoutePolicyProfileWinner {
    pub label: Option<String>,
    pub score: f64,
    pub trace_route_score: f64,
    pub trace_quality_score: f64,
    pub trace_efficiency_score: f64,
    pub response_text_quality_score: f64,
    pub response_prompt_coverage: f64,
    pub response_completion_rate: f64,
    pub latency_ms_mean: f64,
    pub total_tokens: f64,
    pub adapter_runtime_route_score: f64,
    pub adapter_guard_score: f64,
    pub adapter_context_score: f64,
    pub closure_pressure: Option<f64>,
    pub openness: Option<f64>,
    pub context_weight: Option<f64>,
    pub score_evidence: Option<ToposRoutePolicyScoreEvidence>,
}

impl Default for ToposRoutePolicyProfileWinner {
    fn default() -> Self {
        Self {
            label: None,
            score: 0.0,
            trace_route_score: 0.0,
            trace_quality_score: 0.0,
            trace_efficiency_score: 0.0,
            response_text_quality_score: 0.0,
            response_prompt_coverage: 0.0,
            response_completion_rate: 0.0,
            latency_ms_mean: 0.0,
            total_tokens: 0.0,
            adapter_runtime_route_score: 0.0,
            adapter_guard_score: 0.0,
            adapter_context_score: 0.0,
            closure_pressure: None,
            openness: None,
            context_weight: None,
            score_evidence: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToposRoutePolicyEvaluationPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub row_count: usize,
    pub active_row_count: usize,
    pub score_formula_version: &'static str,
    pub score_formula: &'static str,
    pub score_prior_mean: f64,
    pub score_prior_strength: f64,
    pub rows: Vec<ToposRoutePolicyRow>,
    pub profiles: BTreeMap<String, ToposRoutePolicyProfileWinner>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToposRouteRewardsRequest {
    pub rows: Vec<ToposRoutePolicyRow>,
    #[serde(default)]
    pub profile: ToposRoutePolicyProfile,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToposRouteReward {
    pub index: u32,
    pub source_index: u32,
    pub label: String,
    pub profile: ToposRoutePolicyProfile,
    pub reward: f64,
    pub count: u32,
    pub trace_route_score: f64,
    pub response_text_quality_score: f64,
    pub response_completion_rate: f64,
    pub response_incomplete_rate: f64,
    pub adapter_runtime_route_score: f64,
    pub score_evidence: ToposRoutePolicyScoreEvidence,
    pub source_row: ToposRoutePolicyRow,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToposRouteRewardsPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub profile: ToposRoutePolicyProfile,
    pub input_row_count: usize,
    pub active_row_count: usize,
    pub inactive_row_count: usize,
    pub reward_count: usize,
    pub score_formula_version: &'static str,
    pub score_formula: &'static str,
    pub score_prior_mean: f64,
    pub score_prior_strength: f64,
    pub rewards: Vec<ToposRouteReward>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ToposRoutePolicyResolveRequest {
    pub rewards: Vec<ToposRouteReward>,
    pub selected_label: Option<String>,
    pub selected_index: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToposRoutePolicyResolutionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub score_formula_version: &'static str,
    pub resolution: &'static str,
    pub selected_position: Option<usize>,
    pub selected_label: Option<String>,
    pub selected_reward: Option<f64>,
    pub route_reward: Option<ToposRouteReward>,
}

fn finite_or_zero(value: Option<f64>) -> f64 {
    value.unwrap_or(0.0)
}

fn unit_reward(value: Option<f64>) -> Option<f64> {
    value.map(|value| value.clamp(0.0, 1.0))
}

fn inverse_unit_reward(value: Option<f64>) -> Option<f64> {
    unit_reward(value).map(|value| 1.0 - value)
}

fn context_weight_reward(value: Option<f64>) -> Option<f64> {
    value.map(|value| (value / TOPOS_ROUTE_POLICY_CONTEXT_WEIGHT_MAX).clamp(0.0, 1.0))
}

fn latency_reward(value: Option<f64>) -> Option<f64> {
    value.map(|latency| 1.0 - (latency.ln_1p() / 10_000.0_f64.ln_1p()).min(1.0))
}

fn token_reward(value: Option<f64>) -> Option<f64> {
    value.map(|tokens| 1.0 - (tokens.ln_1p() / 4_096.0_f64.ln_1p()).min(1.0))
}

#[derive(Clone, Copy)]
struct WeightedMetric {
    name: &'static str,
    value: Option<f64>,
    weight: f64,
}

impl WeightedMetric {
    const fn new(name: &'static str, value: Option<f64>, weight: f64) -> Self {
        Self {
            name,
            value,
            weight,
        }
    }
}

fn profile_metrics(
    row: &ToposRoutePolicyRow,
    profile: ToposRoutePolicyProfile,
) -> Vec<WeightedMetric> {
    let trace_route = unit_reward(row.trace_route_score);
    let trace_quality = unit_reward(row.trace_quality_score);
    let trace_efficiency = unit_reward(row.trace_efficiency_score);
    let trace_text_quality = unit_reward(row.trace_text_quality_score);
    let response_quality = unit_reward(row.response_text_quality_score);
    let prompt_coverage = unit_reward(row.response_prompt_coverage);
    let completion = unit_reward(row.response_completion_rate);
    let confidence = unit_reward(row.response_confidence);
    let adapter_route = unit_reward(row.adapter_runtime_route_score);
    let adapter_guard = unit_reward(row.adapter_guard_score);
    let adapter_exploration = unit_reward(row.adapter_exploration_score);
    let adapter_context = unit_reward(row.adapter_context_score);
    let closure = inverse_unit_reward(row.closure_pressure);
    let openness = unit_reward(row.openness);
    let context_weight = context_weight_reward(row.context_weight);

    match profile {
        ToposRoutePolicyProfile::Quality => vec![
            WeightedMetric::new("trace_quality_score", trace_quality, 0.25),
            WeightedMetric::new("trace_text_quality_score", trace_text_quality, 0.10),
            WeightedMetric::new("response_text_quality_score", response_quality, 0.25),
            WeightedMetric::new("response_prompt_coverage", prompt_coverage, 0.15),
            WeightedMetric::new("response_confidence", confidence, 0.10),
            WeightedMetric::new("response_completion_rate", completion, 0.10),
            WeightedMetric::new("adapter_runtime_route_score", adapter_route, 0.05),
        ],
        ToposRoutePolicyProfile::Grounded => vec![
            WeightedMetric::new("response_text_quality_score", response_quality, 0.25),
            WeightedMetric::new("response_prompt_coverage", prompt_coverage, 0.20),
            WeightedMetric::new("context_weight_reward", context_weight, 0.15),
            WeightedMetric::new("adapter_guard_score", adapter_guard, 0.15),
            WeightedMetric::new("response_completion_rate", completion, 0.10),
            WeightedMetric::new("adapter_context_score", adapter_context, 0.10),
            WeightedMetric::new("closure_reward", closure, 0.05),
        ],
        ToposRoutePolicyProfile::Efficiency => vec![
            WeightedMetric::new("trace_efficiency_score", trace_efficiency, 0.20),
            WeightedMetric::new("latency_reward", latency_reward(row.latency_ms_mean), 0.20),
            WeightedMetric::new("token_reward", token_reward(row.total_tokens), 0.20),
            WeightedMetric::new("response_text_quality_score", response_quality, 0.10),
            WeightedMetric::new("response_completion_rate", completion, 0.10),
            WeightedMetric::new("adapter_runtime_route_score", adapter_route, 0.10),
            WeightedMetric::new("trace_route_score", trace_route, 0.10),
        ],
        ToposRoutePolicyProfile::Latency => vec![
            WeightedMetric::new("latency_reward", latency_reward(row.latency_ms_mean), 0.40),
            WeightedMetric::new("token_reward", token_reward(row.total_tokens), 0.20),
            WeightedMetric::new("response_text_quality_score", response_quality, 0.15),
            WeightedMetric::new("response_completion_rate", completion, 0.10),
            WeightedMetric::new("trace_route_score", trace_route, 0.10),
            WeightedMetric::new("adapter_runtime_route_score", adapter_route, 0.05),
        ],
        ToposRoutePolicyProfile::Balanced => vec![
            WeightedMetric::new("trace_route_score", trace_route, 0.18),
            WeightedMetric::new("adapter_runtime_route_score", adapter_route, 0.15),
            WeightedMetric::new("response_text_quality_score", response_quality, 0.16),
            WeightedMetric::new("trace_text_quality_score", trace_text_quality, 0.08),
            WeightedMetric::new("response_completion_rate", completion, 0.10),
            WeightedMetric::new("trace_efficiency_score", trace_efficiency, 0.10),
            WeightedMetric::new("closure_reward", closure, 0.08),
            WeightedMetric::new("openness", openness, 0.05),
            WeightedMetric::new("adapter_exploration_score", adapter_exploration, 0.05),
            WeightedMetric::new("adapter_context_score", adapter_context, 0.05),
        ],
    }
}

fn sample_confidence(count: u32) -> f64 {
    let count = f64::from(count);
    count / (count + TOPOS_ROUTE_POLICY_PRIOR_STRENGTH)
}

fn score_row(
    row: &ToposRoutePolicyRow,
    profile: ToposRoutePolicyProfile,
) -> ToposRoutePolicyScoreEvidence {
    let metrics = profile_metrics(row, profile);
    let total_weight = metrics.iter().map(|metric| metric.weight).sum::<f64>();
    let observed_weight = metrics
        .iter()
        .filter(|metric| metric.value.is_some())
        .map(|metric| metric.weight)
        .sum::<f64>();
    let weighted_score = metrics
        .iter()
        .map(|metric| metric.weight * metric.value.unwrap_or(TOPOS_ROUTE_POLICY_PRIOR_MEAN))
        .sum::<f64>()
        / total_weight;
    let incomplete_rate = row
        .response_incomplete_rate
        .unwrap_or(TOPOS_ROUTE_POLICY_PRIOR_MEAN)
        .clamp(0.0, 1.0);
    let incomplete_adjustment = (TOPOS_ROUTE_POLICY_PRIOR_MEAN - incomplete_rate)
        * TOPOS_ROUTE_POLICY_INCOMPLETE_ADJUSTMENT_WEIGHT;
    let raw_score = (weighted_score + incomplete_adjustment).clamp(0.0, 1.0);
    let confidence = sample_confidence(row.count);
    let score = (TOPOS_ROUTE_POLICY_PRIOR_MEAN
        + confidence * (raw_score - TOPOS_ROUTE_POLICY_PRIOR_MEAN))
        .clamp(0.0, 1.0);
    let mut observed_metrics = metrics
        .iter()
        .filter(|metric| metric.value.is_some())
        .map(|metric| metric.name.to_owned())
        .collect::<Vec<_>>();
    let mut missing_metrics = metrics
        .iter()
        .filter(|metric| metric.value.is_none())
        .map(|metric| metric.name.to_owned())
        .collect::<Vec<_>>();
    if row.response_incomplete_rate.is_some() {
        observed_metrics.push("response_incomplete_rate".to_owned());
    } else {
        missing_metrics.push("response_incomplete_rate".to_owned());
    }
    let total_evidence_weight = total_weight + TOPOS_ROUTE_POLICY_INCOMPLETE_ADJUSTMENT_WEIGHT;
    let observed_evidence_weight = observed_weight
        + if row.response_incomplete_rate.is_some() {
            TOPOS_ROUTE_POLICY_INCOMPLETE_ADJUSTMENT_WEIGHT
        } else {
            0.0
        };
    let evidence_coverage = if observed_evidence_weight == 0.0 {
        0.0
    } else {
        (observed_evidence_weight / total_evidence_weight).clamp(0.0, 1.0)
    };

    ToposRoutePolicyScoreEvidence {
        profile,
        formula_version: TOPOS_ROUTE_POLICY_SCORE_FORMULA_VERSION.to_owned(),
        raw_score,
        score,
        evidence_coverage,
        sample_count: row.count,
        sample_confidence: confidence,
        observed_metrics,
        missing_metrics,
    }
}

fn prepare_rows(
    mut rows: Vec<ToposRoutePolicyRow>,
) -> Result<Vec<ToposRoutePolicyRow>, ToposRoutePolicyError> {
    if rows.len() > TOPOS_ROUTE_POLICY_MAX_ROWS {
        return Err(ToposRoutePolicyError::TooManyRows {
            actual: rows.len(),
            max: TOPOS_ROUTE_POLICY_MAX_ROWS,
        });
    }

    let mut labels = BTreeSet::new();
    for (index, row) in rows.iter_mut().enumerate() {
        row.label = row.label.trim().to_owned();
        if row.label.is_empty() {
            return Err(ToposRoutePolicyError::EmptyLabel { index });
        }
        if row.label.len() > TOPOS_ROUTE_POLICY_MAX_LABEL_BYTES {
            return Err(ToposRoutePolicyError::LabelTooLong {
                index,
                actual: row.label.len(),
                max: TOPOS_ROUTE_POLICY_MAX_LABEL_BYTES,
            });
        }
        if !labels.insert(row.label.clone()) {
            return Err(ToposRoutePolicyError::DuplicateLabel {
                label: row.label.clone(),
            });
        }
        row.mode = row
            .mode
            .take()
            .map(|mode| mode.trim().to_owned())
            .filter(|mode| !mode.is_empty());
        if let Some(mode) = &row.mode {
            if mode.len() > TOPOS_ROUTE_POLICY_MAX_MODE_BYTES {
                return Err(ToposRoutePolicyError::ModeTooLong {
                    index,
                    actual: mode.len(),
                    max: TOPOS_ROUTE_POLICY_MAX_MODE_BYTES,
                });
            }
        }

        let unit_metrics = [
            ("trace_route_score", row.trace_route_score),
            ("trace_quality_score", row.trace_quality_score),
            ("trace_efficiency_score", row.trace_efficiency_score),
            ("trace_text_quality_score", row.trace_text_quality_score),
            (
                "response_text_quality_score",
                row.response_text_quality_score,
            ),
            ("response_prompt_coverage", row.response_prompt_coverage),
            ("response_completion_rate", row.response_completion_rate),
            ("response_incomplete_rate", row.response_incomplete_rate),
            ("response_confidence", row.response_confidence),
            (
                "adapter_runtime_route_score",
                row.adapter_runtime_route_score,
            ),
            ("adapter_guard_score", row.adapter_guard_score),
            ("adapter_exploration_score", row.adapter_exploration_score),
            ("adapter_context_score", row.adapter_context_score),
            ("closure_pressure", row.closure_pressure),
            ("openness", row.openness),
        ];
        for (field, value) in unit_metrics {
            if value.is_some_and(|value| !value.is_finite()) {
                return Err(ToposRoutePolicyError::NonFiniteMetric { index, field });
            }
            if value.is_some_and(|value| !(0.0..=1.0).contains(&value)) {
                return Err(ToposRoutePolicyError::MetricOutOfRange { index, field });
            }
        }
        for (field, value) in [
            ("latency_ms_mean", row.latency_ms_mean),
            ("total_tokens", row.total_tokens),
            ("request_temperature", row.request_temperature),
            ("context_weight", row.context_weight),
        ] {
            if value.is_some_and(|value| !value.is_finite()) {
                return Err(ToposRoutePolicyError::NonFiniteMetric { index, field });
            }
            if value.is_some_and(|value| value < 0.0) {
                return Err(ToposRoutePolicyError::NegativeMetric { index, field });
            }
        }
        if row
            .context_weight
            .is_some_and(|value| value > TOPOS_ROUTE_POLICY_CONTEXT_WEIGHT_MAX)
        {
            return Err(ToposRoutePolicyError::ContextWeightOutOfRange { index });
        }

        row.selection_evidence = ToposRoutePolicyProfile::ALL
            .into_iter()
            .map(|profile| (profile.as_str().to_owned(), score_row(row, profile)))
            .collect();
        row.selection_scores = row
            .selection_evidence
            .iter()
            .map(|(profile, evidence)| (profile.clone(), evidence.score))
            .collect();
    }
    Ok(rows)
}

fn winner_from_row(
    row: &ToposRoutePolicyRow,
    profile: ToposRoutePolicyProfile,
) -> ToposRoutePolicyProfileWinner {
    ToposRoutePolicyProfileWinner {
        label: Some(row.label.clone()),
        score: row.selection_scores[profile.as_str()],
        trace_route_score: finite_or_zero(row.trace_route_score),
        trace_quality_score: finite_or_zero(row.trace_quality_score),
        trace_efficiency_score: finite_or_zero(row.trace_efficiency_score),
        response_text_quality_score: finite_or_zero(row.response_text_quality_score),
        response_prompt_coverage: finite_or_zero(row.response_prompt_coverage),
        response_completion_rate: finite_or_zero(row.response_completion_rate),
        latency_ms_mean: finite_or_zero(row.latency_ms_mean),
        total_tokens: finite_or_zero(row.total_tokens),
        adapter_runtime_route_score: finite_or_zero(row.adapter_runtime_route_score),
        adapter_guard_score: finite_or_zero(row.adapter_guard_score),
        adapter_context_score: finite_or_zero(row.adapter_context_score),
        closure_pressure: row.closure_pressure,
        openness: row.openness,
        context_weight: row.context_weight,
        score_evidence: row.selection_evidence.get(profile.as_str()).cloned(),
    }
}

fn candidate_is_better(
    candidate: &ToposRoutePolicyRow,
    current: &ToposRoutePolicyRow,
    profile: ToposRoutePolicyProfile,
) -> bool {
    let candidate_score = candidate.selection_scores[profile.as_str()];
    let current_score = current.selection_scores[profile.as_str()];
    candidate_score
        .total_cmp(&current_score)
        .then_with(|| {
            finite_or_zero(candidate.trace_route_score)
                .total_cmp(&finite_or_zero(current.trace_route_score))
        })
        .then_with(|| candidate.label.cmp(&current.label))
        .is_gt()
}

/// Evaluate every profile and choose deterministic winners for route rows.
pub fn evaluate_topos_route_policy(
    request: ToposRoutePolicyEvaluationRequest,
) -> Result<ToposRoutePolicyEvaluationPayload, ToposRoutePolicyError> {
    let rows = prepare_rows(request.rows)?;

    let mut profiles = BTreeMap::new();
    for profile in ToposRoutePolicyProfile::ALL {
        let mut winner: Option<&ToposRoutePolicyRow> = None;
        for row in rows.iter().filter(|row| row.count > 0) {
            if winner.is_none_or(|current| candidate_is_better(row, current, profile)) {
                winner = Some(row);
            }
        }
        profiles.insert(
            profile.as_str().to_owned(),
            winner
                .map(|row| winner_from_row(row, profile))
                .unwrap_or_default(),
        );
    }

    Ok(ToposRoutePolicyEvaluationPayload {
        kind: TOPOS_ROUTE_POLICY_KIND,
        contract_version: TOPOS_ROUTE_POLICY_CONTRACT_VERSION,
        semantic_owner: TOPOS_ROUTE_POLICY_SEMANTIC_OWNER,
        semantic_backend: TOPOS_ROUTE_POLICY_SEMANTIC_BACKEND,
        row_count: rows.len(),
        active_row_count: rows.iter().filter(|row| row.count > 0).count(),
        score_formula_version: TOPOS_ROUTE_POLICY_SCORE_FORMULA_VERSION,
        score_formula: TOPOS_ROUTE_POLICY_SCORE_FORMULA,
        score_prior_mean: TOPOS_ROUTE_POLICY_PRIOR_MEAN,
        score_prior_strength: TOPOS_ROUTE_POLICY_PRIOR_STRENGTH,
        rows,
        profiles,
    })
}

/// Project evaluated route rows into bounded rewards for one policy profile.
pub fn build_topos_route_rewards(
    request: ToposRouteRewardsRequest,
) -> Result<ToposRouteRewardsPayload, ToposRoutePolicyError> {
    let profile = request.profile;
    let rows = prepare_rows(request.rows)?;
    let input_row_count = rows.len();
    let active_row_count = rows.iter().filter(|row| row.count > 0).count();
    let mut rewards = Vec::with_capacity(active_row_count);
    for (source_index, row) in rows.iter().enumerate() {
        if row.count == 0 {
            continue;
        }
        let score_evidence = row.selection_evidence[profile.as_str()].clone();
        let mut source_row = row.clone();
        source_row.selection_scores.clear();
        source_row.selection_evidence.clear();
        rewards.push(ToposRouteReward {
            index: u32::try_from(rewards.len())
                .expect("validated route reward count fits the wire contract"),
            source_index: u32::try_from(source_index)
                .expect("validated route row count fits the wire contract"),
            label: row.label.clone(),
            profile,
            reward: score_evidence.score,
            count: row.count,
            trace_route_score: finite_or_zero(row.trace_route_score),
            response_text_quality_score: finite_or_zero(row.response_text_quality_score),
            response_completion_rate: finite_or_zero(row.response_completion_rate),
            response_incomplete_rate: finite_or_zero(row.response_incomplete_rate),
            adapter_runtime_route_score: finite_or_zero(row.adapter_runtime_route_score),
            score_evidence,
            source_row,
        });
    }

    Ok(ToposRouteRewardsPayload {
        kind: TOPOS_ROUTE_REWARDS_KIND,
        contract_version: TOPOS_ROUTE_POLICY_CONTRACT_VERSION,
        semantic_owner: TOPOS_ROUTE_POLICY_SEMANTIC_OWNER,
        semantic_backend: TOPOS_ROUTE_POLICY_SEMANTIC_BACKEND,
        profile,
        input_row_count,
        active_row_count,
        inactive_row_count: input_row_count - active_row_count,
        reward_count: rewards.len(),
        score_formula_version: TOPOS_ROUTE_POLICY_SCORE_FORMULA_VERSION,
        score_formula: TOPOS_ROUTE_POLICY_SCORE_FORMULA,
        score_prior_mean: TOPOS_ROUTE_POLICY_PRIOR_MEAN,
        score_prior_strength: TOPOS_ROUTE_POLICY_PRIOR_STRENGTH,
        rewards,
    })
}

fn validate_rewards(rewards: &[ToposRouteReward]) -> Result<(), ToposRoutePolicyError> {
    if rewards.len() > TOPOS_ROUTE_POLICY_MAX_ROWS {
        return Err(ToposRoutePolicyError::TooManyRewards {
            actual: rewards.len(),
            max: TOPOS_ROUTE_POLICY_MAX_ROWS,
        });
    }
    let mut labels = BTreeSet::new();
    let mut expected_profile = None;
    let mut previous_source_index = None;
    for (index, reward) in rewards.iter().enumerate() {
        let expected_index =
            u32::try_from(index).expect("validated route reward count fits the wire contract");
        if reward.index != expected_index {
            return Err(ToposRoutePolicyError::RewardIndexMismatch {
                position: index,
                actual: reward.index,
                expected: expected_index,
            });
        }
        if usize::try_from(reward.source_index)
            .map(|source_index| source_index >= TOPOS_ROUTE_POLICY_MAX_ROWS)
            .unwrap_or(true)
        {
            return Err(ToposRoutePolicyError::RewardEvidenceMismatch {
                index,
                field: "source_index_range",
            });
        }
        if previous_source_index.is_some_and(|previous| reward.source_index <= previous) {
            return Err(ToposRoutePolicyError::RewardEvidenceMismatch {
                index,
                field: "source_index_order",
            });
        }
        previous_source_index = Some(reward.source_index);
        if let Some(profile) = expected_profile {
            if reward.profile != profile {
                return Err(ToposRoutePolicyError::MixedRewardProfiles {
                    index,
                    expected: profile.as_str(),
                    actual: reward.profile.as_str(),
                });
            }
        } else {
            expected_profile = Some(reward.profile);
        }
        if reward.label.trim().is_empty() {
            return Err(ToposRoutePolicyError::EmptyRewardLabel { index });
        }
        if reward.label.len() > TOPOS_ROUTE_POLICY_MAX_LABEL_BYTES {
            return Err(ToposRoutePolicyError::RewardLabelTooLong {
                index,
                actual: reward.label.len(),
                max: TOPOS_ROUTE_POLICY_MAX_LABEL_BYTES,
            });
        }
        if reward.source_row.label.trim().is_empty()
            || reward.score_evidence.formula_version.trim().is_empty()
        {
            return Err(ToposRoutePolicyError::MissingRewardWitness { index });
        }
        if reward.count == 0 {
            return Err(ToposRoutePolicyError::InactiveReward { index });
        }
        if !labels.insert(reward.label.clone()) {
            return Err(ToposRoutePolicyError::DuplicateLabel {
                label: reward.label.clone(),
            });
        }
        for (field, value) in [
            ("reward", reward.reward),
            ("trace_route_score", reward.trace_route_score),
            (
                "response_text_quality_score",
                reward.response_text_quality_score,
            ),
            ("response_completion_rate", reward.response_completion_rate),
            ("response_incomplete_rate", reward.response_incomplete_rate),
            (
                "adapter_runtime_route_score",
                reward.adapter_runtime_route_score,
            ),
            ("raw_score", reward.score_evidence.raw_score),
            ("score", reward.score_evidence.score),
            ("evidence_coverage", reward.score_evidence.evidence_coverage),
            ("sample_confidence", reward.score_evidence.sample_confidence),
        ] {
            if !value.is_finite() {
                return Err(ToposRoutePolicyError::NonFiniteReward { index, field });
            }
        }
        if !(0.0..=1.0).contains(&reward.reward) {
            return Err(ToposRoutePolicyError::RewardOutOfRange { index });
        }
        for (field, value) in [
            ("raw_score", reward.score_evidence.raw_score),
            ("score", reward.score_evidence.score),
            ("evidence_coverage", reward.score_evidence.evidence_coverage),
            ("sample_confidence", reward.score_evidence.sample_confidence),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(ToposRoutePolicyError::RewardEvidenceMismatch { index, field });
            }
        }

        let mut source_rows = prepare_rows(vec![reward.source_row.clone()])?;
        let source = source_rows
            .pop()
            .expect("one validated reward source row is retained");
        let expected_evidence = &source.selection_evidence[reward.profile.as_str()];
        let same_float = |actual: f64, expected: f64| (actual - expected).abs() <= 1e-12;
        for (field, matches) in [
            ("source_row.label", reward.source_row.label == source.label),
            ("source_row.mode", reward.source_row.mode == source.mode),
            ("label", reward.label == source.label),
            ("count", reward.count == source.count),
            (
                "trace_route_score",
                same_float(
                    reward.trace_route_score,
                    finite_or_zero(source.trace_route_score),
                ),
            ),
            (
                "response_text_quality_score",
                same_float(
                    reward.response_text_quality_score,
                    finite_or_zero(source.response_text_quality_score),
                ),
            ),
            (
                "response_completion_rate",
                same_float(
                    reward.response_completion_rate,
                    finite_or_zero(source.response_completion_rate),
                ),
            ),
            (
                "response_incomplete_rate",
                same_float(
                    reward.response_incomplete_rate,
                    finite_or_zero(source.response_incomplete_rate),
                ),
            ),
            (
                "adapter_runtime_route_score",
                same_float(
                    reward.adapter_runtime_route_score,
                    finite_or_zero(source.adapter_runtime_route_score),
                ),
            ),
            (
                "score_evidence.profile",
                reward.score_evidence.profile == reward.profile,
            ),
            (
                "score_evidence.formula_version",
                reward.score_evidence.formula_version == TOPOS_ROUTE_POLICY_SCORE_FORMULA_VERSION,
            ),
            (
                "score_evidence.raw_score",
                same_float(reward.score_evidence.raw_score, expected_evidence.raw_score),
            ),
            (
                "score_evidence.score",
                same_float(reward.score_evidence.score, expected_evidence.score),
            ),
            (
                "score_evidence.evidence_coverage",
                same_float(
                    reward.score_evidence.evidence_coverage,
                    expected_evidence.evidence_coverage,
                ),
            ),
            (
                "score_evidence.sample_count",
                reward.score_evidence.sample_count == expected_evidence.sample_count,
            ),
            (
                "score_evidence.sample_confidence",
                same_float(
                    reward.score_evidence.sample_confidence,
                    expected_evidence.sample_confidence,
                ),
            ),
            (
                "score_evidence.observed_metrics",
                reward.score_evidence.observed_metrics == expected_evidence.observed_metrics,
            ),
            (
                "score_evidence.missing_metrics",
                reward.score_evidence.missing_metrics == expected_evidence.missing_metrics,
            ),
            ("reward", same_float(reward.reward, expected_evidence.score)),
        ] {
            if !matches {
                return Err(ToposRoutePolicyError::RewardEvidenceMismatch { index, field });
            }
        }
    }
    Ok(())
}

/// Resolve a selected route by stable label first, then positional fallback.
pub fn resolve_topos_route_policy(
    request: ToposRoutePolicyResolveRequest,
) -> Result<ToposRoutePolicyResolutionPayload, ToposRoutePolicyError> {
    validate_rewards(&request.rewards)?;
    if let Some(label) = request.selected_label.as_deref() {
        if label.len() > TOPOS_ROUTE_POLICY_MAX_LABEL_BYTES {
            return Err(ToposRoutePolicyError::SelectedLabelTooLong {
                actual: label.len(),
                max: TOPOS_ROUTE_POLICY_MAX_LABEL_BYTES,
            });
        }
    }

    let selected_label = request
        .selected_label
        .as_deref()
        .map(str::trim)
        .filter(|label| !label.is_empty());
    let (resolution, selected_position) = if let Some(label) = selected_label {
        let position = request
            .rewards
            .iter()
            .position(|reward| reward.label == label)
            .ok_or_else(|| ToposRoutePolicyError::SelectedLabelNotFound {
                label: label.to_owned(),
            })?;
        ("label", Some(position))
    } else if request.rewards.is_empty() {
        ("none", None)
    } else {
        let position = usize::try_from(request.selected_index)
            .ok()
            .filter(|position| *position < request.rewards.len())
            .ok_or(ToposRoutePolicyError::SelectedIndexOutOfRange {
                index: request.selected_index,
                len: request.rewards.len(),
            })?;
        ("index", Some(position))
    };
    let route_reward = selected_position.map(|position| request.rewards[position].clone());

    Ok(ToposRoutePolicyResolutionPayload {
        kind: TOPOS_ROUTE_POLICY_RESOLUTION_KIND,
        contract_version: TOPOS_ROUTE_POLICY_CONTRACT_VERSION,
        semantic_owner: TOPOS_ROUTE_POLICY_SEMANTIC_OWNER,
        semantic_backend: TOPOS_ROUTE_POLICY_SEMANTIC_BACKEND,
        score_formula_version: TOPOS_ROUTE_POLICY_SCORE_FORMULA_VERSION,
        resolution,
        selected_position,
        selected_label: route_reward.as_ref().map(|reward| reward.label.clone()),
        selected_reward: route_reward.as_ref().map(|reward| reward.reward),
        route_reward,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(label: &str) -> ToposRoutePolicyRow {
        ToposRoutePolicyRow {
            label: label.to_owned(),
            count: 1,
            trace_route_score: Some(0.8),
            trace_quality_score: Some(0.9),
            trace_efficiency_score: Some(0.7),
            trace_text_quality_score: Some(0.8),
            response_text_quality_score: Some(0.6),
            response_prompt_coverage: Some(0.5),
            response_completion_rate: Some(1.0),
            response_incomplete_rate: Some(0.25),
            response_confidence: Some(0.4),
            latency_ms_mean: Some(0.0),
            total_tokens: Some(0.0),
            adapter_runtime_route_score: Some(0.75),
            adapter_guard_score: Some(0.85),
            adapter_exploration_score: Some(0.6),
            adapter_context_score: Some(0.65),
            closure_pressure: Some(0.2),
            openness: Some(0.7),
            context_weight: Some(0.9),
            ..ToposRoutePolicyRow::default()
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn profile_scores_are_evidence_aware_and_auditable() {
        let payload = evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
            rows: vec![sample_row("guarded")],
        })
        .expect("evaluate route policy");
        let scores = &payload.rows[0].selection_scores;
        assert_close(scores["quality"], 0.61375);
        assert_close(scores["grounded"], 0.60525);
        assert_close(scores["efficiency"], 0.6875);
        assert_close(scores["latency"], 0.71375);
        assert_close(scores["balanced"], 0.634);
        let balanced = &payload.rows[0].selection_evidence["balanced"];
        assert_close(balanced.raw_score, 0.768);
        assert_close(balanced.evidence_coverage, 1.0);
        assert_close(balanced.sample_confidence, 0.5);
        assert_eq!(balanced.sample_count, 1);
        assert!(balanced.missing_metrics.is_empty());
        assert_eq!(
            balanced.observed_metrics.last().map(String::as_str),
            Some("response_incomplete_rate")
        );
        assert_eq!(payload.semantic_backend, "rust");
        assert_eq!(
            payload.score_formula_version,
            TOPOS_ROUTE_POLICY_SCORE_FORMULA_VERSION
        );
        assert_eq!(payload.active_row_count, 1);
    }

    #[test]
    fn winner_uses_route_score_then_label_and_skips_inactive_rows() {
        let mut inactive = sample_row("zz-inactive");
        inactive.count = 0;
        let payload = evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
            rows: vec![sample_row("alpha"), sample_row("zeta"), inactive],
        })
        .expect("evaluate route policy");
        assert_eq!(payload.profiles["balanced"].label.as_deref(), Some("zeta"));
        assert_eq!(payload.active_row_count, 2);
    }

    #[test]
    fn rewards_recompute_scores_and_exclude_inactive_routes() {
        let mut scored = sample_row("guarded");
        scored.selection_scores.insert("grounded".to_owned(), 0.99);
        let inactive = ToposRoutePolicyRow {
            label: "inactive".to_owned(),
            count: 0,
            selection_scores: BTreeMap::from([("grounded".to_owned(), 1.0)]),
            ..ToposRoutePolicyRow::default()
        };
        let sparse = ToposRoutePolicyRow {
            label: "exploratory".to_owned(),
            count: 2,
            trace_route_score: Some(0.8),
            ..ToposRoutePolicyRow::default()
        };
        let payload = build_topos_route_rewards(ToposRouteRewardsRequest {
            rows: vec![scored, inactive, sparse],
            profile: ToposRoutePolicyProfile::Grounded,
        })
        .expect("build route rewards");
        assert_close(payload.rewards[0].reward, 0.60525);
        assert_close(payload.rewards[1].reward, 0.5);
        assert_eq!(
            payload.rewards[1]
                .score_evidence
                .missing_metrics
                .last()
                .map(String::as_str),
            Some("response_incomplete_rate")
        );
        assert_eq!(payload.rewards[1].index, 1);
        assert_eq!(payload.rewards[1].source_index, 2);
        assert_eq!(payload.input_row_count, 3);
        assert_eq!(payload.active_row_count, 2);
        assert_eq!(payload.inactive_row_count, 1);
        assert_eq!(payload.reward_count, 2);
        assert_eq!(
            payload.rewards[0].score_evidence.formula_version,
            TOPOS_ROUTE_POLICY_SCORE_FORMULA_VERSION
        );
        assert_eq!(payload.rewards[0].source_row.label, "guarded");
        let wire = serde_json::to_value(&payload.rewards[0]).expect("serialize reward");
        assert!(wire["source_row"].get("selection_scores").is_none());
        assert!(wire["source_row"].get("selection_evidence").is_none());
    }

    #[test]
    fn selected_route_resolves_by_label_before_index() {
        let rewards = build_topos_route_rewards(ToposRouteRewardsRequest {
            rows: vec![sample_row("guarded"), sample_row("exploratory")],
            profile: ToposRoutePolicyProfile::Balanced,
        })
        .expect("build route rewards")
        .rewards;
        let by_label = resolve_topos_route_policy(ToposRoutePolicyResolveRequest {
            rewards: rewards.clone(),
            selected_label: Some("exploratory".to_owned()),
            selected_index: 0,
        })
        .expect("resolve by label");
        assert_eq!(by_label.resolution, "label");
        assert_eq!(by_label.selected_position, Some(1));

        let by_index = resolve_topos_route_policy(ToposRoutePolicyResolveRequest {
            rewards: rewards.clone(),
            selected_label: None,
            selected_index: 0,
        })
        .expect("resolve by index");
        assert_eq!(by_index.resolution, "index");
        assert_eq!(by_index.selected_label.as_deref(), Some("guarded"));

        assert_eq!(
            resolve_topos_route_policy(ToposRoutePolicyResolveRequest {
                rewards: rewards.clone(),
                selected_label: Some("missing".to_owned()),
                selected_index: 0,
            }),
            Err(ToposRoutePolicyError::SelectedLabelNotFound {
                label: "missing".to_owned()
            })
        );
        assert_eq!(
            resolve_topos_route_policy(ToposRoutePolicyResolveRequest {
                rewards,
                selected_label: None,
                selected_index: 2,
            }),
            Err(ToposRoutePolicyError::SelectedIndexOutOfRange { index: 2, len: 2 })
        );
    }

    #[test]
    fn rejects_duplicate_labels_unknown_profiles_and_non_finite_metrics() {
        let duplicate = evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
            rows: vec![sample_row("same"), sample_row(" same ")],
        });
        assert_eq!(
            duplicate,
            Err(ToposRoutePolicyError::DuplicateLabel {
                label: "same".to_owned()
            })
        );

        let mut out_of_range = sample_row("out-of-range");
        out_of_range.trace_route_score = Some(1.01);
        assert_eq!(
            evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
                rows: vec![out_of_range]
            }),
            Err(ToposRoutePolicyError::MetricOutOfRange {
                index: 0,
                field: "trace_route_score"
            })
        );

        let unknown = serde_json::from_value::<ToposRouteRewardsRequest>(serde_json::json!({
            "rows": [],
            "profile": "commander"
        }));
        assert!(unknown.is_err());

        let mut invalid = sample_row("invalid");
        invalid.response_confidence = Some(f64::NAN);
        assert_eq!(
            evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
                rows: vec![invalid]
            }),
            Err(ToposRoutePolicyError::NonFiniteMetric {
                index: 0,
                field: "response_confidence"
            })
        );
    }

    #[test]
    fn profile_normalization_is_owned_by_the_rust_ingress() {
        let request: ToposRouteRewardsRequest = serde_json::from_value(serde_json::json!({
            "rows": [],
            "profile": " Grounded "
        }))
        .expect("canonicalize profile");
        assert_eq!(request.profile, ToposRoutePolicyProfile::Grounded);
        assert_eq!(
            serde_json::to_value(request.profile).expect("serialize profile"),
            serde_json::json!("grounded")
        );
    }

    #[test]
    fn context_weight_uses_the_tensor_contract_boundary() {
        let mut at_boundary = sample_row("boundary");
        at_boundary.context_weight = Some(TOPOS_ROUTE_POLICY_CONTEXT_WEIGHT_MAX);
        let payload = evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
            rows: vec![at_boundary],
        })
        .expect("accept the st-tensor context-weight maximum");
        assert_close(payload.rows[0].selection_scores["grounded"], 0.62625);

        let mut above_boundary = sample_row("above-boundary");
        above_boundary.context_weight = Some(TOPOS_ROUTE_POLICY_CONTEXT_WEIGHT_MAX + 0.0001);
        assert_eq!(
            evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
                rows: vec![above_boundary],
            }),
            Err(ToposRoutePolicyError::ContextWeightOutOfRange { index: 0 })
        );
    }

    #[test]
    fn evidence_coverage_includes_the_incomplete_adjustment_input() {
        let mut missing_incomplete = sample_row("missing-incomplete");
        missing_incomplete.response_incomplete_rate = None;
        let payload = evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
            rows: vec![missing_incomplete],
        })
        .expect("evaluate incomplete evidence coverage");
        let evidence = &payload.rows[0].selection_evidence["balanced"];
        assert_close(evidence.evidence_coverage, 1.0 / 1.08);
        assert_eq!(
            evidence.missing_metrics.last().map(String::as_str),
            Some("response_incomplete_rate")
        );
    }

    #[test]
    fn client_score_projections_are_discarded_at_wire_ingress() {
        let request: ToposRoutePolicyEvaluationRequest =
            serde_json::from_value(serde_json::json!({
                "rows": [{
                    "label": "wire",
                    "count": 1,
                    "trace_route_score": 0.8,
                    "selection_scores": {"balanced": 1.0},
                    "selection_evidence": {"balanced": {"score": 1.0}}
                }]
            }))
            .expect("accept compatibility projections without trusting them");
        let payload = evaluate_topos_route_policy(request).expect("evaluate canonical score");
        assert_ne!(payload.rows[0].selection_scores["balanced"], 1.0);
        assert_eq!(
            payload.rows[0].selection_evidence["balanced"].formula_version,
            TOPOS_ROUTE_POLICY_SCORE_FORMULA_VERSION
        );
    }

    #[test]
    fn wire_integers_are_target_independent() {
        let too_wide =
            serde_json::from_value::<ToposRoutePolicyEvaluationRequest>(serde_json::json!({
                "rows": [{"label": "wide", "count": u64::from(u32::MAX) + 1}]
            }));
        assert!(too_wide.is_err());

        let too_wide =
            serde_json::from_value::<ToposRoutePolicyResolveRequest>(serde_json::json!({
                "rewards": [],
                "selected_index": u64::from(u32::MAX) + 1
            }));
        assert!(too_wide.is_err());
    }

    #[test]
    fn resolver_rejects_reward_identity_drift() {
        let rewards = build_topos_route_rewards(ToposRouteRewardsRequest {
            rows: vec![sample_row("guarded"), sample_row("exploratory")],
            profile: ToposRoutePolicyProfile::Balanced,
        })
        .expect("build route rewards")
        .rewards;

        let mut wrong_index = rewards.clone();
        wrong_index[1].index = 0;
        assert_eq!(
            resolve_topos_route_policy(ToposRoutePolicyResolveRequest {
                rewards: wrong_index,
                selected_label: None,
                selected_index: 0,
            }),
            Err(ToposRoutePolicyError::RewardIndexMismatch {
                position: 1,
                actual: 0,
                expected: 1,
            })
        );

        let mut tampered = rewards.clone();
        tampered[0].source_row.response_text_quality_score = Some(0.0);
        assert!(matches!(
            resolve_topos_route_policy(ToposRoutePolicyResolveRequest {
                rewards: tampered,
                selected_label: None,
                selected_index: 0,
            }),
            Err(ToposRoutePolicyError::RewardEvidenceMismatch { index: 0, .. })
        ));

        let mut impossible_source_index = rewards.clone();
        impossible_source_index[0].source_index = TOPOS_ROUTE_POLICY_MAX_ROWS as u32;
        assert_eq!(
            resolve_topos_route_policy(ToposRoutePolicyResolveRequest {
                rewards: impossible_source_index,
                selected_label: None,
                selected_index: 0,
            }),
            Err(ToposRoutePolicyError::RewardEvidenceMismatch {
                index: 0,
                field: "source_index_range",
            })
        );

        let mut mixed_profile = rewards;
        mixed_profile[1].profile = ToposRoutePolicyProfile::Grounded;
        assert_eq!(
            resolve_topos_route_policy(ToposRoutePolicyResolveRequest {
                rewards: mixed_profile,
                selected_label: None,
                selected_index: 0,
            }),
            Err(ToposRoutePolicyError::MixedRewardProfiles {
                index: 1,
                expected: "balanced",
                actual: "grounded",
            })
        );
    }

    #[test]
    fn resolver_explains_legacy_rewards_without_v2_witnesses() {
        let legacy_reward: ToposRouteReward = serde_json::from_value(serde_json::json!({
            "index": 0,
            "label": "guarded",
            "profile": "grounded",
            "reward": 0.7,
            "count": 1
        }))
        .expect("decode a legacy-shaped reward for migration diagnostics");
        assert_eq!(
            resolve_topos_route_policy(ToposRoutePolicyResolveRequest {
                rewards: vec![legacy_reward],
                selected_label: Some("guarded".to_owned()),
                selected_index: 0,
            }),
            Err(ToposRoutePolicyError::MissingRewardWitness { index: 0 })
        );
    }
}
