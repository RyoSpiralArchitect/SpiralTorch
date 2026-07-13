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
pub const TOPOS_ROUTE_POLICY_CONTRACT_VERSION: &str = "spiraltorch.topos_route_policy.v1";
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

pub const TOPOS_ROUTE_POLICY_MAX_ROWS: usize = 4_096;
pub const TOPOS_ROUTE_POLICY_MAX_LABEL_BYTES: usize = 256;
pub const TOPOS_ROUTE_POLICY_MAX_MODE_BYTES: usize = 256;

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
    #[error("selection score '{profile}' at route row {index} is not a known profile")]
    UnknownScoreProfile { index: usize, profile: String },
    #[error("selection score '{profile}' at route row {index} must be finite")]
    NonFiniteSelectionScore { index: usize, profile: String },
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
    pub selection_scores: BTreeMap<String, f64>,
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
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToposRouteRewardsPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub profile: ToposRoutePolicyProfile,
    pub input_row_count: usize,
    pub reward_count: usize,
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
    pub resolution: &'static str,
    pub selected_position: Option<usize>,
    pub selected_label: Option<String>,
    pub selected_reward: Option<f64>,
    pub route_reward: Option<ToposRouteReward>,
}

fn finite_or_zero(value: Option<f64>) -> f64 {
    value.unwrap_or(0.0)
}

fn bounded_unit(value: Option<f64>) -> f64 {
    finite_or_zero(value).clamp(0.0, 1.0)
}

fn latency_cost(value: Option<f64>) -> f64 {
    let latency = finite_or_zero(value).max(0.0);
    (latency.ln_1p() / 10_000.0_f64.ln_1p()).min(1.0)
}

fn token_cost(value: Option<f64>) -> f64 {
    let tokens = finite_or_zero(value).max(0.0);
    (tokens.ln_1p() / 4_096.0_f64.ln_1p()).min(1.0)
}

fn score_row(row: &ToposRoutePolicyRow, profile: ToposRoutePolicyProfile) -> f64 {
    let trace_route = bounded_unit(row.trace_route_score);
    let trace_quality = bounded_unit(row.trace_quality_score);
    let trace_efficiency = bounded_unit(row.trace_efficiency_score);
    let response_quality = bounded_unit(row.response_text_quality_score);
    let prompt_coverage = bounded_unit(row.response_prompt_coverage);
    let completion = bounded_unit(row.response_completion_rate);
    let confidence = bounded_unit(row.response_confidence);
    let adapter_route = bounded_unit(row.adapter_runtime_route_score);
    let adapter_guard = bounded_unit(row.adapter_guard_score);
    let adapter_context = bounded_unit(row.adapter_context_score);
    let openness = bounded_unit(row.openness);
    let context_weight = bounded_unit(row.context_weight);
    let closure_reward = 1.0 - bounded_unit(row.closure_pressure);
    let latency_reward = 1.0 - latency_cost(row.latency_ms_mean);
    let token_reward = 1.0 - token_cost(row.total_tokens);
    let incomplete_penalty = 0.08 * bounded_unit(row.response_incomplete_rate);

    let score = match profile {
        ToposRoutePolicyProfile::Quality => {
            0.30 * trace_quality
                + 0.30 * response_quality
                + 0.15 * prompt_coverage
                + 0.10 * confidence
                + 0.10 * completion
                + 0.05 * adapter_route
        }
        ToposRoutePolicyProfile::Grounded => {
            0.30 * response_quality
                + 0.20 * prompt_coverage
                + 0.20 * context_weight
                + 0.15 * adapter_guard
                + 0.10 * completion
                + 0.05 * adapter_context
        }
        ToposRoutePolicyProfile::Efficiency => {
            0.25 * trace_efficiency
                + 0.20 * latency_reward
                + 0.20 * token_reward
                + 0.15 * response_quality
                + 0.10 * completion
                + 0.10 * adapter_route
        }
        ToposRoutePolicyProfile::Latency => {
            0.40 * latency_reward
                + 0.20 * token_reward
                + 0.15 * response_quality
                + 0.10 * completion
                + 0.10 * trace_route
                + 0.05 * adapter_route
        }
        ToposRoutePolicyProfile::Balanced => {
            0.25 * trace_route
                + 0.20 * adapter_route
                + 0.20 * response_quality
                + 0.10 * completion
                + 0.10 * trace_efficiency
                + 0.10 * closure_reward
                + 0.05 * openness
        }
    };

    (score - incomplete_penalty).clamp(0.0, 1.0)
}

fn validate_rows(rows: &[ToposRoutePolicyRow]) -> Result<(), ToposRoutePolicyError> {
    if rows.len() > TOPOS_ROUTE_POLICY_MAX_ROWS {
        return Err(ToposRoutePolicyError::TooManyRows {
            actual: rows.len(),
            max: TOPOS_ROUTE_POLICY_MAX_ROWS,
        });
    }

    let mut labels = BTreeSet::new();
    for (index, row) in rows.iter().enumerate() {
        if row.label.trim().is_empty() {
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
        if let Some(mode) = &row.mode {
            if mode.len() > TOPOS_ROUTE_POLICY_MAX_MODE_BYTES {
                return Err(ToposRoutePolicyError::ModeTooLong {
                    index,
                    actual: mode.len(),
                    max: TOPOS_ROUTE_POLICY_MAX_MODE_BYTES,
                });
            }
        }

        for (field, value) in [
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
            ("latency_ms_mean", row.latency_ms_mean),
            ("total_tokens", row.total_tokens),
            (
                "adapter_runtime_route_score",
                row.adapter_runtime_route_score,
            ),
            ("adapter_guard_score", row.adapter_guard_score),
            ("adapter_exploration_score", row.adapter_exploration_score),
            ("adapter_context_score", row.adapter_context_score),
            ("closure_pressure", row.closure_pressure),
            ("openness", row.openness),
            ("context_weight", row.context_weight),
            ("request_temperature", row.request_temperature),
        ] {
            if value.is_some_and(|value| !value.is_finite()) {
                return Err(ToposRoutePolicyError::NonFiniteMetric { index, field });
            }
        }

        for (profile, score) in &row.selection_scores {
            if ToposRoutePolicyProfile::from_str(profile).is_err() {
                return Err(ToposRoutePolicyError::UnknownScoreProfile {
                    index,
                    profile: profile.clone(),
                });
            }
            if !score.is_finite() {
                return Err(ToposRoutePolicyError::NonFiniteSelectionScore {
                    index,
                    profile: profile.clone(),
                });
            }
        }
    }
    Ok(())
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
    validate_rows(&request.rows)?;
    let mut rows = request.rows;
    for row in &mut rows {
        row.selection_scores = ToposRoutePolicyProfile::ALL
            .into_iter()
            .map(|profile| (profile.as_str().to_owned(), score_row(row, profile)))
            .collect();
    }

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
        rows,
        profiles,
    })
}

/// Project evaluated route rows into bounded rewards for one policy profile.
pub fn build_topos_route_rewards(
    request: ToposRouteRewardsRequest,
) -> Result<ToposRouteRewardsPayload, ToposRoutePolicyError> {
    validate_rows(&request.rows)?;
    let profile = request.profile;
    let mut rewards = Vec::with_capacity(request.rows.len());
    for (source_index, row) in request.rows.iter().enumerate() {
        let reward = row
            .selection_scores
            .get(profile.as_str())
            .copied()
            .unwrap_or_else(|| finite_or_zero(row.trace_route_score))
            .clamp(0.0, 1.0);
        rewards.push(ToposRouteReward {
            index: u32::try_from(rewards.len())
                .expect("validated route reward count fits the wire contract"),
            source_index: u32::try_from(source_index)
                .expect("validated route row count fits the wire contract"),
            label: row.label.clone(),
            profile,
            reward,
            count: row.count,
            trace_route_score: finite_or_zero(row.trace_route_score),
            response_text_quality_score: finite_or_zero(row.response_text_quality_score),
            response_completion_rate: finite_or_zero(row.response_completion_rate),
            response_incomplete_rate: finite_or_zero(row.response_incomplete_rate),
            adapter_runtime_route_score: finite_or_zero(row.adapter_runtime_route_score),
        });
    }

    Ok(ToposRouteRewardsPayload {
        kind: TOPOS_ROUTE_REWARDS_KIND,
        contract_version: TOPOS_ROUTE_POLICY_CONTRACT_VERSION,
        semantic_owner: TOPOS_ROUTE_POLICY_SEMANTIC_OWNER,
        semantic_backend: TOPOS_ROUTE_POLICY_SEMANTIC_BACKEND,
        profile,
        input_row_count: request.rows.len(),
        reward_count: rewards.len(),
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
        ] {
            if !value.is_finite() {
                return Err(ToposRoutePolicyError::NonFiniteReward { index, field });
            }
        }
        if !(0.0..=1.0).contains(&reward.reward) {
            return Err(ToposRoutePolicyError::RewardOutOfRange { index });
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

    let by_label = request
        .selected_label
        .as_deref()
        .filter(|label| !label.trim().is_empty())
        .and_then(|label| {
            request
                .rewards
                .iter()
                .position(|reward| reward.label == label)
        });
    let selected_index = usize::try_from(request.selected_index).ok();
    let (resolution, selected_position) = if let Some(position) = by_label {
        ("label", Some(position))
    } else if let Some(position) =
        selected_index.filter(|position| *position < request.rewards.len())
    {
        ("index", Some(position))
    } else {
        ("none", None)
    };
    let route_reward = selected_position.map(|position| request.rewards[position].clone());

    Ok(ToposRoutePolicyResolutionPayload {
        kind: TOPOS_ROUTE_POLICY_RESOLUTION_KIND,
        contract_version: TOPOS_ROUTE_POLICY_CONTRACT_VERSION,
        semantic_owner: TOPOS_ROUTE_POLICY_SEMANTIC_OWNER,
        semantic_backend: TOPOS_ROUTE_POLICY_SEMANTIC_BACKEND,
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
            response_text_quality_score: Some(0.6),
            response_prompt_coverage: Some(0.5),
            response_completion_rate: Some(1.0),
            response_incomplete_rate: Some(0.25),
            response_confidence: Some(0.4),
            latency_ms_mean: Some(0.0),
            total_tokens: Some(0.0),
            adapter_runtime_route_score: Some(0.75),
            adapter_guard_score: Some(0.85),
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
    fn profile_scores_match_the_previous_python_contract() {
        let payload = evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
            rows: vec![sample_row("guarded")],
        })
        .expect("evaluate route policy");
        let scores = &payload.rows[0].selection_scores;
        assert_close(scores["quality"], 0.6825);
        assert_close(scores["grounded"], 0.70);
        assert_close(scores["efficiency"], 0.82);
        assert_close(scores["latency"], 0.8875);
        assert_close(scores["balanced"], 0.735);
        assert_eq!(payload.semantic_backend, "rust");
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
    fn rewards_prefer_profile_score_and_fall_back_to_route_score() {
        let mut scored = sample_row("guarded");
        scored.selection_scores.insert("grounded".to_owned(), 0.91);
        let fallback = ToposRoutePolicyRow {
            label: "exploratory".to_owned(),
            count: 2,
            trace_route_score: Some(1.2),
            ..ToposRoutePolicyRow::default()
        };
        let payload = build_topos_route_rewards(ToposRouteRewardsRequest {
            rows: vec![scored, fallback],
            profile: ToposRoutePolicyProfile::Grounded,
        })
        .expect("build route rewards");
        assert_eq!(payload.rewards[0].reward, 0.91);
        assert_eq!(payload.rewards[1].reward, 1.0);
        assert_eq!(payload.rewards[1].index, 1);
        assert_eq!(payload.rewards[1].source_index, 1);
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
            rewards,
            selected_label: Some("missing".to_owned()),
            selected_index: 0,
        })
        .expect("resolve by index");
        assert_eq!(by_index.resolution, "index");
        assert_eq!(by_index.selected_label.as_deref(), Some("guarded"));
    }

    #[test]
    fn rejects_duplicate_labels_unknown_profiles_and_non_finite_metrics() {
        let duplicate = evaluate_topos_route_policy(ToposRoutePolicyEvaluationRequest {
            rows: vec![sample_row("same"), sample_row("same")],
        });
        assert_eq!(
            duplicate,
            Err(ToposRoutePolicyError::DuplicateLabel {
                label: "same".to_owned()
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
}
