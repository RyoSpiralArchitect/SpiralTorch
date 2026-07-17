//! Canonical API-LLM trace scoring and route-selection semantics.
//!
//! Host clients gather trace evidence and render reports. Rust owns evidence
//! validation, normalization, score formulas, deterministic ranking, profile
//! winners, and near-best membership.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::route_selection::RouteSelectionProfile;

pub const API_LLM_ROUTE_POLICY_CONTRACT_VERSION: &str = "spiraltorch.api_llm_route_policy.v1";
pub const API_LLM_ROUTE_POLICY_KIND: &str = "spiraltorch.api_llm_route_policy";
pub const API_LLM_ROUTE_POLICY_SEMANTIC_OWNER: &str = "st-core::runtime::api_llm_route_policy";
pub const API_LLM_ROUTE_POLICY_SEMANTIC_BACKEND: &str = "rust";
pub const API_LLM_ROUTE_POLICY_SCORE_FORMULA_VERSION: &str =
    "spiraltorch.api_llm_route_policy.score.v1";
pub const API_LLM_ROUTE_POLICY_SCORE_FORMULA: &str =
    "raw=clamp(weighted_evidence-health_penalty,0,1);coverage=observed_weight/total_weight;sample=n/(n+1);score=0.5+(sample*coverage)*(raw-0.5);token_cost=log1p(total_tokens/max(n,1))/log1p(4096);missing_cost=0.5;count=0_inactive";
pub const API_LLM_ROUTE_POLICY_PRIOR_MEAN: f64 = 0.5;
pub const API_LLM_ROUTE_POLICY_PRIOR_STRENGTH: f64 = 1.0;
pub const API_LLM_ROUTE_POLICY_MAX_ROWS: usize = 4_096;
pub const API_LLM_ROUTE_POLICY_MAX_LABEL_BYTES: usize = 256;

fn discard_client_projection<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default,
{
    let _ = serde::de::IgnoredAny::deserialize(deserializer)?;
    Ok(T::default())
}

#[derive(Debug, Error, PartialEq)]
pub enum ApiLlmRoutePolicyError {
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
    #[error("route label '{label}' appears more than once")]
    DuplicateLabel { label: String },
    #[error("metric '{field}' at route row {index} must be finite")]
    NonFiniteMetric { index: usize, field: &'static str },
    #[error("metric '{field}' at route row {index} must be in the inclusive range [0, 1]")]
    MetricOutOfRange { index: usize, field: &'static str },
    #[error("metric '{field}' at route row {index} must be non-negative")]
    NegativeMetric { index: usize, field: &'static str },
    #[error("near_best_tolerance must be finite and in the inclusive range [0, 1]")]
    InvalidNearBestTolerance,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApiLlmRoutePolicyRow {
    pub label: String,
    pub count: u32,
    pub runtime_ready_rate: Option<f64>,
    pub completion_rate: Option<f64>,
    pub incomplete_rate: Option<f64>,
    pub empty_text_rate: Option<f64>,
    pub refusal_rate: Option<f64>,
    pub total_tokens: Option<f64>,
    pub latency_ms_mean: Option<f64>,
    pub confidence_mean: Option<f64>,
    pub text_quality_score: Option<f64>,
    pub stability_mean: Option<f64>,
    pub frac_mean: Option<f64>,
    pub wasm_loss_mean: Option<f64>,
    pub wasm_stability_hint_mean: Option<f64>,
    pub wasm_webgpu_device_ready_rate: Option<f64>,
    pub topos_context_observed_rate: Option<f64>,
    pub topos_closure_pressure_mean: Option<f64>,
    pub topos_openness_mean: Option<f64>,
    pub topos_training_gradient_bias_scale_mean: Option<f64>,
    pub topos_optimizer_effective_gradient_bias_scale_mean: Option<f64>,
    pub topos_training_clip_scale_mean: Option<f64>,
    pub topos_training_plan_rate_scale_mean: Option<f64>,
    pub topos_optimizer_rate_scale_mean: Option<f64>,
    pub topos_optimizer_raw_rate_scale_mean: Option<f64>,
    pub topos_inference_context_weight_mean: Option<f64>,
    pub topos_inference_plan_context_weight_mean: Option<f64>,
    pub topos_inference_plan_temperature_mean: Option<f64>,
    pub topos_runtime_control_energy_mean: Option<f64>,
    pub topos_runtime_closure_risk_mean: Option<f64>,
    pub topos_runtime_exploration_budget_mean: Option<f64>,
    pub topos_runtime_route_score_mean: Option<f64>,
    pub topos_runtime_route_guard_score_mean: Option<f64>,
    pub topos_runtime_route_exploration_score_mean: Option<f64>,
    pub topos_runtime_route_context_score_mean: Option<f64>,
    #[serde(default, deserialize_with = "discard_client_projection")]
    pub quality_score: f64,
    #[serde(default, deserialize_with = "discard_client_projection")]
    pub latency_cost: f64,
    #[serde(default, deserialize_with = "discard_client_projection")]
    pub token_cost: f64,
    #[serde(default, deserialize_with = "discard_client_projection")]
    pub health_penalty: f64,
    #[serde(default, deserialize_with = "discard_client_projection")]
    pub efficiency_score: f64,
    #[serde(default, deserialize_with = "discard_client_projection")]
    pub route_score: f64,
    #[serde(
        default,
        deserialize_with = "discard_client_projection",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub selection_scores: BTreeMap<String, f64>,
    #[serde(
        default,
        deserialize_with = "discard_client_projection",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub selection_evidence: BTreeMap<String, ApiLlmRoutePolicyScoreEvidence>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ApiLlmRoutePolicyScoreEvidence {
    pub profile: RouteSelectionProfile,
    pub formula_version: String,
    pub raw_score: f64,
    pub score: f64,
    pub evidence_coverage: f64,
    pub sample_count: u32,
    pub sample_confidence: f64,
    pub effective_confidence: f64,
    pub observed_metrics: Vec<String>,
    pub partial_metrics: Vec<String>,
    pub missing_metrics: Vec<String>,
    pub metric_coverage: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ApiLlmRoutePolicyEvaluationRequest {
    pub rows: Vec<ApiLlmRoutePolicyRow>,
    #[serde(default)]
    pub near_best_tolerance: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ApiLlmRoutePolicyProfileWinner {
    pub label: Option<String>,
    pub score: f64,
    pub route_score: f64,
    pub quality_score: f64,
    pub text_quality_score: Option<f64>,
    pub efficiency_score: f64,
    pub latency_ms_mean: Option<f64>,
    pub total_tokens: Option<f64>,
    pub completion_rate: Option<f64>,
    pub wasm_loss_mean: Option<f64>,
    pub wasm_webgpu_device_ready_rate: Option<f64>,
    pub score_evidence: Option<ApiLlmRoutePolicyScoreEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ApiLlmRoutePolicyNearBest {
    pub label: String,
    pub route_score: f64,
    pub route_score_delta: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ApiLlmRoutePolicyEvaluationPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub row_count: usize,
    pub active_row_count: usize,
    pub inactive_row_count: usize,
    pub near_best_tolerance: f64,
    pub score_formula_version: &'static str,
    pub score_formula: &'static str,
    pub score_prior_mean: f64,
    pub score_prior_strength: f64,
    pub rows: Vec<ApiLlmRoutePolicyRow>,
    pub ranked_labels: Vec<String>,
    pub near_best: Vec<ApiLlmRoutePolicyNearBest>,
    pub profiles: BTreeMap<String, ApiLlmRoutePolicyProfileWinner>,
    pub winners: BTreeMap<String, Option<String>>,
}

#[derive(Clone, Copy)]
struct EvidenceMetric {
    name: &'static str,
    value: f64,
    weight: f64,
    coverage: f64,
    adverse: bool,
}

impl EvidenceMetric {
    fn reward(name: &'static str, value: Option<f64>, weight: f64) -> Self {
        Self {
            name,
            value: value.unwrap_or(API_LLM_ROUTE_POLICY_PRIOR_MEAN),
            weight,
            coverage: if value.is_some() { 1.0 } else { 0.0 },
            adverse: false,
        }
    }

    fn derived(name: &'static str, value: f64, weight: f64, coverage: f64) -> Self {
        Self {
            name,
            value,
            weight,
            coverage,
            adverse: false,
        }
    }

    fn adverse(name: &'static str, value: Option<f64>, weight: f64) -> Self {
        Self {
            name,
            value: value.unwrap_or(0.0),
            weight,
            coverage: if value.is_some() { 1.0 } else { 0.0 },
            adverse: true,
        }
    }
}

#[derive(Clone, Copy)]
struct DerivedScore {
    raw_score: f64,
    score: f64,
    coverage: f64,
}

fn sample_confidence(count: u32) -> f64 {
    let count = f64::from(count);
    count / (count + API_LLM_ROUTE_POLICY_PRIOR_STRENGTH)
}

fn score_metrics(
    metrics: &[EvidenceMetric],
    count: u32,
    profile: RouteSelectionProfile,
) -> ApiLlmRoutePolicyScoreEvidence {
    let total_weight = metrics.iter().map(|metric| metric.weight).sum::<f64>();
    let observed_weight = metrics
        .iter()
        .map(|metric| metric.weight * metric.coverage)
        .sum::<f64>();
    let raw_score = metrics
        .iter()
        .map(|metric| {
            let signed = if metric.adverse { -1.0 } else { 1.0 };
            signed * metric.weight * metric.value
        })
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let evidence_coverage = if total_weight > 0.0 {
        (observed_weight / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let sample_confidence = sample_confidence(count);
    let effective_confidence = sample_confidence * evidence_coverage;
    let score = (API_LLM_ROUTE_POLICY_PRIOR_MEAN
        + effective_confidence * (raw_score - API_LLM_ROUTE_POLICY_PRIOR_MEAN))
        .clamp(0.0, 1.0);
    let metric_coverage = metrics
        .iter()
        .map(|metric| (metric.name.to_owned(), metric.coverage))
        .collect::<BTreeMap<_, _>>();
    let observed_metrics = metrics
        .iter()
        .filter(|metric| metric.coverage >= 1.0)
        .map(|metric| metric.name.to_owned())
        .collect();
    let partial_metrics = metrics
        .iter()
        .filter(|metric| metric.coverage > 0.0 && metric.coverage < 1.0)
        .map(|metric| metric.name.to_owned())
        .collect();
    let missing_metrics = metrics
        .iter()
        .filter(|metric| metric.coverage <= 0.0)
        .map(|metric| metric.name.to_owned())
        .collect();

    ApiLlmRoutePolicyScoreEvidence {
        profile,
        formula_version: API_LLM_ROUTE_POLICY_SCORE_FORMULA_VERSION.to_owned(),
        raw_score,
        score,
        evidence_coverage,
        sample_count: count,
        sample_confidence,
        effective_confidence,
        observed_metrics,
        partial_metrics,
        missing_metrics,
        metric_coverage,
    }
}

fn derived_score(metrics: &[EvidenceMetric], count: u32) -> DerivedScore {
    let evidence = score_metrics(metrics, count, RouteSelectionProfile::Balanced);
    DerivedScore {
        raw_score: evidence.raw_score,
        score: evidence.score,
        coverage: evidence.evidence_coverage,
    }
}

fn normalized_cost(value: Option<f64>, ceiling: f64) -> f64 {
    value
        .map(|value| (value.ln_1p() / ceiling.ln_1p()).min(1.0))
        .unwrap_or(API_LLM_ROUTE_POLICY_PRIOR_MEAN)
}

fn normalized_token_cost(total_tokens: Option<f64>, count: u32) -> f64 {
    normalized_cost(
        total_tokens.map(|total| total / f64::from(count.max(1))),
        4_096.0,
    )
}

fn health_metrics(row: &ApiLlmRoutePolicyRow) -> [EvidenceMetric; 3] {
    [
        EvidenceMetric::adverse("empty_text_rate", row.empty_text_rate, 0.35),
        EvidenceMetric::adverse("refusal_rate", row.refusal_rate, 0.25),
        EvidenceMetric::adverse("incomplete_rate", row.incomplete_rate, 0.08),
    ]
}

fn quality_metrics(row: &ApiLlmRoutePolicyRow) -> [EvidenceMetric; 3] {
    [
        EvidenceMetric::reward("confidence_mean", row.confidence_mean, 0.50),
        EvidenceMetric::reward("stability_mean", row.stability_mean, 0.30),
        EvidenceMetric::reward("frac_mean", row.frac_mean, 0.20),
    ]
}

fn profile_metrics(
    row: &ApiLlmRoutePolicyRow,
    quality: DerivedScore,
    profile: RouteSelectionProfile,
) -> Vec<EvidenceMetric> {
    let latency_coverage = if row.latency_ms_mean.is_some() {
        1.0
    } else {
        0.0
    };
    let token_coverage = if row.total_tokens.is_some() { 1.0 } else { 0.0 };
    let mut metrics = match profile {
        RouteSelectionProfile::Balanced => vec![
            EvidenceMetric::derived("quality_score", quality.raw_score, 1.00, quality.coverage),
            EvidenceMetric::reward("runtime_ready_rate", row.runtime_ready_rate, 0.05),
            EvidenceMetric::derived("latency_cost", row.latency_cost, 0.05, latency_coverage),
            EvidenceMetric::derived("token_cost", row.token_cost, 0.05, token_coverage),
        ],
        RouteSelectionProfile::Quality => vec![
            EvidenceMetric::derived("quality_score", quality.raw_score, 0.65, quality.coverage),
            EvidenceMetric::reward("text_quality_score", row.text_quality_score, 0.20),
            EvidenceMetric::reward("completion_rate", row.completion_rate, 0.10),
            EvidenceMetric::reward("runtime_ready_rate", row.runtime_ready_rate, 0.05),
        ],
        RouteSelectionProfile::Grounded => vec![
            EvidenceMetric::reward("text_quality_score", row.text_quality_score, 0.55),
            EvidenceMetric::derived("quality_score", quality.raw_score, 0.25, quality.coverage),
            EvidenceMetric::reward("completion_rate", row.completion_rate, 0.15),
            EvidenceMetric::reward("runtime_ready_rate", row.runtime_ready_rate, 0.05),
        ],
        RouteSelectionProfile::Efficiency => vec![
            EvidenceMetric::derived("quality_score", quality.raw_score, 0.40, quality.coverage),
            EvidenceMetric::reward("text_quality_score", row.text_quality_score, 0.20),
            EvidenceMetric::derived(
                "latency_reward",
                1.0 - row.latency_cost,
                0.20,
                latency_coverage,
            ),
            EvidenceMetric::derived("token_reward", 1.0 - row.token_cost, 0.15, token_coverage),
            EvidenceMetric::reward("runtime_ready_rate", row.runtime_ready_rate, 0.05),
        ],
        RouteSelectionProfile::Latency => vec![
            EvidenceMetric::derived(
                "latency_reward",
                1.0 - row.latency_cost,
                0.35,
                latency_coverage,
            ),
            EvidenceMetric::derived("quality_score", quality.raw_score, 0.25, quality.coverage),
            EvidenceMetric::reward("text_quality_score", row.text_quality_score, 0.20),
            EvidenceMetric::reward("completion_rate", row.completion_rate, 0.15),
            EvidenceMetric::reward("runtime_ready_rate", row.runtime_ready_rate, 0.05),
        ],
    };
    if profile == RouteSelectionProfile::Balanced {
        metrics[2].adverse = true;
        metrics[3].adverse = true;
    }
    metrics.extend(health_metrics(row));
    metrics
}

fn prepare_rows(
    mut rows: Vec<ApiLlmRoutePolicyRow>,
) -> Result<Vec<ApiLlmRoutePolicyRow>, ApiLlmRoutePolicyError> {
    if rows.len() > API_LLM_ROUTE_POLICY_MAX_ROWS {
        return Err(ApiLlmRoutePolicyError::TooManyRows {
            actual: rows.len(),
            max: API_LLM_ROUTE_POLICY_MAX_ROWS,
        });
    }
    let mut labels = BTreeSet::new();
    for (index, row) in rows.iter_mut().enumerate() {
        if row.label.trim().is_empty() {
            return Err(ApiLlmRoutePolicyError::EmptyLabel { index });
        }
        if row.label.len() > API_LLM_ROUTE_POLICY_MAX_LABEL_BYTES {
            return Err(ApiLlmRoutePolicyError::LabelTooLong {
                index,
                actual: row.label.len(),
                max: API_LLM_ROUTE_POLICY_MAX_LABEL_BYTES,
            });
        }
        if !labels.insert(row.label.clone()) {
            return Err(ApiLlmRoutePolicyError::DuplicateLabel {
                label: row.label.clone(),
            });
        }

        for (field, value) in [
            ("runtime_ready_rate", row.runtime_ready_rate),
            ("completion_rate", row.completion_rate),
            ("incomplete_rate", row.incomplete_rate),
            ("empty_text_rate", row.empty_text_rate),
            ("refusal_rate", row.refusal_rate),
            ("confidence_mean", row.confidence_mean),
            ("text_quality_score", row.text_quality_score),
            ("stability_mean", row.stability_mean),
            ("frac_mean", row.frac_mean),
            ("wasm_stability_hint_mean", row.wasm_stability_hint_mean),
            (
                "wasm_webgpu_device_ready_rate",
                row.wasm_webgpu_device_ready_rate,
            ),
            (
                "topos_context_observed_rate",
                row.topos_context_observed_rate,
            ),
            (
                "topos_closure_pressure_mean",
                row.topos_closure_pressure_mean,
            ),
            ("topos_openness_mean", row.topos_openness_mean),
            (
                "topos_runtime_closure_risk_mean",
                row.topos_runtime_closure_risk_mean,
            ),
            (
                "topos_runtime_route_score_mean",
                row.topos_runtime_route_score_mean,
            ),
            (
                "topos_runtime_route_guard_score_mean",
                row.topos_runtime_route_guard_score_mean,
            ),
            (
                "topos_runtime_route_exploration_score_mean",
                row.topos_runtime_route_exploration_score_mean,
            ),
            (
                "topos_runtime_route_context_score_mean",
                row.topos_runtime_route_context_score_mean,
            ),
            (
                "topos_runtime_exploration_budget_mean",
                row.topos_runtime_exploration_budget_mean,
            ),
        ] {
            if value.is_some_and(|value| !value.is_finite()) {
                return Err(ApiLlmRoutePolicyError::NonFiniteMetric { index, field });
            }
            if value.is_some_and(|value| !(0.0..=1.0).contains(&value)) {
                return Err(ApiLlmRoutePolicyError::MetricOutOfRange { index, field });
            }
        }
        for (field, value) in [
            ("total_tokens", row.total_tokens),
            ("latency_ms_mean", row.latency_ms_mean),
            ("wasm_loss_mean", row.wasm_loss_mean),
            (
                "topos_optimizer_rate_scale_mean",
                row.topos_optimizer_rate_scale_mean,
            ),
            (
                "topos_optimizer_raw_rate_scale_mean",
                row.topos_optimizer_raw_rate_scale_mean,
            ),
            (
                "topos_training_clip_scale_mean",
                row.topos_training_clip_scale_mean,
            ),
            (
                "topos_training_plan_rate_scale_mean",
                row.topos_training_plan_rate_scale_mean,
            ),
            (
                "topos_inference_context_weight_mean",
                row.topos_inference_context_weight_mean,
            ),
            (
                "topos_inference_plan_context_weight_mean",
                row.topos_inference_plan_context_weight_mean,
            ),
            (
                "topos_inference_plan_temperature_mean",
                row.topos_inference_plan_temperature_mean,
            ),
            (
                "topos_runtime_control_energy_mean",
                row.topos_runtime_control_energy_mean,
            ),
        ] {
            if value.is_some_and(|value| !value.is_finite()) {
                return Err(ApiLlmRoutePolicyError::NonFiniteMetric { index, field });
            }
            if value.is_some_and(|value| value < 0.0) {
                return Err(ApiLlmRoutePolicyError::NegativeMetric { index, field });
            }
        }
        if row
            .topos_training_gradient_bias_scale_mean
            .is_some_and(|value| !value.is_finite())
        {
            return Err(ApiLlmRoutePolicyError::NonFiniteMetric {
                index,
                field: "topos_training_gradient_bias_scale_mean",
            });
        }
        if row
            .topos_optimizer_effective_gradient_bias_scale_mean
            .is_some_and(|value| !value.is_finite())
        {
            return Err(ApiLlmRoutePolicyError::NonFiniteMetric {
                index,
                field: "topos_optimizer_effective_gradient_bias_scale_mean",
            });
        }

        let quality = derived_score(&quality_metrics(row), row.count);
        row.quality_score = quality.score;
        row.latency_cost = normalized_cost(row.latency_ms_mean, 10_000.0);
        row.token_cost = normalized_token_cost(row.total_tokens, row.count);
        row.health_penalty = health_metrics(row)
            .iter()
            .map(|metric| metric.weight * metric.value)
            .sum::<f64>();

        let efficiency_metrics = [
            EvidenceMetric::derived("quality_score", quality.raw_score, 0.60, quality.coverage),
            EvidenceMetric::derived(
                "latency_reward",
                1.0 - row.latency_cost,
                0.175,
                if row.latency_ms_mean.is_some() {
                    1.0
                } else {
                    0.0
                },
            ),
            EvidenceMetric::derived(
                "token_reward",
                1.0 - row.token_cost,
                0.175,
                if row.total_tokens.is_some() { 1.0 } else { 0.0 },
            ),
            EvidenceMetric::reward("runtime_ready_rate", row.runtime_ready_rate, 0.05),
            health_metrics(row)[0],
            health_metrics(row)[1],
            health_metrics(row)[2],
        ];
        row.efficiency_score = derived_score(&efficiency_metrics, row.count).score;

        row.selection_evidence = RouteSelectionProfile::ALL
            .into_iter()
            .map(|profile| {
                let evidence =
                    score_metrics(&profile_metrics(row, quality, profile), row.count, profile);
                (profile.as_str().to_owned(), evidence)
            })
            .collect();
        row.selection_scores = row
            .selection_evidence
            .iter()
            .map(|(profile, evidence)| (profile.clone(), evidence.score))
            .collect();
        row.route_score = row.selection_scores[RouteSelectionProfile::Balanced.as_str()];
    }
    Ok(rows)
}

fn candidate_order(
    candidate: &ApiLlmRoutePolicyRow,
    current: &ApiLlmRoutePolicyRow,
    candidate_value: f64,
    current_value: f64,
    higher_is_better: bool,
) -> Ordering {
    let primary = if higher_is_better {
        candidate_value.total_cmp(&current_value)
    } else {
        current_value.total_cmp(&candidate_value)
    };
    primary
        .then_with(|| candidate.route_score.total_cmp(&current.route_score))
        .then_with(|| candidate.label.cmp(&current.label))
}

fn metric_winner<F>(
    rows: &[ApiLlmRoutePolicyRow],
    value: F,
    higher_is_better: bool,
) -> Option<String>
where
    F: Fn(&ApiLlmRoutePolicyRow) -> Option<f64>,
{
    let mut winner: Option<(&ApiLlmRoutePolicyRow, f64)> = None;
    for row in rows.iter().filter(|row| row.count > 0) {
        let Some(candidate_value) = value(row) else {
            continue;
        };
        if winner.is_none_or(|(current, current_value)| {
            candidate_order(
                row,
                current,
                candidate_value,
                current_value,
                higher_is_better,
            )
            .is_gt()
        }) {
            winner = Some((row, candidate_value));
        }
    }
    winner.map(|(row, _)| row.label.clone())
}

fn profile_winner(
    rows: &[ApiLlmRoutePolicyRow],
    profile: RouteSelectionProfile,
) -> ApiLlmRoutePolicyProfileWinner {
    let winner = rows
        .iter()
        .filter(|row| row.count > 0)
        .max_by(|candidate, current| {
            candidate_order(
                candidate,
                current,
                candidate.selection_scores[profile.as_str()],
                current.selection_scores[profile.as_str()],
                true,
            )
        });
    let Some(row) = winner else {
        return ApiLlmRoutePolicyProfileWinner::default();
    };
    ApiLlmRoutePolicyProfileWinner {
        label: Some(row.label.clone()),
        score: row.selection_scores[profile.as_str()],
        route_score: row.route_score,
        quality_score: row.quality_score,
        text_quality_score: row.text_quality_score,
        efficiency_score: row.efficiency_score,
        latency_ms_mean: row.latency_ms_mean,
        total_tokens: row.total_tokens,
        completion_rate: row.completion_rate,
        wasm_loss_mean: row.wasm_loss_mean,
        wasm_webgpu_device_ready_rate: row.wasm_webgpu_device_ready_rate,
        score_evidence: row.selection_evidence.get(profile.as_str()).cloned(),
    }
}

type RouteWinnerMetric = (&'static str, bool, fn(&ApiLlmRoutePolicyRow) -> Option<f64>);

fn route_winners(rows: &[ApiLlmRoutePolicyRow]) -> BTreeMap<String, Option<String>> {
    let specs: [RouteWinnerMetric; 34] = [
        ("best_score", true, |row| Some(row.route_score)),
        ("highest_quality", true, |row| Some(row.quality_score)),
        ("highest_text_quality", true, |row| row.text_quality_score),
        ("highest_efficiency", true, |row| Some(row.efficiency_score)),
        ("highest_confidence", true, |row| row.confidence_mean),
        ("highest_stability", true, |row| row.stability_mean),
        ("highest_completion_rate", true, |row| row.completion_rate),
        ("lowest_empty_text", false, |row| row.empty_text_rate),
        ("lowest_refusal", false, |row| row.refusal_rate),
        ("lowest_latency", false, |row| row.latency_ms_mean),
        ("lowest_total_tokens", false, |row| row.total_tokens),
        ("highest_runtime_ready", true, |row| row.runtime_ready_rate),
        ("lowest_wasm_loss", false, |row| row.wasm_loss_mean),
        ("highest_wasm_stability_hint", true, |row| {
            row.wasm_stability_hint_mean
        }),
        ("highest_wasm_webgpu_device_ready", true, |row| {
            row.wasm_webgpu_device_ready_rate
        }),
        ("highest_topos_context_observed", true, |row| {
            row.topos_context_observed_rate
        }),
        ("lowest_topos_closure_pressure", false, |row| {
            row.topos_closure_pressure_mean
        }),
        ("highest_topos_openness", true, |row| {
            row.topos_openness_mean
        }),
        ("highest_topos_training_gradient_bias", true, |row| {
            row.topos_training_gradient_bias_scale_mean
        }),
        (
            "highest_topos_effective_training_gradient_bias",
            true,
            |row| row.topos_optimizer_effective_gradient_bias_scale_mean,
        ),
        ("lowest_topos_training_clip_scale", false, |row| {
            row.topos_training_clip_scale_mean
        }),
        ("lowest_topos_training_plan_rate_scale", false, |row| {
            row.topos_training_plan_rate_scale_mean
        }),
        ("lowest_topos_optimizer_rate_scale", false, |row| {
            row.topos_optimizer_rate_scale_mean
        }),
        ("lowest_topos_optimizer_raw_rate_scale", false, |row| {
            row.topos_optimizer_raw_rate_scale_mean
        }),
        ("highest_topos_inference_context_weight", true, |row| {
            row.topos_inference_context_weight_mean
        }),
        ("highest_topos_inference_plan_context_weight", true, |row| {
            row.topos_inference_plan_context_weight_mean
        }),
        ("lowest_topos_inference_plan_temperature", false, |row| {
            row.topos_inference_plan_temperature_mean
        }),
        ("highest_topos_runtime_control_energy", true, |row| {
            row.topos_runtime_control_energy_mean
        }),
        ("lowest_topos_runtime_closure_risk", false, |row| {
            row.topos_runtime_closure_risk_mean
        }),
        ("highest_topos_runtime_exploration_budget", true, |row| {
            row.topos_runtime_exploration_budget_mean
        }),
        ("highest_topos_runtime_route_score", true, |row| {
            row.topos_runtime_route_score_mean
        }),
        ("highest_topos_runtime_route_guard_score", true, |row| {
            row.topos_runtime_route_guard_score_mean
        }),
        (
            "highest_topos_runtime_route_exploration_score",
            true,
            |row| row.topos_runtime_route_exploration_score_mean,
        ),
        ("highest_topos_runtime_route_context_score", true, |row| {
            row.topos_runtime_route_context_score_mean
        }),
    ];
    specs
        .into_iter()
        .map(|(name, higher_is_better, value)| {
            (
                name.to_owned(),
                metric_winner(rows, value, higher_is_better),
            )
        })
        .collect()
}

pub fn evaluate_api_llm_route_policy(
    request: ApiLlmRoutePolicyEvaluationRequest,
) -> Result<ApiLlmRoutePolicyEvaluationPayload, ApiLlmRoutePolicyError> {
    if !request.near_best_tolerance.is_finite()
        || !(0.0..=1.0).contains(&request.near_best_tolerance)
    {
        return Err(ApiLlmRoutePolicyError::InvalidNearBestTolerance);
    }
    let mut rows = prepare_rows(request.rows)?;
    rows.sort_by(|candidate, current| {
        (candidate.count > 0)
            .cmp(&(current.count > 0))
            .then_with(|| candidate.route_score.total_cmp(&current.route_score))
            .then_with(|| {
                candidate
                    .confidence_mean
                    .unwrap_or(API_LLM_ROUTE_POLICY_PRIOR_MEAN)
                    .total_cmp(
                        &current
                            .confidence_mean
                            .unwrap_or(API_LLM_ROUTE_POLICY_PRIOR_MEAN),
                    )
            })
            .then_with(|| candidate.label.cmp(&current.label))
            .reverse()
    });

    let active_rows = rows.iter().filter(|row| row.count > 0).collect::<Vec<_>>();
    let best_score = active_rows.first().map(|row| row.route_score);
    let near_best = best_score
        .map(|best| {
            active_rows
                .iter()
                .filter_map(|row| {
                    let delta = best - row.route_score;
                    (delta <= request.near_best_tolerance).then(|| ApiLlmRoutePolicyNearBest {
                        label: row.label.clone(),
                        route_score: row.route_score,
                        route_score_delta: delta,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let profiles = RouteSelectionProfile::ALL
        .into_iter()
        .map(|profile| (profile.as_str().to_owned(), profile_winner(&rows, profile)))
        .collect();
    let winners = route_winners(&rows);
    let row_count = rows.len();
    let active_row_count = active_rows.len();
    let ranked_labels = active_rows.iter().map(|row| row.label.clone()).collect();

    Ok(ApiLlmRoutePolicyEvaluationPayload {
        kind: API_LLM_ROUTE_POLICY_KIND,
        contract_version: API_LLM_ROUTE_POLICY_CONTRACT_VERSION,
        semantic_owner: API_LLM_ROUTE_POLICY_SEMANTIC_OWNER,
        semantic_backend: API_LLM_ROUTE_POLICY_SEMANTIC_BACKEND,
        row_count,
        active_row_count,
        inactive_row_count: row_count - active_row_count,
        near_best_tolerance: request.near_best_tolerance,
        score_formula_version: API_LLM_ROUTE_POLICY_SCORE_FORMULA_VERSION,
        score_formula: API_LLM_ROUTE_POLICY_SCORE_FORMULA,
        score_prior_mean: API_LLM_ROUTE_POLICY_PRIOR_MEAN,
        score_prior_strength: API_LLM_ROUTE_POLICY_PRIOR_STRENGTH,
        rows,
        ranked_labels,
        near_best,
        profiles,
        winners,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(label: &str, count: u32) -> ApiLlmRoutePolicyRow {
        ApiLlmRoutePolicyRow {
            label: label.to_owned(),
            count,
            completion_rate: Some(1.0),
            incomplete_rate: Some(0.0),
            empty_text_rate: Some(0.0),
            refusal_rate: Some(0.0),
            ..ApiLlmRoutePolicyRow::default()
        }
    }

    #[test]
    fn missing_costs_are_neutral_and_zero_count_is_inactive() {
        let mut missing = row("missing", 4);
        missing.confidence_mean = Some(0.8);
        let mut observed = row("observed", 4);
        observed.confidence_mean = Some(0.8);
        observed.latency_ms_mean = Some(100.0);
        observed.total_tokens = Some(32.0);
        let mut inactive = row("inactive", 0);
        inactive.confidence_mean = Some(1.0);

        let payload = evaluate_api_llm_route_policy(ApiLlmRoutePolicyEvaluationRequest {
            rows: vec![missing, observed, inactive],
            near_best_tolerance: 1.0,
        })
        .expect("valid policy");
        let by_label = payload
            .rows
            .iter()
            .map(|row| (row.label.as_str(), row))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(by_label["missing"].latency_cost, 0.5);
        assert_eq!(by_label["missing"].token_cost, 0.5);
        assert!(by_label["missing"].selection_scores["latency"] < 0.6);
        assert_eq!(payload.active_row_count, 2);
        assert!(!payload.ranked_labels.contains(&"inactive".to_owned()));
        assert_ne!(
            payload.winners["lowest_latency"],
            Some("missing".to_owned())
        );
    }

    #[test]
    fn balanced_score_applies_health_penalty_once() {
        let mut healthy = row("healthy", 8);
        healthy.confidence_mean = Some(0.8);
        healthy.stability_mean = Some(0.8);
        healthy.frac_mean = Some(0.8);
        healthy.runtime_ready_rate = Some(1.0);
        healthy.latency_ms_mean = Some(100.0);
        healthy.total_tokens = Some(32.0);
        healthy.text_quality_score = Some(0.8);
        let mut refusal = healthy.clone();
        refusal.label = "refusal".to_owned();
        refusal.empty_text_rate = Some(1.0);
        refusal.refusal_rate = Some(1.0);

        let payload = evaluate_api_llm_route_policy(ApiLlmRoutePolicyEvaluationRequest {
            rows: vec![refusal, healthy],
            near_best_tolerance: 1.0,
        })
        .expect("valid policy");
        let by_label = payload
            .rows
            .iter()
            .map(|row| (row.label.as_str(), row))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(
            by_label["healthy"].selection_scores["balanced"],
            by_label["healthy"].route_score
        );
        assert_eq!(
            by_label["refusal"].selection_scores["balanced"],
            by_label["refusal"].route_score
        );
        assert!(by_label["healthy"].route_score > by_label["refusal"].route_score);
        assert_eq!(payload.winners["best_score"], Some("healthy".to_owned()));
    }

    #[test]
    fn token_cost_is_invariant_to_the_number_of_observations() {
        let mut one = row("one", 1);
        one.total_tokens = Some(64.0);
        let mut many = row("many", 8);
        many.total_tokens = Some(512.0);

        let payload = evaluate_api_llm_route_policy(ApiLlmRoutePolicyEvaluationRequest {
            rows: vec![one, many],
            near_best_tolerance: 1.0,
        })
        .expect("valid policy");
        let by_label = payload
            .rows
            .iter()
            .map(|row| (row.label.as_str(), row))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(by_label["one"].token_cost, by_label["many"].token_cost);
    }

    #[test]
    fn tie_breaks_and_near_best_are_rust_owned() {
        let left = row("left", 2);
        let right = row("right", 2);
        let payload = evaluate_api_llm_route_policy(ApiLlmRoutePolicyEvaluationRequest {
            rows: vec![left, right],
            near_best_tolerance: 0.0,
        })
        .expect("valid policy");

        assert_eq!(payload.ranked_labels, ["right", "left"]);
        assert_eq!(payload.winners["best_score"], Some("right".to_owned()));
        assert_eq!(payload.profiles["grounded"].label, Some("right".to_owned()));
        assert_eq!(
            payload
                .near_best
                .iter()
                .map(|row| row.label.as_str())
                .collect::<Vec<_>>(),
            ["right", "left"]
        );
    }

    #[test]
    fn client_scores_are_ignored_and_invalid_evidence_fails_closed() {
        let request: ApiLlmRoutePolicyEvaluationRequest =
            serde_json::from_value(serde_json::json!({
                "rows": [{
                    "label": "guarded",
                    "count": 1,
                    "confidence_mean": 0.8,
                    "route_score": 1.0,
                    "selection_scores": {"balanced": 1.0}
                }],
                "near_best_tolerance": 0.02
            }))
            .expect("compatibility projections decode");
        let payload = evaluate_api_llm_route_policy(request).expect("valid policy");
        assert_ne!(payload.rows[0].route_score, 1.0);
        assert_ne!(payload.rows[0].selection_scores["balanced"], 1.0);

        let error = evaluate_api_llm_route_policy(ApiLlmRoutePolicyEvaluationRequest {
            rows: vec![ApiLlmRoutePolicyRow {
                label: "bad".to_owned(),
                count: 1,
                confidence_mean: Some(f64::NAN),
                ..ApiLlmRoutePolicyRow::default()
            }],
            near_best_tolerance: 0.0,
        })
        .expect_err("non-finite evidence must fail");
        assert!(matches!(
            error,
            ApiLlmRoutePolicyError::NonFiniteMetric {
                field: "confidence_mean",
                ..
            }
        ));
    }
}
