// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical Z-space generation control semantics.
//!
//! This module owns the deterministic transformation from selected candidate
//! logits and token history into repression-adjusted Z-space probabilities.
//! Language bindings may select candidates and scatter the returned values,
//! but must not reconstruct these formulas.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

pub const ZSPACE_GENERATION_CONTROL_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_generation_control.v1";
pub const ZSPACE_GENERATION_CONTROL_KIND: &str = "spiraltorch.zspace_generation_control";
pub const ZSPACE_GENERATION_CONTROL_SEMANTIC_OWNER: &str = "st-core::inference::generation_control";
pub const ZSPACE_GENERATION_CONTROL_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_GENERATION_CONTROL_BACKEND: &str = "spiraltorch_generation_control_core";

pub const ZSPACE_SOFTMAX_LOG_FLOOR: f64 = 1.0e-12;
pub const ZSPACE_SOFTMAX_ADJUST_MIN: f64 = 0.25;
pub const ZSPACE_SOFTMAX_ADJUST_MAX: f64 = 4.0;

#[derive(Debug, Error, PartialEq)]
pub enum ZSpaceGenerationControlError {
    #[error("generation control field '{field}' must be finite")]
    NonFinite { field: &'static str },
    #[error("generation control field '{field}' must be positive")]
    NonPositive { field: &'static str },
    #[error("generation control field '{field}' must be non-negative")]
    Negative { field: &'static str },
    #[error("generation control field '{field}' must be in [{min}, {max}]")]
    OutOfRange {
        field: &'static str,
        min: &'static str,
        max: &'static str,
    },
    #[error("generation control temperature bounds require min_temperature <= max_temperature")]
    InvalidTemperatureBounds,
    #[error("generation control candidate lengths differ: logits={logits}, token_ids={token_ids}")]
    CandidateLengthMismatch { logits: usize, token_ids: usize },
    #[error("derived generation control field '{field}' must be finite")]
    NonFiniteDerived { field: &'static str },
}

fn require_finite(field: &'static str, value: f64) -> Result<f64, ZSpaceGenerationControlError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ZSpaceGenerationControlError::NonFinite { field })
    }
}

fn require_positive(field: &'static str, value: f64) -> Result<f64, ZSpaceGenerationControlError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(ZSpaceGenerationControlError::NonPositive { field })
    }
}

fn require_non_negative(
    field: &'static str,
    value: f64,
) -> Result<f64, ZSpaceGenerationControlError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(ZSpaceGenerationControlError::Negative { field })
    }
}

fn require_derived_finite(
    field: &'static str,
    value: f64,
) -> Result<f64, ZSpaceGenerationControlError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ZSpaceGenerationControlError::NonFiniteDerived { field })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZSpaceSoftmaxConfig {
    curvature: f64,
    temperature: f64,
    min_temperature: f64,
    max_temperature: f64,
    entropy_target: Option<f64>,
    entropy_tolerance: f64,
    entropy_gain: f64,
}

impl ZSpaceSoftmaxConfig {
    pub fn new(curvature: f64, temperature: f64) -> Result<Self, ZSpaceGenerationControlError> {
        require_finite("curvature", curvature)?;
        if curvature >= 0.0 {
            return Err(ZSpaceGenerationControlError::OutOfRange {
                field: "curvature",
                min: "-infinity",
                max: "0 (exclusive)",
            });
        }
        require_positive("temperature", temperature)?;
        let min_temperature =
            require_derived_finite("default_min_temperature", (temperature * 0.1).max(1.0e-3))?;
        let max_temperature =
            require_derived_finite("default_max_temperature", temperature * 10.0)?;
        Ok(Self {
            curvature,
            temperature,
            min_temperature,
            max_temperature,
            entropy_target: None,
            entropy_tolerance: 1.0e-4,
            entropy_gain: 0.5,
        })
    }

    pub fn with_entropy_target(
        mut self,
        target: f64,
        tolerance: f64,
        gain: f64,
    ) -> Result<Self, ZSpaceGenerationControlError> {
        self.entropy_target = Some(require_non_negative("entropy_target", target)?);
        self.entropy_tolerance = require_non_negative("entropy_tolerance", tolerance)?;
        self.entropy_gain = require_non_negative("entropy_gain", gain)?;
        Ok(self)
    }

    pub fn with_temperature_bounds(
        mut self,
        min_temperature: f64,
        max_temperature: f64,
    ) -> Result<Self, ZSpaceGenerationControlError> {
        self.min_temperature = require_positive("min_temperature", min_temperature)?;
        self.max_temperature = require_positive("max_temperature", max_temperature)?;
        if self.min_temperature > self.max_temperature {
            return Err(ZSpaceGenerationControlError::InvalidTemperatureBounds);
        }
        Ok(self)
    }

    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn min_temperature(&self) -> f64 {
        self.min_temperature
    }

    pub fn max_temperature(&self) -> f64 {
        self.max_temperature
    }

    pub fn entropy_target(&self) -> Option<f64> {
        self.entropy_target
    }

    pub fn entropy_tolerance(&self) -> f64 {
        self.entropy_tolerance
    }

    pub fn entropy_gain(&self) -> f64 {
        self.entropy_gain
    }

    pub fn fixed_temperature(&self) -> f64 {
        self.temperature
            .clamp(self.min_temperature, self.max_temperature)
    }

    pub fn curvature_scale(&self) -> f64 {
        (-self.curvature).sqrt()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ZSpaceSoftmaxProjection {
    pub probabilities: Vec<f64>,
    pub base_probabilities: Vec<f64>,
    pub entropy: f64,
    pub base_entropy: f64,
    pub effective_temperature: f64,
    pub base_temperature: f64,
    pub adaptive_temperature: bool,
    pub temperature_gradient_active: bool,
    pub scale: f64,
    pub base_scale: f64,
}

fn stable_softmax(logits: &[f64], scale: f64) -> Result<Vec<f64>, ZSpaceGenerationControlError> {
    require_derived_finite("zspace_softmax_scale", scale)?;
    if logits.is_empty() {
        return Ok(Vec::new());
    }

    let scaled = logits
        .iter()
        .copied()
        .map(|value| {
            require_finite("logit", value)?;
            require_derived_finite("zspace_softmax_scaled_logit", value * scale)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let max_logit = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    require_derived_finite("zspace_softmax_max_logit", max_logit)?;

    let weights = scaled
        .iter()
        .copied()
        .map(|value| {
            let delta = value - max_logit;
            let weight = if delta == f64::NEG_INFINITY {
                0.0
            } else {
                delta.exp()
            };
            require_derived_finite("zspace_softmax_weight", weight)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total = weights.iter().copied().try_fold(0.0, |sum, weight| {
        require_derived_finite("zspace_softmax_weight_sum", sum + weight)
    })?;
    if total <= 0.0 {
        return Err(ZSpaceGenerationControlError::NonPositive {
            field: "zspace_softmax_weight_sum",
        });
    }

    weights
        .into_iter()
        .map(|weight| require_derived_finite("zspace_softmax_probability", weight / total))
        .collect()
}

pub fn zspace_entropy(probabilities: &[f64]) -> Result<f64, ZSpaceGenerationControlError> {
    probabilities
        .iter()
        .copied()
        .try_fold(0.0, |entropy, probability| {
            require_finite("probability", probability)?;
            if !(0.0..=1.0).contains(&probability) {
                return Err(ZSpaceGenerationControlError::OutOfRange {
                    field: "probability",
                    min: "0",
                    max: "1",
                });
            }
            let term = if probability > 0.0 {
                -probability * probability.max(ZSPACE_SOFTMAX_LOG_FLOOR).ln()
            } else {
                0.0
            };
            require_derived_finite("zspace_softmax_entropy", entropy + term)
        })
}

pub fn project_zspace_softmax(
    logits: &[f64],
    config: ZSpaceSoftmaxConfig,
) -> Result<ZSpaceSoftmaxProjection, ZSpaceGenerationControlError> {
    let base_temperature = config.fixed_temperature();
    require_derived_finite("zspace_softmax_base_temperature", base_temperature)?;
    let curvature_scale = config.curvature_scale();
    require_derived_finite("zspace_softmax_curvature_scale", curvature_scale)?;
    let base_scale = require_derived_finite(
        "zspace_softmax_base_scale",
        curvature_scale / base_temperature,
    )?;
    let base_probabilities = stable_softmax(logits, base_scale)?;
    let base_entropy = zspace_entropy(&base_probabilities)?;

    let mut effective_temperature = base_temperature;
    let mut adaptive_temperature = false;
    let mut temperature_gradient_active = false;
    if let Some(target) = config.entropy_target {
        let delta = require_derived_finite("zspace_softmax_entropy_delta", target - base_entropy)?;
        if delta.abs() > config.entropy_tolerance {
            let raw_adjust = require_derived_finite(
                "zspace_softmax_temperature_adjust",
                1.0 + config.entropy_gain * delta,
            )?;
            let adjust = raw_adjust.clamp(ZSPACE_SOFTMAX_ADJUST_MIN, ZSPACE_SOFTMAX_ADJUST_MAX);
            let raw_temperature = require_derived_finite(
                "zspace_softmax_effective_temperature",
                base_temperature * adjust,
            )?;
            effective_temperature =
                raw_temperature.clamp(config.min_temperature, config.max_temperature);
            adaptive_temperature = true;
            temperature_gradient_active = config.entropy_gain > 0.0
                && raw_adjust > ZSPACE_SOFTMAX_ADJUST_MIN
                && raw_adjust < ZSPACE_SOFTMAX_ADJUST_MAX
                && raw_temperature > config.min_temperature
                && raw_temperature < config.max_temperature;
        }
    }

    let scale = require_derived_finite(
        "zspace_softmax_scale",
        curvature_scale / effective_temperature,
    )?;
    let probabilities = if adaptive_temperature {
        stable_softmax(logits, scale)?
    } else {
        base_probabilities.clone()
    };
    let entropy = if adaptive_temperature {
        zspace_entropy(&probabilities)?
    } else {
        base_entropy
    };

    Ok(ZSpaceSoftmaxProjection {
        probabilities,
        base_probabilities,
        entropy,
        base_entropy,
        effective_temperature,
        base_temperature,
        adaptive_temperature,
        temperature_gradient_active,
        scale,
        base_scale,
    })
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpaceGenerationControlConfig {
    pub curvature: f64,
    pub temperature: f64,
    pub entropy_target: Option<f64>,
    pub entropy_tolerance: f64,
    pub entropy_gain: f64,
    pub min_temperature: Option<f64>,
    pub max_temperature: Option<f64>,
    pub repression_window: usize,
    pub repression_strength: f64,
    pub last_token_repression: f64,
    pub ngram_size: usize,
    pub ngram_window: usize,
    pub ngram_repression_strength: f64,
    pub ngram_decay: f64,
}

impl Default for ZSpaceGenerationControlConfig {
    fn default() -> Self {
        Self {
            curvature: -0.04,
            temperature: 1.0,
            entropy_target: None,
            entropy_tolerance: 1.0e-4,
            entropy_gain: 0.5,
            min_temperature: None,
            max_temperature: None,
            repression_window: 32,
            repression_strength: 1.0,
            last_token_repression: 0.5,
            ngram_size: 0,
            ngram_window: 0,
            ngram_repression_strength: 0.0,
            ngram_decay: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResolvedZSpaceGenerationControlConfig {
    pub curvature: f64,
    pub temperature: f64,
    pub entropy_target: Option<f64>,
    pub entropy_tolerance: f64,
    pub entropy_gain: f64,
    pub min_temperature: f64,
    pub max_temperature: f64,
    pub repression_window: usize,
    pub repression_strength: f64,
    pub last_token_repression: f64,
    pub ngram_size: usize,
    pub ngram_window: usize,
    pub ngram_repression_strength: f64,
    pub ngram_decay: f64,
}

impl ZSpaceGenerationControlConfig {
    pub fn resolve(
        &self,
    ) -> Result<ResolvedZSpaceGenerationControlConfig, ZSpaceGenerationControlError> {
        let mut softmax = ZSpaceSoftmaxConfig::new(self.curvature, self.temperature)?;
        if let Some(target) = self.entropy_target {
            softmax =
                softmax.with_entropy_target(target, self.entropy_tolerance, self.entropy_gain)?;
        } else {
            require_non_negative("entropy_tolerance", self.entropy_tolerance)?;
            require_non_negative("entropy_gain", self.entropy_gain)?;
        }
        let min_temperature = self.min_temperature.unwrap_or(softmax.min_temperature());
        let max_temperature = self.max_temperature.unwrap_or(softmax.max_temperature());
        softmax = softmax.with_temperature_bounds(min_temperature, max_temperature)?;
        require_non_negative("repression_strength", self.repression_strength)?;
        require_non_negative("last_token_repression", self.last_token_repression)?;
        require_non_negative("ngram_repression_strength", self.ngram_repression_strength)?;
        require_finite("ngram_decay", self.ngram_decay)?;
        if !(0.0..=1.0).contains(&self.ngram_decay) {
            return Err(ZSpaceGenerationControlError::OutOfRange {
                field: "ngram_decay",
                min: "0",
                max: "1",
            });
        }
        Ok(ResolvedZSpaceGenerationControlConfig {
            curvature: softmax.curvature(),
            temperature: softmax.temperature(),
            entropy_target: softmax.entropy_target(),
            entropy_tolerance: softmax.entropy_tolerance(),
            entropy_gain: softmax.entropy_gain(),
            min_temperature: softmax.min_temperature(),
            max_temperature: softmax.max_temperature(),
            repression_window: self.repression_window,
            repression_strength: self.repression_strength,
            last_token_repression: self.last_token_repression,
            ngram_size: self.ngram_size,
            ngram_window: self.ngram_window,
            ngram_repression_strength: self.ngram_repression_strength,
            ngram_decay: self.ngram_decay,
        })
    }
}

impl ResolvedZSpaceGenerationControlConfig {
    pub fn softmax_config(&self) -> ZSpaceSoftmaxConfig {
        ZSpaceSoftmaxConfig {
            curvature: self.curvature,
            temperature: self.temperature,
            min_temperature: self.min_temperature,
            max_temperature: self.max_temperature,
            entropy_target: self.entropy_target,
            entropy_tolerance: self.entropy_tolerance,
            entropy_gain: self.entropy_gain,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpaceGenerationControlRequest {
    pub logits: Vec<f64>,
    pub token_ids: Vec<u64>,
    pub recent_tokens: Vec<u64>,
    pub config: ZSpaceGenerationControlConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceGenerationControlPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub backend: &'static str,
    pub config: ResolvedZSpaceGenerationControlConfig,
    pub token_ids: Vec<u64>,
    pub adjusted_logits: Vec<f64>,
    pub repression_penalties: Vec<f64>,
    pub ngram_repression_penalties: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub log_probabilities: Vec<f64>,
    pub base_entropy: f64,
    pub entropy: f64,
    pub base_temperature: f64,
    pub effective_temperature: f64,
    pub adaptive_temperature: bool,
    pub temperature_gradient_active: bool,
    pub candidate_count: usize,
    pub history_length: usize,
    pub repression_history_length: usize,
    pub ngram_history_length: usize,
    pub repressed_token_count: usize,
    pub max_repression: f64,
    pub ngram_repressed_token_count: usize,
    pub max_ngram_repression: f64,
    pub before_top_position: Option<usize>,
    pub after_top_position: Option<usize>,
    pub before_top_token: Option<u64>,
    pub after_top_token: Option<u64>,
    pub top_token_changed: bool,
}

fn checked_add(
    field: &'static str,
    left: f64,
    right: f64,
) -> Result<f64, ZSpaceGenerationControlError> {
    require_derived_finite(field, left + right)
}

fn checked_mul(
    field: &'static str,
    left: f64,
    right: f64,
) -> Result<f64, ZSpaceGenerationControlError> {
    require_derived_finite(field, left * right)
}

fn ngram_penalty(
    recent_tokens: &[u64],
    candidate_token: u64,
    config: &ResolvedZSpaceGenerationControlConfig,
) -> Result<f64, ZSpaceGenerationControlError> {
    if config.ngram_size <= 1
        || config.ngram_repression_strength <= 0.0
        || recent_tokens.len() < config.ngram_size
    {
        return Ok(0.0);
    }

    let prefix_size = config.ngram_size - 1;
    let prefix = &recent_tokens[recent_tokens.len() - prefix_size..];
    let latest_start = recent_tokens.len() - config.ngram_size;
    let mut weighted_matches = 0.0;
    for start in 0..=latest_start {
        let candidate_matches = recent_tokens[start..start + prefix_size] == *prefix
            && recent_tokens[start + prefix_size] == candidate_token;
        if !candidate_matches {
            continue;
        }
        let distance = latest_start - start;
        let weight = require_derived_finite(
            "ngram_decay_weight",
            config.ngram_decay.powf(distance as f64),
        )?;
        weighted_matches = checked_add("ngram_weighted_matches", weighted_matches, weight)?;
    }
    checked_mul(
        "ngram_repression_penalty",
        config.ngram_repression_strength,
        weighted_matches,
    )
}

fn first_argmax(values: &[f64]) -> Option<usize> {
    let mut best = None;
    for (index, &value) in values.iter().enumerate() {
        if best.is_none_or(|best_index| value > values[best_index]) {
            best = Some(index);
        }
    }
    best
}

pub fn apply_zspace_generation_control(
    request: ZSpaceGenerationControlRequest,
) -> Result<ZSpaceGenerationControlPayload, ZSpaceGenerationControlError> {
    if request.logits.len() != request.token_ids.len() {
        return Err(ZSpaceGenerationControlError::CandidateLengthMismatch {
            logits: request.logits.len(),
            token_ids: request.token_ids.len(),
        });
    }
    for &logit in &request.logits {
        require_finite("logit", logit)?;
    }
    let config = request.config.resolve()?;

    let repression_start = request
        .recent_tokens
        .len()
        .saturating_sub(config.repression_window);
    let repression_history = &request.recent_tokens[repression_start..];
    let mut token_counts = HashMap::<u64, usize>::new();
    for &token in repression_history {
        *token_counts.entry(token).or_default() += 1;
    }
    let last_token = repression_history.last().copied();

    let effective_ngram_window = if config.ngram_window == 0 {
        config.repression_window
    } else {
        config.ngram_window
    };
    let ngram_start = request
        .recent_tokens
        .len()
        .saturating_sub(effective_ngram_window);
    let ngram_history = &request.recent_tokens[ngram_start..];

    let mut adjusted_logits = Vec::with_capacity(request.logits.len());
    let mut repression_penalties = Vec::with_capacity(request.logits.len());
    let mut ngram_repression_penalties = Vec::with_capacity(request.logits.len());
    for (&logit, &token) in request.logits.iter().zip(request.token_ids.iter()) {
        let count = token_counts.get(&token).copied().unwrap_or(0) as f64;
        let mut penalty = checked_mul(
            "token_repression_penalty",
            config.repression_strength,
            count,
        )?;
        if last_token == Some(token) {
            penalty = checked_add(
                "last_token_repression_penalty",
                penalty,
                config.last_token_repression,
            )?;
        }
        let candidate_ngram_penalty = ngram_penalty(ngram_history, token, &config)?;
        penalty = checked_add("total_repression_penalty", penalty, candidate_ngram_penalty)?;
        let adjusted = require_derived_finite("adjusted_logit", logit - penalty)?;
        adjusted_logits.push(adjusted);
        repression_penalties.push(penalty);
        ngram_repression_penalties.push(candidate_ngram_penalty);
    }

    let projection = project_zspace_softmax(&adjusted_logits, config.softmax_config())?;
    let log_probabilities = projection
        .probabilities
        .iter()
        .copied()
        .map(|probability| {
            require_derived_finite(
                "log_probability",
                probability.max(ZSPACE_SOFTMAX_LOG_FLOOR).ln(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let before_top_position = first_argmax(&request.logits);
    let after_top_position = first_argmax(&adjusted_logits);
    let before_top_token = before_top_position.map(|index| request.token_ids[index]);
    let after_top_token = after_top_position.map(|index| request.token_ids[index]);
    let max_repression = repression_penalties.iter().copied().fold(0.0, f64::max);
    let max_ngram_repression = ngram_repression_penalties
        .iter()
        .copied()
        .fold(0.0, f64::max);
    let repressed_token_count = repression_penalties
        .iter()
        .filter(|&&penalty| penalty > 0.0)
        .count();
    let ngram_repressed_token_count = ngram_repression_penalties
        .iter()
        .filter(|&&penalty| penalty > 0.0)
        .count();

    Ok(ZSpaceGenerationControlPayload {
        kind: ZSPACE_GENERATION_CONTROL_KIND,
        contract_version: ZSPACE_GENERATION_CONTROL_CONTRACT_VERSION,
        semantic_owner: ZSPACE_GENERATION_CONTROL_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_GENERATION_CONTROL_SEMANTIC_BACKEND,
        backend: ZSPACE_GENERATION_CONTROL_BACKEND,
        config,
        token_ids: request.token_ids,
        adjusted_logits,
        repression_penalties,
        ngram_repression_penalties,
        probabilities: projection.probabilities,
        log_probabilities,
        base_entropy: projection.base_entropy,
        entropy: projection.entropy,
        base_temperature: projection.base_temperature,
        effective_temperature: projection.effective_temperature,
        adaptive_temperature: projection.adaptive_temperature,
        temperature_gradient_active: projection.temperature_gradient_active,
        candidate_count: request.logits.len(),
        history_length: request.recent_tokens.len(),
        repression_history_length: repression_history.len(),
        ngram_history_length: ngram_history.len(),
        repressed_token_count,
        max_repression,
        ngram_repressed_token_count,
        max_ngram_repression,
        before_top_position,
        after_top_position,
        before_top_token,
        after_top_token,
        top_token_changed: before_top_token != after_top_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_probability_contract(payload: &ZSpaceGenerationControlPayload) {
        assert!(payload.probabilities.iter().all(|value| value.is_finite()));
        assert!(payload
            .log_probabilities
            .iter()
            .all(|value| value.is_finite()));
        let sum = payload.probabilities.iter().sum::<f64>();
        if payload.candidate_count == 0 {
            assert_eq!(sum, 0.0);
        } else {
            assert!((sum - 1.0).abs() < 1e-12, "probability sum={sum}");
        }
    }

    #[test]
    fn repetition_repression_changes_the_greedy_token() {
        let payload = apply_zspace_generation_control(ZSpaceGenerationControlRequest {
            logits: vec![4.0, 3.5, 1.0],
            token_ids: vec![0, 1, 2],
            recent_tokens: vec![0, 0, 0],
            config: ZSpaceGenerationControlConfig {
                curvature: -1.0,
                temperature: 1.0,
                entropy_target: Some(1.0),
                min_temperature: Some(0.5),
                max_temperature: Some(2.0),
                repression_window: 4,
                repression_strength: 2.0,
                last_token_repression: 1.0,
                ..ZSpaceGenerationControlConfig::default()
            },
        })
        .expect("valid generation control");

        assert_eq!(payload.repression_penalties, vec![7.0, 0.0, 0.0]);
        assert_eq!(payload.adjusted_logits, vec![-3.0, 3.5, 1.0]);
        assert_eq!(payload.before_top_token, Some(0));
        assert_eq!(payload.after_top_token, Some(1));
        assert!(payload.top_token_changed);
        assert_eq!(payload.repressed_token_count, 1);
        assert_eq!(
            payload.semantic_owner,
            ZSPACE_GENERATION_CONTROL_SEMANTIC_OWNER
        );
        assert_probability_contract(&payload);
    }

    #[test]
    fn ngram_repression_matches_phrase_completion_semantics() {
        let payload = apply_zspace_generation_control(ZSpaceGenerationControlRequest {
            logits: vec![4.0, 3.9, 0.0],
            token_ids: vec![3, 4, 0],
            recent_tokens: vec![1, 2, 3, 1, 2],
            config: ZSpaceGenerationControlConfig {
                curvature: -1.0,
                repression_window: 8,
                repression_strength: 0.0,
                last_token_repression: 0.0,
                ngram_size: 3,
                ngram_repression_strength: 2.0,
                ..ZSpaceGenerationControlConfig::default()
            },
        })
        .expect("valid ngram control");

        assert_eq!(payload.ngram_repression_penalties, vec![2.0, 0.0, 0.0]);
        assert_eq!(payload.before_top_token, Some(3));
        assert_eq!(payload.after_top_token, Some(4));
        assert_eq!(payload.ngram_repressed_token_count, 1);
        assert_probability_contract(&payload);
    }

    #[test]
    fn adaptive_softmax_uses_one_step_entropy_projection() {
        let config = ZSpaceSoftmaxConfig::new(-1.0, 1.5)
            .unwrap()
            .with_entropy_target(1.3, 1.0e-6, 0.8)
            .unwrap()
            .with_temperature_bounds(0.1, 4.0)
            .unwrap();
        let projection = project_zspace_softmax(&[2.0, 0.5, -1.0], config).unwrap();

        assert!(projection.adaptive_temperature);
        assert!(projection.effective_temperature > projection.base_temperature);
        assert!(projection.entropy > projection.base_entropy);
        assert!((projection.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_candidates_still_return_a_resolved_auditable_contract() {
        let payload = apply_zspace_generation_control(Default::default()).unwrap();

        assert_eq!(payload.candidate_count, 0);
        assert_eq!(payload.before_top_token, None);
        assert_eq!(payload.after_top_token, None);
        assert!(!payload.top_token_changed);
        assert_eq!(payload.config.min_temperature, 0.1);
        assert_eq!(payload.config.max_temperature, 10.0);
        assert_probability_contract(&payload);
    }

    #[test]
    fn contract_rejects_invalid_shapes_config_and_unknown_fields() {
        let mismatch = apply_zspace_generation_control(ZSpaceGenerationControlRequest {
            logits: vec![1.0],
            token_ids: vec![],
            ..ZSpaceGenerationControlRequest::default()
        })
        .unwrap_err();
        assert_eq!(
            mismatch,
            ZSpaceGenerationControlError::CandidateLengthMismatch {
                logits: 1,
                token_ids: 0
            }
        );

        let negative = apply_zspace_generation_control(ZSpaceGenerationControlRequest {
            config: ZSpaceGenerationControlConfig {
                repression_strength: -1.0,
                ..ZSpaceGenerationControlConfig::default()
            },
            ..ZSpaceGenerationControlRequest::default()
        })
        .unwrap_err();
        assert_eq!(
            negative,
            ZSpaceGenerationControlError::Negative {
                field: "repression_strength"
            }
        );

        let error = serde_json::from_value::<ZSpaceGenerationControlRequest>(json!({
            "logits": [1.0],
            "token_ids": [1],
            "config": {"represson_strength": 1.0}
        }))
        .expect_err("unknown config fields must fail closed");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn derived_overflow_fails_closed() {
        let error = apply_zspace_generation_control(ZSpaceGenerationControlRequest {
            logits: vec![-f64::MAX],
            token_ids: vec![7],
            recent_tokens: vec![7, 7],
            config: ZSpaceGenerationControlConfig {
                repression_strength: f64::MAX,
                last_token_repression: f64::MAX,
                ..ZSpaceGenerationControlConfig::default()
            },
        })
        .unwrap_err();
        assert!(matches!(
            error,
            ZSpaceGenerationControlError::NonFiniteDerived { .. }
        ));
    }

    #[test]
    fn decayed_ngram_matches_weight_recent_occurrences_more_strongly() {
        let payload = apply_zspace_generation_control(ZSpaceGenerationControlRequest {
            logits: vec![1.0],
            token_ids: vec![3],
            recent_tokens: vec![1, 2, 3, 1, 2, 3, 1, 2],
            config: ZSpaceGenerationControlConfig {
                repression_window: 8,
                repression_strength: 0.0,
                last_token_repression: 0.0,
                ngram_size: 3,
                ngram_repression_strength: 2.0,
                ngram_decay: 0.5,
                ..ZSpaceGenerationControlConfig::default()
            },
        })
        .unwrap();

        assert!((payload.ngram_repression_penalties[0] - 0.5625).abs() < 1e-12);
    }
}
