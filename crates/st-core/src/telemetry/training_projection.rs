//! Canonical projection of trainer observations into bounded telemetry.
//!
//! The resulting desire and psi values are explicitly log-derived surrogates,
//! not native PSI observations. Bindings should only translate client inputs
//! into this contract and attach client-specific event metadata.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Stable contract identifier shared by Rust, Python, and WASM clients.
pub const TRAINING_TELEMETRY_PROJECTION_CONTRACT_VERSION: &str =
    "spiraltorch.training_telemetry_projection.v1";
/// Stable payload kind for trainer-log telemetry projection.
pub const TRAINING_TELEMETRY_PROJECTION_KIND: &str = "spiraltorch.training_telemetry_projection";
/// Crate/module that owns the projection semantics.
pub const TRAINING_TELEMETRY_PROJECTION_SEMANTIC_OWNER: &str =
    "st-core::telemetry::training_projection";
/// Backend label attached to payloads produced by the canonical implementation.
pub const TRAINING_TELEMETRY_PROJECTION_SEMANTIC_BACKEND: &str = "rust";
/// Provenance label distinguishing trainer-log proxies from native PSI signals.
pub const TRAINING_TELEMETRY_PROJECTION_SIGNAL_SOURCE: &str = "trainer_log_proxy";
/// Semantic strength of the projected desire and psi values.
pub const TRAINING_TELEMETRY_PROJECTION_SIGNAL_SEMANTICS: &str = "surrogate";

#[derive(Debug, Error, PartialEq)]
pub enum TrainingTelemetryProjectionError {
    #[error("training telemetry observation '{field}' must be finite")]
    NonFiniteObservation { field: &'static str },
    #[error("training telemetry observation '{field}' must be non-negative")]
    NegativeObservation { field: &'static str },
    #[error("training telemetry config '{field}' must be finite")]
    NonFiniteConfig { field: &'static str },
    #[error("training telemetry config '{field}' must be non-negative")]
    NegativeConfig { field: &'static str },
    #[error("derived training telemetry field '{field}' must be finite")]
    NonFiniteDerived { field: &'static str },
}

/// Raw scalar observations supplied by a training client.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingTelemetryObservation {
    pub step: Option<f64>,
    pub max_steps: Option<f64>,
    pub epoch: Option<f64>,
    pub loss: Option<f64>,
    pub previous_loss: Option<f64>,
    pub grad_norm: Option<f64>,
    pub learning_rate: Option<f64>,
}

/// Tunable gains for the bounded surrogate projection.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingTelemetryProjectionConfig {
    pub desire_gain: f64,
    pub psi_gain: f64,
    pub learning_rate_scale: f64,
}

impl Default for TrainingTelemetryProjectionConfig {
    fn default() -> Self {
        Self {
            desire_gain: 1.0,
            psi_gain: 1.0,
            learning_rate_scale: 10_000.0,
        }
    }
}

/// Canonical request accepted by every language binding.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingTelemetryProjectionRequest {
    pub observation: TrainingTelemetryObservation,
    pub config: TrainingTelemetryProjectionConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainingTelemetryDesireProjection {
    pub gain: f64,
    pub pressure: Option<f64>,
    pub stability: Option<f64>,
    pub saturation: Option<f64>,
    pub improvement_pressure: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainingTelemetryPsiProjection {
    pub gain: f64,
    pub total: Option<f64>,
    pub loss_component: Option<f64>,
    pub gradient_component: Option<f64>,
    pub learning_rate_component: Option<f64>,
}

/// Rust-owned projection payload returned to all clients.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainingTelemetryProjectionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub signal_source: &'static str,
    pub signal_semantics: &'static str,
    pub step: Option<f64>,
    pub max_steps: Option<f64>,
    pub epoch: Option<f64>,
    pub progress: Option<f64>,
    pub loss: Option<f64>,
    pub previous_loss: Option<f64>,
    pub loss_delta: Option<f64>,
    pub loss_improvement: Option<f64>,
    pub grad_norm: Option<f64>,
    pub learning_rate: Option<f64>,
    pub desire: TrainingTelemetryDesireProjection,
    pub psi: TrainingTelemetryPsiProjection,
    pub telemetry: BTreeMap<String, f64>,
    pub observation_count: usize,
    pub psi_component_count: usize,
}

fn validate_observation(
    field: &'static str,
    value: Option<f64>,
    non_negative: bool,
) -> Result<(), TrainingTelemetryProjectionError> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() {
        return Err(TrainingTelemetryProjectionError::NonFiniteObservation { field });
    }
    if non_negative && value < 0.0 {
        return Err(TrainingTelemetryProjectionError::NegativeObservation { field });
    }
    Ok(())
}

fn validate_config(
    field: &'static str,
    value: f64,
) -> Result<(), TrainingTelemetryProjectionError> {
    if !value.is_finite() {
        return Err(TrainingTelemetryProjectionError::NonFiniteConfig { field });
    }
    if value < 0.0 {
        return Err(TrainingTelemetryProjectionError::NegativeConfig { field });
    }
    Ok(())
}

fn bounded_abs(value: f64) -> f64 {
    let magnitude = value.abs();
    if magnitude <= 1.0 {
        magnitude / (1.0 + magnitude)
    } else {
        1.0 - 1.0 / (1.0 + magnitude)
    }
}

fn bounded_scaled_abs(value: f64, scale: f64) -> f64 {
    if value == 0.0 || scale == 0.0 {
        return 0.0;
    }
    let scaled = value.abs() * scale;
    if scaled.is_finite() {
        bounded_abs(scaled)
    } else {
        1.0
    }
}

fn saturating_gain(value: f64, gain: f64) -> f64 {
    let scaled = value * gain;
    if scaled.is_finite() {
        scaled.min(1.0)
    } else {
        1.0
    }
}

fn checked_difference(
    left: f64,
    right: f64,
    field: &'static str,
) -> Result<f64, TrainingTelemetryProjectionError> {
    let value = left - right;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(TrainingTelemetryProjectionError::NonFiniteDerived { field })
    }
}

fn insert_optional(telemetry: &mut BTreeMap<String, f64>, key: &'static str, value: Option<f64>) {
    if let Some(value) = value {
        telemetry.insert(key.to_owned(), value);
    }
}

/// Project finite trainer observations into bounded desire/psi surrogate telemetry.
pub fn project_training_telemetry(
    request: TrainingTelemetryProjectionRequest,
) -> Result<TrainingTelemetryProjectionPayload, TrainingTelemetryProjectionError> {
    let observation = request.observation;
    let config = request.config;

    for (field, value, non_negative) in [
        ("step", observation.step, true),
        ("max_steps", observation.max_steps, true),
        ("epoch", observation.epoch, true),
        ("loss", observation.loss, false),
        ("previous_loss", observation.previous_loss, false),
        ("grad_norm", observation.grad_norm, true),
        ("learning_rate", observation.learning_rate, true),
    ] {
        validate_observation(field, value, non_negative)?;
    }
    validate_config("desire_gain", config.desire_gain)?;
    validate_config("psi_gain", config.psi_gain)?;
    validate_config("learning_rate_scale", config.learning_rate_scale)?;

    let progress = match (observation.step, observation.max_steps) {
        (Some(step), Some(max_steps)) if max_steps > 0.0 => {
            Some((step / max_steps).clamp(0.0, 1.0))
        }
        _ => None,
    };
    let loss_delta = match (observation.loss, observation.previous_loss) {
        (Some(loss), Some(previous_loss)) => {
            Some(checked_difference(loss, previous_loss, "loss_delta")?)
        }
        _ => None,
    };
    let loss_improvement = loss_delta.map(|delta| -delta);
    let loss_pressure = observation.loss.map(bounded_abs);
    let gradient_pressure = observation.grad_norm.map(bounded_abs);
    let learning_rate_pressure = observation
        .learning_rate
        .map(|value| bounded_scaled_abs(value, config.learning_rate_scale));
    let stability = loss_delta.map(|delta| 1.0 / (1.0 + delta.abs()));
    let improvement_pressure = loss_improvement.map(bounded_abs);
    let desire_pressure = loss_pressure.map(|value| saturating_gain(value, config.desire_gain));

    let saturation_terms: Vec<f64> = [loss_pressure, gradient_pressure]
        .into_iter()
        .flatten()
        .collect();
    let desire_saturation = (!saturation_terms.is_empty()).then(|| {
        saturating_gain(
            saturation_terms.iter().sum::<f64>() / saturation_terms.len() as f64,
            config.desire_gain,
        )
    });

    let psi_components: Vec<f64> = [loss_pressure, gradient_pressure, learning_rate_pressure]
        .into_iter()
        .flatten()
        .collect();
    let psi_total = (!psi_components.is_empty()).then(|| {
        saturating_gain(
            psi_components.iter().sum::<f64>() / psi_components.len() as f64,
            config.psi_gain,
        )
    });

    let desire = TrainingTelemetryDesireProjection {
        gain: config.desire_gain,
        pressure: desire_pressure,
        stability,
        saturation: desire_saturation,
        improvement_pressure,
    };
    let psi = TrainingTelemetryPsiProjection {
        gain: config.psi_gain,
        total: psi_total,
        loss_component: loss_pressure,
        gradient_component: gradient_pressure,
        learning_rate_component: learning_rate_pressure,
    };

    let mut telemetry = BTreeMap::new();
    insert_optional(&mut telemetry, "step", observation.step);
    insert_optional(&mut telemetry, "max_steps", observation.max_steps);
    insert_optional(&mut telemetry, "epoch", observation.epoch);
    insert_optional(&mut telemetry, "progress", progress);
    insert_optional(&mut telemetry, "loss", observation.loss);
    insert_optional(&mut telemetry, "loss_delta", loss_delta);
    insert_optional(&mut telemetry, "loss_improvement", loss_improvement);
    insert_optional(&mut telemetry, "grad_norm", observation.grad_norm);
    insert_optional(&mut telemetry, "learning_rate", observation.learning_rate);
    telemetry.insert("desire.gain".to_owned(), desire.gain);
    insert_optional(&mut telemetry, "desire.pressure", desire.pressure);
    insert_optional(&mut telemetry, "desire.stability", desire.stability);
    insert_optional(&mut telemetry, "desire.saturation", desire.saturation);
    insert_optional(
        &mut telemetry,
        "desire.improvement_pressure",
        desire.improvement_pressure,
    );
    telemetry.insert("psi.gain".to_owned(), psi.gain);
    insert_optional(&mut telemetry, "psi.total", psi.total);
    insert_optional(&mut telemetry, "psi.loss_component", psi.loss_component);
    insert_optional(
        &mut telemetry,
        "psi.gradient_component",
        psi.gradient_component,
    );
    insert_optional(
        &mut telemetry,
        "psi.learning_rate_component",
        psi.learning_rate_component,
    );

    let observation_count = [
        observation.step,
        observation.max_steps,
        observation.epoch,
        observation.loss,
        observation.previous_loss,
        observation.grad_norm,
        observation.learning_rate,
    ]
    .into_iter()
    .flatten()
    .count();

    Ok(TrainingTelemetryProjectionPayload {
        kind: TRAINING_TELEMETRY_PROJECTION_KIND,
        contract_version: TRAINING_TELEMETRY_PROJECTION_CONTRACT_VERSION,
        semantic_owner: TRAINING_TELEMETRY_PROJECTION_SEMANTIC_OWNER,
        semantic_backend: TRAINING_TELEMETRY_PROJECTION_SEMANTIC_BACKEND,
        signal_source: TRAINING_TELEMETRY_PROJECTION_SIGNAL_SOURCE,
        signal_semantics: TRAINING_TELEMETRY_PROJECTION_SIGNAL_SEMANTICS,
        step: observation.step,
        max_steps: observation.max_steps,
        epoch: observation.epoch,
        progress,
        loss: observation.loss,
        previous_loss: observation.previous_loss,
        loss_delta,
        loss_improvement,
        grad_norm: observation.grad_norm,
        learning_rate: observation.learning_rate,
        desire,
        psi,
        telemetry,
        observation_count,
        psi_component_count: psi_components.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn example_request() -> TrainingTelemetryProjectionRequest {
        TrainingTelemetryProjectionRequest {
            observation: TrainingTelemetryObservation {
                step: Some(4.0),
                max_steps: Some(10.0),
                epoch: Some(0.4),
                loss: Some(2.0),
                previous_loss: Some(2.5),
                grad_norm: Some(4.0),
                learning_rate: Some(5e-5),
            },
            config: TrainingTelemetryProjectionConfig {
                desire_gain: 1.2,
                psi_gain: 0.8,
                ..TrainingTelemetryProjectionConfig::default()
            },
        }
    }

    #[test]
    fn projection_matches_the_established_hf_proxy_values() {
        let payload = project_training_telemetry(example_request()).expect("valid projection");

        assert_eq!(payload.kind, TRAINING_TELEMETRY_PROJECTION_KIND);
        assert_eq!(payload.signal_source, "trainer_log_proxy");
        assert_eq!(payload.signal_semantics, "surrogate");
        assert_eq!(payload.progress, Some(0.4));
        assert_eq!(payload.loss_delta, Some(-0.5));
        assert_eq!(payload.loss_improvement, Some(0.5));
        assert!((payload.desire.pressure.unwrap() - 0.8).abs() < 1e-12);
        assert!((payload.desire.saturation.unwrap() - 0.88).abs() < 1e-12);
        assert!((payload.desire.stability.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert!((payload.psi.total.unwrap() - 0.48).abs() < 1e-12);
        assert_eq!(payload.observation_count, 7);
        assert_eq!(payload.psi_component_count, 3);
        assert_eq!(
            payload.telemetry.get("psi.total"),
            payload.psi.total.as_ref()
        );
    }

    #[test]
    fn empty_observations_retain_auditable_gains() {
        let payload = project_training_telemetry(Default::default()).expect("empty projection");

        assert_eq!(payload.observation_count, 0);
        assert_eq!(payload.psi_component_count, 0);
        assert_eq!(payload.desire.pressure, None);
        assert_eq!(payload.psi.total, None);
        assert_eq!(payload.telemetry.len(), 2);
        assert_eq!(payload.telemetry["desire.gain"], 1.0);
        assert_eq!(payload.telemetry["psi.gain"], 1.0);
    }

    #[test]
    fn projection_rejects_invalid_observations_and_config() {
        let mut request = example_request();
        request.observation.grad_norm = Some(f64::NAN);
        assert_eq!(
            project_training_telemetry(request).unwrap_err(),
            TrainingTelemetryProjectionError::NonFiniteObservation { field: "grad_norm" }
        );

        let mut request = example_request();
        request.observation.learning_rate = Some(-1e-4);
        assert_eq!(
            project_training_telemetry(request).unwrap_err(),
            TrainingTelemetryProjectionError::NegativeObservation {
                field: "learning_rate"
            }
        );

        let mut request = example_request();
        request.config.psi_gain = f64::INFINITY;
        assert_eq!(
            project_training_telemetry(request).unwrap_err(),
            TrainingTelemetryProjectionError::NonFiniteConfig { field: "psi_gain" }
        );
    }

    #[test]
    fn extreme_finite_values_remain_bounded_and_finite() {
        let payload = project_training_telemetry(TrainingTelemetryProjectionRequest {
            observation: TrainingTelemetryObservation {
                loss: Some(f64::MAX),
                grad_norm: Some(f64::MAX),
                learning_rate: Some(f64::MAX),
                ..TrainingTelemetryObservation::default()
            },
            config: TrainingTelemetryProjectionConfig {
                desire_gain: f64::MAX,
                psi_gain: f64::MAX,
                learning_rate_scale: f64::MAX,
            },
        })
        .expect("extreme finite inputs remain projectable");

        assert_eq!(payload.desire.pressure, Some(1.0));
        assert_eq!(payload.psi.learning_rate_component, Some(1.0));
        assert_eq!(payload.psi.total, Some(1.0));
        assert!(payload.telemetry.values().all(|value| value.is_finite()));
    }

    #[test]
    fn derived_overflow_fails_closed() {
        let request = TrainingTelemetryProjectionRequest {
            observation: TrainingTelemetryObservation {
                loss: Some(f64::MAX),
                previous_loss: Some(-f64::MAX),
                ..TrainingTelemetryObservation::default()
            },
            ..TrainingTelemetryProjectionRequest::default()
        };
        assert_eq!(
            project_training_telemetry(request).unwrap_err(),
            TrainingTelemetryProjectionError::NonFiniteDerived {
                field: "loss_delta"
            }
        );
    }

    #[test]
    fn serde_ingress_rejects_unknown_fields() {
        let error = serde_json::from_value::<TrainingTelemetryProjectionRequest>(json!({
            "observation": {"loss": 2.0, "lozz": 1.0}
        }))
        .expect_err("unknown observation fields must fail closed");
        assert!(error.to_string().contains("unknown field"));
    }
}
