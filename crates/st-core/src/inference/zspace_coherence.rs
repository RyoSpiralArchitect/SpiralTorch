// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical projection from coherence diagnostics into Z-space partial metrics.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const ZSPACE_COHERENCE_PROJECTION_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_coherence_projection.v1";
pub const ZSPACE_COHERENCE_PROJECTION_KIND: &str = "spiraltorch.zspace_coherence_projection";
pub const ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER: &str = "st-core::inference::zspace_coherence";
pub const ZSPACE_COHERENCE_PROJECTION_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_COHERENCE_PROJECTION_FORMULA: &str =
    "speed=tanh(speed_gain*C);C=C_HHI(normalized_weights)|1-H_norm(diagnostic_entropy);memory=tanh(z_bias);stability=tanh(stability_gain*(1-2*H_norm));frac=tanh(frac_gain*fractional_order);drs=tanh(drs_gain*(energy_ratio-0.5))";
pub const ZSPACE_COHERENCE_SUMMARY_FORMULA: &str =
    "p=w/sum(w);H=-sum(p*ln(p));H_norm=H/ln(N);C_HHI=(sum(p^2)-1/N)/(1-1/N);N_eff=exp(H);response_mean=mean(coherence);peak=max(values)";
pub const ZSPACE_COHERENCE_MAX_CHANNELS: usize = 4096;
pub const ZSPACE_COHERENCE_PARTIAL_METRICS: [&str; 22] = [
    "coherence_mean",
    "coherence_entropy",
    "coherence_energy_ratio",
    "coherence_z_bias",
    "coherence_fractional_order",
    "coherence_channels",
    "coherence_preserved",
    "coherence_discarded",
    "coherence_dominant",
    "coherence_peak",
    "coherence_weight_mass",
    "coherence_weight_entropy",
    "coherence_normalized_entropy",
    "coherence_entropy_residual",
    "coherence_concentration",
    "coherence_effective_channels",
    "coherence_response_peak",
    "coherence_response_mean",
    "coherence_strength",
    "coherence_prosody",
    "coherence_articulation",
    "coherence_timbre_spread",
];
const NUMERIC_TOLERANCE_FLOOR: f64 = 1.0e-6;

/// Resolve a coherence projection metric to the spelling owned by this module.
pub fn canonical_zspace_coherence_metric_name(name: &str) -> Option<&'static str> {
    let normalized = name.to_ascii_lowercase();
    ZSPACE_COHERENCE_PARTIAL_METRICS
        .iter()
        .copied()
        .find(|metric| *metric == normalized)
}

fn default_gain() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceCoherenceDiagnosticsInput {
    pub mean_coherence: f64,
    #[serde(alias = "entropy")]
    pub coherence_entropy: f64,
    pub energy_ratio: f64,
    pub z_bias: f64,
    pub fractional_order: f64,
    #[serde(default)]
    pub normalized_weights: Vec<f64>,
    #[serde(default)]
    pub preserved_channels: Option<usize>,
    #[serde(default)]
    pub discarded_channels: Option<usize>,
    #[serde(default)]
    pub dominant_channel: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceCoherenceContourInput {
    pub coherence_strength: f64,
    pub prosody_index: f64,
    pub articulation_bias: f64,
    #[serde(default)]
    pub timbre_spread: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceCoherenceProjectionConfig {
    #[serde(default = "default_gain")]
    pub speed_gain: f64,
    #[serde(default = "default_gain")]
    pub stability_gain: f64,
    #[serde(default = "default_gain")]
    pub frac_gain: f64,
    #[serde(default = "default_gain")]
    pub drs_gain: f64,
}

impl Default for ZSpaceCoherenceProjectionConfig {
    fn default() -> Self {
        Self {
            speed_gain: 1.0,
            stability_gain: 1.0,
            frac_gain: 1.0,
            drs_gain: 1.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceCoherenceProjectionRequest {
    pub diagnostics: ZSpaceCoherenceDiagnosticsInput,
    #[serde(default)]
    pub coherence: Vec<f64>,
    #[serde(default)]
    pub contour: Option<ZSpaceCoherenceContourInput>,
    #[serde(default)]
    pub config: ZSpaceCoherenceProjectionConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceCoherenceProjectionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub projection_formula: &'static str,
    pub summary_formula: &'static str,
    pub diagnostics: ZSpaceCoherenceDiagnosticsInput,
    pub coherence: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contour: Option<ZSpaceCoherenceContourInput>,
    pub config: ZSpaceCoherenceProjectionConfig,
    pub derived: ZSpaceCoherenceProjectionDerived,
    pub partial: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceCoherenceProjectionDerived {
    pub channels: usize,
    pub distribution_source: &'static str,
    pub weight_mass: f64,
    pub weight_entropy: f64,
    pub normalized_entropy: f64,
    pub concentration: f64,
    pub effective_channels: f64,
    pub response_peak: f64,
    pub response_mean: f64,
}

#[derive(Debug, Error, PartialEq)]
pub enum ZSpaceCoherenceProjectionError {
    #[error("Z-space coherence field '{field}' must be finite")]
    NonFiniteScalar { field: &'static str },
    #[error("Z-space coherence field '{field}' entry {index} must be finite")]
    NonFiniteVector { field: &'static str, index: usize },
    #[error("normalized coherence weight {index} must be non-negative")]
    NegativeWeight { index: usize },
    #[error("normalized coherence weight {index} must not exceed one")]
    WeightAboveOne { index: usize },
    #[error("normalized coherence weights must sum to one, received {mass}")]
    InvalidWeightMass { mass: f64 },
    #[error("Z-space coherence projection requires channel evidence")]
    EmptyChannels,
    #[error("Z-space coherence gain '{field}' must be finite and non-negative")]
    InvalidGain { field: &'static str },
    #[error("Z-space coherence channel count {actual} exceeds limit {max}")]
    ChannelLimit { actual: usize, max: usize },
    #[error("dominant coherence channel {dominant} is outside {channels} channels")]
    DominantChannelOutOfRange { dominant: usize, channels: usize },
    #[error("Z-space coherence field '{field}' describes {actual} channels, expected {channels}")]
    ChannelCountMismatch {
        field: &'static str,
        actual: usize,
        channels: usize,
    },
    #[error("Z-space coherence field '{field}' must lie in [{min}, {max}], received {value}")]
    ScalarOutOfRange {
        field: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    #[error("coherence entropy {entropy} exceeds ln({channels})={maximum}")]
    EntropyAboveMaximum {
        entropy: f64,
        channels: usize,
        maximum: f64,
    },
    #[error("derived Z-space coherence metric '{metric}' is not finite")]
    NonFiniteDerived { metric: String },
}

fn validate_scalar(field: &'static str, value: f64) -> Result<(), ZSpaceCoherenceProjectionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ZSpaceCoherenceProjectionError::NonFiniteScalar { field })
    }
}

fn validate_gain(field: &'static str, value: f64) -> Result<(), ZSpaceCoherenceProjectionError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ZSpaceCoherenceProjectionError::InvalidGain { field })
    }
}

fn validate_range(
    field: &'static str,
    value: f64,
    min: f64,
    max: f64,
) -> Result<(), ZSpaceCoherenceProjectionError> {
    validate_scalar(field, value)?;
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(ZSpaceCoherenceProjectionError::ScalarOutOfRange {
            field,
            value,
            min,
            max,
        })
    }
}

fn f32_accumulation_tolerance(channels: usize, scale: f64) -> f64 {
    (channels as f64 * f32::EPSILON as f64 * 4.0 * scale.max(1.0)).max(NUMERIC_TOLERANCE_FLOOR)
}

fn diagnostic_channel_count(diagnostics: &ZSpaceCoherenceDiagnosticsInput) -> usize {
    match (
        diagnostics.preserved_channels,
        diagnostics.discarded_channels,
    ) {
        (Some(preserved), Some(discarded)) => preserved.saturating_add(discarded),
        _ => 0,
    }
}

fn request_channel_count(request: &ZSpaceCoherenceProjectionRequest) -> usize {
    diagnostic_channel_count(&request.diagnostics)
        .max(request.diagnostics.preserved_channels.unwrap_or(0))
        .max(request.diagnostics.discarded_channels.unwrap_or(0))
        .max(request.diagnostics.normalized_weights.len())
        .max(request.coherence.len())
}

fn validate_request(
    request: &ZSpaceCoherenceProjectionRequest,
) -> Result<(), ZSpaceCoherenceProjectionError> {
    let diagnostics = &request.diagnostics;
    validate_range("mean_coherence", diagnostics.mean_coherence, 0.0, 1.0)?;
    validate_range(
        "coherence_entropy",
        diagnostics.coherence_entropy,
        0.0,
        f64::MAX,
    )?;
    validate_range("energy_ratio", diagnostics.energy_ratio, 0.0, 1.0)?;
    validate_scalar("z_bias", diagnostics.z_bias)?;
    validate_range("fractional_order", diagnostics.fractional_order, 0.0, 1.0)?;

    for (index, value) in diagnostics.normalized_weights.iter().enumerate() {
        if !value.is_finite() {
            return Err(ZSpaceCoherenceProjectionError::NonFiniteVector {
                field: "normalized_weights",
                index,
            });
        }
        if *value < 0.0 {
            return Err(ZSpaceCoherenceProjectionError::NegativeWeight { index });
        }
        if *value > 1.0 {
            return Err(ZSpaceCoherenceProjectionError::WeightAboveOne { index });
        }
    }
    if !diagnostics.normalized_weights.is_empty() {
        let mass = diagnostics.normalized_weights.iter().sum::<f64>();
        let tolerance = f32_accumulation_tolerance(diagnostics.normalized_weights.len(), 1.0);
        if (mass - 1.0).abs() > tolerance {
            return Err(ZSpaceCoherenceProjectionError::InvalidWeightMass { mass });
        }
    }
    for (index, value) in request.coherence.iter().enumerate() {
        if !value.is_finite() {
            return Err(ZSpaceCoherenceProjectionError::NonFiniteVector {
                field: "coherence",
                index,
            });
        }
    }

    for count in [
        diagnostics.normalized_weights.len(),
        request.coherence.len(),
        diagnostics.preserved_channels.unwrap_or(0),
        diagnostics.discarded_channels.unwrap_or(0),
    ] {
        if count > ZSPACE_COHERENCE_MAX_CHANNELS {
            return Err(ZSpaceCoherenceProjectionError::ChannelLimit {
                actual: count,
                max: ZSPACE_COHERENCE_MAX_CHANNELS,
            });
        }
    }

    for (field, value) in [
        ("speed_gain", request.config.speed_gain),
        ("stability_gain", request.config.stability_gain),
        ("frac_gain", request.config.frac_gain),
        ("drs_gain", request.config.drs_gain),
    ] {
        validate_gain(field, value)?;
    }

    if let Some(contour) = &request.contour {
        validate_range("coherence_strength", contour.coherence_strength, 0.0, 1.0)?;
        validate_range("prosody_index", contour.prosody_index, 0.0, 1.0)?;
        validate_range("articulation_bias", contour.articulation_bias, 0.0, 4.0)?;
        if let Some(value) = contour.timbre_spread {
            validate_range("timbre_spread", value, 0.0, f64::MAX)?;
        }
    }

    let weight_channels = diagnostics.normalized_weights.len();
    let response_channels = request.coherence.len();
    if weight_channels > 0 && response_channels > 0 && weight_channels != response_channels {
        return Err(ZSpaceCoherenceProjectionError::ChannelCountMismatch {
            field: "coherence",
            actual: response_channels,
            channels: weight_channels,
        });
    }
    let vector_channels = weight_channels.max(response_channels);
    let diagnostic_channels = diagnostic_channel_count(diagnostics);
    let has_complete_counts =
        diagnostics.preserved_channels.is_some() && diagnostics.discarded_channels.is_some();
    if has_complete_counts && vector_channels > 0 && diagnostic_channels != vector_channels {
        return Err(ZSpaceCoherenceProjectionError::ChannelCountMismatch {
            field: "preserved_channels+discarded_channels",
            actual: diagnostic_channels,
            channels: vector_channels,
        });
    }
    for (field, count) in [
        ("preserved_channels", diagnostics.preserved_channels),
        ("discarded_channels", diagnostics.discarded_channels),
    ] {
        if let Some(count) = count {
            if vector_channels > 0 && count > vector_channels {
                return Err(ZSpaceCoherenceProjectionError::ChannelCountMismatch {
                    field,
                    actual: count,
                    channels: vector_channels,
                });
            }
        }
    }
    let channels = request_channel_count(request);
    if channels == 0 {
        return Err(ZSpaceCoherenceProjectionError::EmptyChannels);
    }
    if channels > ZSPACE_COHERENCE_MAX_CHANNELS {
        return Err(ZSpaceCoherenceProjectionError::ChannelLimit {
            actual: channels,
            max: ZSPACE_COHERENCE_MAX_CHANNELS,
        });
    }
    let maximum_entropy = if channels > 1 {
        (channels as f64).ln()
    } else {
        0.0
    };
    let entropy_tolerance = f32_accumulation_tolerance(channels, maximum_entropy);
    if diagnostics.coherence_entropy > maximum_entropy + entropy_tolerance {
        return Err(ZSpaceCoherenceProjectionError::EntropyAboveMaximum {
            entropy: diagnostics.coherence_entropy,
            channels,
            maximum: maximum_entropy,
        });
    }
    if let Some(dominant) = diagnostics.dominant_channel {
        if dominant >= channels {
            return Err(ZSpaceCoherenceProjectionError::DominantChannelOutOfRange {
                dominant,
                channels,
            });
        }
    }
    Ok(())
}

fn peak(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(0.0)
}

fn derive_projection_summary(
    request: &ZSpaceCoherenceProjectionRequest,
) -> ZSpaceCoherenceProjectionDerived {
    let channels = request_channel_count(request);
    let diagnostics = &request.diagnostics;
    let response_mean = if request.coherence.is_empty() {
        0.0
    } else {
        request.coherence.iter().sum::<f64>() / request.coherence.len() as f64
    };

    if diagnostics.normalized_weights.is_empty() {
        let maximum_entropy = if channels > 1 {
            (channels as f64).ln()
        } else {
            0.0
        };
        let normalized_entropy = if maximum_entropy > 0.0 {
            (diagnostics.coherence_entropy / maximum_entropy).clamp(0.0, 1.0)
        } else {
            0.0
        };
        return ZSpaceCoherenceProjectionDerived {
            channels,
            distribution_source: "diagnostic_entropy",
            weight_mass: 0.0,
            weight_entropy: diagnostics.coherence_entropy,
            normalized_entropy,
            concentration: 1.0 - normalized_entropy,
            effective_channels: diagnostics.coherence_entropy.exp().min(channels as f64),
            response_peak: peak(&request.coherence),
            response_mean,
        };
    }

    let weight_mass = diagnostics.normalized_weights.iter().sum::<f64>();
    let probabilities = diagnostics
        .normalized_weights
        .iter()
        .map(|weight| *weight / weight_mass);
    let (weight_entropy, squared_mass) =
        probabilities.fold((0.0, 0.0), |(entropy, squares), probability| {
            let entropy_term = if probability > 0.0 {
                -probability * probability.ln()
            } else {
                0.0
            };
            (entropy + entropy_term, squares + probability * probability)
        });
    let distribution_channels = diagnostics.normalized_weights.len();
    let maximum_entropy = if distribution_channels > 1 {
        (distribution_channels as f64).ln()
    } else {
        0.0
    };
    let normalized_entropy = if maximum_entropy > 0.0 {
        (weight_entropy / maximum_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let concentration = if distribution_channels > 1 {
        let uniform_concentration = 1.0 / distribution_channels as f64;
        ((squared_mass - uniform_concentration) / (1.0 - uniform_concentration)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    ZSpaceCoherenceProjectionDerived {
        channels,
        distribution_source: "normalized_weights",
        weight_mass,
        weight_entropy,
        normalized_entropy,
        concentration,
        effective_channels: weight_entropy.exp().min(distribution_channels as f64),
        response_peak: peak(&request.coherence),
        response_mean,
    }
}

/// Project validated coherence diagnostics into canonical Z-space partial metrics.
pub fn project_zspace_coherence(
    request: ZSpaceCoherenceProjectionRequest,
) -> Result<ZSpaceCoherenceProjectionPayload, ZSpaceCoherenceProjectionError> {
    validate_request(&request)?;
    let diagnostics = &request.diagnostics;
    let config = &request.config;
    let derived = derive_projection_summary(&request);
    let channels = derived.channels;
    let observed_preserved = if !diagnostics.normalized_weights.is_empty() {
        diagnostics
            .normalized_weights
            .iter()
            .filter(|weight| **weight > 0.0)
            .count()
    } else if !request.coherence.is_empty() {
        request
            .coherence
            .iter()
            .filter(|weight| **weight > 0.0)
            .count()
    } else {
        channels
    };
    let preserved = diagnostics.preserved_channels.unwrap_or_else(|| {
        diagnostics
            .discarded_channels
            .map(|discarded| channels.saturating_sub(discarded))
            .unwrap_or(observed_preserved)
    });
    let discarded = diagnostics
        .discarded_channels
        .unwrap_or_else(|| channels.saturating_sub(preserved));

    let mut partial = BTreeMap::from([
        (
            "speed".to_owned(),
            (config.speed_gain * derived.concentration).tanh(),
        ),
        ("memory".to_owned(), diagnostics.z_bias.tanh()),
        (
            "stability".to_owned(),
            (config.stability_gain * (1.0 - 2.0 * derived.normalized_entropy)).tanh(),
        ),
        (
            "frac".to_owned(),
            (config.frac_gain * diagnostics.fractional_order).tanh(),
        ),
        (
            "drs".to_owned(),
            (config.drs_gain * (diagnostics.energy_ratio - 0.5)).tanh(),
        ),
        ("coherence_mean".to_owned(), diagnostics.mean_coherence),
        (
            "coherence_entropy".to_owned(),
            diagnostics.coherence_entropy,
        ),
        (
            "coherence_energy_ratio".to_owned(),
            diagnostics.energy_ratio,
        ),
        ("coherence_z_bias".to_owned(), diagnostics.z_bias),
        (
            "coherence_fractional_order".to_owned(),
            diagnostics.fractional_order,
        ),
        ("coherence_channels".to_owned(), channels as f64),
        ("coherence_preserved".to_owned(), preserved as f64),
        ("coherence_discarded".to_owned(), discarded as f64),
        (
            "coherence_peak".to_owned(),
            peak(&diagnostics.normalized_weights),
        ),
        ("coherence_weight_mass".to_owned(), derived.weight_mass),
        (
            "coherence_weight_entropy".to_owned(),
            derived.weight_entropy,
        ),
        (
            "coherence_normalized_entropy".to_owned(),
            derived.normalized_entropy,
        ),
        (
            "coherence_entropy_residual".to_owned(),
            diagnostics.coherence_entropy - derived.weight_entropy,
        ),
        ("coherence_concentration".to_owned(), derived.concentration),
        (
            "coherence_effective_channels".to_owned(),
            derived.effective_channels,
        ),
        ("coherence_response_peak".to_owned(), derived.response_peak),
        ("coherence_response_mean".to_owned(), derived.response_mean),
    ]);
    if let Some(dominant) = diagnostics.dominant_channel {
        partial.insert("coherence_dominant".to_owned(), dominant as f64);
    }
    if let Some(contour) = &request.contour {
        partial.insert("coherence_strength".to_owned(), contour.coherence_strength);
        partial.insert("coherence_prosody".to_owned(), contour.prosody_index);
        partial.insert(
            "coherence_articulation".to_owned(),
            contour.articulation_bias,
        );
        if let Some(timbre_spread) = contour.timbre_spread {
            partial.insert("coherence_timbre_spread".to_owned(), timbre_spread);
        }
    }
    if let Some((metric, _)) = partial.iter().find(|(_, value)| !value.is_finite()) {
        return Err(ZSpaceCoherenceProjectionError::NonFiniteDerived {
            metric: metric.clone(),
        });
    }

    Ok(ZSpaceCoherenceProjectionPayload {
        kind: ZSPACE_COHERENCE_PROJECTION_KIND,
        contract_version: ZSPACE_COHERENCE_PROJECTION_CONTRACT_VERSION,
        semantic_owner: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_BACKEND,
        projection_formula: ZSPACE_COHERENCE_PROJECTION_FORMULA,
        summary_formula: ZSPACE_COHERENCE_SUMMARY_FORMULA,
        diagnostics: request.diagnostics,
        coherence: request.coherence,
        contour: request.contour,
        config: request.config,
        derived,
        partial,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use serde_json::json;

    fn request() -> ZSpaceCoherenceProjectionRequest {
        let weight_entropy = 1.0296530140645737;
        ZSpaceCoherenceProjectionRequest {
            diagnostics: ZSpaceCoherenceDiagnosticsInput {
                mean_coherence: 0.25,
                coherence_entropy: weight_entropy,
                energy_ratio: 0.7,
                z_bias: -0.12,
                fractional_order: 0.4,
                normalized_weights: vec![0.5, 0.3, 0.2, 0.0],
                preserved_channels: Some(3),
                discarded_channels: Some(1),
                dominant_channel: Some(0),
            },
            coherence: vec![0.5, 0.3, 0.2, 0.0],
            contour: None,
            config: ZSpaceCoherenceProjectionConfig::default(),
        }
    }

    #[test]
    fn projection_uses_dimension_normalized_distribution_controls() {
        let payload = project_zspace_coherence(request()).expect("valid coherence projection");
        let expected_concentration = (0.38_f64 - 0.25) / 0.75;
        let expected_normalized_entropy = 1.0296530140645737_f64 / 4.0_f64.ln();

        assert_eq!(payload.semantic_backend, "rust");
        assert_eq!(payload.derived.distribution_source, "normalized_weights");
        assert_relative_eq!(
            payload.partial["speed"],
            expected_concentration.tanh(),
            epsilon = 1e-14
        );
        assert_relative_eq!(
            payload.partial["stability"],
            (1.0 - 2.0 * expected_normalized_entropy).tanh(),
            epsilon = 1e-14
        );
        assert_relative_eq!(
            payload.partial["memory"],
            (-0.12_f64).tanh(),
            epsilon = 1e-14
        );
        assert_relative_eq!(
            payload.partial["coherence_response_mean"],
            0.25,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            payload.partial["coherence_weight_entropy"],
            1.0296530140645737,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            payload.partial["coherence_concentration"],
            expected_concentration,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            payload.partial["coherence_normalized_entropy"],
            expected_normalized_entropy,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            payload.partial["coherence_entropy_residual"],
            0.0,
            epsilon = 1e-14
        );
        for metric in payload
            .partial
            .keys()
            .filter(|metric| metric.starts_with("coherence_"))
        {
            assert_eq!(
                canonical_zspace_coherence_metric_name(metric),
                Some(metric.as_str())
            );
        }
        assert_eq!(payload.partial["coherence_channels"], 4.0);

        let mut inferred_counts = request();
        inferred_counts.diagnostics.discarded_channels = None;
        let inferred = project_zspace_coherence(inferred_counts).expect("one-sided count");
        assert_eq!(inferred.partial["coherence_preserved"], 3.0);
        assert_eq!(inferred.partial["coherence_discarded"], 1.0);

        let mut inferred_counts = request();
        inferred_counts.diagnostics.preserved_channels = None;
        inferred_counts.diagnostics.discarded_channels = None;
        let inferred = project_zspace_coherence(inferred_counts).expect("weight-derived counts");
        assert_eq!(inferred.partial["coherence_preserved"], 3.0);
        assert_eq!(inferred.partial["coherence_discarded"], 1.0);
    }

    #[test]
    fn uniform_and_concentrated_distributions_are_dimension_invariant() {
        fn distribution_request(
            channels: usize,
            concentrated: bool,
        ) -> ZSpaceCoherenceProjectionRequest {
            let mut weights = vec![1.0 / channels as f64; channels];
            if concentrated {
                weights.fill(0.0);
                weights[0] = 1.0;
            }
            ZSpaceCoherenceProjectionRequest {
                diagnostics: ZSpaceCoherenceDiagnosticsInput {
                    mean_coherence: 1.0 / channels as f64,
                    coherence_entropy: if concentrated {
                        0.0
                    } else {
                        (channels as f64).ln()
                    },
                    energy_ratio: 0.5,
                    z_bias: 0.0,
                    fractional_order: 0.5,
                    normalized_weights: weights.clone(),
                    preserved_channels: Some(channels),
                    discarded_channels: Some(0),
                    dominant_channel: Some(0),
                },
                coherence: weights,
                contour: None,
                config: ZSpaceCoherenceProjectionConfig::default(),
            }
        }

        let uniform_four = project_zspace_coherence(distribution_request(4, false)).unwrap();
        let uniform_sixteen = project_zspace_coherence(distribution_request(16, false)).unwrap();
        assert_relative_eq!(uniform_four.partial["speed"], 0.0, epsilon = 1e-14);
        assert_relative_eq!(
            uniform_four.partial["speed"],
            uniform_sixteen.partial["speed"],
            epsilon = 1e-14
        );
        assert_relative_eq!(
            uniform_four.partial["stability"],
            uniform_sixteen.partial["stability"],
            epsilon = 1e-14
        );

        let concentrated_four = project_zspace_coherence(distribution_request(4, true)).unwrap();
        let concentrated_sixteen =
            project_zspace_coherence(distribution_request(16, true)).unwrap();
        assert_relative_eq!(
            concentrated_four.partial["speed"],
            1.0_f64.tanh(),
            epsilon = 1e-14
        );
        assert_relative_eq!(
            concentrated_four.partial["speed"],
            concentrated_sixteen.partial["speed"],
            epsilon = 1e-14
        );
        assert_relative_eq!(
            concentrated_four.partial["stability"],
            concentrated_sixteen.partial["stability"],
            epsilon = 1e-14
        );
    }

    #[test]
    fn trace_entropy_alias_and_contour_are_part_of_the_same_contract() {
        let request: ZSpaceCoherenceProjectionRequest = serde_json::from_value(json!({
            "diagnostics": {
                "mean_coherence": 0.2,
                "entropy": 0.3,
                "energy_ratio": 0.6,
                "z_bias": 0.1,
                "fractional_order": 0.5,
                "preserved_channels": 2,
                "discarded_channels": 1,
                "dominant_channel": 1
            },
            "coherence": [0.1, 0.4, 0.2],
            "contour": {
                "coherence_strength": 0.8,
                "prosody_index": 0.6,
                "articulation_bias": 0.1,
                "timbre_spread": 0.25
            }
        }))
        .expect("trace request");
        let payload = project_zspace_coherence(request).expect("valid trace projection");

        assert_eq!(payload.partial["coherence_entropy"], 0.3);
        assert_eq!(payload.partial["coherence_response_peak"], 0.4);
        assert_eq!(payload.partial["coherence_timbre_spread"], 0.25);
        assert_eq!(payload.partial["coherence_channels"], 3.0);
        assert_eq!(payload.derived.distribution_source, "diagnostic_entropy");
    }

    #[test]
    fn projection_rejects_invalid_inputs_instead_of_silently_repairing_them() {
        let mut invalid = request();
        invalid.config.speed_gain = -1.0;
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::InvalidGain {
                field: "speed_gain"
            })
        ));

        let mut invalid = request();
        invalid.diagnostics.normalized_weights[1] = -0.1;
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::NegativeWeight { index: 1 })
        ));

        let mut invalid = request();
        invalid.diagnostics.normalized_weights[1] = 1.1;
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::WeightAboveOne { index: 1 })
        ));

        let mut invalid = request();
        invalid.diagnostics.normalized_weights[1] = 0.2;
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::InvalidWeightMass { .. })
        ));

        let mut invalid = request();
        invalid.diagnostics.normalized_weights.pop();
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::ChannelCountMismatch { .. })
        ));

        let mut invalid = request();
        invalid.coherence.pop();
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::ChannelCountMismatch {
                field: "coherence",
                ..
            })
        ));

        let mut invalid = request();
        invalid.diagnostics.coherence_entropy = 4.0_f64.ln() + 0.01;
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::EntropyAboveMaximum { .. })
        ));

        let mut invalid = request();
        invalid.coherence[0] = f64::NAN;
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::NonFiniteVector {
                field: "coherence",
                index: 0
            })
        ));

        let mut invalid = request();
        invalid.diagnostics.normalized_weights.clear();
        invalid.coherence.clear();
        invalid.diagnostics.preserved_channels = Some(ZSPACE_COHERENCE_MAX_CHANNELS);
        invalid.diagnostics.discarded_channels = Some(1);
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::ChannelLimit { .. })
        ));

        let mut invalid = request();
        invalid.diagnostics.normalized_weights.clear();
        invalid.coherence.clear();
        invalid.diagnostics.preserved_channels = None;
        invalid.diagnostics.discarded_channels = None;
        assert_eq!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::EmptyChannels)
        );

        let mut invalid = request();
        invalid.coherence = vec![f64::MAX; 4];
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::NonFiniteDerived { .. })
        ));
    }
}
