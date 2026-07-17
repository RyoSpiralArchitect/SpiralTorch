// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical projection from coherence diagnostics into Z-space partial metrics.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const ZSPACE_COHERENCE_PROJECTION_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_coherence_projection.v2";
pub const ZSPACE_COHERENCE_PROJECTION_KIND: &str = "spiraltorch.zspace_coherence_projection";
pub const ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER: &str = "st-core::inference::zspace_coherence";
pub const ZSPACE_COHERENCE_PROJECTION_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_COHERENCE_PROJECTION_FORMULA: &str =
    "speed=tanh(speed_gain*C);C=C_HHI(normalized_weights)|1-H_norm(diagnostic_entropy);memory=tanh(z_bias);stability=tanh(stability_gain*(1-2*H_norm));frac=tanh(frac_gain*fractional_order);drs=tanh(drs_gain*(energy_ratio-0.5))";
pub const ZSPACE_COHERENCE_SUMMARY_FORMULA: &str =
    "p=w/sum(w);H=-sum(p*ln(p));H_norm=H/ln(N);C_HHI=(sum(p^2)-1/N)/(1-1/N);N_eff=exp(H);response_mean=mean(coherence);peak=max(values)";
pub const ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_KIND: &str =
    "spiraltorch.zspace_coherence_distribution_witness";
pub const ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_coherence_distribution_witness.v1";
pub const ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_FORMULA: &str =
    "normalized_weights are the complete simplex witness;all distribution summaries and controls must be re-derived by st-core";
pub const ZSPACE_COHERENCE_EVIDENCE_VALIDATION_FORMULA: &str =
    "normalized_weights=>coherence_entropy~=H(p),preserved_channels=count(p>0),discarded_channels=count(p=0),dominant_channel in argmax(p);coherence=>mean_coherence~=mean(coherence);comparisons use channel-scaled f32 tolerance";
pub const ZSPACE_COHERENCE_CLASSIFICATION_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_coherence_classification.v1";
pub const ZSPACE_COHERENCE_CLASSIFICATION_KIND: &str =
    "spiraltorch.zspace_coherence_classification";
pub const ZSPACE_COHERENCE_CLASSIFICATION_FORMULA: &str =
    "threshold comparisons tolerate one relative f32 epsilon;background if energy_ratio<=background_energy_ratio_max;symmetric_pulse if swap_invariant;cascade_imbalance if energy_ratio>=cascade_energy_ratio_min;diffuse_drift otherwise";
pub const ZSPACE_COHERENCE_CONTROL_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_coherence_control.v1";
pub const ZSPACE_COHERENCE_CONTROL_KIND: &str = "spiraltorch.zspace_coherence_control";
pub const ZSPACE_COHERENCE_CONTROL_FORMULA: &str =
    "p=w/sum(w);rho(p*p^T)=sum(p^2);spectral_radius=(rho-1/N)/(1-1/N)=C_HHI (N=1 => 1);spectral_entropy=H/ln(N) (N=1 => 0);spectral_pressure=energy_ratio*(1-spectral_entropy);N_eff=exp(H)";
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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpaceCoherenceClassificationPolicy {
    pub background_energy_ratio_max: f64,
    pub cascade_energy_ratio_min: f64,
}

impl Default for ZSpaceCoherenceClassificationPolicy {
    fn default() -> Self {
        Self {
            background_energy_ratio_max: 1.0e-5,
            cascade_energy_ratio_min: 0.7,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ZSpaceCoherenceLabel {
    Background,
    SymmetricPulse,
    CascadeImbalance,
    DiffuseDrift,
}

impl ZSpaceCoherenceLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Background => "background",
            Self::SymmetricPulse => "symmetric_pulse",
            Self::CascadeImbalance => "cascade_imbalance",
            Self::DiffuseDrift => "diffuse_drift",
        }
    }
}

impl std::fmt::Display for ZSpaceCoherenceLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceCoherenceClassificationRequest {
    pub energy_ratio: f64,
    pub swap_invariant: bool,
    #[serde(default)]
    pub policy: ZSpaceCoherenceClassificationPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ZSpaceCoherenceClassificationPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub classification_formula: &'static str,
    pub label: ZSpaceCoherenceLabel,
    pub reason: &'static str,
    pub energy_ratio: f64,
    pub swap_invariant: bool,
    pub policy: ZSpaceCoherenceClassificationPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct ZSpaceCoherenceDistributionSummary {
    pub channels: usize,
    pub weight_mass: f64,
    pub weight_entropy: f64,
    pub normalized_entropy: f64,
    pub concentration: f64,
    pub effective_channels: f64,
}

/// Complete, portable evidence for reconstructing coherence distribution semantics.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceCoherenceDistributionWitness {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub witness_formula: String,
    pub normalized_weights: Vec<f64>,
}

/// Dimensionless coherence controls derived by the canonical Rust contract.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ZSpaceCoherenceControlPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub control_formula: &'static str,
    pub channels: usize,
    pub raw_mean_coherence: f64,
    pub raw_coherence_entropy: f64,
    pub spectral_radius: f64,
    pub spectral_entropy: f64,
    pub spectral_pressure: f64,
    pub effective_channels: f64,
    pub energy_ratio: f64,
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
    #[serde(default)]
    pub classification_policy: ZSpaceCoherenceClassificationPolicy,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceCoherenceProjectionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub projection_formula: &'static str,
    pub summary_formula: &'static str,
    pub evidence_validation_formula: &'static str,
    pub diagnostics: ZSpaceCoherenceDiagnosticsInput,
    pub coherence: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contour: Option<ZSpaceCoherenceContourInput>,
    pub config: ZSpaceCoherenceProjectionConfig,
    pub derived: ZSpaceCoherenceProjectionDerived,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification: Option<ZSpaceCoherenceClassificationPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control: Option<ZSpaceCoherenceControlPayload>,
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
    #[error(
        "Z-space coherence background energy-ratio maximum {background_energy_ratio_max} must not exceed cascade energy-ratio minimum {cascade_energy_ratio_min}"
    )]
    InvalidClassificationPolicy {
        background_energy_ratio_max: f64,
        cascade_energy_ratio_min: f64,
    },
    #[error(
        "Z-space coherence distribution metric '{field}' is inconsistent: received {value}, expected {expected}"
    )]
    InconsistentDistributionMetric {
        field: &'static str,
        value: f64,
        expected: f64,
    },
    #[error(
        "Z-space coherence field '{field}' reports {actual} supported channels, observed {expected}"
    )]
    InconsistentChannelSupport {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error(
        "dominant coherence channel {dominant} has normalized weight {value}, below observed peak {peak}"
    )]
    InconsistentDominantChannel {
        dominant: usize,
        value: f64,
        peak: f64,
    },
    #[error(
        "Z-space coherence distribution witness field '{field}' must be '{expected}', received '{actual}'"
    )]
    InvalidDistributionWitnessContract {
        field: &'static str,
        actual: String,
        expected: &'static str,
    },
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

fn at_or_below_f32_boundary(value: f64, threshold: f64) -> bool {
    value <= threshold || within_f32_boundary(value, threshold)
}

fn at_or_above_f32_boundary(value: f64, threshold: f64) -> bool {
    value >= threshold || within_f32_boundary(value, threshold)
}

fn within_f32_boundary(value: f64, threshold: f64) -> bool {
    let scale = value
        .abs()
        .max(threshold.abs())
        .max(f32::MIN_POSITIVE as f64);
    (value - threshold).abs() <= f32::EPSILON as f64 * scale
}

/// Summarize a normalized coherence simplex with the canonical dimensionless metrics.
pub fn summarize_zspace_coherence_distribution<T>(
    weights: &[T],
) -> Result<ZSpaceCoherenceDistributionSummary, ZSpaceCoherenceProjectionError>
where
    T: Copy,
    f64: From<T>,
{
    if weights.is_empty() {
        return Err(ZSpaceCoherenceProjectionError::EmptyChannels);
    }
    if weights.len() > ZSPACE_COHERENCE_MAX_CHANNELS {
        return Err(ZSpaceCoherenceProjectionError::ChannelLimit {
            actual: weights.len(),
            max: ZSPACE_COHERENCE_MAX_CHANNELS,
        });
    }

    let mut weight_mass = 0.0;
    for (index, weight) in weights.iter().copied().map(f64::from).enumerate() {
        if !weight.is_finite() {
            return Err(ZSpaceCoherenceProjectionError::NonFiniteVector {
                field: "normalized_weights",
                index,
            });
        }
        if weight < 0.0 {
            return Err(ZSpaceCoherenceProjectionError::NegativeWeight { index });
        }
        if weight > 1.0 {
            return Err(ZSpaceCoherenceProjectionError::WeightAboveOne { index });
        }
        weight_mass += weight;
    }
    let tolerance = f32_accumulation_tolerance(weights.len(), 1.0);
    if (weight_mass - 1.0).abs() > tolerance {
        return Err(ZSpaceCoherenceProjectionError::InvalidWeightMass { mass: weight_mass });
    }

    let (weight_entropy, squared_mass) = weights
        .iter()
        .copied()
        .map(f64::from)
        .map(|weight| weight / weight_mass)
        .fold((0.0, 0.0), |(entropy, squares), probability| {
            let entropy_term = if probability > 0.0 {
                -probability * probability.ln()
            } else {
                0.0
            };
            (entropy + entropy_term, squares + probability * probability)
        });
    let channels = weights.len();
    let maximum_entropy = if channels > 1 {
        (channels as f64).ln()
    } else {
        0.0
    };
    let normalized_entropy = if maximum_entropy > 0.0 {
        (weight_entropy / maximum_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let concentration = if channels > 1 {
        let uniform_concentration = 1.0 / channels as f64;
        ((squared_mass - uniform_concentration) / (1.0 - uniform_concentration)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    Ok(ZSpaceCoherenceDistributionSummary {
        channels,
        weight_mass,
        weight_entropy,
        normalized_entropy,
        concentration,
        effective_channels: weight_entropy.exp().min(channels as f64),
    })
}

/// Build a portable simplex witness after validating the canonical Rust distribution contract.
pub fn build_zspace_coherence_distribution_witness<T>(
    weights: &[T],
) -> Result<ZSpaceCoherenceDistributionWitness, ZSpaceCoherenceProjectionError>
where
    T: Copy,
    f64: From<T>,
{
    summarize_zspace_coherence_distribution(weights)?;
    Ok(ZSpaceCoherenceDistributionWitness {
        kind: ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_KIND.to_owned(),
        contract_version: ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_CONTRACT_VERSION.to_owned(),
        semantic_owner: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER.to_owned(),
        semantic_backend: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_BACKEND.to_owned(),
        witness_formula: ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_FORMULA.to_owned(),
        normalized_weights: weights.iter().copied().map(f64::from).collect(),
    })
}

fn validate_distribution_witness_contract_field(
    field: &'static str,
    actual: &str,
    expected: &'static str,
) -> Result<(), ZSpaceCoherenceProjectionError> {
    if actual == expected {
        Ok(())
    } else {
        Err(
            ZSpaceCoherenceProjectionError::InvalidDistributionWitnessContract {
                field,
                actual: actual.to_owned(),
                expected,
            },
        )
    }
}

/// Validate witness identity and reconstruct all distribution metrics from its simplex.
pub fn validate_zspace_coherence_distribution_witness(
    witness: &ZSpaceCoherenceDistributionWitness,
) -> Result<ZSpaceCoherenceDistributionSummary, ZSpaceCoherenceProjectionError> {
    validate_distribution_witness_contract_field(
        "kind",
        &witness.kind,
        ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_KIND,
    )?;
    validate_distribution_witness_contract_field(
        "contract_version",
        &witness.contract_version,
        ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_CONTRACT_VERSION,
    )?;
    validate_distribution_witness_contract_field(
        "semantic_owner",
        &witness.semantic_owner,
        ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER,
    )?;
    validate_distribution_witness_contract_field(
        "semantic_backend",
        &witness.semantic_backend,
        ZSPACE_COHERENCE_PROJECTION_SEMANTIC_BACKEND,
    )?;
    validate_distribution_witness_contract_field(
        "witness_formula",
        &witness.witness_formula,
        ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_FORMULA,
    )?;
    summarize_zspace_coherence_distribution(&witness.normalized_weights)
}

/// Validate observed support and dominant-channel identity against a witness.
pub fn validate_zspace_coherence_distribution_observation(
    witness: &ZSpaceCoherenceDistributionWitness,
    preserved_channels: usize,
    dominant_channel: Option<usize>,
) -> Result<ZSpaceCoherenceDistributionSummary, ZSpaceCoherenceProjectionError> {
    let distribution = validate_zspace_coherence_distribution_witness(witness)?;
    validate_distribution_observation(
        &witness.normalized_weights,
        distribution,
        Some(preserved_channels),
        None,
        dominant_channel,
    )?;
    Ok(distribution)
}

/// Derive dimensionless trainer/runtime controls from a canonical distribution summary.
pub fn derive_zspace_coherence_control(
    distribution: ZSpaceCoherenceDistributionSummary,
    energy_ratio: f64,
    raw_mean_coherence: f64,
) -> Result<ZSpaceCoherenceControlPayload, ZSpaceCoherenceProjectionError> {
    if distribution.channels == 0 {
        return Err(ZSpaceCoherenceProjectionError::EmptyChannels);
    }
    if distribution.channels > ZSPACE_COHERENCE_MAX_CHANNELS {
        return Err(ZSpaceCoherenceProjectionError::ChannelLimit {
            actual: distribution.channels,
            max: ZSPACE_COHERENCE_MAX_CHANNELS,
        });
    }
    validate_range("energy_ratio", energy_ratio, 0.0, 1.0)?;
    validate_range("mean_coherence", raw_mean_coherence, 0.0, 1.0)?;
    validate_range(
        "normalized_entropy",
        distribution.normalized_entropy,
        0.0,
        1.0,
    )?;
    validate_range("concentration", distribution.concentration, 0.0, 1.0)?;
    validate_range(
        "effective_channels",
        distribution.effective_channels,
        1.0,
        distribution.channels as f64,
    )?;
    validate_scalar("weight_mass", distribution.weight_mass)?;
    validate_scalar("weight_entropy", distribution.weight_entropy)?;
    let mass_tolerance = f32_accumulation_tolerance(distribution.channels, 1.0);
    if (distribution.weight_mass - 1.0).abs() > mass_tolerance {
        return Err(ZSpaceCoherenceProjectionError::InvalidWeightMass {
            mass: distribution.weight_mass,
        });
    }

    let maximum_entropy = if distribution.channels > 1 {
        (distribution.channels as f64).ln()
    } else {
        0.0
    };
    let entropy_tolerance = f32_accumulation_tolerance(distribution.channels, maximum_entropy);
    if distribution.weight_entropy < 0.0
        || distribution.weight_entropy > maximum_entropy + entropy_tolerance
    {
        return Err(ZSpaceCoherenceProjectionError::ScalarOutOfRange {
            field: "weight_entropy",
            value: distribution.weight_entropy,
            min: 0.0,
            max: maximum_entropy,
        });
    }
    let expected_normalized_entropy = if maximum_entropy > 0.0 {
        (distribution.weight_entropy / maximum_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if (distribution.normalized_entropy - expected_normalized_entropy).abs() > entropy_tolerance {
        return Err(
            ZSpaceCoherenceProjectionError::InconsistentDistributionMetric {
                field: "normalized_entropy",
                value: distribution.normalized_entropy,
                expected: expected_normalized_entropy,
            },
        );
    }
    let expected_effective_channels = distribution
        .weight_entropy
        .exp()
        .min(distribution.channels as f64);
    if (distribution.effective_channels - expected_effective_channels).abs() > entropy_tolerance {
        return Err(
            ZSpaceCoherenceProjectionError::InconsistentDistributionMetric {
                field: "effective_channels",
                value: distribution.effective_channels,
                expected: expected_effective_channels,
            },
        );
    }

    let spectral_radius = distribution.concentration;
    let spectral_entropy = distribution.normalized_entropy;
    let spectral_pressure = energy_ratio * (1.0 - spectral_entropy);
    for (metric, value) in [
        ("spectral_radius", spectral_radius),
        ("spectral_entropy", spectral_entropy),
        ("spectral_pressure", spectral_pressure),
    ] {
        if !value.is_finite() {
            return Err(ZSpaceCoherenceProjectionError::NonFiniteDerived {
                metric: metric.to_owned(),
            });
        }
    }

    Ok(ZSpaceCoherenceControlPayload {
        kind: ZSPACE_COHERENCE_CONTROL_KIND,
        contract_version: ZSPACE_COHERENCE_CONTROL_CONTRACT_VERSION,
        semantic_owner: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_BACKEND,
        control_formula: ZSPACE_COHERENCE_CONTROL_FORMULA,
        channels: distribution.channels,
        raw_mean_coherence,
        raw_coherence_entropy: distribution.weight_entropy,
        spectral_radius,
        spectral_entropy,
        spectral_pressure,
        effective_channels: distribution.effective_channels,
        energy_ratio,
    })
}

/// Return whether a non-empty, fully finite arrangement is equal within the shared tolerance.
pub fn is_zspace_coherence_swap_invariant<T>(arrangement: &[T]) -> bool
where
    T: Copy,
    f64: From<T>,
{
    if arrangement.is_empty() {
        return false;
    }

    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for value in arrangement.iter().copied().map(f64::from) {
        if !value.is_finite() {
            return false;
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    let scale = minimum.abs().max(maximum.abs());
    maximum - minimum <= (scale * 1.0e-4).max(NUMERIC_TOLERANCE_FLOOR)
}

/// Classify a coherence observation through the canonical Rust policy.
pub fn classify_zspace_coherence(
    request: ZSpaceCoherenceClassificationRequest,
) -> Result<ZSpaceCoherenceClassificationPayload, ZSpaceCoherenceProjectionError> {
    validate_range("energy_ratio", request.energy_ratio, 0.0, 1.0)?;
    validate_range(
        "background_energy_ratio_max",
        request.policy.background_energy_ratio_max,
        0.0,
        1.0,
    )?;
    validate_range(
        "cascade_energy_ratio_min",
        request.policy.cascade_energy_ratio_min,
        0.0,
        1.0,
    )?;
    if request.policy.background_energy_ratio_max > request.policy.cascade_energy_ratio_min {
        return Err(
            ZSpaceCoherenceProjectionError::InvalidClassificationPolicy {
                background_energy_ratio_max: request.policy.background_energy_ratio_max,
                cascade_energy_ratio_min: request.policy.cascade_energy_ratio_min,
            },
        );
    }

    let (label, reason) = if at_or_below_f32_boundary(
        request.energy_ratio,
        request.policy.background_energy_ratio_max,
    ) {
        (
            ZSpaceCoherenceLabel::Background,
            "energy_ratio_at_or_below_background_max",
        )
    } else if request.swap_invariant {
        (ZSpaceCoherenceLabel::SymmetricPulse, "swap_invariant")
    } else if at_or_above_f32_boundary(
        request.energy_ratio,
        request.policy.cascade_energy_ratio_min,
    ) {
        (
            ZSpaceCoherenceLabel::CascadeImbalance,
            "dominant_energy_ratio_at_or_above_cascade_min",
        )
    } else {
        (
            ZSpaceCoherenceLabel::DiffuseDrift,
            "asymmetric_energy_below_cascade_ratio",
        )
    };

    Ok(ZSpaceCoherenceClassificationPayload {
        kind: ZSPACE_COHERENCE_CLASSIFICATION_KIND,
        contract_version: ZSPACE_COHERENCE_CLASSIFICATION_CONTRACT_VERSION,
        semantic_owner: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_BACKEND,
        classification_formula: ZSPACE_COHERENCE_CLASSIFICATION_FORMULA,
        label,
        reason,
        energy_ratio: request.energy_ratio,
        swap_invariant: request.swap_invariant,
        policy: request.policy,
    })
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

fn stable_mean(values: &[f64]) -> f64 {
    let scale = values
        .iter()
        .copied()
        .map(f64::abs)
        .reduce(f64::max)
        .unwrap_or(0.0);
    if scale == 0.0 {
        return 0.0;
    }
    let mut normalized_sum = 0.0;
    let mut compensation = 0.0;
    for value in values.iter().map(|value| value / scale) {
        let next = normalized_sum + value;
        // Neumaier compensation retains small residuals when large terms cancel.
        compensation += if normalized_sum.abs() >= value.abs() {
            (normalized_sum - next) + value
        } else {
            (value - next) + normalized_sum
        };
        normalized_sum = next;
    }
    let normalized_mean = (normalized_sum + compensation) / values.len() as f64;
    scale * normalized_mean.clamp(-1.0, 1.0)
}

fn validate_distribution_observation(
    normalized_weights: &[f64],
    distribution: ZSpaceCoherenceDistributionSummary,
    preserved_channels: Option<usize>,
    discarded_channels: Option<usize>,
    dominant_channel: Option<usize>,
) -> Result<(), ZSpaceCoherenceProjectionError> {
    let preserved = normalized_weights
        .iter()
        .filter(|weight| **weight > 0.0)
        .count();
    let discarded = distribution.channels - preserved;
    for (field, actual, expected) in [
        ("preserved_channels", preserved_channels, preserved),
        ("discarded_channels", discarded_channels, discarded),
    ] {
        if let Some(actual) = actual {
            if actual != expected {
                return Err(ZSpaceCoherenceProjectionError::InconsistentChannelSupport {
                    field,
                    actual,
                    expected,
                });
            }
        }
    }

    if let Some(dominant) = dominant_channel {
        if dominant >= distribution.channels {
            return Err(ZSpaceCoherenceProjectionError::DominantChannelOutOfRange {
                dominant,
                channels: distribution.channels,
            });
        }
        let dominant_weight = normalized_weights[dominant];
        let observed_peak = peak(normalized_weights);
        let peak_tolerance = f32_accumulation_tolerance(
            distribution.channels,
            dominant_weight.abs().max(observed_peak.abs()),
        );
        if observed_peak - dominant_weight > peak_tolerance {
            return Err(
                ZSpaceCoherenceProjectionError::InconsistentDominantChannel {
                    dominant,
                    value: dominant_weight,
                    peak: observed_peak,
                },
            );
        }
    }
    Ok(())
}

fn validate_request(
    request: &ZSpaceCoherenceProjectionRequest,
) -> Result<Option<ZSpaceCoherenceDistributionSummary>, ZSpaceCoherenceProjectionError> {
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

    let distribution = (!diagnostics.normalized_weights.is_empty())
        .then(|| summarize_zspace_coherence_distribution(&diagnostics.normalized_weights))
        .transpose()?;
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

    if let Some(distribution) = distribution {
        let metric_tolerance = f32_accumulation_tolerance(
            distribution.channels,
            diagnostics
                .coherence_entropy
                .abs()
                .max(distribution.weight_entropy.abs()),
        );
        if (diagnostics.coherence_entropy - distribution.weight_entropy).abs() > metric_tolerance {
            return Err(
                ZSpaceCoherenceProjectionError::InconsistentDistributionMetric {
                    field: "coherence_entropy",
                    value: diagnostics.coherence_entropy,
                    expected: distribution.weight_entropy,
                },
            );
        }

        validate_distribution_observation(
            &diagnostics.normalized_weights,
            distribution,
            diagnostics.preserved_channels,
            diagnostics.discarded_channels,
            diagnostics.dominant_channel,
        )?;
    }

    if !request.coherence.is_empty() {
        let observed_mean = stable_mean(&request.coherence);
        let mean_tolerance = f32_accumulation_tolerance(
            request.coherence.len(),
            diagnostics.mean_coherence.abs().max(observed_mean.abs()),
        );
        if (diagnostics.mean_coherence - observed_mean).abs() > mean_tolerance {
            return Err(
                ZSpaceCoherenceProjectionError::InconsistentDistributionMetric {
                    field: "mean_coherence",
                    value: diagnostics.mean_coherence,
                    expected: observed_mean,
                },
            );
        }
    }

    Ok(distribution)
}

fn peak(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(0.0)
}

fn derive_projection_summary(
    request: &ZSpaceCoherenceProjectionRequest,
    distribution: Option<ZSpaceCoherenceDistributionSummary>,
) -> Result<ZSpaceCoherenceProjectionDerived, ZSpaceCoherenceProjectionError> {
    let channels = request_channel_count(request);
    let diagnostics = &request.diagnostics;
    let response_mean = if request.coherence.is_empty() {
        0.0
    } else {
        stable_mean(&request.coherence)
    };

    let Some(distribution) = distribution else {
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
        return Ok(ZSpaceCoherenceProjectionDerived {
            channels,
            distribution_source: "diagnostic_entropy",
            weight_mass: 0.0,
            weight_entropy: diagnostics.coherence_entropy,
            normalized_entropy,
            concentration: 1.0 - normalized_entropy,
            effective_channels: diagnostics.coherence_entropy.exp().min(channels as f64),
            response_peak: peak(&request.coherence),
            response_mean,
        });
    };

    Ok(ZSpaceCoherenceProjectionDerived {
        channels,
        distribution_source: "normalized_weights",
        weight_mass: distribution.weight_mass,
        weight_entropy: distribution.weight_entropy,
        normalized_entropy: distribution.normalized_entropy,
        concentration: distribution.concentration,
        effective_channels: distribution.effective_channels,
        response_peak: peak(&request.coherence),
        response_mean,
    })
}

/// Project validated coherence diagnostics into canonical Z-space partial metrics.
pub fn project_zspace_coherence(
    request: ZSpaceCoherenceProjectionRequest,
) -> Result<ZSpaceCoherenceProjectionPayload, ZSpaceCoherenceProjectionError> {
    let distribution = validate_request(&request)?;
    let diagnostics = &request.diagnostics;
    let config = &request.config;
    let derived = derive_projection_summary(&request, distribution)?;
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
    let classification = if distribution.is_some() {
        Some(classify_zspace_coherence(
            ZSpaceCoherenceClassificationRequest {
                energy_ratio: diagnostics.energy_ratio,
                swap_invariant: is_zspace_coherence_swap_invariant(&diagnostics.normalized_weights),
                policy: request.classification_policy,
            },
        )?)
    } else {
        None
    };
    let control = if let Some(distribution) = distribution {
        Some(derive_zspace_coherence_control(
            distribution,
            diagnostics.energy_ratio,
            diagnostics.mean_coherence,
        )?)
    } else {
        None
    };

    Ok(ZSpaceCoherenceProjectionPayload {
        kind: ZSPACE_COHERENCE_PROJECTION_KIND,
        contract_version: ZSPACE_COHERENCE_PROJECTION_CONTRACT_VERSION,
        semantic_owner: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_COHERENCE_PROJECTION_SEMANTIC_BACKEND,
        projection_formula: ZSPACE_COHERENCE_PROJECTION_FORMULA,
        summary_formula: ZSPACE_COHERENCE_SUMMARY_FORMULA,
        evidence_validation_formula: ZSPACE_COHERENCE_EVIDENCE_VALIDATION_FORMULA,
        diagnostics: request.diagnostics,
        coherence: request.coherence,
        contour: request.contour,
        config: request.config,
        derived,
        classification,
        control,
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
            classification_policy: ZSpaceCoherenceClassificationPolicy::default(),
        }
    }

    #[test]
    fn projection_uses_dimension_normalized_distribution_controls() {
        let payload = project_zspace_coherence(request()).expect("valid coherence projection");
        let expected_concentration = (0.38_f64 - 0.25) / 0.75;
        let expected_normalized_entropy = 1.0296530140645737_f64 / 4.0_f64.ln();

        assert_eq!(payload.semantic_backend, "rust");
        assert_eq!(
            payload.contract_version,
            "spiraltorch.zspace_coherence_projection.v2"
        );
        assert_eq!(
            payload.evidence_validation_formula,
            ZSPACE_COHERENCE_EVIDENCE_VALIDATION_FORMULA
        );
        assert_eq!(payload.derived.distribution_source, "normalized_weights");
        let classification = payload.classification.expect("weight classification");
        assert_eq!(classification.label, ZSpaceCoherenceLabel::CascadeImbalance);
        assert_eq!(
            classification.reason,
            "dominant_energy_ratio_at_or_above_cascade_min"
        );
        let control = payload.control.expect("weight control");
        assert_eq!(control.kind, ZSPACE_COHERENCE_CONTROL_KIND);
        assert_eq!(
            control.contract_version,
            ZSPACE_COHERENCE_CONTROL_CONTRACT_VERSION
        );
        assert_eq!(control.semantic_backend, "rust");
        assert_relative_eq!(
            control.spectral_radius,
            expected_concentration,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            control.spectral_entropy,
            expected_normalized_entropy,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            control.spectral_pressure,
            0.7 * (1.0 - expected_normalized_entropy),
            epsilon = 1e-14
        );
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
                    energy_ratio: 0.85,
                    z_bias: 0.0,
                    fractional_order: 0.5,
                    normalized_weights: weights.clone(),
                    preserved_channels: Some(if concentrated { 1 } else { channels }),
                    discarded_channels: Some(if concentrated { channels - 1 } else { 0 }),
                    dominant_channel: Some(0),
                },
                coherence: weights,
                contour: None,
                config: ZSpaceCoherenceProjectionConfig::default(),
                classification_policy: ZSpaceCoherenceClassificationPolicy::default(),
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
        assert_eq!(
            uniform_four.classification.unwrap().label,
            ZSpaceCoherenceLabel::SymmetricPulse
        );
        assert_eq!(
            uniform_sixteen.classification.unwrap().label,
            ZSpaceCoherenceLabel::SymmetricPulse
        );
        let uniform_four_control = uniform_four.control.unwrap();
        let uniform_sixteen_control = uniform_sixteen.control.unwrap();
        assert_relative_eq!(uniform_four_control.spectral_radius, 0.0, epsilon = 1e-14);
        assert_relative_eq!(uniform_four_control.spectral_entropy, 1.0, epsilon = 1e-14);
        assert_relative_eq!(uniform_four_control.spectral_pressure, 0.0, epsilon = 1e-14);
        assert_relative_eq!(
            uniform_four_control.spectral_radius,
            uniform_sixteen_control.spectral_radius,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            uniform_four_control.spectral_entropy,
            uniform_sixteen_control.spectral_entropy,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            uniform_four_control.spectral_pressure,
            uniform_sixteen_control.spectral_pressure,
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
        assert_eq!(
            concentrated_four.classification.unwrap().label,
            ZSpaceCoherenceLabel::CascadeImbalance
        );
        assert_eq!(
            concentrated_sixteen.classification.unwrap().label,
            ZSpaceCoherenceLabel::CascadeImbalance
        );
        let concentrated_four_control = concentrated_four.control.unwrap();
        let concentrated_sixteen_control = concentrated_sixteen.control.unwrap();
        assert_relative_eq!(
            concentrated_four_control.spectral_radius,
            concentrated_sixteen_control.spectral_radius,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            concentrated_four_control.spectral_entropy,
            concentrated_sixteen_control.spectral_entropy,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            concentrated_four_control.spectral_pressure,
            concentrated_sixteen_control.spectral_pressure,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            concentrated_four_control.spectral_pressure,
            0.85,
            epsilon = 1e-14
        );
    }

    #[test]
    fn control_rejects_inconsistent_or_non_simplex_distribution_summaries() {
        let summary = summarize_zspace_coherence_distribution(&[0.5_f64, 0.3, 0.2]).unwrap();
        let mut inconsistent = summary;
        inconsistent.normalized_entropy = 0.0;
        assert!(matches!(
            derive_zspace_coherence_control(inconsistent, 0.5, 1.0 / 3.0),
            Err(
                ZSpaceCoherenceProjectionError::InconsistentDistributionMetric {
                    field: "normalized_entropy",
                    ..
                }
            )
        ));

        let mut invalid_mass = summary;
        invalid_mass.weight_mass = 0.5;
        assert!(matches!(
            derive_zspace_coherence_control(invalid_mass, 0.5, 1.0 / 3.0),
            Err(ZSpaceCoherenceProjectionError::InvalidWeightMass { .. })
        ));

        for (field, mut non_finite) in [("weight_mass", summary), ("weight_entropy", summary)] {
            if field == "weight_mass" {
                non_finite.weight_mass = f64::NAN;
            } else {
                non_finite.weight_entropy = f64::NAN;
            }
            assert!(matches!(
                derive_zspace_coherence_control(non_finite, 0.5, 1.0 / 3.0),
                Err(ZSpaceCoherenceProjectionError::NonFiniteScalar {
                    field: rejected,
                    ..
                }) if rejected == field
            ));
        }
    }

    #[test]
    fn distribution_summary_and_classification_share_f32_and_f64_semantics() {
        let f32_summary = summarize_zspace_coherence_distribution(&[0.6_f32, 0.3, 0.1]).unwrap();
        let f64_summary = summarize_zspace_coherence_distribution(&[0.6_f64, 0.3, 0.1]).unwrap();
        assert_relative_eq!(
            f32_summary.concentration,
            f64_summary.concentration,
            epsilon = 1.0e-6
        );
        assert_relative_eq!(
            f32_summary.normalized_entropy,
            f64_summary.normalized_entropy,
            epsilon = 1.0e-6
        );

        let background = classify_zspace_coherence(ZSpaceCoherenceClassificationRequest {
            energy_ratio: 0.0,
            swap_invariant: true,
            policy: ZSpaceCoherenceClassificationPolicy::default(),
        })
        .unwrap();
        assert_eq!(background.label, ZSpaceCoherenceLabel::Background);

        let f32_boundary = classify_zspace_coherence(ZSpaceCoherenceClassificationRequest {
            energy_ratio: f64::from(0.7_f32),
            swap_invariant: false,
            policy: ZSpaceCoherenceClassificationPolicy::default(),
        })
        .unwrap();
        assert_eq!(f32_boundary.label, ZSpaceCoherenceLabel::CascadeImbalance);

        let below_boundary = classify_zspace_coherence(ZSpaceCoherenceClassificationRequest {
            energy_ratio: 0.6999,
            swap_invariant: false,
            policy: ZSpaceCoherenceClassificationPolicy::default(),
        })
        .unwrap();
        assert_eq!(below_boundary.label, ZSpaceCoherenceLabel::DiffuseDrift);

        let invalid = classify_zspace_coherence(ZSpaceCoherenceClassificationRequest {
            energy_ratio: 0.5,
            swap_invariant: false,
            policy: ZSpaceCoherenceClassificationPolicy {
                background_energy_ratio_max: 0.8,
                cascade_energy_ratio_min: 0.7,
            },
        });
        assert!(matches!(
            invalid,
            Err(ZSpaceCoherenceProjectionError::InvalidClassificationPolicy { .. })
        ));

        assert!(is_zspace_coherence_swap_invariant(&[0.5_f64, 0.5]));
        assert!(!is_zspace_coherence_swap_invariant(&[] as &[f64]));
        assert!(!is_zspace_coherence_swap_invariant(&[f64::NAN, f64::NAN]));
        assert!(!is_zspace_coherence_swap_invariant(&[0.5, f64::INFINITY]));
    }

    #[test]
    fn distribution_witness_round_trips_only_through_the_rust_contract() {
        let witness = build_zspace_coherence_distribution_witness(&[0.6_f32, 0.3, 0.1]).unwrap();
        assert_eq!(
            witness.contract_version,
            ZSPACE_COHERENCE_DISTRIBUTION_WITNESS_CONTRACT_VERSION
        );
        assert_eq!(witness.semantic_backend, "rust");

        let summary = validate_zspace_coherence_distribution_witness(&witness).unwrap();
        assert_relative_eq!(summary.weight_mass, 1.0, epsilon = 1.0e-6);
        assert_relative_eq!(
            summary.concentration,
            summarize_zspace_coherence_distribution(&[0.6_f32, 0.3, 0.1])
                .unwrap()
                .concentration,
            epsilon = 1.0e-12
        );
        assert_eq!(
            validate_zspace_coherence_distribution_observation(&witness, 3, Some(0)).unwrap(),
            summary
        );
        assert!(matches!(
            validate_zspace_coherence_distribution_observation(&witness, 2, Some(0)),
            Err(ZSpaceCoherenceProjectionError::InconsistentChannelSupport {
                field: "preserved_channels",
                ..
            })
        ));
        assert!(matches!(
            validate_zspace_coherence_distribution_observation(&witness, 3, Some(1)),
            Err(ZSpaceCoherenceProjectionError::InconsistentDominantChannel { dominant: 1, .. })
        ));

        let mut wrong_version = witness.clone();
        wrong_version.contract_version =
            "spiraltorch.zspace_coherence_distribution_witness.v0".to_owned();
        assert!(matches!(
            validate_zspace_coherence_distribution_witness(&wrong_version),
            Err(
                ZSpaceCoherenceProjectionError::InvalidDistributionWitnessContract {
                    field: "contract_version",
                    ..
                }
            )
        ));

        let mut invalid_simplex = witness;
        invalid_simplex.normalized_weights[0] = 0.7;
        assert!(matches!(
            validate_zspace_coherence_distribution_witness(&invalid_simplex),
            Err(ZSpaceCoherenceProjectionError::InvalidWeightMass { .. })
        ));
    }

    #[test]
    fn projection_accepts_f32_diagnostics_at_the_channel_limit() {
        let channels = ZSPACE_COHERENCE_MAX_CHANNELS;
        let weight = 1.0_f32 / channels as f32;
        let weights = vec![weight; channels];
        let entropy = -weights
            .iter()
            .map(|weight| *weight * weight.ln())
            .sum::<f32>();
        let payload = project_zspace_coherence(ZSpaceCoherenceProjectionRequest {
            diagnostics: ZSpaceCoherenceDiagnosticsInput {
                mean_coherence: f64::from(weight),
                coherence_entropy: f64::from(entropy),
                energy_ratio: 0.8,
                z_bias: 0.0,
                fractional_order: 0.5,
                normalized_weights: weights.iter().map(|value| f64::from(*value)).collect(),
                preserved_channels: Some(channels),
                discarded_channels: Some(0),
                dominant_channel: Some(0),
            },
            coherence: weights.iter().map(|value| f64::from(*value)).collect(),
            contour: None,
            config: ZSpaceCoherenceProjectionConfig::default(),
            classification_policy: ZSpaceCoherenceClassificationPolicy::default(),
        })
        .expect("f32 accumulation drift must remain inside the shared tolerance");

        assert_eq!(payload.derived.channels, channels);
        assert_eq!(
            payload.classification.expect("classification").label,
            ZSpaceCoherenceLabel::SymmetricPulse
        );
    }

    #[test]
    fn trace_entropy_alias_and_contour_are_part_of_the_same_contract() {
        let request: ZSpaceCoherenceProjectionRequest = serde_json::from_value(json!({
            "diagnostics": {
                "mean_coherence": 0.23333333333333334,
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
        invalid.diagnostics.coherence_entropy = 0.5;
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(
                ZSpaceCoherenceProjectionError::InconsistentDistributionMetric {
                    field: "coherence_entropy",
                    ..
                }
            )
        ));

        let mut invalid = request();
        invalid.diagnostics.preserved_channels = Some(4);
        invalid.diagnostics.discarded_channels = Some(0);
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::InconsistentChannelSupport {
                field: "preserved_channels",
                actual: 4,
                expected: 3,
            })
        ));

        let mut invalid = request();
        invalid.diagnostics.dominant_channel = Some(1);
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(ZSpaceCoherenceProjectionError::InconsistentDominantChannel { dominant: 1, .. })
        ));

        let mut invalid = request();
        invalid.coherence = vec![0.8, 0.2, 0.1, 0.1];
        assert!(matches!(
            project_zspace_coherence(invalid),
            Err(
                ZSpaceCoherenceProjectionError::InconsistentDistributionMetric {
                    field: "mean_coherence",
                    ..
                }
            )
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

        let mut high_dynamic_range = request();
        high_dynamic_range.coherence = vec![1.0e300, 1.0, -1.0e300, 0.0];
        let projected = project_zspace_coherence(high_dynamic_range)
            .expect("finite high-dynamic-range residual must survive cancellation");
        assert_relative_eq!(projected.derived.response_mean, 0.25, epsilon = 1.0e-15);
        assert_eq!(projected.derived.response_peak, 1.0e300);

        let mut permuted = request();
        permuted.coherence = vec![1.0e300, -1.0e300, 1.0, 0.0];
        let permuted = project_zspace_coherence(permuted)
            .expect("response mean must not depend on cancellation order");
        assert_relative_eq!(permuted.derived.response_mean, 0.25, epsilon = 1.0e-15);
    }
}
