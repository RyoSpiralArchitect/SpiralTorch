// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical latent Z-space posterior decoding and projection semantics.
//!
//! Rust owns spectral decoding, metric reconstruction, barycentric projection,
//! residual confidence, and telemetry adjustment. Python and WASM clients
//! transport requests and adapt the versioned payload; they must not rebuild
//! these formulas.

use crate::telemetry::zspace_fusion::{
    fuse_zspace_partials, fuse_zspace_telemetry, ZSpaceFusionError, ZSpaceFusionStrategy,
    ZSpaceMetricInput, ZSpacePartialFusionRequest, ZSpacePartialInput,
    ZSpaceTelemetryFusionPayload,
};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex};
use thiserror::Error;

pub const ZSPACE_POSTERIOR_CONTRACT_VERSION: &str = "spiraltorch.zspace_posterior.v1";
pub const ZSPACE_POSTERIOR_DECODE_KIND: &str = "spiraltorch.zspace_posterior_decode";
pub const ZSPACE_POSTERIOR_PROJECTION_KIND: &str = "spiraltorch.zspace_posterior_projection";
pub const ZSPACE_POSTERIOR_SEMANTIC_OWNER: &str = "st-core::inference::zspace_posterior";
pub const ZSPACE_POSTERIOR_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_POSTERIOR_MAX_DIMENSION: usize = 4_096;
pub const ZSPACE_POSTERIOR_METRIC_FORMULA: &str =
    "speed=tanh(mean_abs_dz+0.25*||z||_2/n);memory=tanh(mean(z)+0.25*||z||_2^2/n);stability=tanh(1.5*((1+mean_abs_d2z)^-1+(1+mean_abs_dz)^-1)-1);frac=tanh(E_alpha/(||z||_2^2+1e-9));drs=tanh((E_high-E_low)/(E_high+E_low+1e-9))";
pub const ZSPACE_POSTERIOR_FRACTIONAL_FORMULA: &str =
    "E_alpha=mean_k((k/k_max)^(2*alpha)*|DFT(z)_k|^2),k=0..floor(n/2)";
pub const ZSPACE_POSTERIOR_GRADIENT_FORMULA: &str =
    "g_i=0.5*((z_i-z_(i-1))+(z_(i+1)-z_i)),with zero exterior boundaries;if ||g||_inf>1 then g=tanh(g/||g||_inf)";
pub const ZSPACE_POSTERIOR_BARYCENTRIC_FORMULA: &str =
    "bary=normalize(softplus(speed),softplus(memory),softplus(stability))";
pub const ZSPACE_POSTERIOR_PROJECTION_FORMULA: &str =
    "bary=normalize(smoothing*bary_prior+(1-smoothing)*bary_override);residual=rms_5(projected-prior);confidence=exp(-residual)";
pub const ZSPACE_POSTERIOR_TELEMETRY_FORMULA: &str =
    "residual'=residual/max(1,1+variance);confidence'=clamp(confidence*max(0.25,1+0.25*focus)*min(1.5,1+0.05*energy),0,1)";

const BASE_METRICS: [&str; 5] = ["speed", "memory", "stability", "frac", "drs"];
const DEFAULT_FRACTIONAL_ORDER: f64 = 0.35;
const DEFAULT_SMOOTHING: f64 = 0.35;
const FFT_PLAN_CACHE_CAPACITY: usize = 32;
type FftPlan = Arc<dyn Fft<f64>>;
static FFT_PLAN_CACHE: LazyLock<Mutex<VecDeque<(usize, FftPlan)>>> =
    LazyLock::new(|| Mutex::new(VecDeque::with_capacity(FFT_PLAN_CACHE_CAPACITY)));

const fn default_fractional_order() -> f64 {
    DEFAULT_FRACTIONAL_ORDER
}

const fn default_smoothing() -> f64 {
    DEFAULT_SMOOTHING
}

#[derive(Debug, Error, PartialEq)]
pub enum ZSpacePosteriorError {
    #[error("Z-space posterior state must contain at least one value")]
    EmptyState,
    #[error("Z-space posterior dimension {actual} exceeds limit {max}")]
    DimensionLimit { actual: usize, max: usize },
    #[error("Z-space posterior state entry {index} must be finite")]
    NonFiniteState { index: usize },
    #[error("fractional order must be finite and positive, received {value}")]
    InvalidFractionalOrder { value: f64 },
    #[error("smoothing must be finite and lie in [0, 1], received {value}")]
    InvalidSmoothing { value: f64 },
    #[error("derived Z-space posterior field '{field}' is not finite")]
    NonFiniteDerived { field: &'static str },
    #[error("canonical Z-space posterior metric '{metric}' is missing")]
    MissingMetric { metric: &'static str },
    #[error("Z-space posterior barycentric mass is invalid")]
    InvalidBarycentricMass,
    #[error("Z-space posterior fusion failed: {0}")]
    Fusion(#[from] ZSpaceFusionError),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ZSpacePosteriorDecodeRequest {
    pub z_state: Vec<f64>,
    #[serde(default = "default_fractional_order")]
    pub alpha: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ZSpacePosteriorProjectionRequest {
    pub z_state: Vec<f64>,
    #[serde(default = "default_fractional_order")]
    pub alpha: f64,
    #[serde(default)]
    pub partial: BTreeMap<String, ZSpaceMetricInput>,
    #[serde(default = "default_smoothing")]
    pub smoothing: f64,
    #[serde(default)]
    pub telemetry: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePosteriorDecodePayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub metric_formula: &'static str,
    pub fractional_formula: &'static str,
    pub gradient_formula: &'static str,
    pub barycentric_formula: &'static str,
    pub z_state: Vec<f64>,
    pub alpha: f64,
    pub metrics: BTreeMap<String, f64>,
    pub gradient: Vec<f64>,
    pub barycentric: [f64; 3],
    pub energy: f64,
    pub frac_energy: f64,
    pub spectral_bins: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ZSpacePosteriorTelemetryAdjustment {
    pub variance_damping: f64,
    pub focus_gain: f64,
    pub energy_gain: f64,
    pub residual_before: f64,
    pub confidence_before: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePosteriorProjectionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub projection_formula: &'static str,
    pub telemetry_formula: &'static str,
    pub smoothing: f64,
    pub metrics: BTreeMap<String, f64>,
    pub gradient: Vec<f64>,
    pub barycentric: [f64; 3],
    pub residual: f64,
    pub confidence: f64,
    pub applied: BTreeMap<String, ZSpaceMetricInput>,
    pub prior: ZSpacePosteriorDecodePayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<ZSpaceTelemetryFusionPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_adjustment: Option<ZSpacePosteriorTelemetryAdjustment>,
}

struct CanonicalPosteriorPartial {
    metrics: BTreeMap<String, f64>,
    gradient: Option<Vec<f64>>,
}

fn ensure_finite(field: &'static str, value: f64) -> Result<f64, ZSpacePosteriorError> {
    if !value.is_finite() {
        return Err(ZSpacePosteriorError::NonFiniteDerived { field });
    }
    Ok(value)
}

fn validate_decode_request(
    request: &ZSpacePosteriorDecodeRequest,
) -> Result<(), ZSpacePosteriorError> {
    if request.z_state.is_empty() {
        return Err(ZSpacePosteriorError::EmptyState);
    }
    if request.z_state.len() > ZSPACE_POSTERIOR_MAX_DIMENSION {
        return Err(ZSpacePosteriorError::DimensionLimit {
            actual: request.z_state.len(),
            max: ZSPACE_POSTERIOR_MAX_DIMENSION,
        });
    }
    if let Some((index, _)) = request
        .z_state
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ZSpacePosteriorError::NonFiniteState { index });
    }
    if !request.alpha.is_finite() || request.alpha <= 0.0 {
        return Err(ZSpacePosteriorError::InvalidFractionalOrder {
            value: request.alpha,
        });
    }
    Ok(())
}

fn forward_spectrum(values: &[f64]) -> Result<Vec<Complex<f64>>, ZSpacePosteriorError> {
    let mut spectrum: Vec<Complex<f64>> = values
        .iter()
        .copied()
        .map(|value| Complex::new(value, 0.0))
        .collect();
    let fft = forward_fft_plan(values.len());
    fft.process(&mut spectrum);
    spectrum.truncate(values.len() / 2 + 1);
    if spectrum
        .iter()
        .any(|coefficient| !coefficient.re.is_finite() || !coefficient.im.is_finite())
    {
        return Err(ZSpacePosteriorError::NonFiniteDerived { field: "spectrum" });
    }
    Ok(spectrum)
}

fn forward_fft_plan(dimension: usize) -> FftPlan {
    let mut cache = FFT_PLAN_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(position) = cache.iter().position(|(cached, _)| *cached == dimension) {
        let (_, plan) = cache.remove(position).expect("cached FFT position exists");
        cache.push_back((dimension, Arc::clone(&plan)));
        return plan;
    }

    let plan = FftPlanner::<f64>::new().plan_fft_forward(dimension);
    if cache.len() == FFT_PLAN_CACHE_CAPACITY {
        cache.pop_front();
    }
    cache.push_back((dimension, Arc::clone(&plan)));
    plan
}

fn fractional_energy(spectrum: &[Complex<f64>], alpha: f64) -> Result<f64, ZSpacePosteriorError> {
    if spectrum.len() <= 1 {
        return Ok(0.0);
    }
    let denominator = (spectrum.len() - 1) as f64;
    let mut energy = 0.0;
    for (index, coefficient) in spectrum.iter().enumerate() {
        let omega = index as f64 / denominator;
        energy += omega.powf(2.0 * alpha) * coefficient.norm_sqr();
    }
    ensure_finite("frac_energy", energy / spectrum.len() as f64)
}

fn normalized_gradient(values: &[f64], length: usize) -> Result<Vec<f64>, ZSpacePosteriorError> {
    let mut gradient = values.to_vec();
    gradient.resize(length, 0.0);
    gradient.truncate(length);
    let scale = gradient.iter().copied().map(f64::abs).fold(0.0, f64::max);
    if scale > 1.0 {
        for value in &mut gradient {
            *value = (*value / scale).tanh();
        }
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(ZSpacePosteriorError::NonFiniteDerived { field: "gradient" });
    }
    Ok(gradient)
}

fn latent_gradient(values: &[f64]) -> Result<Vec<f64>, ZSpacePosteriorError> {
    let mut gradient = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        let left = if index > 0 {
            values[index] - values[index - 1]
        } else {
            values[index]
        };
        let right = if index + 1 < values.len() {
            values[index + 1] - values[index]
        } else {
            -values[index]
        };
        gradient.push(0.5 * (left + right));
    }
    normalized_gradient(&gradient, values.len())
}

fn softplus(value: f64) -> f64 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn barycentric_from_metrics(
    metrics: &BTreeMap<String, f64>,
) -> Result<[f64; 3], ZSpacePosteriorError> {
    let mut weights = [0.0; 3];
    for (index, metric) in ["speed", "memory", "stability"].iter().enumerate() {
        let value = metrics
            .get(*metric)
            .copied()
            .ok_or(ZSpacePosteriorError::MissingMetric { metric })?;
        weights[index] = softplus(value);
    }
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(ZSpacePosteriorError::InvalidBarycentricMass);
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Ok(weights)
}

/// Decode one finite latent Z vector into canonical posterior metrics.
pub fn decode_zspace_posterior(
    request: ZSpacePosteriorDecodeRequest,
) -> Result<ZSpacePosteriorDecodePayload, ZSpacePosteriorError> {
    validate_decode_request(&request)?;
    let values = request.z_state;
    let dimension = values.len();
    let differences: Vec<f64> = values.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let curvature: Vec<f64> = values
        .windows(3)
        .map(|window| window[2] - 2.0 * window[1] + window[0])
        .collect();
    let mean_velocity = ensure_finite(
        "mean_velocity",
        differences.iter().map(|value| value.abs()).sum::<f64>() / differences.len().max(1) as f64,
    )?;
    let curvature_energy = ensure_finite(
        "curvature_energy",
        curvature.iter().map(|value| value.abs()).sum::<f64>() / curvature.len().max(1) as f64,
    )?;
    let energy = ensure_finite("energy", values.iter().map(|value| value * value).sum())?;
    let l2 = ensure_finite("l2", energy.sqrt())?;
    let centre = ensure_finite("centre", values.iter().sum::<f64>() / dimension as f64)?;
    let spectrum = forward_spectrum(&values)?;
    let frac_energy = fractional_energy(&spectrum, request.alpha)?;
    let gradient = latent_gradient(&values)?;

    let speed = ensure_finite(
        "speed",
        (mean_velocity + 0.25 * l2 / dimension as f64).tanh(),
    )?;
    let memory = ensure_finite("memory", (centre + 0.25 * energy / dimension as f64).tanh())?;
    let smoothness = 1.0 / (1.0 + curvature_energy);
    let drift = 1.0 / (1.0 + mean_velocity);
    let stability = ensure_finite("stability", ((smoothness + drift) * 1.5 - 1.0).tanh())?;
    let drs = if spectrum.len() > 1 {
        let half = (spectrum.len() / 2).max(1);
        let high = spectrum[half..].iter().map(Complex::norm_sqr).sum::<f64>();
        let low = spectrum[..half].iter().map(Complex::norm_sqr).sum::<f64>();
        ensure_finite("drs", ((high - low) / (high + low + 1e-9)).tanh())?
    } else {
        0.0
    };
    let frac = ensure_finite("frac", (frac_energy / (energy + 1e-9)).tanh())?;
    let metrics = BTreeMap::from([
        ("speed".to_owned(), speed),
        ("memory".to_owned(), memory),
        ("stability".to_owned(), stability),
        ("frac".to_owned(), frac),
        ("drs".to_owned(), drs),
    ]);
    let barycentric = barycentric_from_metrics(&metrics)?;

    Ok(ZSpacePosteriorDecodePayload {
        kind: ZSPACE_POSTERIOR_DECODE_KIND,
        contract_version: ZSPACE_POSTERIOR_CONTRACT_VERSION,
        semantic_owner: ZSPACE_POSTERIOR_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_POSTERIOR_SEMANTIC_BACKEND,
        metric_formula: ZSPACE_POSTERIOR_METRIC_FORMULA,
        fractional_formula: ZSPACE_POSTERIOR_FRACTIONAL_FORMULA,
        gradient_formula: ZSPACE_POSTERIOR_GRADIENT_FORMULA,
        barycentric_formula: ZSPACE_POSTERIOR_BARYCENTRIC_FORMULA,
        z_state: values,
        alpha: request.alpha,
        metrics,
        gradient,
        barycentric,
        energy,
        frac_energy,
        spectral_bins: spectrum.len(),
    })
}

fn canonical_partial(
    partial: BTreeMap<String, ZSpaceMetricInput>,
) -> Result<CanonicalPosteriorPartial, ZSpacePosteriorError> {
    if partial.is_empty() {
        return Ok(CanonicalPosteriorPartial {
            metrics: BTreeMap::new(),
            gradient: None,
        });
    }
    let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
        partials: vec![Some(ZSpacePartialInput {
            metrics: partial,
            weight: 1.0,
            origin: Some("zspace_posterior_projection".to_owned()),
            telemetry: None,
        })],
        weights: None,
        strategy: ZSpaceFusionStrategy::Last,
        gradient_alignment: Default::default(),
        telemetry: Vec::new(),
    })?;
    Ok(CanonicalPosteriorPartial {
        metrics: fused.metrics,
        gradient: fused.gradient,
    })
}

/// Project canonical partial metrics and telemetry onto a latent posterior.
pub fn project_zspace_posterior(
    request: ZSpacePosteriorProjectionRequest,
) -> Result<ZSpacePosteriorProjectionPayload, ZSpacePosteriorError> {
    if !request.smoothing.is_finite() || !(0.0..=1.0).contains(&request.smoothing) {
        return Err(ZSpacePosteriorError::InvalidSmoothing {
            value: request.smoothing,
        });
    }
    let prior = decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
        z_state: request.z_state,
        alpha: request.alpha,
    })?;
    let partial = canonical_partial(request.partial)?;
    let mut metrics = prior.metrics.clone();
    let mut applied = BTreeMap::new();
    for (metric, value) in partial.metrics {
        metrics.insert(metric.clone(), value);
        applied.insert(metric, ZSpaceMetricInput::Scalar(value));
    }
    let gradient = if let Some(gradient) = partial.gradient {
        let gradient = normalized_gradient(&gradient, prior.z_state.len())?;
        applied.insert(
            "gradient".to_owned(),
            ZSpaceMetricInput::Vector(gradient.clone()),
        );
        gradient
    } else {
        prior.gradient.clone()
    };

    let override_barycentric = barycentric_from_metrics(&metrics)?;
    let mut barycentric = [0.0; 3];
    for index in 0..3 {
        barycentric[index] = request.smoothing * prior.barycentric[index]
            + (1.0 - request.smoothing) * override_barycentric[index];
    }
    let barycentric_mass = barycentric.iter().sum::<f64>();
    if !barycentric_mass.is_finite() || barycentric_mass <= 0.0 {
        return Err(ZSpacePosteriorError::InvalidBarycentricMass);
    }
    for value in &mut barycentric {
        *value /= barycentric_mass;
    }

    let mut squared_residual = 0.0;
    for metric in BASE_METRICS {
        let prior_value = prior
            .metrics
            .get(metric)
            .copied()
            .ok_or(ZSpacePosteriorError::MissingMetric { metric })?;
        let projected_value = metrics
            .get(metric)
            .copied()
            .ok_or(ZSpacePosteriorError::MissingMetric { metric })?;
        squared_residual += (projected_value - prior_value).powi(2);
    }
    let residual_before = ensure_finite(
        "residual",
        (squared_residual / BASE_METRICS.len() as f64).sqrt(),
    )?;
    let confidence_before = ensure_finite("confidence", (-residual_before).exp())?;
    let mut residual = residual_before;
    let mut confidence = confidence_before;

    let telemetry = if request.telemetry.is_empty() {
        None
    } else {
        Some(fuse_zspace_telemetry(&request.telemetry)?)
    };
    let telemetry_adjustment = telemetry.as_ref().map(|telemetry| {
        let variance_damping = (1.0 + telemetry.summary.variance).max(1.0);
        let focus_gain = (1.0 + 0.25 * telemetry.summary.focus).max(0.25);
        let energy_gain = (1.0 + 0.05 * telemetry.summary.energy).min(1.5);
        residual /= variance_damping;
        confidence = (confidence * focus_gain * energy_gain).clamp(0.0, 1.0);
        ZSpacePosteriorTelemetryAdjustment {
            variance_damping,
            focus_gain,
            energy_gain,
            residual_before,
            confidence_before,
        }
    });
    ensure_finite("adjusted_residual", residual)?;
    ensure_finite("adjusted_confidence", confidence)?;

    Ok(ZSpacePosteriorProjectionPayload {
        kind: ZSPACE_POSTERIOR_PROJECTION_KIND,
        contract_version: ZSPACE_POSTERIOR_CONTRACT_VERSION,
        semantic_owner: ZSPACE_POSTERIOR_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_POSTERIOR_SEMANTIC_BACKEND,
        projection_formula: ZSPACE_POSTERIOR_PROJECTION_FORMULA,
        telemetry_formula: ZSPACE_POSTERIOR_TELEMETRY_FORMULA,
        smoothing: request.smoothing,
        metrics,
        gradient,
        barycentric,
        residual,
        confidence,
        applied,
        prior,
        telemetry,
        telemetry_adjustment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use serde_json::json;

    fn sample_state() -> Vec<f64> {
        vec![0.12, -0.03, 0.48, -0.2]
    }

    #[test]
    fn decode_matches_the_previous_python_posterior_contract() {
        let decoded = decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
            z_state: sample_state(),
            alpha: 0.35,
        })
        .expect("decode posterior");

        assert_eq!(decoded.semantic_backend, "rust");
        assert_eq!(decoded.spectral_bins, 3);
        assert_relative_eq!(decoded.energy, 0.2857, epsilon = 1.0e-14);
        assert_relative_eq!(decoded.frac_energy, 0.2621560649191949, epsilon = 1.0e-12);
        assert_relative_eq!(
            decoded.metrics["speed"],
            0.4463024613684294,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            decoded.metrics["memory"],
            0.10991043037290935,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            decoded.metrics["stability"],
            0.672934550609384,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            decoded.metrics["drs"],
            0.6180432815695912,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            decoded.metrics["frac"],
            0.7247563119052868,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            decoded.barycentric.iter().sum::<f64>(),
            1.0,
            epsilon = 1.0e-14
        );
        assert_eq!(decoded.gradient.len(), sample_state().len());
    }

    #[test]
    fn projection_canonicalizes_aliases_and_recomputes_telemetry_summary() {
        let projected = project_zspace_posterior(ZSpacePosteriorProjectionRequest {
            z_state: sample_state(),
            alpha: 0.35,
            partial: BTreeMap::from([
                ("speed".to_owned(), ZSpaceMetricInput::Scalar(0.3)),
                ("mem".to_owned(), ZSpaceMetricInput::Scalar(-0.2)),
                (
                    "gradient".to_owned(),
                    ZSpaceMetricInput::Vector(vec![2.0, -1.0]),
                ),
            ]),
            smoothing: 0.35,
            telemetry: vec![json!({"psi": {"energy": 2.0, "focus": 0.4}})],
        })
        .expect("project posterior");

        assert_relative_eq!(projected.metrics["memory"], -0.2, epsilon = 1.0e-14);
        assert_relative_eq!(projected.gradient[0], 0.7615941559557649, epsilon = 1.0e-12);
        assert_relative_eq!(
            projected.gradient[1],
            -0.46211715726000974,
            epsilon = 1.0e-12
        );
        assert_eq!(projected.gradient[2..], [0.0, 0.0]);
        assert_relative_eq!(projected.residual, 0.09345350599255713, epsilon = 1.0e-12);
        assert_eq!(projected.confidence, 1.0);
        let telemetry = projected.telemetry.expect("telemetry contract");
        assert_eq!(telemetry.payload["psi.energy"], 2.0);
        assert_eq!(telemetry.payload["psi.focus"], 0.4);
        assert_relative_eq!(telemetry.summary.variance, 0.64, epsilon = 1.0e-14);
    }

    #[test]
    fn projection_rejects_alias_collisions_and_invalid_controls() {
        let collision = project_zspace_posterior(ZSpacePosteriorProjectionRequest {
            z_state: sample_state(),
            alpha: 0.35,
            partial: BTreeMap::from([
                ("speed".to_owned(), ZSpaceMetricInput::Scalar(0.1)),
                ("velocity".to_owned(), ZSpaceMetricInput::Scalar(0.2)),
            ]),
            smoothing: 0.35,
            telemetry: Vec::new(),
        });
        assert!(matches!(
            collision,
            Err(ZSpacePosteriorError::Fusion(
                ZSpaceFusionError::AliasCollision { .. }
            ))
        ));

        let invalid_smoothing = project_zspace_posterior(ZSpacePosteriorProjectionRequest {
            z_state: sample_state(),
            alpha: 0.35,
            partial: BTreeMap::new(),
            smoothing: 1.5,
            telemetry: Vec::new(),
        });
        assert_eq!(
            invalid_smoothing,
            Err(ZSpacePosteriorError::InvalidSmoothing { value: 1.5 })
        );
    }

    #[test]
    fn decoder_fails_closed_on_invalid_state_and_fractional_order() {
        assert_eq!(
            decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
                z_state: Vec::new(),
                alpha: 0.35,
            }),
            Err(ZSpacePosteriorError::EmptyState)
        );
        assert_eq!(
            decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
                z_state: vec![0.0, f64::NAN],
                alpha: 0.35,
            }),
            Err(ZSpacePosteriorError::NonFiniteState { index: 1 })
        );
        assert_eq!(
            decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
                z_state: sample_state(),
                alpha: 0.0,
            }),
            Err(ZSpacePosteriorError::InvalidFractionalOrder { value: 0.0 })
        );
    }

    #[test]
    fn fft_plan_cache_reuses_plans_and_stays_bounded() {
        let first = forward_fft_plan(7);
        let second = forward_fft_plan(7);
        assert!(Arc::ptr_eq(&first, &second));

        for dimension in 1..=(FFT_PLAN_CACHE_CAPACITY + 8) {
            let _ = forward_fft_plan(dimension);
        }
        let cache = FFT_PLAN_CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(cache.len() <= FFT_PLAN_CACHE_CAPACITY);
    }
}
