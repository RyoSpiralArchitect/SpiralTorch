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

pub const ZSPACE_POSTERIOR_CONTRACT_VERSION: &str = "spiraltorch.zspace_posterior.v2";
pub const ZSPACE_POSTERIOR_DECODE_KIND: &str = "spiraltorch.zspace_posterior_decode";
pub const ZSPACE_POSTERIOR_PROJECTION_KIND: &str = "spiraltorch.zspace_posterior_projection";
pub const ZSPACE_POSTERIOR_SEMANTIC_OWNER: &str = "st-core::inference::zspace_posterior";
pub const ZSPACE_POSTERIOR_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS: &str =
    "spiraltorch.zspace.latent.central_difference.zero_boundary.v1";
pub const ZSPACE_POSTERIOR_MAX_DIMENSION: usize = 4_096;
pub const ZSPACE_POSTERIOR_METRIC_FORMULA: &str =
    "speed=tanh(mean_abs_dz+0.25*||z||_2/n);memory=tanh(mean(z)+0.25*||z||_2^2/n);stability=tanh(1.5*((1+mean_abs_d2z)^-1+(1+mean_abs_dz)^-1)-1);frac=tanh(E_alpha/||z||_2^2),frac(0)=0;drs=tanh((E_high-E_low)/(E_high+E_low)),drs(0)=0";
pub const ZSPACE_POSTERIOR_FRACTIONAL_FORMULA: &str =
    "P_k=m_k*|DFT(z)_k|^2/n,m_0=1,m_nyquist=1,m_other=2;E_alpha=sum_k((2k/n)^(2*alpha)*P_k),k=0..floor(n/2)";
pub const ZSPACE_POSTERIOR_SPECTRAL_FORMULA: &str =
    "sum_k(P_k)=sum_i(z_i^2) by one-sided Parseval normalization;centroid=sum_k((2k/n)*P_k)/sum_k(P_k)";
pub const ZSPACE_POSTERIOR_GRADIENT_FORMULA: &str =
    "g_i=0.5*((z_i-z_(i-1))+(z_(i+1)-z_i)),with zero exterior boundaries;g=g/max(1,||g||_inf)";
pub const ZSPACE_POSTERIOR_BARYCENTRIC_FORMULA: &str =
    "bary=normalize(softplus(speed),softplus(memory),softplus(stability))";
pub const ZSPACE_POSTERIOR_PROJECTION_FORMULA: &str =
    "bary=normalize(smoothing*bary_prior+(1-smoothing)*bary_override);residual=rms_observed(projected-prior);confidence_geometry=exp(-residual);external gradients remain basis-tagged controls and never replace the latent gradient";
pub const ZSPACE_POSTERIOR_TELEMETRY_FORMULA: &str =
    "r_variance=(1+variance)^-1;r_focus=clamp(1+0.25*focus,0.75,1);reliability=r_variance*r_focus;residual'=residual;confidence'=confidence_geometry*reliability";

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
    #[error("external Z-space posterior gradients require an explicit gradient basis")]
    MissingControlGradientBasis,
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
    #[serde(default)]
    pub gradient_basis: Option<String>,
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
    pub spectral_formula: &'static str,
    pub gradient_formula: &'static str,
    pub gradient_basis: &'static str,
    pub barycentric_formula: &'static str,
    pub z_state: Vec<f64>,
    pub alpha: f64,
    pub metrics: BTreeMap<String, f64>,
    pub gradient: Vec<f64>,
    pub barycentric: [f64; 3],
    pub energy: f64,
    pub spectral_energy: f64,
    pub parseval_relative_error: f64,
    pub frac_energy: f64,
    pub fractional_energy_ratio: f64,
    pub spectral_centroid: f64,
    pub spectral_bins: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ZSpacePosteriorTelemetryAdjustment {
    pub variance_reliability: f64,
    pub focus_reliability: f64,
    pub combined_reliability: f64,
    pub residual_before: f64,
    pub confidence_before: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePosteriorControlGradient {
    pub source: &'static str,
    pub basis: String,
    pub values: Vec<f64>,
    pub dimension: usize,
    pub l2: f64,
    pub linf: f64,
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
    pub gradient_basis: &'static str,
    pub gradient_source: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_gradient: Option<ZSpacePosteriorControlGradient>,
    pub barycentric: [f64; 3],
    pub residual: f64,
    pub residual_metric_count: usize,
    pub confidence: f64,
    pub telemetry_reliability: f64,
    pub applied: BTreeMap<String, ZSpaceMetricInput>,
    pub prior: ZSpacePosteriorDecodePayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<ZSpaceTelemetryFusionPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_adjustment: Option<ZSpacePosteriorTelemetryAdjustment>,
}

struct CanonicalPosteriorPartial {
    metrics: BTreeMap<String, f64>,
    control_gradient: Option<ZSpacePosteriorControlGradient>,
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

fn one_sided_spectral_energy(
    spectrum: &[Complex<f64>],
    dimension: usize,
) -> Result<Vec<f64>, ZSpacePosteriorError> {
    let scale = 1.0 / dimension as f64;
    let nyquist = dimension / 2;
    let mut energy = Vec::with_capacity(spectrum.len());
    for (index, coefficient) in spectrum.iter().enumerate() {
        let multiplicity = if index == 0 || (dimension.is_multiple_of(2) && index == nyquist) {
            1.0
        } else {
            2.0
        };
        energy.push(ensure_finite(
            "spectral_bin_energy",
            multiplicity * coefficient.norm_sqr() * scale,
        )?);
    }
    Ok(energy)
}

fn normalized_frequency(index: usize, dimension: usize) -> f64 {
    2.0 * index as f64 / dimension as f64
}

fn fractional_energy(
    spectral_energy: &[f64],
    dimension: usize,
    alpha: f64,
) -> Result<f64, ZSpacePosteriorError> {
    if spectral_energy.len() <= 1 {
        return Ok(0.0);
    }
    let mut energy = 0.0;
    for (index, contribution) in spectral_energy.iter().enumerate() {
        let omega = normalized_frequency(index, dimension);
        energy += omega.powf(2.0 * alpha) * contribution;
    }
    ensure_finite("frac_energy", energy)
}

fn spectral_centroid(
    spectral_energy: &[f64],
    dimension: usize,
    total_energy: f64,
) -> Result<f64, ZSpacePosteriorError> {
    if total_energy == 0.0 {
        return Ok(0.0);
    }
    let weighted = spectral_energy
        .iter()
        .enumerate()
        .map(|(index, contribution)| normalized_frequency(index, dimension) * contribution)
        .sum::<f64>();
    ensure_finite("spectral_centroid", weighted / total_energy)
}

fn spectral_drift_response(
    spectral_energy: &[f64],
    dimension: usize,
) -> Result<f64, ZSpacePosteriorError> {
    if spectral_energy.len() <= 1 {
        return Ok(0.0);
    }
    let mut low = 0.0;
    let mut high = 0.0;
    for (index, contribution) in spectral_energy.iter().enumerate() {
        if normalized_frequency(index, dimension) < 0.5 {
            low += contribution;
        } else {
            high += contribution;
        }
    }
    let total = high + low;
    if total == 0.0 {
        return Ok(0.0);
    }
    ensure_finite("drs", ((high - low) / total).tanh())
}

fn normalized_gradient(values: &[f64], length: usize) -> Result<Vec<f64>, ZSpacePosteriorError> {
    let mut gradient = values.to_vec();
    gradient.resize(length, 0.0);
    gradient.truncate(length);
    let scale = gradient.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let denominator = scale.max(1.0);
    if denominator > 1.0 {
        for value in &mut gradient {
            *value /= denominator;
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
    let spectral_bins = one_sided_spectral_energy(&spectrum, dimension)?;
    let spectral_energy = ensure_finite(
        "spectral_energy",
        spectral_bins.iter().copied().sum::<f64>(),
    )?;
    let parseval_relative_error = ensure_finite(
        "parseval_relative_error",
        if energy == 0.0 {
            0.0
        } else {
            (spectral_energy - energy).abs() / energy
        },
    )?;
    let frac_energy = fractional_energy(&spectral_bins, dimension, request.alpha)?;
    let fractional_energy_ratio = ensure_finite(
        "fractional_energy_ratio",
        if energy == 0.0 {
            0.0
        } else {
            frac_energy / energy
        },
    )?;
    let spectral_centroid = spectral_centroid(&spectral_bins, dimension, spectral_energy)?;
    let gradient = latent_gradient(&values)?;

    let speed = ensure_finite(
        "speed",
        (mean_velocity + 0.25 * l2 / dimension as f64).tanh(),
    )?;
    let memory = ensure_finite("memory", (centre + 0.25 * energy / dimension as f64).tanh())?;
    let smoothness = 1.0 / (1.0 + curvature_energy);
    let drift = 1.0 / (1.0 + mean_velocity);
    let stability = ensure_finite("stability", ((smoothness + drift) * 1.5 - 1.0).tanh())?;
    let drs = spectral_drift_response(&spectral_bins, dimension)?;
    let frac = ensure_finite("frac", fractional_energy_ratio.tanh())?;
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
        spectral_formula: ZSPACE_POSTERIOR_SPECTRAL_FORMULA,
        gradient_formula: ZSPACE_POSTERIOR_GRADIENT_FORMULA,
        gradient_basis: ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS,
        barycentric_formula: ZSPACE_POSTERIOR_BARYCENTRIC_FORMULA,
        z_state: values,
        alpha: request.alpha,
        metrics,
        gradient,
        barycentric,
        energy,
        spectral_energy,
        parseval_relative_error,
        frac_energy,
        fractional_energy_ratio,
        spectral_centroid,
        spectral_bins: spectrum.len(),
    })
}

fn canonical_partial(
    partial: BTreeMap<String, ZSpaceMetricInput>,
    gradient_basis: Option<String>,
) -> Result<CanonicalPosteriorPartial, ZSpacePosteriorError> {
    if partial.is_empty() {
        if gradient_basis.is_some() {
            return Err(ZSpacePosteriorError::Fusion(
                ZSpaceFusionError::GradientBasisWithoutGradient { index: 0 },
            ));
        }
        return Ok(CanonicalPosteriorPartial {
            metrics: BTreeMap::new(),
            control_gradient: None,
        });
    }
    let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
        partials: vec![Some(ZSpacePartialInput {
            metrics: partial,
            weight: 1.0,
            origin: Some("zspace_posterior_projection".to_owned()),
            gradient_basis,
            telemetry: None,
        })],
        weights: None,
        strategy: ZSpaceFusionStrategy::Last,
        gradient_alignment: Default::default(),
        metric_gradient_dimension: None,
        telemetry: Vec::new(),
    })?;
    let control_gradient = match fused.gradient {
        Some(values) => {
            let basis = fused
                .gradient_basis
                .ok_or(ZSpacePosteriorError::MissingControlGradientBasis)?;
            let l2 = ensure_finite(
                "control_gradient_l2",
                values
                    .iter()
                    .fold(0.0_f64, |norm, value| norm.hypot(*value)),
            )?;
            let linf = ensure_finite(
                "control_gradient_linf",
                values.iter().copied().map(f64::abs).fold(0.0, f64::max),
            )?;
            Some(ZSpacePosteriorControlGradient {
                source: "partial",
                basis,
                dimension: values.len(),
                values,
                l2,
                linf,
            })
        }
        None => None,
    };
    Ok(CanonicalPosteriorPartial {
        metrics: fused.metrics,
        control_gradient,
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
    let partial = canonical_partial(request.partial, request.gradient_basis)?;
    let mut metrics = prior.metrics.clone();
    let mut applied = BTreeMap::new();
    let mut squared_residual = 0.0;
    let mut residual_metric_count = 0;
    for (metric, value) in partial.metrics {
        if let Some(base_metric) = BASE_METRICS
            .iter()
            .copied()
            .find(|candidate| *candidate == metric)
        {
            let prior_value = prior.metrics.get(base_metric).copied().ok_or(
                ZSpacePosteriorError::MissingMetric {
                    metric: base_metric,
                },
            )?;
            squared_residual += (value - prior_value).powi(2);
            residual_metric_count += 1;
        }
        metrics.insert(metric.clone(), value);
        applied.insert(metric, ZSpaceMetricInput::Scalar(value));
    }
    let gradient = prior.gradient.clone();
    let control_gradient = partial.control_gradient;

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

    let residual_before = ensure_finite(
        "residual",
        if residual_metric_count == 0 {
            0.0
        } else {
            (squared_residual / residual_metric_count as f64).sqrt()
        },
    )?;
    let confidence_before = ensure_finite("confidence", (-residual_before).exp())?;
    let residual = residual_before;
    let mut confidence = confidence_before;
    let mut telemetry_reliability = 1.0;

    let telemetry = if request.telemetry.is_empty() {
        None
    } else {
        Some(fuse_zspace_telemetry(&request.telemetry)?)
    };
    let telemetry_adjustment = telemetry.as_ref().map(|telemetry| {
        let variance_reliability = 1.0 / (1.0 + telemetry.summary.variance.max(0.0));
        let focus_reliability = (1.0 + 0.25 * telemetry.summary.focus).clamp(0.75, 1.0);
        let combined_reliability = variance_reliability * focus_reliability;
        telemetry_reliability = combined_reliability;
        confidence = (confidence * combined_reliability).clamp(0.0, confidence_before);
        ZSpacePosteriorTelemetryAdjustment {
            variance_reliability,
            focus_reliability,
            combined_reliability,
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
        gradient_basis: ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS,
        gradient_source: "latent_state",
        control_gradient,
        barycentric,
        residual,
        residual_metric_count,
        confidence,
        telemetry_reliability,
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
    fn decode_is_parseval_normalized_and_auditable() {
        let decoded = decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
            z_state: sample_state(),
            alpha: 0.35,
        })
        .expect("decode posterior");

        assert_eq!(decoded.semantic_backend, "rust");
        assert_eq!(decoded.contract_version, "spiraltorch.zspace_posterior.v2");
        assert_eq!(
            decoded.gradient_basis,
            ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
        );
        assert_eq!(decoded.spectral_bins, 3);
        assert_relative_eq!(decoded.energy, 0.2857, epsilon = 1.0e-14);
        assert_relative_eq!(decoded.spectral_energy, decoded.energy, epsilon = 1.0e-14);
        assert!(decoded.parseval_relative_error <= 1.0e-14);
        assert_relative_eq!(decoded.frac_energy, 0.2210090973787923, epsilon = 1.0e-12);
        assert_relative_eq!(
            decoded.fractional_energy_ratio,
            0.7735705193517406,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            decoded.spectral_centroid,
            0.7415120756037802,
            epsilon = 1.0e-12
        );
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
            0.6413201743341207,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            decoded.metrics["frac"],
            0.6490008507207798,
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
            gradient_basis: Some("test.posterior.control.v1".to_owned()),
            smoothing: 0.35,
            telemetry: vec![json!({"psi": {"energy": 2.0, "focus": 0.4}})],
        })
        .expect("project posterior");

        assert_relative_eq!(projected.metrics["memory"], -0.2, epsilon = 1.0e-14);
        assert_eq!(
            projected.gradient_basis,
            ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS
        );
        assert_eq!(projected.gradient_source, "latent_state");
        for (actual, expected) in projected.gradient.iter().zip([-0.015, 0.18, -0.085, -0.24]) {
            assert_relative_eq!(*actual, expected, epsilon = 1.0e-14);
        }
        assert!(!projected.applied.contains_key("gradient"));
        let control = projected
            .control_gradient
            .as_ref()
            .expect("basis-tagged control gradient");
        assert_eq!(control.values, [2.0, -1.0]);
        assert_eq!(control.dimension, 2);
        assert_eq!(control.basis, "test.posterior.control.v1");
        assert_relative_eq!(control.l2, 5.0_f64.sqrt(), epsilon = 1.0e-14);
        assert_relative_eq!(control.linf, 2.0, epsilon = 1.0e-14);
        assert_eq!(projected.residual_metric_count, 2);
        assert_relative_eq!(projected.residual, 0.24233126609703365, epsilon = 1.0e-12);
        assert_relative_eq!(
            projected.telemetry_reliability,
            1.0 / 1.64,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(projected.confidence, 0.4785342427598339, epsilon = 1.0e-12);
        let telemetry = projected.telemetry.expect("telemetry contract");
        assert_eq!(telemetry.payload["psi.energy"], 2.0);
        assert_eq!(telemetry.payload["psi.focus"], 0.4);
        assert_relative_eq!(telemetry.summary.variance, 0.64, epsilon = 1.0e-14);
        let adjustment = projected
            .telemetry_adjustment
            .expect("telemetry reliability audit");
        assert_relative_eq!(
            adjustment.combined_reliability,
            projected.telemetry_reliability,
            epsilon = 1.0e-14
        );
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
            gradient_basis: None,
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
            gradient_basis: None,
            smoothing: 1.5,
            telemetry: Vec::new(),
        });
        assert_eq!(
            invalid_smoothing,
            Err(ZSpacePosteriorError::InvalidSmoothing { value: 1.5 })
        );

        let untagged_gradient = project_zspace_posterior(ZSpacePosteriorProjectionRequest {
            z_state: sample_state(),
            alpha: 0.35,
            partial: BTreeMap::from([(
                "gradient".to_owned(),
                ZSpaceMetricInput::Vector(vec![0.2, -0.1]),
            )]),
            gradient_basis: None,
            smoothing: 0.35,
            telemetry: Vec::new(),
        });
        assert_eq!(
            untagged_gradient,
            Err(ZSpacePosteriorError::MissingControlGradientBasis)
        );
    }

    #[test]
    fn spectral_energy_is_scale_consistent_and_frequency_bounded() {
        let base = decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
            z_state: sample_state(),
            alpha: 0.35,
        })
        .expect("base decode");
        let scaled = decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
            z_state: sample_state()
                .into_iter()
                .map(|value| value * 7.0)
                .collect(),
            alpha: 0.35,
        })
        .expect("scaled decode");
        let tiny = decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
            z_state: sample_state()
                .into_iter()
                .map(|value| value * 1.0e-8)
                .collect(),
            alpha: 0.35,
        })
        .expect("tiny decode");
        assert_relative_eq!(
            base.fractional_energy_ratio,
            scaled.fractional_energy_ratio,
            epsilon = 1.0e-8
        );
        assert_relative_eq!(
            base.fractional_energy_ratio,
            tiny.fractional_energy_ratio,
            epsilon = 1.0e-8
        );
        assert_relative_eq!(base.metrics["drs"], scaled.metrics["drs"], epsilon = 1.0e-8);
        assert_relative_eq!(base.metrics["drs"], tiny.metrics["drs"], epsilon = 1.0e-8);
        assert!(base.frac_energy <= base.energy + 1.0e-12);
        assert!((0.0..=1.0).contains(&base.spectral_centroid));

        let constant = decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
            z_state: vec![1.0; 8],
            alpha: 0.35,
        })
        .expect("constant decode");
        assert_relative_eq!(constant.frac_energy, 0.0, epsilon = 1.0e-14);
        assert_relative_eq!(constant.spectral_centroid, 0.0, epsilon = 1.0e-14);

        let alternating = decode_zspace_posterior(ZSpacePosteriorDecodeRequest {
            z_state: vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            alpha: 0.35,
        })
        .expect("Nyquist decode");
        assert_relative_eq!(
            alternating.spectral_energy,
            alternating.energy,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            alternating.frac_energy,
            alternating.energy,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(alternating.spectral_centroid, 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn latent_gradient_normalization_is_bounded_and_continuous_at_unit_scale() {
        let epsilon = 1.0e-9;
        let below = normalized_gradient(&[1.0 - epsilon, 0.5], 2).expect("below threshold");
        let above = normalized_gradient(&[1.0 + epsilon, 0.5], 2).expect("above threshold");

        assert!(below.iter().chain(&above).all(|value| value.abs() <= 1.0));
        assert_relative_eq!(below[0], above[0], epsilon = 2.0e-9);
        assert_relative_eq!(below[1], above[1], epsilon = 1.0e-9);
    }

    #[test]
    fn telemetry_reliability_is_conservative_and_variance_monotonic() {
        let project = |telemetry| {
            project_zspace_posterior(ZSpacePosteriorProjectionRequest {
                z_state: sample_state(),
                alpha: 0.35,
                partial: BTreeMap::from([("speed".to_owned(), ZSpaceMetricInput::Scalar(0.3))]),
                gradient_basis: None,
                smoothing: 0.35,
                telemetry,
            })
            .expect("posterior projection")
        };
        let no_telemetry = project(Vec::new());
        let zero_variance = project(vec![json!({"a": 1.0, "b": 1.0})]);
        let high_variance = project(vec![json!({"a": 0.0, "b": 4.0})]);

        assert_relative_eq!(
            no_telemetry.residual,
            zero_variance.residual,
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            zero_variance.residual,
            high_variance.residual,
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            no_telemetry.confidence,
            zero_variance.confidence,
            epsilon = 1.0e-14
        );
        assert!(high_variance.confidence < zero_variance.confidence);
        assert!(
            high_variance.confidence
                <= high_variance
                    .telemetry_adjustment
                    .expect("telemetry adjustment")
                    .confidence_before
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
