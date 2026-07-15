//! Canonical Z-space partial and telemetry fusion semantics.
//!
//! Rust owns normalization, reduction, and audit metadata. Language bindings
//! should only translate their native values into these request types.

use crate::inference::zspace_coherence::canonical_zspace_coherence_metric_name;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use thiserror::Error;

/// Stable contract identifier shared by Rust, Python, and WASM telemetry clients.
pub const ZSPACE_TELEMETRY_FUSION_CONTRACT_VERSION: &str = "spiraltorch.zspace_telemetry_fusion.v1";
/// Stable payload kind for canonical Z-space telemetry fusion.
pub const ZSPACE_TELEMETRY_FUSION_KIND: &str = "spiraltorch.zspace_telemetry_fusion";
/// Stable contract identifier shared by Rust, Python, and WASM partial clients.
pub const ZSPACE_PARTIAL_FUSION_CONTRACT_VERSION: &str = "spiraltorch.zspace_partial_fusion.v1";
/// Stable payload kind for canonical Z-space partial fusion.
pub const ZSPACE_PARTIAL_FUSION_KIND: &str = "spiraltorch.zspace_partial_fusion";
/// Crate/module that owns Z-space fusion semantics.
pub const ZSPACE_FUSION_SEMANTIC_OWNER: &str = "st-core::telemetry::zspace_fusion";
/// Backend label attached to payloads produced by the canonical implementation.
pub const ZSPACE_FUSION_SEMANTIC_BACKEND: &str = "rust";

pub const ZSPACE_FUSION_MAX_PARTIALS: usize = 4_096;
pub const ZSPACE_FUSION_MAX_METRICS_PER_PARTIAL: usize = 4_096;
pub const ZSPACE_FUSION_MAX_GRADIENT_DIM: usize = 4_096;
pub const ZSPACE_FUSION_MAX_TELEMETRY_ENTRIES: usize = 16_384;
pub const ZSPACE_FUSION_MAX_TELEMETRY_DEPTH: usize = 64;

#[derive(Debug, Error, PartialEq)]
pub enum ZSpaceFusionError {
    #[error("partial count {actual} exceeds limit {max}")]
    TooManyPartials { actual: usize, max: usize },
    #[error("weights length {actual} must match partial count {expected}")]
    WeightsLength { actual: usize, expected: usize },
    #[error("weight at partial {index} must be finite")]
    NonFiniteWeight { index: usize },
    #[error("telemetry summary field '{field}' must be finite")]
    NonFiniteTelemetrySummary { field: &'static str },
    #[error("fused metric '{metric}' must be finite")]
    NonFiniteMetricReduction { metric: String },
    #[error("fused gradient entry {entry} must be finite")]
    NonFiniteGradientReduction { entry: usize },
    #[error("partial {index} has {actual} metrics, exceeding limit {max}")]
    TooManyMetrics {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error("unknown Z-space metric '{metric}' at partial {index}")]
    UnknownMetric { index: usize, metric: String },
    #[error("metrics '{first}' and '{second}' both resolve to '{canonical}' at partial {index}")]
    AliasCollision {
        index: usize,
        first: String,
        second: String,
        canonical: String,
    },
    #[error("metric '{metric}' at partial {index} must be a finite scalar")]
    InvalidScalarMetric { index: usize, metric: String },
    #[error("metric '{metric}' at partial {index} must be a numeric vector")]
    InvalidVectorMetric { index: usize, metric: String },
    #[error("gradient dimension {actual} at partial {index} exceeds limit {max}")]
    GradientTooLong {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error("gradient entry {entry} at partial {index} must be finite")]
    NonFiniteGradient { index: usize, entry: usize },
    #[error("telemetry payload {index} must be an object")]
    TelemetryNotObject { index: usize },
    #[error("telemetry payload {index} exceeds nesting limit {max_depth}")]
    TelemetryTooDeep { index: usize, max_depth: usize },
    #[error("flattened telemetry exceeds entry limit {max}")]
    TooManyTelemetryEntries { max: usize },
    #[error("telemetry value '{key}' in payload {index} must be finite")]
    NonFiniteTelemetry { index: usize, key: String },
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ZSpaceFusionStrategy {
    #[default]
    Mean,
    Last,
    Max,
    Min,
}

impl ZSpaceFusionStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Last => "last",
            Self::Max => "max",
            Self::Min => "min",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ZSpaceMetricInput {
    Scalar(f64),
    Vector(Vec<f64>),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ZSpacePartialInput {
    pub metrics: BTreeMap<String, ZSpaceMetricInput>,
    #[serde(default = "default_weight")]
    pub weight: f64,
    #[serde(default)]
    pub origin: Option<String>,
    #[serde(default)]
    pub telemetry: Option<Value>,
}

const fn default_weight() -> f64 {
    1.0
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ZSpacePartialFusionRequest {
    pub partials: Vec<Option<ZSpacePartialInput>>,
    #[serde(default)]
    pub weights: Option<Vec<f64>>,
    #[serde(default)]
    pub strategy: ZSpaceFusionStrategy,
    #[serde(default)]
    pub telemetry: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceTelemetrySummary {
    pub count: usize,
    pub l1: f64,
    pub l2: f64,
    pub linf: f64,
    pub mean: f64,
    pub variance: f64,
    pub energy: f64,
    pub amplitude: f64,
    pub positive: f64,
    pub negative: f64,
    pub balance: f64,
    pub focus: f64,
}

impl ZSpaceTelemetrySummary {
    fn try_from_values<'a>(
        values: impl Iterator<Item = &'a f64>,
    ) -> Result<Self, ZSpaceFusionError> {
        let data: Vec<f64> = values.copied().collect();
        if data.is_empty() {
            return Ok(Self {
                count: 0,
                l1: 0.0,
                l2: 0.0,
                linf: 0.0,
                mean: 0.0,
                variance: 0.0,
                energy: 0.0,
                amplitude: 0.0,
                positive: 0.0,
                negative: 0.0,
                balance: 0.0,
                focus: 0.0,
            });
        }

        let count = data.len();
        let l1 = data.iter().map(|value| value.abs()).sum();
        let energy = data.iter().map(|value| value * value).sum::<f64>();
        let linf = data.iter().map(|value| value.abs()).fold(0.0, f64::max);
        let mean = data.iter().enumerate().fold(0.0, |mean, (index, value)| {
            let count = (index + 1) as f64;
            mean * ((count - 1.0) / count) + value * (1.0 / count)
        });
        let variance = data
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / count as f64;
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let positive = data.iter().filter(|value| **value > 0.0).sum::<f64>();
        let negative = -data.iter().filter(|value| **value < 0.0).sum::<f64>();
        let balance = (positive - negative) / (positive + negative + 1e-9);

        let summary = Self {
            count,
            l1,
            l2: energy.sqrt(),
            linf,
            mean,
            variance,
            energy,
            amplitude: max - min,
            positive,
            negative,
            balance,
            focus: (balance * 1.5).tanh(),
        };
        for (field, value) in [
            ("l1", summary.l1),
            ("l2", summary.l2),
            ("linf", summary.linf),
            ("mean", summary.mean),
            ("variance", summary.variance),
            ("energy", summary.energy),
            ("amplitude", summary.amplitude),
            ("positive", summary.positive),
            ("negative", summary.negative),
            ("balance", summary.balance),
            ("focus", summary.focus),
        ] {
            if !value.is_finite() {
                return Err(ZSpaceFusionError::NonFiniteTelemetrySummary { field });
            }
        }
        Ok(summary)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceTelemetrySourceAudit {
    pub index: usize,
    pub flattened_count: usize,
    pub ignored_value_count: usize,
    pub conflict_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceTelemetryFusionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub payload: BTreeMap<String, f64>,
    pub summary: ZSpaceTelemetrySummary,
    pub input_count: usize,
    pub active_input_count: usize,
    pub ignored_value_count: usize,
    pub conflict_count: usize,
    pub sources: Vec<ZSpaceTelemetrySourceAudit>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePartialSourceAudit {
    pub index: usize,
    pub origin: Option<String>,
    pub weight: Option<f64>,
    pub status: &'static str,
    pub metric_count: usize,
    pub gradient_dim: usize,
    pub telemetry_entry_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePartialFusionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub strategy: &'static str,
    pub metrics: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient: Option<Vec<f64>>,
    pub telemetry: ZSpaceTelemetryFusionPayload,
    pub input_count: usize,
    pub active_count: usize,
    pub suppressed_count: usize,
    pub null_count: usize,
    pub sources: Vec<ZSpacePartialSourceAudit>,
}

#[derive(Default)]
struct FlattenAudit {
    ignored: usize,
    conflicts: usize,
}

fn flatten_telemetry(
    value: &Value,
    index: usize,
    prefix: &str,
    depth: usize,
    output: &mut BTreeMap<String, f64>,
    audit: &mut FlattenAudit,
) -> Result<(), ZSpaceFusionError> {
    if depth > ZSPACE_FUSION_MAX_TELEMETRY_DEPTH {
        return Err(ZSpaceFusionError::TelemetryTooDeep {
            index,
            max_depth: ZSPACE_FUSION_MAX_TELEMETRY_DEPTH,
        });
    }
    let object = value
        .as_object()
        .ok_or(ZSpaceFusionError::TelemetryNotObject { index })?;

    for (key, value) in object {
        let label = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        if value.is_object() {
            flatten_telemetry(value, index, &label, depth + 1, output, audit)?;
            continue;
        }

        let numeric = match value {
            Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
            Value::Number(value) => {
                let numeric =
                    value
                        .as_f64()
                        .ok_or_else(|| ZSpaceFusionError::NonFiniteTelemetry {
                            index,
                            key: label.clone(),
                        })?;
                if !numeric.is_finite() {
                    return Err(ZSpaceFusionError::NonFiniteTelemetry { index, key: label });
                }
                Some(numeric)
            }
            Value::String(value) => value.parse::<f64>().ok().filter(|value| value.is_finite()),
            _ => None,
        };

        let Some(numeric) = numeric else {
            audit.ignored += 1;
            continue;
        };
        if output.len() >= ZSPACE_FUSION_MAX_TELEMETRY_ENTRIES && !output.contains_key(&label) {
            return Err(ZSpaceFusionError::TooManyTelemetryEntries {
                max: ZSPACE_FUSION_MAX_TELEMETRY_ENTRIES,
            });
        }
        if output.insert(label, numeric).is_some() {
            audit.conflicts += 1;
        }
    }
    Ok(())
}

/// Flatten and merge telemetry payloads using deterministic last-writer-wins semantics.
pub fn fuse_zspace_telemetry(
    payloads: &[Value],
) -> Result<ZSpaceTelemetryFusionPayload, ZSpaceFusionError> {
    if payloads.len() > ZSPACE_FUSION_MAX_PARTIALS {
        return Err(ZSpaceFusionError::TooManyPartials {
            actual: payloads.len(),
            max: ZSPACE_FUSION_MAX_PARTIALS,
        });
    }

    let mut merged = BTreeMap::new();
    let mut ignored_value_count = 0;
    let mut conflict_count = 0;
    let mut active_input_count = 0;
    let mut sources = Vec::with_capacity(payloads.len());

    for (index, payload) in payloads.iter().enumerate() {
        let mut flattened = BTreeMap::new();
        let mut audit = FlattenAudit::default();
        flatten_telemetry(payload, index, "", 0, &mut flattened, &mut audit)?;
        let flattened_count = flattened.len();
        if flattened_count > 0 {
            active_input_count += 1;
        }
        for (key, value) in flattened {
            if merged.len() >= ZSPACE_FUSION_MAX_TELEMETRY_ENTRIES && !merged.contains_key(&key) {
                return Err(ZSpaceFusionError::TooManyTelemetryEntries {
                    max: ZSPACE_FUSION_MAX_TELEMETRY_ENTRIES,
                });
            }
            if merged.insert(key, value).is_some() {
                audit.conflicts += 1;
            }
        }
        ignored_value_count += audit.ignored;
        conflict_count += audit.conflicts;
        sources.push(ZSpaceTelemetrySourceAudit {
            index,
            flattened_count,
            ignored_value_count: audit.ignored,
            conflict_count: audit.conflicts,
        });
    }

    Ok(ZSpaceTelemetryFusionPayload {
        kind: ZSPACE_TELEMETRY_FUSION_KIND,
        contract_version: ZSPACE_TELEMETRY_FUSION_CONTRACT_VERSION,
        semantic_owner: ZSPACE_FUSION_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_FUSION_SEMANTIC_BACKEND,
        summary: ZSpaceTelemetrySummary::try_from_values(merged.values())?,
        payload: merged,
        input_count: payloads.len(),
        active_input_count,
        ignored_value_count,
        conflict_count,
        sources,
    })
}

fn stable_weighted_mean<'a>(values: impl Iterator<Item = (&'a f64, &'a f64)>) -> f64 {
    let values: Vec<(&f64, &f64)> = values.collect();
    let max_weight = values
        .iter()
        .map(|(_, weight)| **weight)
        .fold(0.0, f64::max);
    let mut total_weight = 0.0;
    let mut mean = 0.0;
    for (value, weight) in values {
        let scaled_weight = *weight / max_weight;
        let next_weight = total_weight + scaled_weight;
        mean = mean * (total_weight / next_weight) + value * (scaled_weight / next_weight);
        total_weight = next_weight;
    }
    mean
}

fn reduce_scalars(values: &[(f64, f64)], strategy: ZSpaceFusionStrategy) -> f64 {
    match strategy {
        ZSpaceFusionStrategy::Mean => {
            stable_weighted_mean(values.iter().map(|(value, weight)| (value, weight)))
        }
        ZSpaceFusionStrategy::Last => values.last().map(|(value, _)| *value).unwrap_or(0.0),
        ZSpaceFusionStrategy::Max => values
            .iter()
            .map(|(value, _)| *value)
            .fold(f64::NEG_INFINITY, f64::max),
        ZSpaceFusionStrategy::Min => values
            .iter()
            .map(|(value, _)| *value)
            .fold(f64::INFINITY, f64::min),
    }
}

fn reduce_gradients(
    gradients: &[(Vec<f64>, f64)],
    strategy: ZSpaceFusionStrategy,
) -> Option<Vec<f64>> {
    let length = gradients
        .iter()
        .map(|(gradient, _)| gradient.len())
        .max()
        .unwrap_or(0);
    if length == 0 {
        return None;
    }

    let mut result = vec![0.0; length];
    match strategy {
        ZSpaceFusionStrategy::Mean => {
            let max_weight = gradients
                .iter()
                .map(|(_, weight)| *weight)
                .fold(0.0, f64::max);
            let mut total_weight = 0.0;
            for (gradient, weight) in gradients {
                let scaled_weight = weight / max_weight;
                let next_weight = total_weight + scaled_weight;
                for (index, output) in result.iter_mut().enumerate() {
                    *output = *output * (total_weight / next_weight)
                        + gradient.get(index).copied().unwrap_or(0.0)
                            * (scaled_weight / next_weight);
                }
                total_weight = next_weight;
            }
        }
        ZSpaceFusionStrategy::Last => {
            let (gradient, _) = gradients.last().expect("non-empty gradients");
            result[..gradient.len()].copy_from_slice(gradient);
        }
        ZSpaceFusionStrategy::Max | ZSpaceFusionStrategy::Min => {
            for (index, output) in result.iter_mut().enumerate() {
                *output = gradients
                    .iter()
                    .map(|(gradient, _)| gradient.get(index).copied().unwrap_or(0.0))
                    .fold(
                        if strategy == ZSpaceFusionStrategy::Max {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        },
                        if strategy == ZSpaceFusionStrategy::Max {
                            f64::max
                        } else {
                            f64::min
                        },
                    );
            }
        }
    }
    Some(result)
}

/// Fuse Z-space metric partials and attached telemetry under one Rust-owned contract.
pub fn fuse_zspace_partials(
    request: ZSpacePartialFusionRequest,
) -> Result<ZSpacePartialFusionPayload, ZSpaceFusionError> {
    let input_count = request.partials.len();
    if input_count > ZSPACE_FUSION_MAX_PARTIALS {
        return Err(ZSpaceFusionError::TooManyPartials {
            actual: input_count,
            max: ZSPACE_FUSION_MAX_PARTIALS,
        });
    }
    if let Some(weights) = &request.weights {
        if weights.len() != input_count {
            return Err(ZSpaceFusionError::WeightsLength {
                actual: weights.len(),
                expected: input_count,
            });
        }
        if let Some((index, _)) = weights
            .iter()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite())
        {
            return Err(ZSpaceFusionError::NonFiniteWeight { index });
        }
    }

    let mut aggregated: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    let mut gradients = Vec::new();
    let mut telemetry = Vec::new();
    let mut sources = Vec::with_capacity(input_count);
    let mut active_count = 0;
    let mut suppressed_count = 0;
    let mut null_count = 0;

    for (index, partial) in request.partials.into_iter().enumerate() {
        let Some(partial) = partial else {
            null_count += 1;
            sources.push(ZSpacePartialSourceAudit {
                index,
                origin: None,
                weight: None,
                status: "null",
                metric_count: 0,
                gradient_dim: 0,
                telemetry_entry_count: 0,
            });
            continue;
        };
        let weight = request
            .weights
            .as_ref()
            .map(|weights| weights[index])
            .unwrap_or(partial.weight);
        if !weight.is_finite() {
            return Err(ZSpaceFusionError::NonFiniteWeight { index });
        }
        if weight <= 0.0 {
            suppressed_count += 1;
            sources.push(ZSpacePartialSourceAudit {
                index,
                origin: partial.origin,
                weight: Some(weight),
                status: "suppressed",
                metric_count: 0,
                gradient_dim: 0,
                telemetry_entry_count: 0,
            });
            continue;
        }
        if partial.metrics.len() > ZSPACE_FUSION_MAX_METRICS_PER_PARTIAL {
            return Err(ZSpaceFusionError::TooManyMetrics {
                index,
                actual: partial.metrics.len(),
                max: ZSPACE_FUSION_MAX_METRICS_PER_PARTIAL,
            });
        }

        let mut canonical_metrics: BTreeMap<&'static str, (String, ZSpaceMetricInput)> =
            BTreeMap::new();
        for (key, value) in partial.metrics {
            let Some(canonical) = canonical_metric_name(&key) else {
                return Err(ZSpaceFusionError::UnknownMetric { index, metric: key });
            };
            if let Some((first, _)) = canonical_metrics.get(canonical) {
                return Err(ZSpaceFusionError::AliasCollision {
                    index,
                    first: first.clone(),
                    second: key,
                    canonical: canonical.to_owned(),
                });
            }
            canonical_metrics.insert(canonical, (key, value));
        }

        let mut metric_count = 0;
        let mut gradient_dim = 0;
        for (canonical, (_, value)) in canonical_metrics {
            if canonical == "gradient" {
                let ZSpaceMetricInput::Vector(gradient) = value else {
                    return Err(ZSpaceFusionError::InvalidVectorMetric {
                        index,
                        metric: canonical.to_owned(),
                    });
                };
                if gradient.len() > ZSPACE_FUSION_MAX_GRADIENT_DIM {
                    return Err(ZSpaceFusionError::GradientTooLong {
                        index,
                        actual: gradient.len(),
                        max: ZSPACE_FUSION_MAX_GRADIENT_DIM,
                    });
                }
                if let Some((entry, _)) = gradient
                    .iter()
                    .enumerate()
                    .find(|(_, value)| !value.is_finite())
                {
                    return Err(ZSpaceFusionError::NonFiniteGradient { index, entry });
                }
                gradient_dim = gradient.len();
                gradients.push((gradient, weight));
                continue;
            }
            let ZSpaceMetricInput::Scalar(value) = value else {
                return Err(ZSpaceFusionError::InvalidScalarMetric {
                    index,
                    metric: canonical.to_owned(),
                });
            };
            if !value.is_finite() {
                return Err(ZSpaceFusionError::InvalidScalarMetric {
                    index,
                    metric: canonical.to_owned(),
                });
            }
            metric_count += 1;
            aggregated
                .entry(canonical.to_owned())
                .or_default()
                .push((value, weight));
        }

        let telemetry_entry_count = if let Some(payload) = partial.telemetry {
            let mut flattened = BTreeMap::new();
            let mut audit = FlattenAudit::default();
            flatten_telemetry(&payload, index, "", 0, &mut flattened, &mut audit)?;
            let count = flattened.len();
            telemetry.push(payload);
            count
        } else {
            0
        };
        active_count += 1;
        sources.push(ZSpacePartialSourceAudit {
            index,
            origin: partial.origin,
            weight: Some(weight),
            status: "active",
            metric_count,
            gradient_dim,
            telemetry_entry_count,
        });
    }

    telemetry.extend(request.telemetry);
    let telemetry = fuse_zspace_telemetry(&telemetry)?;
    let mut metrics = BTreeMap::new();
    for (key, values) in aggregated {
        let value = reduce_scalars(&values, request.strategy);
        if !value.is_finite() {
            return Err(ZSpaceFusionError::NonFiniteMetricReduction { metric: key });
        }
        metrics.insert(key, value);
    }
    let gradient = reduce_gradients(&gradients, request.strategy);
    if let Some((entry, _)) = gradient
        .iter()
        .flatten()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ZSpaceFusionError::NonFiniteGradientReduction { entry });
    }

    Ok(ZSpacePartialFusionPayload {
        kind: ZSPACE_PARTIAL_FUSION_KIND,
        contract_version: ZSPACE_PARTIAL_FUSION_CONTRACT_VERSION,
        semantic_owner: ZSPACE_FUSION_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_FUSION_SEMANTIC_BACKEND,
        strategy: request.strategy.as_str(),
        metrics,
        gradient,
        telemetry,
        input_count,
        active_count,
        suppressed_count,
        null_count,
        sources,
    })
}

/// Resolve a public Z-space metric alias to its canonical spelling.
pub fn canonical_metric_name(name: &str) -> Option<&'static str> {
    match name.to_ascii_lowercase().as_str() {
        "speed" | "velocity" => Some("speed"),
        "mem" | "memory" => Some("memory"),
        "stab" | "stability" => Some("stability"),
        "frac" | "frac_reg" | "fractality" => Some("frac"),
        "drs" | "drift" => Some("drs"),
        "gradient" | "grad" => Some("gradient"),
        "canvas_energy" => Some("canvas_energy"),
        "canvas_mean" => Some("canvas_mean"),
        "canvas_peak" => Some("canvas_peak"),
        "canvas_balance" => Some("canvas_balance"),
        "canvas_l1" => Some("canvas_l1"),
        "canvas_l2" => Some("canvas_l2"),
        "canvas_linf" => Some("canvas_linf"),
        "canvas_pixels" => Some("canvas_pixels"),
        "canvas_patch_energy" => Some("canvas_patch_energy"),
        "canvas_patch_mean" => Some("canvas_patch_mean"),
        "canvas_patch_peak" => Some("canvas_patch_peak"),
        "canvas_patch_pixels" => Some("canvas_patch_pixels"),
        "canvas_patch_balance" => Some("canvas_patch_balance"),
        "hypergrad_norm" => Some("hypergrad_norm"),
        "hypergrad_balance" => Some("hypergrad_balance"),
        "hypergrad_mean" => Some("hypergrad_mean"),
        "hypergrad_l1" => Some("hypergrad_l1"),
        "hypergrad_l2" => Some("hypergrad_l2"),
        "hypergrad_linf" => Some("hypergrad_linf"),
        "realgrad_norm" => Some("realgrad_norm"),
        "realgrad_balance" => Some("realgrad_balance"),
        "realgrad_mean" => Some("realgrad_mean"),
        "realgrad_l1" => Some("realgrad_l1"),
        "realgrad_l2" => Some("realgrad_l2"),
        "realgrad_linf" => Some("realgrad_linf"),
        "import_l1" => Some("import_l1"),
        "import_l2" => Some("import_l2"),
        "import_linf" => Some("import_linf"),
        "import_mean" => Some("import_mean"),
        "import_variance" => Some("import_variance"),
        "import_energy" => Some("import_energy"),
        "import_count" => Some("import_count"),
        "import_amplitude" => Some("import_amplitude"),
        "import_balance" => Some("import_balance"),
        "import_focus" => Some("import_focus"),
        "elliptic_curvature" | "curvature_radius" | "elliptic_curvature_radius" => {
            Some("elliptic_curvature")
        }
        "elliptic_geodesic" | "geodesic_radius" => Some("elliptic_geodesic"),
        "elliptic_normalized" | "normalized_radius" => Some("elliptic_normalized"),
        "elliptic_alignment" | "spin_alignment" => Some("elliptic_alignment"),
        "elliptic_bias" | "normal_bias" => Some("elliptic_bias"),
        "elliptic_sheet_position" | "sheet_position" => Some("elliptic_sheet_position"),
        "elliptic_sheet_index" | "sheet_index" => Some("elliptic_sheet_index"),
        "elliptic_sheet_count" | "sheet_count" => Some("elliptic_sheet_count"),
        "elliptic_sector" | "topological_sector" => Some("elliptic_sector"),
        "elliptic_homology" | "homology_index" => Some("elliptic_homology"),
        "elliptic_resonance" | "resonance_heat" => Some("elliptic_resonance"),
        "elliptic_noise" | "noise_density" => Some("elliptic_noise"),
        _ => canonical_zspace_coherence_metric_name(name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn partial(
        metrics: Value,
        weight: f64,
        origin: &str,
        telemetry: Option<Value>,
    ) -> ZSpacePartialInput {
        ZSpacePartialInput {
            metrics: serde_json::from_value(metrics).expect("valid metrics"),
            weight,
            origin: Some(origin.to_owned()),
            telemetry,
        }
    }

    #[test]
    fn telemetry_contract_flattens_summarises_and_audits_conflicts() {
        let fused = fuse_zspace_telemetry(&[
            json!({"psi": {"energy": 2.0, "ready": true}, "skip": [1]}),
            json!({"psi.energy": "4.0", "note": "ignored"}),
        ])
        .expect("valid telemetry");

        assert_eq!(fused.kind, ZSPACE_TELEMETRY_FUSION_KIND);
        assert_eq!(fused.semantic_owner, ZSPACE_FUSION_SEMANTIC_OWNER);
        assert_eq!(fused.payload["psi.energy"], 4.0);
        assert_eq!(fused.payload["psi.ready"], 1.0);
        assert_eq!(fused.summary.count, 2);
        assert_eq!(fused.conflict_count, 1);
        assert_eq!(fused.ignored_value_count, 2);
    }

    #[test]
    fn weighted_mean_uses_explicit_weights_and_suppresses_telemetry() {
        let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![
                Some(partial(
                    json!({"velocity": 1.0, "grad": [1.0, 2.0]}),
                    100.0,
                    "first",
                    Some(json!({"suppressed_only": 1.0})),
                )),
                Some(partial(
                    json!({"speed": 5.0, "gradient": [5.0]}),
                    1.0,
                    "second",
                    Some(json!({"source": 2.0})),
                )),
            ],
            weights: Some(vec![0.0, 3.0]),
            strategy: ZSpaceFusionStrategy::Mean,
            telemetry: vec![json!({"external": 7.0})],
        })
        .expect("valid fusion");

        assert_eq!(fused.metrics["speed"], 5.0);
        assert_eq!(fused.gradient, Some(vec![5.0]));
        assert_eq!(fused.suppressed_count, 1);
        assert!(!fused.telemetry.payload.contains_key("suppressed_only"));
        assert_eq!(fused.telemetry.payload["source"], 2.0);
        assert_eq!(fused.telemetry.payload["external"], 7.0);
    }

    #[test]
    fn coherence_projection_metrics_flow_through_partial_fusion() {
        use crate::inference::zspace_coherence::{
            project_zspace_coherence, ZSpaceCoherenceContourInput, ZSpaceCoherenceDiagnosticsInput,
            ZSpaceCoherenceProjectionConfig, ZSpaceCoherenceProjectionRequest,
        };

        let entropy = -(0.6_f64 * 0.6_f64.ln() + 0.3_f64 * 0.3_f64.ln() + 0.1_f64 * 0.1_f64.ln());
        let projection = project_zspace_coherence(ZSpaceCoherenceProjectionRequest {
            diagnostics: ZSpaceCoherenceDiagnosticsInput {
                mean_coherence: 1.0 / 3.0,
                coherence_entropy: entropy,
                energy_ratio: 0.7,
                z_bias: -0.1,
                fractional_order: 0.4,
                normalized_weights: vec![0.6, 0.3, 0.1],
                preserved_channels: Some(3),
                discarded_channels: Some(0),
                dominant_channel: Some(0),
            },
            coherence: vec![0.6, 0.3, 0.1],
            contour: Some(ZSpaceCoherenceContourInput {
                coherence_strength: 0.46,
                prosody_index: 0.4,
                articulation_bias: 0.2,
                timbre_spread: Some(0.3),
            }),
            config: ZSpaceCoherenceProjectionConfig::default(),
            classification_policy: Default::default(),
        })
        .expect("valid coherence projection");
        let expected_concentration = projection.partial["coherence_concentration"];
        let metrics = projection
            .partial
            .into_iter()
            .map(|(name, value)| (name, ZSpaceMetricInput::Scalar(value)))
            .collect();

        let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![Some(ZSpacePartialInput {
                metrics,
                weight: 1.0,
                origin: Some("coherence".to_owned()),
                telemetry: None,
            })],
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("all Rust-owned coherence metrics must be registered");

        assert_eq!(fused.active_count, 1);
        assert_eq!(
            fused.metrics["coherence_concentration"],
            expected_concentration
        );
        assert_eq!(fused.metrics["coherence_timbre_spread"], 0.3);
    }

    #[test]
    fn every_strategy_applies_to_scalars_and_gradients() {
        let inputs = || {
            vec![
                Some(partial(
                    json!({"speed": -2.0, "gradient": [-2.0, 4.0]}),
                    1.0,
                    "a",
                    None,
                )),
                Some(partial(
                    json!({"speed": 3.0, "gradient": [3.0]}),
                    1.0,
                    "b",
                    None,
                )),
            ]
        };
        let expected = [
            (ZSpaceFusionStrategy::Mean, 0.5, vec![0.5, 2.0]),
            (ZSpaceFusionStrategy::Last, 3.0, vec![3.0, 0.0]),
            (ZSpaceFusionStrategy::Max, 3.0, vec![3.0, 4.0]),
            (ZSpaceFusionStrategy::Min, -2.0, vec![-2.0, 0.0]),
        ];

        for (strategy, scalar, gradient) in expected {
            let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
                partials: inputs(),
                strategy,
                ..ZSpacePartialFusionRequest::default()
            })
            .expect("valid fusion");
            assert_eq!(fused.metrics["speed"], scalar);
            assert_eq!(fused.gradient, Some(gradient));
        }
    }

    #[test]
    fn alias_collisions_and_bad_weight_lengths_fail_closed() {
        let collision = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![Some(partial(
                json!({"speed": 1.0, "velocity": 2.0}),
                1.0,
                "collision",
                None,
            ))],
            ..ZSpacePartialFusionRequest::default()
        });
        assert!(matches!(
            collision,
            Err(ZSpaceFusionError::AliasCollision { .. })
        ));

        let bad_weights = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![None],
            weights: Some(vec![]),
            ..ZSpacePartialFusionRequest::default()
        });
        assert_eq!(
            bad_weights,
            Err(ZSpaceFusionError::WeightsLength {
                actual: 0,
                expected: 1
            })
        );
    }

    #[test]
    fn non_finite_direct_inputs_and_size_guards_fail_closed() {
        let non_finite = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![Some(ZSpacePartialInput {
                metrics: BTreeMap::from([(
                    "speed".to_owned(),
                    ZSpaceMetricInput::Scalar(f64::NAN),
                )]),
                weight: 1.0,
                origin: None,
                telemetry: None,
            })],
            ..ZSpacePartialFusionRequest::default()
        });
        assert!(matches!(
            non_finite,
            Err(ZSpaceFusionError::InvalidScalarMetric { .. })
        ));

        let too_long = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![Some(ZSpacePartialInput {
                metrics: BTreeMap::from([(
                    "gradient".to_owned(),
                    ZSpaceMetricInput::Vector(vec![0.0; ZSPACE_FUSION_MAX_GRADIENT_DIM + 1]),
                )]),
                weight: 1.0,
                origin: None,
                telemetry: None,
            })],
            ..ZSpacePartialFusionRequest::default()
        });
        assert!(matches!(
            too_long,
            Err(ZSpaceFusionError::GradientTooLong { .. })
        ));

        let null_with_non_finite_weight = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![None],
            weights: Some(vec![f64::NAN]),
            ..ZSpacePartialFusionRequest::default()
        });
        assert_eq!(
            null_with_non_finite_weight,
            Err(ZSpaceFusionError::NonFiniteWeight { index: 0 })
        );

        let overflowing_summary = fuse_zspace_telemetry(&[json!({"huge": f64::MAX})]);
        assert!(matches!(
            overflowing_summary,
            Err(ZSpaceFusionError::NonFiniteTelemetrySummary { .. })
        ));
    }

    #[test]
    fn weighted_mean_stays_finite_for_extreme_finite_inputs() {
        let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![
                Some(partial(
                    json!({"speed": f64::MAX, "gradient": [f64::MAX]}),
                    f64::MAX,
                    "a",
                    None,
                )),
                Some(partial(
                    json!({"speed": f64::MAX, "gradient": [f64::MAX]}),
                    f64::MAX,
                    "b",
                    None,
                )),
            ],
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("finite convex combination");

        assert!(fused.metrics["speed"].is_finite());
        assert!(fused.gradient.expect("gradient")[0].is_finite());
    }
}
