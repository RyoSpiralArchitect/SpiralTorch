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
pub const ZSPACE_PARTIAL_FUSION_CONTRACT_VERSION: &str = "spiraltorch.zspace_partial_fusion.v3";
/// Stable payload kind for canonical Z-space partial fusion.
pub const ZSPACE_PARTIAL_FUSION_KIND: &str = "spiraltorch.zspace_partial_fusion";
/// Stable contract identifier for canonical metric-to-gradient projection.
pub const ZSPACE_METRIC_GRADIENT_PROJECTION_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_metric_gradient_projection.v1";
/// Stable payload kind for canonical metric-to-gradient projection.
pub const ZSPACE_METRIC_GRADIENT_PROJECTION_KIND: &str =
    "spiraltorch.zspace_metric_gradient_projection";
/// Basis identity for a gradient signal projected from the five base metrics.
pub const ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS: &str =
    "spiraltorch.zspace.canonical_metric_cycle.v1";
/// Exact projection rule shared by every client.
pub const ZSPACE_CANONICAL_METRIC_GRADIENT_FORMULA: &str =
    "g_i=m_(i mod 5),m=[speed,memory,stability,frac,drs]";
/// Ordered channels tiled by the canonical metric-gradient projection.
pub const ZSPACE_CANONICAL_METRIC_GRADIENT_CHANNELS: [&str; 5] =
    ["speed", "memory", "stability", "frac", "drs"];
/// Crate/module that owns Z-space fusion semantics.
pub const ZSPACE_FUSION_SEMANTIC_OWNER: &str = "st-core::telemetry::zspace_fusion";
/// Backend label attached to payloads produced by the canonical implementation.
pub const ZSPACE_FUSION_SEMANTIC_BACKEND: &str = "rust";

pub const ZSPACE_FUSION_MAX_PARTIALS: usize = 4_096;
pub const ZSPACE_FUSION_MAX_METRICS_PER_PARTIAL: usize = 4_096;
pub const ZSPACE_FUSION_MAX_GRADIENT_DIM: usize = 4_096;
pub const ZSPACE_FUSION_MAX_TELEMETRY_ENTRIES: usize = 16_384;
pub const ZSPACE_FUSION_MAX_TELEMETRY_DEPTH: usize = 64;
pub const ZSPACE_FUSION_MAX_METRIC_LABEL_BYTES: usize = 256;
pub const ZSPACE_FUSION_MAX_GRADIENT_BASIS_BYTES: usize = 256;
pub const ZSPACE_FUSION_MAX_ORIGIN_BYTES: usize = 4_096;
pub const ZSPACE_FUSION_MAX_TELEMETRY_PATH_BYTES: usize = 4_096;

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
    #[error("metric label at partial {index} has {actual} bytes, exceeding limit {max}")]
    MetricLabelTooLong {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error("origin at partial {index} has {actual} bytes, exceeding limit {max}")]
    OriginTooLong {
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
    #[error(
        "gradient dimension {actual} at partial {index} does not match dimension {expected} at partial {expected_index}; set gradient_alignment to 'pad_zero' to opt into zero padding"
    )]
    GradientDimensionMismatch {
        index: usize,
        actual: usize,
        expected_index: usize,
        expected: usize,
    },
    #[error("gradient entry {entry} at partial {index} must be finite")]
    NonFiniteGradient { index: usize, entry: usize },
    #[error("gradient at partial {index} must contain at least one entry")]
    EmptyGradient { index: usize },
    #[error("gradient basis at partial {index} requires a gradient vector")]
    GradientBasisWithoutGradient { index: usize },
    #[error("gradient basis at partial {index} must be a non-empty trimmed label")]
    InvalidGradientBasis { index: usize },
    #[error("gradient basis at partial {index} has {actual} bytes, exceeding limit {max}")]
    GradientBasisTooLong {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error(
        "gradient basis '{actual}' at partial {index} does not match basis '{expected}' at partial {expected_index}"
    )]
    GradientBasisMismatch {
        index: usize,
        actual: String,
        expected_index: usize,
        expected: String,
    },
    #[error("metric-gradient projection dimension must be in 1..={max}, received {actual}")]
    InvalidMetricGradientDimension { actual: usize, max: usize },
    #[error("metric-gradient projection is missing canonical metric '{metric}'")]
    MissingMetricGradientMetric { metric: &'static str },
    #[error("metric-gradient projection metric '{metric}' must be finite")]
    InvalidMetricGradientMetric { metric: String },
    #[error("metric-gradient projection does not support metric '{metric}'")]
    UnsupportedMetricGradientMetric { metric: String },
    #[error(
        "metric-gradient projection metrics '{first}' and '{second}' both resolve to '{canonical}'"
    )]
    MetricGradientAliasCollision {
        first: String,
        second: String,
        canonical: String,
    },
    #[error("telemetry payload {index} must be an object")]
    TelemetryNotObject { index: usize },
    #[error("telemetry payload {index} exceeds nesting limit {max_depth}")]
    TelemetryTooDeep { index: usize, max_depth: usize },
    #[error("telemetry path in payload {index} has {actual} bytes, exceeding limit {max}")]
    TelemetryPathTooLong {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error("flattened telemetry exceeds entry limit {max}")]
    TooManyTelemetryEntries { max: usize },
    #[error("telemetry value '{key}' in payload {index} must be finite")]
    NonFiniteTelemetry { index: usize, key: String },
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ZSpaceFusionStrategy {
    /// Numerically stable mean weighted by each active partial.
    #[default]
    Mean,
    /// Value from the last active partial that defines the metric.
    Last,
    /// Largest value across active partials; weights only gate participation.
    Max,
    /// Smallest value across active partials; weights only gate participation.
    Min,
    /// Weighted median, interpolating a balanced boundary between neighbours.
    Median,
    /// Compensated weighted sum across active partials.
    Sum,
}

impl ZSpaceFusionStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Last => "last",
            Self::Max => "max",
            Self::Min => "min",
            Self::Median => "median",
            Self::Sum => "sum",
        }
    }
}

/// Policy for reconciling gradient dimensions across active partials.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ZSpaceGradientAlignment {
    /// Reject mismatched dimensions rather than inventing missing coordinates.
    #[default]
    Strict,
    /// Preserve the legacy behavior by padding shorter gradients with zeroes.
    PadZero,
}

impl ZSpaceGradientAlignment {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::PadZero => "pad_zero",
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
    pub gradient_basis: Option<String>,
    #[serde(default)]
    pub telemetry: Option<Value>,
}

/// Request for projecting canonical Z-space metrics into an optimizer gradient basis.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceMetricGradientProjectionRequest {
    pub metrics: BTreeMap<String, f64>,
    pub dimension: usize,
}

/// Rust-owned metric-to-gradient projection shared by native, Python, and WASM clients.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceMetricGradientProjectionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub basis: &'static str,
    pub formula: &'static str,
    pub dimension: usize,
    pub base_dimension: usize,
    pub source_metrics: BTreeMap<String, f64>,
    pub coordinate_channels: Vec<&'static str>,
    pub gradient: Vec<f64>,
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
    pub gradient_alignment: ZSpaceGradientAlignment,
    /// Replace positional input gradients with one projection of the fused base metrics.
    #[serde(default)]
    pub metric_gradient_dimension: Option<usize>,
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
    pub gradient_present: bool,
    pub gradient_dim: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_basis: Option<String>,
    pub gradient_replaced_by_metric_projection: bool,
    pub gradient_padded: bool,
    pub telemetry_entry_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpacePartialFusionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub strategy: &'static str,
    pub gradient_alignment: &'static str,
    pub gradient_source: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_formula: Option<&'static str>,
    pub metrics: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_basis: Option<String>,
    pub telemetry: ZSpaceTelemetryFusionPayload,
    pub input_count: usize,
    pub active_count: usize,
    pub suppressed_count: usize,
    pub null_count: usize,
    pub gradient_input_count: usize,
    pub gradient_dim: usize,
    pub metric_gradient_projection_applied: bool,
    pub gradient_replaced_source_count: usize,
    pub gradient_padding_applied: bool,
    pub gradient_padded_source_count: usize,
    pub sources: Vec<ZSpacePartialSourceAudit>,
}

#[derive(Default)]
struct FlattenAudit {
    ignored: usize,
    conflicts: usize,
}

fn checked_telemetry_path(
    index: usize,
    prefix: &str,
    key: &str,
) -> Result<String, ZSpaceFusionError> {
    let actual = prefix
        .len()
        .saturating_add(usize::from(!prefix.is_empty()))
        .saturating_add(key.len());
    if actual > ZSPACE_FUSION_MAX_TELEMETRY_PATH_BYTES {
        return Err(ZSpaceFusionError::TelemetryPathTooLong {
            index,
            actual,
            max: ZSPACE_FUSION_MAX_TELEMETRY_PATH_BYTES,
        });
    }
    Ok(if prefix.is_empty() {
        key.to_owned()
    } else {
        format!("{prefix}.{key}")
    })
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
        let label = checked_telemetry_path(index, prefix, key)?;
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

/// Project the five canonical observation metrics into a dimensioned gradient signal.
///
/// This intentionally centralizes the periodic projection already consumed by
/// the Z-space optimizer. Clients provide named observations and never assign
/// positional meanings to gradient coordinates themselves.
pub fn project_zspace_metric_gradient(
    request: ZSpaceMetricGradientProjectionRequest,
) -> Result<ZSpaceMetricGradientProjectionPayload, ZSpaceFusionError> {
    if !(1..=ZSPACE_FUSION_MAX_GRADIENT_DIM).contains(&request.dimension) {
        return Err(ZSpaceFusionError::InvalidMetricGradientDimension {
            actual: request.dimension,
            max: ZSPACE_FUSION_MAX_GRADIENT_DIM,
        });
    }

    let mut canonical = BTreeMap::new();
    let mut original_names = BTreeMap::new();
    for (name, value) in request.metrics {
        let Some(canonical_name) = canonical_metric_name(&name) else {
            return Err(ZSpaceFusionError::UnsupportedMetricGradientMetric { metric: name });
        };
        if !ZSPACE_CANONICAL_METRIC_GRADIENT_CHANNELS.contains(&canonical_name) {
            return Err(ZSpaceFusionError::UnsupportedMetricGradientMetric { metric: name });
        }
        if !value.is_finite() {
            return Err(ZSpaceFusionError::InvalidMetricGradientMetric { metric: name });
        }
        if let Some(first) = original_names.insert(canonical_name, name.clone()) {
            return Err(ZSpaceFusionError::MetricGradientAliasCollision {
                first,
                second: name,
                canonical: canonical_name.to_owned(),
            });
        }
        canonical.insert(canonical_name.to_owned(), value);
    }

    let base = ZSPACE_CANONICAL_METRIC_GRADIENT_CHANNELS
        .map(|metric| {
            canonical
                .get(metric)
                .copied()
                .ok_or(ZSpaceFusionError::MissingMetricGradientMetric { metric })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let coordinate_channels = (0..request.dimension)
        .map(|index| ZSPACE_CANONICAL_METRIC_GRADIENT_CHANNELS[index % base.len()])
        .collect::<Vec<_>>();
    let gradient = (0..request.dimension)
        .map(|index| base[index % base.len()])
        .collect();

    Ok(ZSpaceMetricGradientProjectionPayload {
        kind: ZSPACE_METRIC_GRADIENT_PROJECTION_KIND,
        contract_version: ZSPACE_METRIC_GRADIENT_PROJECTION_CONTRACT_VERSION,
        semantic_owner: ZSPACE_FUSION_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_FUSION_SEMANTIC_BACKEND,
        basis: ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS,
        formula: ZSPACE_CANONICAL_METRIC_GRADIENT_FORMULA,
        dimension: request.dimension,
        base_dimension: ZSPACE_CANONICAL_METRIC_GRADIENT_CHANNELS.len(),
        source_metrics: canonical,
        coordinate_channels,
        gradient,
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
        if scaled_weight == 0.0 {
            continue;
        }
        let next_weight = total_weight + scaled_weight;
        mean = mean * (total_weight / next_weight) + value * (scaled_weight / next_weight);
        total_weight = next_weight;
    }
    mean
}

fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let next = sum + value;
        compensation += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + compensation
}

fn stable_midpoint(lower: f64, upper: f64) -> f64 {
    if (lower < 0.0) == (upper < 0.0) {
        lower + (upper - lower) * 0.5
    } else {
        lower * 0.5 + upper * 0.5
    }
}

fn stable_weighted_sum(values: impl Iterator<Item = (f64, f64)>, max_weight: f64) -> f64 {
    debug_assert!(max_weight.is_finite() && max_weight > 0.0);
    let scaled_sum = compensated_sum(values.map(|(value, weight)| value * (weight / max_weight)));
    scaled_sum * max_weight
}

fn weighted_median(mut values: Vec<(f64, f64)>) -> f64 {
    debug_assert!(!values.is_empty());
    values.sort_by(|(left, _), (right, _)| left.total_cmp(right));
    let max_weight = values.iter().map(|(_, weight)| *weight).fold(0.0, f64::max);
    let midpoint_weight =
        compensated_sum(values.iter().map(|(_, weight)| weight / max_weight)) * 0.5;
    let mut cumulative_weight = 0.0;
    for (index, (value, weight)) in values.iter().enumerate() {
        cumulative_weight += weight / max_weight;
        if cumulative_weight < midpoint_weight {
            continue;
        }
        if cumulative_weight == midpoint_weight {
            if let Some((upper, _)) = values.get(index + 1) {
                return stable_midpoint(*value, *upper);
            }
        }
        return *value;
    }
    values.last().map(|(value, _)| *value).unwrap_or(0.0)
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
        ZSpaceFusionStrategy::Median => weighted_median(values.to_vec()),
        ZSpaceFusionStrategy::Sum => {
            let max_weight = values.iter().map(|(_, weight)| *weight).fold(0.0, f64::max);
            stable_weighted_sum(values.iter().copied(), max_weight)
        }
    }
}

#[derive(Default)]
struct GradientReduction {
    gradient: Option<Vec<f64>>,
    basis: Option<String>,
    input_count: usize,
    dimension: usize,
    padded_source_indices: Vec<usize>,
}

fn gradient_basis_label(basis: Option<&str>) -> String {
    basis.unwrap_or("<unspecified>").to_owned()
}

fn reduce_gradients(
    gradients: &[(usize, Vec<f64>, f64, Option<String>)],
    strategy: ZSpaceFusionStrategy,
    alignment: ZSpaceGradientAlignment,
) -> Result<GradientReduction, ZSpaceFusionError> {
    let Some((expected_index, expected_gradient, _, expected_basis)) = gradients.first() else {
        return Ok(GradientReduction::default());
    };
    if let Some((index, _, _, actual_basis)) = gradients
        .iter()
        .skip(1)
        .find(|(_, _, _, basis)| basis != expected_basis)
    {
        return Err(ZSpaceFusionError::GradientBasisMismatch {
            index: *index,
            actual: gradient_basis_label(actual_basis.as_deref()),
            expected_index: *expected_index,
            expected: gradient_basis_label(expected_basis.as_deref()),
        });
    }
    if alignment == ZSpaceGradientAlignment::Strict {
        if let Some((index, gradient, _, _)) = gradients
            .iter()
            .skip(1)
            .find(|(_, gradient, _, _)| gradient.len() != expected_gradient.len())
        {
            return Err(ZSpaceFusionError::GradientDimensionMismatch {
                index: *index,
                actual: gradient.len(),
                expected_index: *expected_index,
                expected: expected_gradient.len(),
            });
        }
    }

    let length = match alignment {
        ZSpaceGradientAlignment::Strict => expected_gradient.len(),
        ZSpaceGradientAlignment::PadZero => gradients
            .iter()
            .map(|(_, gradient, _, _)| gradient.len())
            .max()
            .unwrap_or(0),
    };
    let padded_source_indices = if alignment == ZSpaceGradientAlignment::PadZero {
        gradients
            .iter()
            .filter_map(|(index, gradient, _, _)| (gradient.len() < length).then_some(*index))
            .collect()
    } else {
        Vec::new()
    };
    let mut reduction = GradientReduction {
        gradient: None,
        basis: expected_basis.clone(),
        input_count: gradients.len(),
        dimension: length,
        padded_source_indices,
    };
    if length == 0 {
        return Ok(reduction);
    }

    let mut result = vec![0.0; length];
    match strategy {
        ZSpaceFusionStrategy::Mean => {
            let max_weight = gradients
                .iter()
                .map(|(_, _, weight, _)| *weight)
                .fold(0.0, f64::max);
            let mut total_weight = 0.0;
            for (_, gradient, weight, _) in gradients {
                let scaled_weight = weight / max_weight;
                if scaled_weight == 0.0 {
                    continue;
                }
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
            let (_, gradient, _, _) = gradients.last().expect("non-empty gradients");
            result[..gradient.len()].copy_from_slice(gradient);
        }
        ZSpaceFusionStrategy::Max | ZSpaceFusionStrategy::Min => {
            for (index, output) in result.iter_mut().enumerate() {
                *output = gradients
                    .iter()
                    .map(|(_, gradient, _, _)| gradient.get(index).copied().unwrap_or(0.0))
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
        ZSpaceFusionStrategy::Median => {
            for (index, output) in result.iter_mut().enumerate() {
                *output = weighted_median(
                    gradients
                        .iter()
                        .map(|(_, gradient, weight, _)| {
                            (gradient.get(index).copied().unwrap_or(0.0), *weight)
                        })
                        .collect(),
                );
            }
        }
        ZSpaceFusionStrategy::Sum => {
            let max_weight = gradients
                .iter()
                .map(|(_, _, weight, _)| *weight)
                .fold(0.0, f64::max);
            for (index, output) in result.iter_mut().enumerate() {
                *output = stable_weighted_sum(
                    gradients.iter().map(|(_, gradient, weight, _)| {
                        (gradient.get(index).copied().unwrap_or(0.0), *weight)
                    }),
                    max_weight,
                );
            }
        }
    }
    reduction.gradient = Some(result);
    Ok(reduction)
}

/// Fuse Z-space metric partials and attached telemetry under one Rust-owned contract.
pub fn fuse_zspace_partials(
    request: ZSpacePartialFusionRequest,
) -> Result<ZSpacePartialFusionPayload, ZSpaceFusionError> {
    let input_count = request.partials.len();
    let metric_gradient_dimension = request.metric_gradient_dimension;
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
    if let Some(dimension) = metric_gradient_dimension {
        if !(1..=ZSPACE_FUSION_MAX_GRADIENT_DIM).contains(&dimension) {
            return Err(ZSpaceFusionError::InvalidMetricGradientDimension {
                actual: dimension,
                max: ZSPACE_FUSION_MAX_GRADIENT_DIM,
            });
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
                gradient_present: false,
                gradient_dim: 0,
                gradient_basis: None,
                gradient_replaced_by_metric_projection: false,
                gradient_padded: false,
                telemetry_entry_count: 0,
            });
            continue;
        };
        if let Some(origin) = partial.origin.as_deref() {
            if origin.len() > ZSPACE_FUSION_MAX_ORIGIN_BYTES {
                return Err(ZSpaceFusionError::OriginTooLong {
                    index,
                    actual: origin.len(),
                    max: ZSPACE_FUSION_MAX_ORIGIN_BYTES,
                });
            }
        }
        let weight = request
            .weights
            .as_ref()
            .map(|weights| weights[index])
            .unwrap_or(partial.weight);
        if !weight.is_finite() {
            return Err(ZSpaceFusionError::NonFiniteWeight { index });
        }
        if partial.metrics.len() > ZSPACE_FUSION_MAX_METRICS_PER_PARTIAL {
            return Err(ZSpaceFusionError::TooManyMetrics {
                index,
                actual: partial.metrics.len(),
                max: ZSPACE_FUSION_MAX_METRICS_PER_PARTIAL,
            });
        }
        if let Some(key) = partial
            .metrics
            .keys()
            .find(|key| key.len() > ZSPACE_FUSION_MAX_METRIC_LABEL_BYTES)
        {
            return Err(ZSpaceFusionError::MetricLabelTooLong {
                index,
                actual: key.len(),
                max: ZSPACE_FUSION_MAX_METRIC_LABEL_BYTES,
            });
        }
        if weight <= 0.0 {
            suppressed_count += 1;
            sources.push(ZSpacePartialSourceAudit {
                index,
                origin: partial.origin,
                weight: Some(weight),
                status: "suppressed",
                metric_count: 0,
                gradient_present: false,
                gradient_dim: 0,
                gradient_basis: None,
                gradient_replaced_by_metric_projection: false,
                gradient_padded: false,
                telemetry_entry_count: 0,
            });
            continue;
        }

        let gradient_basis = partial.gradient_basis;
        if let Some(basis) = gradient_basis.as_deref() {
            if basis.is_empty() || basis.trim() != basis {
                return Err(ZSpaceFusionError::InvalidGradientBasis { index });
            }
            if basis.len() > ZSPACE_FUSION_MAX_GRADIENT_BASIS_BYTES {
                return Err(ZSpaceFusionError::GradientBasisTooLong {
                    index,
                    actual: basis.len(),
                    max: ZSPACE_FUSION_MAX_GRADIENT_BASIS_BYTES,
                });
            }
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
        let mut gradient_present = false;
        let mut gradient_dim = 0;
        for (canonical, (_, value)) in canonical_metrics {
            if canonical == "gradient" {
                let ZSpaceMetricInput::Vector(gradient) = value else {
                    return Err(ZSpaceFusionError::InvalidVectorMetric {
                        index,
                        metric: canonical.to_owned(),
                    });
                };
                if gradient.is_empty() {
                    return Err(ZSpaceFusionError::EmptyGradient { index });
                }
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
                gradient_present = true;
                gradient_dim = gradient.len();
                gradients.push((index, gradient, weight, gradient_basis.clone()));
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
        if gradient_basis.is_some() && !gradient_present {
            return Err(ZSpaceFusionError::GradientBasisWithoutGradient { index });
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
            gradient_present,
            gradient_dim,
            gradient_basis,
            gradient_replaced_by_metric_projection: metric_gradient_dimension.is_some()
                && gradient_present,
            gradient_padded: false,
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
    let gradient_reduction = if let Some(dimension) = metric_gradient_dimension {
        let projection_metrics = ZSPACE_CANONICAL_METRIC_GRADIENT_CHANNELS
            .iter()
            .filter_map(|metric| {
                metrics
                    .get(*metric)
                    .map(|value| ((*metric).to_owned(), *value))
            })
            .collect();
        let projection = project_zspace_metric_gradient(ZSpaceMetricGradientProjectionRequest {
            metrics: projection_metrics,
            dimension,
        })?;
        GradientReduction {
            gradient: Some(projection.gradient),
            basis: Some(projection.basis.to_owned()),
            input_count: gradients.len(),
            dimension: projection.dimension,
            padded_source_indices: Vec::new(),
        }
    } else {
        reduce_gradients(&gradients, request.strategy, request.gradient_alignment)?
    };
    for source in &mut sources {
        source.gradient_padded = gradient_reduction
            .padded_source_indices
            .binary_search(&source.index)
            .is_ok();
    }
    let gradient_input_count = gradient_reduction.input_count;
    let gradient_dim = gradient_reduction.dimension;
    let gradient_padded_source_count = gradient_reduction.padded_source_indices.len();
    let gradient_replaced_source_count = sources
        .iter()
        .filter(|source| source.gradient_replaced_by_metric_projection)
        .count();
    let gradient_basis = gradient_reduction.basis;
    let gradient = gradient_reduction.gradient;
    let gradient_source = if metric_gradient_dimension.is_some() {
        "canonical_metrics"
    } else if gradient.is_some() {
        "explicit"
    } else {
        "none"
    };
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
        gradient_alignment: request.gradient_alignment.as_str(),
        gradient_source,
        gradient_formula: metric_gradient_dimension
            .map(|_| ZSPACE_CANONICAL_METRIC_GRADIENT_FORMULA),
        metrics,
        gradient,
        gradient_basis,
        telemetry,
        input_count,
        active_count,
        suppressed_count,
        null_count,
        gradient_input_count,
        gradient_dim,
        metric_gradient_projection_applied: metric_gradient_dimension.is_some(),
        gradient_replaced_source_count,
        gradient_padding_applied: gradient_padded_source_count > 0,
        gradient_padded_source_count,
        sources,
    })
}

/// Resolve a public Z-space metric alias to its canonical spelling.
pub fn canonical_metric_name(name: &str) -> Option<&'static str> {
    if name.len() > ZSPACE_FUSION_MAX_METRIC_LABEL_BYTES {
        return None;
    }
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
            gradient_basis: None,
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
    fn metric_gradient_projection_owns_dimension_and_coordinate_basis() {
        let projection = project_zspace_metric_gradient(ZSpaceMetricGradientProjectionRequest {
            metrics: BTreeMap::from([
                ("velocity".to_owned(), 1.0),
                ("mem".to_owned(), 2.0),
                ("stab".to_owned(), 3.0),
                ("frac_reg".to_owned(), 4.0),
                ("drift".to_owned(), 5.0),
            ]),
            dimension: 7,
        })
        .expect("canonical metric projection");

        assert_eq!(
            projection.contract_version,
            ZSPACE_METRIC_GRADIENT_PROJECTION_CONTRACT_VERSION
        );
        assert_eq!(projection.basis, ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS);
        assert_eq!(projection.base_dimension, 5);
        assert_eq!(
            projection.coordinate_channels,
            [
                "speed",
                "memory",
                "stability",
                "frac",
                "drs",
                "speed",
                "memory"
            ]
        );
        assert_eq!(projection.gradient, [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]);
        assert_eq!(projection.source_metrics["speed"], 1.0);
    }

    #[test]
    fn metric_gradient_projection_fails_closed_on_incomplete_or_ambiguous_inputs() {
        let missing = project_zspace_metric_gradient(ZSpaceMetricGradientProjectionRequest {
            metrics: BTreeMap::from([("speed".to_owned(), 1.0)]),
            dimension: 4,
        });
        assert_eq!(
            missing,
            Err(ZSpaceFusionError::MissingMetricGradientMetric { metric: "memory" })
        );

        let collision = project_zspace_metric_gradient(ZSpaceMetricGradientProjectionRequest {
            metrics: BTreeMap::from([
                ("speed".to_owned(), 1.0),
                ("velocity".to_owned(), 2.0),
                ("memory".to_owned(), 3.0),
                ("stability".to_owned(), 4.0),
                ("frac".to_owned(), 5.0),
                ("drs".to_owned(), 6.0),
            ]),
            dimension: 4,
        });
        assert!(matches!(
            collision,
            Err(ZSpaceFusionError::MetricGradientAliasCollision { .. })
        ));
    }

    #[test]
    fn gradient_basis_identity_is_enforced_before_positional_reduction() {
        let tagged = |basis: Option<&str>, gradient: Vec<f64>| {
            let mut input = partial(
                json!({"speed": 1.0, "gradient": gradient}),
                1.0,
                "basis-test",
                None,
            );
            input.gradient_basis = basis.map(str::to_owned);
            Some(input)
        };

        let mismatch = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![
                tagged(Some("basis.a.v1"), vec![1.0, 2.0]),
                tagged(Some("basis.b.v1"), vec![3.0, 4.0]),
            ],
            ..ZSpacePartialFusionRequest::default()
        });
        assert!(matches!(
            mismatch,
            Err(ZSpaceFusionError::GradientBasisMismatch { index: 1, .. })
        ));

        let mixed = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![
                tagged(Some("basis.a.v1"), vec![1.0]),
                tagged(None, vec![2.0]),
            ],
            ..ZSpacePartialFusionRequest::default()
        });
        assert!(matches!(
            mixed,
            Err(ZSpaceFusionError::GradientBasisMismatch { index: 1, .. })
        ));

        let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![
                tagged(Some("basis.a.v1"), vec![1.0, 2.0]),
                tagged(Some("basis.a.v1"), vec![3.0, 4.0]),
            ],
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("matching bases fuse");
        assert_eq!(fused.gradient, Some(vec![2.0, 3.0]));
        assert_eq!(fused.gradient_basis.as_deref(), Some("basis.a.v1"));
        assert_eq!(
            fused.sources[0].gradient_basis.as_deref(),
            Some("basis.a.v1")
        );
    }

    #[test]
    fn metric_projection_replaces_heterogeneous_input_gradients_after_scalar_fusion() {
        let mut first = partial(
            json!({
                "speed": 1.0,
                "memory": 2.0,
                "stability": 3.0,
                "frac": 4.0,
                "drs": 5.0,
                "gradient": [99.0, 98.0]
            }),
            1.0,
            "first",
            None,
        );
        first.gradient_basis = Some("feature.first.v1".to_owned());
        let mut second = partial(
            json!({
                "speed": 3.0,
                "memory": 4.0,
                "stability": 5.0,
                "frac": 6.0,
                "drs": 7.0,
                "gradient": [-9.0, -8.0, -7.0]
            }),
            1.0,
            "second",
            None,
        );
        second.gradient_basis = Some("feature.second.v1".to_owned());

        let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![Some(first), Some(second)],
            metric_gradient_dimension: Some(4),
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("named metrics replace heterogeneous positional gradients");

        assert_eq!(fused.gradient_source, "canonical_metrics");
        assert_eq!(
            fused.gradient_basis.as_deref(),
            Some(ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS)
        );
        assert_eq!(fused.gradient, Some(vec![2.0, 3.0, 4.0, 5.0]));
        assert_eq!(fused.gradient_input_count, 2);
        assert_eq!(fused.gradient_replaced_source_count, 2);
        assert!(fused.metric_gradient_projection_applied);
        assert!(fused
            .sources
            .iter()
            .all(|source| source.gradient_replaced_by_metric_projection));
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
            gradient_alignment: ZSpaceGradientAlignment::Strict,
            metric_gradient_dimension: None,
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
                gradient_basis: None,
                telemetry: None,
            })],
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("all Rust-owned coherence metrics must be registered");

        assert_eq!(fused.active_count, 1);
        assert_eq!(fused.gradient_source, "none");
        assert_eq!(fused.gradient, None);
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
                    json!({"speed": 0.0, "gradient": [0.0, 6.0]}),
                    1.0,
                    "a",
                    None,
                )),
                Some(partial(
                    json!({"speed": 2.0, "gradient": [2.0, 4.0]}),
                    1.0,
                    "b",
                    None,
                )),
                Some(partial(
                    json!({"speed": 10.0, "gradient": [10.0, 8.0]}),
                    1.0,
                    "c",
                    None,
                )),
            ]
        };
        let expected = [
            (ZSpaceFusionStrategy::Mean, 4.0, vec![4.0, 6.0]),
            (ZSpaceFusionStrategy::Last, 10.0, vec![10.0, 8.0]),
            (ZSpaceFusionStrategy::Max, 10.0, vec![10.0, 8.0]),
            (ZSpaceFusionStrategy::Min, 0.0, vec![0.0, 4.0]),
            (ZSpaceFusionStrategy::Median, 2.0, vec![2.0, 6.0]),
            (ZSpaceFusionStrategy::Sum, 12.0, vec![12.0, 18.0]),
        ];

        for (strategy, scalar, gradient) in expected {
            let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
                partials: inputs(),
                strategy,
                ..ZSpacePartialFusionRequest::default()
            })
            .expect("valid fusion");
            assert!((fused.metrics["speed"] - scalar).abs() < 1e-12);
            let fused_gradient = fused.gradient.expect("gradient");
            assert_eq!(fused_gradient.len(), gradient.len());
            assert!(fused_gradient
                .iter()
                .zip(&gradient)
                .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
            assert_eq!(fused.gradient_alignment, "strict");
            assert_eq!(fused.gradient_input_count, 3);
            assert_eq!(fused.gradient_dim, 2);
            assert!(!fused.gradient_padding_applied);
        }
    }

    #[test]
    fn sum_and_median_preserve_representable_low_order_signals() {
        let summed = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![
                Some(partial(
                    json!({"speed": 1e16, "gradient": [1e16]}),
                    1.0,
                    "large-positive",
                    None,
                )),
                Some(partial(
                    json!({"speed": 1.0, "gradient": [1.0]}),
                    1.0,
                    "low-order",
                    None,
                )),
                Some(partial(
                    json!({"speed": -1e16, "gradient": [-1e16]}),
                    1.0,
                    "large-negative",
                    None,
                )),
            ],
            strategy: ZSpaceFusionStrategy::Sum,
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("compensated sum remains finite");
        assert_eq!(summed.metrics["speed"], 1.0);
        assert_eq!(summed.gradient, Some(vec![1.0]));

        let smallest = f64::from_bits(1);
        let midpoint = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![
                Some(partial(
                    json!({"speed": smallest, "gradient": [smallest]}),
                    1.0,
                    "a",
                    None,
                )),
                Some(partial(
                    json!({"speed": smallest, "gradient": [smallest]}),
                    1.0,
                    "b",
                    None,
                )),
            ],
            strategy: ZSpaceFusionStrategy::Median,
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("median midpoint preserves identical subnormal values");
        assert_eq!(midpoint.metrics["speed"], smallest);
        assert_eq!(midpoint.gradient, Some(vec![smallest]));
    }

    #[test]
    fn sum_and_median_honor_positive_partial_weights() {
        let inputs = || {
            vec![
                Some(partial(
                    json!({"speed": 0.0, "gradient": [0.0]}),
                    1.0,
                    "light",
                    None,
                )),
                Some(partial(
                    json!({"speed": 10.0, "gradient": [10.0]}),
                    3.0,
                    "heavy",
                    None,
                )),
            ]
        };
        let summed = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: inputs(),
            strategy: ZSpaceFusionStrategy::Sum,
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("weighted sum");
        assert_eq!(summed.metrics["speed"], 30.0);
        assert_eq!(summed.gradient, Some(vec![30.0]));

        let median = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: inputs(),
            strategy: ZSpaceFusionStrategy::Median,
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("weighted median");
        assert_eq!(median.metrics["speed"], 10.0);
        assert_eq!(median.gradient, Some(vec![10.0]));
    }

    #[test]
    fn strict_gradient_alignment_rejects_ragged_inputs() {
        let error = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![
                Some(partial(
                    json!({"gradient": [1.0, 2.0]}),
                    1.0,
                    "reference",
                    None,
                )),
                Some(partial(json!({"gradient": [3.0]}), 1.0, "ragged", None)),
            ],
            ..ZSpacePartialFusionRequest::default()
        })
        .expect_err("strict alignment must reject a ragged gradient");

        assert_eq!(
            error,
            ZSpaceFusionError::GradientDimensionMismatch {
                index: 1,
                actual: 1,
                expected_index: 0,
                expected: 2,
            }
        );
    }

    #[test]
    fn pad_zero_preserves_legacy_reduction_and_audits_padding() {
        let inputs = || {
            vec![
                Some(partial(
                    json!({"speed": -2.0, "gradient": [-2.0, 4.0]}),
                    1.0,
                    "full",
                    None,
                )),
                Some(partial(
                    json!({"speed": 3.0, "gradient": [3.0]}),
                    1.0,
                    "short",
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
                gradient_alignment: ZSpaceGradientAlignment::PadZero,
                ..ZSpacePartialFusionRequest::default()
            })
            .expect("explicit compatibility mode permits padding");
            assert_eq!(fused.metrics["speed"], scalar);
            assert_eq!(fused.gradient, Some(gradient));
            assert_eq!(fused.gradient_alignment, "pad_zero");
            assert_eq!(fused.gradient_input_count, 2);
            assert_eq!(fused.gradient_dim, 2);
            assert!(fused.gradient_padding_applied);
            assert_eq!(fused.gradient_padded_source_count, 1);
            assert!(!fused.sources[0].gradient_padded);
            assert!(fused.sources[1].gradient_padded);
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
                gradient_basis: None,
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
                gradient_basis: None,
                telemetry: None,
            })],
            ..ZSpacePartialFusionRequest::default()
        });
        assert!(matches!(
            too_long,
            Err(ZSpaceFusionError::GradientTooLong { .. })
        ));

        let empty = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![Some(partial(
                json!({"gradient": []}),
                1.0,
                "empty-gradient",
                None,
            ))],
            ..ZSpacePartialFusionRequest::default()
        });
        assert_eq!(empty, Err(ZSpaceFusionError::EmptyGradient { index: 0 }));

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

    #[test]
    fn weighted_mean_skips_relative_weights_that_underflow_to_zero() {
        let fused = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![
                Some(partial(
                    json!({"speed": -3.0, "gradient": [-3.0, 9.0]}),
                    f64::MIN_POSITIVE,
                    "tiny",
                    None,
                )),
                Some(partial(
                    json!({"speed": 7.0, "gradient": [7.0, -5.0]}),
                    f64::MAX,
                    "dominant",
                    None,
                )),
            ],
            ..ZSpacePartialFusionRequest::default()
        })
        .expect("underflowed relative weights are negligible, not invalid");

        assert_eq!(fused.metrics["speed"], 7.0);
        assert_eq!(fused.gradient, Some(vec![7.0, -5.0]));
    }

    #[test]
    fn oversized_labels_and_origins_fail_before_normalization() {
        let metric_label = "x".repeat(ZSPACE_FUSION_MAX_METRIC_LABEL_BYTES + 1);
        let metric_error = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![Some(ZSpacePartialInput {
                metrics: BTreeMap::from([(metric_label, ZSpaceMetricInput::Scalar(1.0))]),
                weight: 1.0,
                origin: None,
                gradient_basis: None,
                telemetry: None,
            })],
            ..ZSpacePartialFusionRequest::default()
        })
        .expect_err("oversized metric labels must fail closed");
        assert!(matches!(
            metric_error,
            ZSpaceFusionError::MetricLabelTooLong { index: 0, .. }
        ));

        let suppressed_metric_error = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![Some(ZSpacePartialInput {
                metrics: BTreeMap::from([(
                    "x".repeat(ZSPACE_FUSION_MAX_METRIC_LABEL_BYTES + 1),
                    ZSpaceMetricInput::Scalar(1.0),
                )]),
                weight: 1.0,
                origin: None,
                gradient_basis: None,
                telemetry: None,
            })],
            weights: Some(vec![0.0]),
            ..ZSpacePartialFusionRequest::default()
        })
        .expect_err("suppression must not bypass metric label bounds");
        assert!(matches!(
            suppressed_metric_error,
            ZSpaceFusionError::MetricLabelTooLong { index: 0, .. }
        ));

        let origin_error = fuse_zspace_partials(ZSpacePartialFusionRequest {
            partials: vec![Some(partial(
                json!({"speed": 1.0}),
                1.0,
                &"o".repeat(ZSPACE_FUSION_MAX_ORIGIN_BYTES + 1),
                None,
            ))],
            ..ZSpacePartialFusionRequest::default()
        })
        .expect_err("oversized origins must fail closed");
        assert!(matches!(
            origin_error,
            ZSpaceFusionError::OriginTooLong { index: 0, .. }
        ));

        let telemetry_key = "t".repeat(ZSPACE_FUSION_MAX_TELEMETRY_PATH_BYTES + 1);
        let telemetry_error = fuse_zspace_telemetry(&[Value::Object(
            [(telemetry_key, json!(1.0))].into_iter().collect(),
        )])
        .expect_err("oversized telemetry paths must fail closed");
        assert!(matches!(
            telemetry_error,
            ZSpaceFusionError::TelemetryPathTooLong { index: 0, .. }
        ));
    }
}
