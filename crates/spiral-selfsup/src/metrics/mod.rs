//! Telemetry descriptors used by self-supervised objectives.
//!
//! The original implementation depended on the external `st-metrics` crate to
//! provide a registry of descriptors and gauge values.  That crate is no longer
//! part of the SpiralTorch workspace, so we keep a lightweight in-crate
//! replacement that offers the same surface-level functionality required by the
//! examples and tutorials: registering well-known descriptors and producing
//! gauge values for InfoNCE epochs.

use crate::contrastive::InfoNCEResult;
use once_cell::sync::Lazy;
use std::sync::RwLock;

/// Units associated with a metric descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricUnit {
    /// Dimensionless scalar value (losses, ratios, etc.).
    Scalar,
    /// Raw count of occurrences, batches, or steps.
    Count,
}

/// Descriptor describing a metric that can be emitted by self-supervised code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricDescriptor {
    /// Canonical metric name registered with the telemetry layer.
    pub name: &'static str,
    /// Unit associated with the metric value.
    pub unit: MetricUnit,
    /// Human readable description for dashboards and registries.
    pub description: &'static str,
}

/// Gauge value paired with a descriptor name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricValue {
    /// Name matching a registered descriptor.
    pub name: &'static str,
    /// Recorded value.
    pub value: f32,
    /// Unit associated with the reading.
    pub unit: MetricUnit,
}

static REGISTRY: Lazy<RwLock<Vec<MetricDescriptor>>> = Lazy::new(|| RwLock::new(Vec::new()));

/// Registers a collection of metric descriptors, ignoring duplicates.
pub fn register_descriptors(descriptors: &[MetricDescriptor]) {
    let mut registry = REGISTRY
        .write()
        .expect("metric registry write lock should not be poisoned");
    for descriptor in descriptors {
        if registry
            .iter()
            .all(|existing| existing.name != descriptor.name)
        {
            registry.push(*descriptor);
        }
    }
}

/// Returns the list of descriptors that were registered so far.
pub fn descriptors() -> Vec<MetricDescriptor> {
    REGISTRY
        .read()
        .expect("metric registry read lock should not be poisoned")
        .clone()
}

/// Canonical descriptors exposed by the InfoNCE objective.
pub const INFO_NCE_DESCRIPTORS: &[MetricDescriptor] = &[
    MetricDescriptor {
        name: "selfsup.info_nce.loss",
        unit: MetricUnit::Scalar,
        description: "Mean InfoNCE loss observed across a batch.",
    },
    MetricDescriptor {
        name: "selfsup.info_nce.top1_accuracy",
        unit: MetricUnit::Scalar,
        description: "Share of anchors whose positives win the top-1 logit.",
    },
    MetricDescriptor {
        name: "selfsup.info_nce.margin",
        unit: MetricUnit::Scalar,
        description: "Mean margin between positive and hardest negative logits.",
    },
    MetricDescriptor {
        name: "selfsup.info_nce.positive_log_prob",
        unit: MetricUnit::Scalar,
        description: "Mean log probability of the positives (nats).",
    },
];

/// Convenience wrapper that registers the built-in InfoNCE descriptors.
pub fn register_info_nce_metrics() {
    register_descriptors(INFO_NCE_DESCRIPTORS);
}

/// Summary statistics derived from an [`InfoNCEResult`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfoNCEMetricSummary {
    /// Mean InfoNCE loss recorded for the batch.
    pub loss: f32,
    /// Ratio of anchors whose positive sample achieved the top-1 logit.
    pub top1_accuracy: f32,
    /// Mean margin between the positive logit and the hardest negative.
    pub mean_positive_margin: f32,
    /// Mean log probability assigned to the positives.
    pub mean_positive_log_probability: f32,
}

impl InfoNCEMetricSummary {
    /// Builds the summary from a raw InfoNCE evaluation result.
    pub fn from_result(result: &InfoNCEResult) -> Self {
        let batch = result.batch.max(1);
        let mut top1_hits = 0usize;
        let mut margin_sum = 0.0f32;
        let mut log_prob_sum = 0.0f32;

        for anchor in 0..result.batch {
            let row = &result.logits[anchor * result.batch..(anchor + 1) * result.batch];
            let positive_logit = row[anchor];

            let mut hardest_negative = f32::NEG_INFINITY;
            for (idx, &logit) in row.iter().enumerate() {
                if idx != anchor {
                    hardest_negative = hardest_negative.max(logit);
                }
            }
            if hardest_negative.is_finite() {
                margin_sum += positive_logit - hardest_negative;
            }

            if row
                .iter()
                .enumerate()
                .all(|(idx, &logit)| idx == anchor || positive_logit >= logit)
            {
                top1_hits += 1;
            }

            let max_logit = row.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
            let exp_sum: f32 = row
                .iter()
                .map(|&v| ((v - max_logit) as f64).exp() as f32)
                .sum();
            if exp_sum > 0.0 {
                let positive_log_prob = positive_logit - max_logit - exp_sum.ln();
                log_prob_sum += positive_log_prob;
            }
        }

        let batch_f32 = batch as f32;
        Self {
            loss: result.loss,
            top1_accuracy: top1_hits as f32 / batch_f32,
            mean_positive_margin: margin_sum / batch_f32,
            mean_positive_log_probability: log_prob_sum / batch_f32,
        }
    }
}

/// Evaluate the registered InfoNCE descriptors against a result.
pub fn evaluate_registered_info_nce(
    result: &InfoNCEResult,
) -> Vec<(MetricDescriptor, MetricValue)> {
    let summary = InfoNCEMetricSummary::from_result(result);
    let mut values = Vec::new();
    for (name, value) in [
        ("selfsup.info_nce.loss", summary.loss),
        ("selfsup.info_nce.top1_accuracy", summary.top1_accuracy),
        ("selfsup.info_nce.margin", summary.mean_positive_margin),
        (
            "selfsup.info_nce.positive_log_prob",
            summary.mean_positive_log_probability,
        ),
    ] {
        if let Some(descriptor) = INFO_NCE_DESCRIPTORS
            .iter()
            .copied()
            .find(|descriptor| descriptor.name == name)
        {
            values.push((
                descriptor,
                MetricValue {
                    name: descriptor.name,
                    value,
                    unit: descriptor.unit,
                },
            ));
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registering_descriptors_is_idempotent() {
        register_info_nce_metrics();
        register_info_nce_metrics();
        let registered = descriptors();
        assert_eq!(registered.len(), INFO_NCE_DESCRIPTORS.len());
        assert!(registered
            .iter()
            .any(|descriptor| descriptor.name == "selfsup.info_nce.loss"));
    }

    #[test]
    fn summary_matches_expected_values() {
        let result = InfoNCEResult {
            loss: 0.5,
            logits: vec![
                2.0, 0.5, 1.0, // anchor 0
                0.1, 1.5, 0.4, // anchor 1
                0.3, 0.2, 1.2, // anchor 2
            ],
            labels: vec![0, 1, 2],
            batch: 3,
        };

        let summary = InfoNCEMetricSummary::from_result(&result);
        assert!((summary.loss - 0.5).abs() < 1e-6);
        assert!((summary.top1_accuracy - 1.0).abs() < 1e-6);
        assert!(summary.mean_positive_margin > 0.0);
        assert!(summary.mean_positive_log_probability < 0.0);

        let evaluated = evaluate_registered_info_nce(&result);
        assert_eq!(evaluated.len(), 4);
        assert!(evaluated.iter().any(|(descriptor, value)| descriptor.name
            == "selfsup.info_nce.loss"
            && value.value == 0.5));
    }
}
