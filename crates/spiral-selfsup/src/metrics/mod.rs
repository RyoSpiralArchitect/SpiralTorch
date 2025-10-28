//! Metrics and telemetry helpers for self-supervised objectives.

use crate::contrastive::InfoNCEResult;
use st_metrics::registry::{self, MetricDescriptor, MetricUnit, MetricValue};

/// Aggregated statistics derived from an [`InfoNCEResult`].
#[derive(Clone, Debug, PartialEq)]
pub struct InfoNCEMetricSummary {
    pub loss: f32,
    pub top1_accuracy: f32,
    pub mean_positive_logit: f32,
    pub positive_logit_std: f32,
    pub mean_negative_logit: f32,
    pub negative_logit_std: f32,
    pub mean_positive_margin: f32,
    pub mean_positive_log_probability: f32,
    pub positive_log_probability_std: f32,
    pub batch: usize,
}

impl InfoNCEMetricSummary {
    /// Builds a summary by recomputing the logits statistics for the provided result.
    pub fn from_result(result: &InfoNCEResult) -> Self {
        let stats = info_nce_stats(result);
        let (mean_pos, std_pos) = mean_std(&stats.positives);
        let (mean_neg, std_neg) = mean_std(&stats.negatives);
        let (mean_log_prob, std_log_prob) = mean_std(&stats.log_probs);
        let margin_mean = if stats.margins.is_empty() {
            0.0
        } else {
            stats.margins.iter().sum::<f32>() / stats.margins.len() as f32
        };

        Self {
            loss: result.loss,
            top1_accuracy: stats.top1_accuracy,
            mean_positive_logit: mean_pos,
            positive_logit_std: std_pos,
            mean_negative_logit: mean_neg,
            negative_logit_std: std_neg,
            mean_positive_margin: margin_mean,
            mean_positive_log_probability: mean_log_prob,
            positive_log_probability_std: std_log_prob,
            batch: result.batch,
        }
    }
}

struct InfoNCEStats {
    positives: Vec<f32>,
    negatives: Vec<f32>,
    margins: Vec<f32>,
    log_probs: Vec<f32>,
    top1_accuracy: f32,
}

fn info_nce_stats(result: &InfoNCEResult) -> InfoNCEStats {
    let batch = result.batch;
    if batch == 0 {
        return InfoNCEStats {
            positives: Vec::new(),
            negatives: Vec::new(),
            margins: Vec::new(),
            log_probs: Vec::new(),
            top1_accuracy: 0.0,
        };
    }

    let mut positives = Vec::with_capacity(batch);
    let mut negatives = Vec::with_capacity(batch.saturating_mul(batch.saturating_sub(1)));
    let mut margins = Vec::with_capacity(batch);
    let mut log_probs = Vec::with_capacity(batch);
    let mut correct = 0usize;

    for row_idx in 0..batch {
        let row = &result.logits[row_idx * batch..(row_idx + 1) * batch];
        let positive = row[row_idx];
        positives.push(positive);

        let mut max_negative = f32::NEG_INFINITY;
        let mut max_index = 0usize;
        let mut max_value = f32::NEG_INFINITY;
        for (col_idx, &logit) in row.iter().enumerate() {
            if col_idx != row_idx {
                negatives.push(logit);
                if logit > max_negative {
                    max_negative = logit;
                }
            }
            if logit > max_value {
                max_value = logit;
                max_index = col_idx;
            }
        }
        if max_index == row_idx {
            correct += 1;
        }
        if max_negative.is_finite() {
            margins.push(positive - max_negative);
        } else {
            margins.push(0.0);
        }
        let max_logit = if max_value.is_finite() {
            max_value
        } else {
            row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        };
        let exp_sum: f32 = row
            .iter()
            .map(|&value| ((value - max_logit) as f64).exp() as f32)
            .sum();
        let log_prob = positive - max_logit - exp_sum.ln();
        log_probs.push(log_prob);
    }

    let top1_accuracy = correct as f32 / batch as f32;

    InfoNCEStats {
        positives,
        negatives,
        margins,
        log_probs,
        top1_accuracy,
    }
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    (mean, variance.max(0.0).sqrt())
}

/// Registers InfoNCE metrics in the shared [`st_metrics`] registry.
pub fn register_info_nce_metrics() {
    registry::register_metric::<InfoNCEResult, _>(
        MetricDescriptor {
            name: "selfsup.info_nce.loss",
            description: "Average InfoNCE loss across the batch",
            unit: MetricUnit::Loss,
            tags: &["selfsup", "contrastive", "loss"],
            higher_is_better: Some(false),
        },
        |result| Some(MetricValue::Scalar(result.loss as f64)),
    );

    registry::register_metric::<InfoNCEResult, _>(
        MetricDescriptor {
            name: "selfsup.info_nce.top1_accuracy",
            description: "Fraction of anchors whose positive pair is the top-1 logit",
            unit: MetricUnit::Probability,
            tags: &["selfsup", "contrastive", "accuracy"],
            higher_is_better: Some(true),
        },
        |result| {
            let stats = info_nce_stats(result);
            Some(MetricValue::Scalar(stats.top1_accuracy as f64))
        },
    );

    registry::register_metric::<InfoNCEResult, _>(
        MetricDescriptor {
            name: "selfsup.info_nce.margin",
            description: "Mean margin between the positive logit and the strongest negative",
            unit: MetricUnit::Logit,
            tags: &["selfsup", "contrastive", "margin"],
            higher_is_better: Some(true),
        },
        |result| {
            let stats = info_nce_stats(result);
            if stats.margins.is_empty() {
                None
            } else {
                let mean = stats.margins.iter().sum::<f32>() / stats.margins.len() as f32;
                Some(MetricValue::Scalar(mean as f64))
            }
        },
    );

    registry::register_metric::<InfoNCEResult, _>(
        MetricDescriptor {
            name: "selfsup.info_nce.positive_log_prob",
            description: "Mean log probability assigned to the positive pairs",
            unit: MetricUnit::Custom("nats"),
            tags: &["selfsup", "contrastive", "log_prob"],
            higher_is_better: Some(true),
        },
        |result| {
            let stats = info_nce_stats(result);
            if stats.log_probs.is_empty() {
                None
            } else {
                let mean = stats.log_probs.iter().sum::<f32>() / stats.log_probs.len() as f32;
                Some(MetricValue::Scalar(mean as f64))
            }
        },
    );

    registry::register_metric::<InfoNCEResult, _>(
        MetricDescriptor {
            name: "selfsup.info_nce.positive_logits",
            description: "Distribution of positive logits across the batch",
            unit: MetricUnit::Logit,
            tags: &["selfsup", "contrastive", "distribution"],
            higher_is_better: None,
        },
        |result| {
            let stats = info_nce_stats(result);
            if stats.positives.is_empty() {
                None
            } else {
                Some(MetricValue::Distribution(
                    stats.positives.iter().map(|value| *value as f64).collect(),
                ))
            }
        },
    );

    registry::register_metric::<InfoNCEResult, _>(
        MetricDescriptor {
            name: "selfsup.info_nce.negative_logits",
            description: "Distribution of negative logits across the batch",
            unit: MetricUnit::Logit,
            tags: &["selfsup", "contrastive", "distribution"],
            higher_is_better: None,
        },
        |result| {
            let stats = info_nce_stats(result);
            if stats.negatives.is_empty() {
                None
            } else {
                Some(MetricValue::Distribution(
                    stats.negatives.iter().map(|value| *value as f64).collect(),
                ))
            }
        },
    );
}

/// Evaluates all registered InfoNCE metrics for the provided result.
pub fn evaluate_registered_info_nce(
    result: &InfoNCEResult,
) -> Vec<(MetricDescriptor, MetricValue)> {
    registry::evaluate(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result() -> InfoNCEResult {
        InfoNCEResult {
            loss: 0.42,
            logits: vec![2.0, 0.5, 0.25, 1.5],
            labels: vec![0, 1],
            batch: 2,
        }
    }

    #[test]
    fn summary_matches_expected_statistics() {
        let summary = InfoNCEMetricSummary::from_result(&sample_result());
        assert!((summary.mean_positive_logit - 1.75).abs() < 1e-3);
        assert!((summary.mean_negative_logit - 0.375).abs() < 1e-3);
        assert!((summary.top1_accuracy - 1.0).abs() < 1e-6);
        assert!((summary.mean_positive_margin - 1.375).abs() < 1e-3);
        assert!(summary.positive_logit_std > 0.0);
        assert!(summary.positive_log_probability_std >= 0.0);
    }

    #[test]
    fn registry_exposes_metrics() {
        register_info_nce_metrics();
        let descriptors = registry::descriptors_for::<InfoNCEResult>();
        assert!(descriptors
            .iter()
            .any(|d| d.name == "selfsup.info_nce.loss"));
        assert!(descriptors
            .iter()
            .any(|d| d.name == "selfsup.info_nce.top1_accuracy"));

        let metrics = evaluate_registered_info_nce(&sample_result());
        assert!(metrics
            .iter()
            .any(|(d, _)| d.name == "selfsup.info_nce.loss"));
    }
}
