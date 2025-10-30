//! Telemetry descriptors used by self-supervised objectives.
//!
//! The original implementation depended on the external `st-metrics` crate to
//! provide a registry of descriptors and gauge values.  That crate is no longer
//! part of the SpiralTorch workspace, so we keep a lightweight in-crate
//! replacement that offers the same surface-level functionality required by the
//! examples and tutorials: registering well-known descriptors and producing
//! gauge values for InfoNCE epochs.

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
        description: "Mean InfoNCE loss observed during the epoch.",
    },
    MetricDescriptor {
        name: "selfsup.info_nce.batches",
        unit: MetricUnit::Count,
        description: "Total number of batches processed in the epoch.",
    },
];

/// Convenience wrapper that registers the built-in InfoNCE descriptors.
pub fn register_info_nce_descriptors() {
    register_descriptors(INFO_NCE_DESCRIPTORS);
}

/// Metric payload summarising an InfoNCE epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfoNCEEpochMetrics {
    /// Mean InfoNCE loss recorded for the epoch.
    pub mean_loss: f32,
    /// Total number of batches seen in the epoch.
    pub batches: usize,
}

impl InfoNCEEpochMetrics {
    /// Builds gauge values suitable for publishing to the telemetry layer.
    pub fn to_values(self) -> [MetricValue; 2] {
        [
            MetricValue {
                name: "selfsup.info_nce.loss",
                value: self.mean_loss,
                unit: MetricUnit::Scalar,
            },
            MetricValue {
                name: "selfsup.info_nce.batches",
                value: self.batches as f32,
                unit: MetricUnit::Count,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registering_descriptors_is_idempotent() {
        register_info_nce_descriptors();
        register_info_nce_descriptors();
        let registered = descriptors();
        assert_eq!(registered.len(), INFO_NCE_DESCRIPTORS.len());
        assert!(registered
            .iter()
            .any(|descriptor| descriptor.name == "selfsup.info_nce.loss"));
    }

    #[test]
    fn epoch_metrics_convert_to_values() {
        let metrics = InfoNCEEpochMetrics {
            mean_loss: 0.42,
            batches: 17,
        };
        let values = metrics.to_values();
        assert_eq!(values[0].name, "selfsup.info_nce.loss");
        assert_eq!(values[0].value, 0.42);
        assert_eq!(values[0].unit, MetricUnit::Scalar);
        assert_eq!(values[1].name, "selfsup.info_nce.batches");
        assert_eq!(values[1].value, 17.0);
        assert_eq!(values[1].unit, MetricUnit::Count);
    }
}
