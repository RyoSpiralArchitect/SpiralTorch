// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Metrics descriptors and helpers for self-supervised objectives.

use once_cell::sync::Lazy;

/// Unit attached to a metric descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricUnit {
    /// Dimensionless ratio (e.g. losses, accuracies).
    Ratio,
    /// Discrete count (e.g. number of batches, tokens).
    Count,
}

impl MetricUnit {
    fn as_str(self) -> &'static str {
        match self {
            MetricUnit::Ratio => "ratio",
            MetricUnit::Count => "count",
        }
    }
}

/// Description of a scalar metric exposed by the self-supervised objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricDescriptor {
    /// Fully-qualified metric identifier.
    pub name: &'static str,
    /// Human-friendly description of what the metric represents.
    pub help: &'static str,
    /// Unit attached to the metric.
    pub unit: MetricUnit,
}

impl MetricDescriptor {
    const fn new(name: &'static str, help: &'static str, unit: MetricUnit) -> Self {
        Self { name, help, unit }
    }
}

/// Value recorded for a metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricValue {
    /// Descriptor attached to the value.
    pub descriptor: &'static MetricDescriptor,
    /// Observed value.
    pub value: f32,
}

impl MetricValue {
    const fn new(descriptor: &'static MetricDescriptor, value: f32) -> Self {
        Self { descriptor, value }
    }
}

static INFO_NCE_METRICS: Lazy<[MetricDescriptor; 2]> = Lazy::new(|| {
    [
        MetricDescriptor::new(
            "selfsup.info_nce.loss",
            "Mean InfoNCE loss aggregated across the epoch",
            MetricUnit::Ratio,
        ),
        MetricDescriptor::new(
            "selfsup.info_nce.batches",
            "Number of batches that contributed to the epoch aggregation",
            MetricUnit::Count,
        ),
    ]
});

static MASKED_MSE_METRICS: Lazy<[MetricDescriptor; 2]> = Lazy::new(|| {
    [
        MetricDescriptor::new(
            "selfsup.masked_mse.loss",
            "Mean squared error over the masked tokens",
            MetricUnit::Ratio,
        ),
        MetricDescriptor::new(
            "selfsup.masked_mse.total_masked",
            "Total number of masked elements processed during the epoch",
            MetricUnit::Count,
        ),
    ]
});

/// Returns the descriptors emitted by the InfoNCE objective.
pub fn info_nce_descriptors() -> &'static [MetricDescriptor] {
    &*INFO_NCE_METRICS
}

/// Returns the descriptors emitted by the masked MSE objective.
pub fn masked_mse_descriptors() -> &'static [MetricDescriptor] {
    &*MASKED_MSE_METRICS
}

/// Builds metric values for an InfoNCE epoch summary.
pub fn info_nce_values(mean_loss: f32, batches: usize) -> [MetricValue; 2] {
    [
        MetricValue::new(&INFO_NCE_METRICS[0], mean_loss),
        MetricValue::new(&INFO_NCE_METRICS[1], batches as f32),
    ]
}

/// Builds metric values for a masked MSE epoch summary.
pub fn masked_mse_values(mean_loss: f32, total_masked: usize) -> [MetricValue; 2] {
    [
        MetricValue::new(&MASKED_MSE_METRICS[0], mean_loss),
        MetricValue::new(&MASKED_MSE_METRICS[1], total_masked as f32),
    ]
}

/// In-memory registry that keeps track of descriptors and the latest published values.
pub mod registry {
    use super::{MetricDescriptor, MetricValue};
    use once_cell::sync::Lazy;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    static DESCRIPTORS: Lazy<Mutex<BTreeMap<&'static str, &'static MetricDescriptor>>> =
        Lazy::new(|| Mutex::new(BTreeMap::new()));
    static VALUES: Lazy<Mutex<BTreeMap<&'static str, f32>>> =
        Lazy::new(|| Mutex::new(BTreeMap::new()));

    /// Registers the provided metric descriptors with the registry.
    pub fn register(metrics: &'static [MetricDescriptor]) {
        let mut registry = DESCRIPTORS.lock().expect("descriptor registry poisoned");
        for descriptor in metrics {
            registry.entry(descriptor.name).or_insert(descriptor);
        }
    }

    /// Registers a single metric descriptor.
    pub fn register_one(descriptor: &'static MetricDescriptor) {
        let mut registry = DESCRIPTORS.lock().expect("descriptor registry poisoned");
        registry.entry(descriptor.name).or_insert(descriptor);
    }

    /// Returns all registered metric descriptors sorted by their identifier.
    pub fn descriptors() -> Vec<&'static MetricDescriptor> {
        let registry = DESCRIPTORS.lock().expect("descriptor registry poisoned");
        registry.values().copied().collect()
    }

    /// Publishes the provided metric values, overwriting any previously recorded value.
    pub fn publish(values: impl IntoIterator<Item = MetricValue>) {
        let mut latest = VALUES.lock().expect("metric value registry poisoned");
        for value in values {
            latest.insert(value.descriptor.name, value.value);
        }
    }

    /// Returns the most recently recorded value for the provided metric identifier.
    pub fn latest(name: &str) -> Option<f32> {
        let latest = VALUES.lock().expect("metric value registry poisoned");
        latest.get(name).copied()
    }

    /// Returns a snapshot of all metric values that have been published at least once.
    pub fn snapshot() -> Vec<MetricValue> {
        let registry = DESCRIPTORS.lock().expect("descriptor registry poisoned");
        let latest = VALUES.lock().expect("metric value registry poisoned");

        registry
            .values()
            .filter_map(|descriptor| {
                latest
                    .get(descriptor.name)
                    .copied()
                    .map(|value| MetricValue::new(descriptor, value))
            })
            .collect()
    }

    /// Returns a human-readable summary of the registered descriptors.
    pub fn describe() -> Vec<String> {
        descriptors()
            .into_iter()
            .map(|descriptor| {
                format!(
                    "{} [{}]: {}",
                    descriptor.name,
                    descriptor.unit.as_str(),
                    descriptor.help
                )
            })
            .collect()
    }

    #[cfg(test)]
    pub fn clear() {
        DESCRIPTORS
            .lock()
            .expect("descriptor registry poisoned")
            .clear();
        VALUES
            .lock()
            .expect("metric value registry poisoned")
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_nce_metric_values_are_reported() {
        registry::clear();
        registry::register(info_nce_descriptors());
        let metrics = info_nce_values(0.42, 8);
        registry::publish(metrics);

        assert_eq!(registry::latest("selfsup.info_nce.loss").unwrap(), 0.42);
        assert_eq!(registry::latest("selfsup.info_nce.batches").unwrap(), 8.0);

        let snapshot = registry::snapshot();
        assert_eq!(snapshot.len(), 2);
        assert!(snapshot
            .iter()
            .any(|value| value.descriptor.name == "selfsup.info_nce.loss"));
    }

    #[test]
    fn masked_metrics_are_registered_and_published() {
        registry::clear();
        registry::register(masked_mse_descriptors());
        registry::publish(masked_mse_values(1.5, 32));

        assert_eq!(registry::latest("selfsup.masked_mse.loss").unwrap(), 1.5);
        assert_eq!(
            registry::latest("selfsup.masked_mse.total_masked").unwrap(),
            32.0
        );

        let description = registry::describe();
        assert!(description
            .iter()
            .any(|entry| entry.contains("selfsup.masked_mse.loss")));
    }
}
