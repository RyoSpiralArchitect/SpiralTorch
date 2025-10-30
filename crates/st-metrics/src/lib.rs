//! Telemetry metric registry utilities shared across SpiralTorch crates.

pub mod registry;

#[cfg(test)]
mod tests {
    use super::registry::{self, MetricDescriptor, MetricUnit, MetricValue};

    #[test]
    fn registers_and_evaluates_scalar_metrics() {
        registry::clear_for_tests();
        registry::register_metric::<u32, _>(
            MetricDescriptor {
                name: "metric.scalar",
                description: "Test metric returning a scalar",
                unit: MetricUnit::Unitless,
                tags: &["test"],
                higher_is_better: Some(true),
            },
            |value| Some(MetricValue::Scalar((*value as f64) * 2.0)),
        );

        let results = registry::evaluate(&5u32);
        assert_eq!(results.len(), 1);
        let (descriptor, value) = &results[0];
        assert_eq!(descriptor.name, "metric.scalar");
        match value {
            MetricValue::Scalar(v) => assert!((*v - 10.0).abs() < f64::EPSILON),
            _ => panic!("unexpected metric value"),
        }
    }

    #[test]
    fn skips_metrics_with_type_mismatch() {
        registry::clear_for_tests();
        registry::register_metric::<u32, _>(
            MetricDescriptor {
                name: "metric.scalar",
                description: "Test metric returning a scalar",
                unit: MetricUnit::Unitless,
                tags: &["test"],
                higher_is_better: Some(true),
            },
            |value| Some(MetricValue::Scalar((*value as f64) * 2.0)),
        );

        let results = registry::evaluate(&5i32);
        assert!(results.is_empty());
    }
}
