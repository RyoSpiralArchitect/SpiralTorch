//! Telemetry metric registry utilities shared across SpiralTorch crates.

pub mod registry;

#[cfg(test)]
mod tests {
    use super::registry::{self, MetricDescriptor, MetricUnit, MetricValue};
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn registers_and_evaluates_scalar_metrics() {
        let _guard = TEST_LOCK.lock().unwrap();
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
        let _guard = TEST_LOCK.lock().unwrap();
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

    #[cfg(feature = "json")]
    #[test]
    fn exports_metrics_to_json() {
        let _guard = TEST_LOCK.lock().unwrap();
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

        let json = registry::evaluate_json(&5u32);
        let entries = json.as_array().expect("expected JSON array");
        assert_eq!(entries.len(), 1);
        let entry = entries[0].as_object().expect("expected JSON object");
        assert!(entry.contains_key("descriptor"));
        assert!(entry.contains_key("value"));
    }
}
