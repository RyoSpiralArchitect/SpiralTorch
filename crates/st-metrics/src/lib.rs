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

    #[test]
    fn exports_metrics_to_scalar_map() {
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
        registry::register_metric::<u32, _>(
            MetricDescriptor {
                name: "metric.dist",
                description: "Test metric returning a distribution",
                unit: MetricUnit::Loss,
                tags: &["test"],
                higher_is_better: Some(false),
            },
            |_value| Some(MetricValue::Distribution(vec![1.0, 2.0, 3.0])),
        );

        let results = registry::evaluate(&5u32);
        let map = registry::evaluation_to_scalar_map(&results);

        assert_eq!(map.get("metric.scalar"), Some(&10.0));
        assert_eq!(map.get("metric.dist.count"), Some(&3.0));
        assert_eq!(map.get("metric.dist.mean"), Some(&2.0));
        assert_eq!(map.get("metric.dist.min"), Some(&1.0));
        assert_eq!(map.get("metric.dist.max"), Some(&3.0));
    }

    #[cfg(feature = "dashboard")]
    #[test]
    fn exports_metrics_to_dashboard_frame() {
        use std::collections::HashMap;

        let _guard = TEST_LOCK.lock().unwrap();
        registry::clear_for_tests();
        registry::register_metric::<u32, _>(
            MetricDescriptor {
                name: "metric.loss",
                description: "Test metric returning a scalar loss",
                unit: MetricUnit::Loss,
                tags: &["test"],
                higher_is_better: Some(false),
            },
            |value| Some(MetricValue::Scalar(*value as f64)),
        );
        registry::register_metric::<u32, _>(
            MetricDescriptor {
                name: "metric.samples",
                description: "Test metric returning a distribution",
                unit: MetricUnit::Probability,
                tags: &["test"],
                higher_is_better: None,
            },
            |_value| Some(MetricValue::Distribution(vec![0.25, 0.5, 0.75])),
        );

        let results = registry::evaluate(&4u32);
        let frame = registry::evaluation_to_dashboard_frame(&results);

        let metrics: HashMap<&str, (f64, Option<&str>)> = frame
            .metrics
            .iter()
            .map(|metric| (metric.name.as_str(), (metric.value, metric.unit.as_deref())))
            .collect();

        assert_eq!(metrics.get("metric.loss"), Some(&(4.0, Some("loss"))));
        assert_eq!(
            metrics.get("metric.samples.count"),
            Some(&(3.0, Some("count")))
        );
        assert_eq!(
            metrics.get("metric.samples.mean"),
            Some(&(0.5, Some("probability")))
        );
        assert_eq!(
            metrics.get("metric.samples.min"),
            Some(&(0.25, Some("probability")))
        );
        assert_eq!(
            metrics.get("metric.samples.max"),
            Some(&(0.75, Some("probability")))
        );
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
