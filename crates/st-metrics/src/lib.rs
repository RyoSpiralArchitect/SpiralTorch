//! Telemetry metric registry utilities shared across SpiralTorch crates.

#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

pub mod registry;

#[cfg(test)]
mod tests {
    use super::registry::{self, MetricDescriptor, MetricUnit, MetricValue};

    #[test]
    fn registers_and_evaluates_scalar_metrics() {
        let _guard = registry::test_guard();
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
        let _guard = registry::test_guard();
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
    fn evaluators_run_without_the_registry_lock_and_reentrant_metrics_are_deferred() {
        let _guard = registry::test_guard();
        registry::clear_for_tests();
        registry::register_metric::<u64, _>(
            MetricDescriptor {
                name: "metric.parent",
                description: "Registers a child metric while evaluating",
                unit: MetricUnit::Unitless,
                tags: &["test", "reentrant"],
                higher_is_better: None,
            },
            |value| {
                assert!(registry::write_available_for_tests());
                registry::register_metric::<u64, _>(
                    MetricDescriptor {
                        name: "metric.child",
                        description: "Metric registered reentrantly",
                        unit: MetricUnit::Unitless,
                        tags: &["test", "reentrant"],
                        higher_is_better: None,
                    },
                    |child| Some(MetricValue::Scalar(*child as f64 + 1.0)),
                );
                Some(MetricValue::Scalar(*value as f64))
            },
        );

        let first = registry::evaluate_report(&3_u64);
        assert!(first.is_clean());
        assert_eq!(first.values.len(), 1);
        assert_eq!(first.values[0].0.name, "metric.parent");

        let second = registry::evaluate_report(&3_u64);
        assert!(second.is_clean());
        assert_eq!(second.values.len(), 2);
        assert_eq!(second.values[0].0.name, "metric.parent");
        assert_eq!(second.values[1].0.name, "metric.child");
    }

    #[test]
    fn panicking_evaluator_is_reported_without_skipping_healthy_metrics() {
        let _guard = registry::test_guard();
        registry::clear_for_tests();
        registry::register_metric::<i64, _>(
            MetricDescriptor {
                name: "metric.panics",
                description: "Panicking test metric",
                unit: MetricUnit::Unitless,
                tags: &["test"],
                higher_is_better: None,
            },
            |_value| panic!("metric evaluator boom"),
        );
        registry::register_metric::<i64, _>(
            MetricDescriptor {
                name: "metric.healthy",
                description: "Healthy test metric",
                unit: MetricUnit::Unitless,
                tags: &["test"],
                higher_is_better: None,
            },
            |value| Some(MetricValue::Scalar(*value as f64)),
        );

        let report = registry::evaluate_report(&7_i64);
        assert_eq!(report.values.len(), 1);
        assert_eq!(report.values[0].0.name, "metric.healthy");
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].descriptor.name, "metric.panics");
        assert_eq!(report.failures[0].message, "metric evaluator boom");
    }

    #[test]
    fn hostile_panic_payload_and_poisoned_registry_are_recovered() {
        struct PanicOnDrop;

        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                panic!("panic while dropping metric payload");
            }
        }

        let _guard = registry::test_guard();
        registry::clear_for_tests();
        registry::poison_for_tests();
        registry::register_metric::<u8, _>(
            MetricDescriptor {
                name: "metric.hostile",
                description: "Metric with a hostile panic payload",
                unit: MetricUnit::Unitless,
                tags: &["test"],
                higher_is_better: None,
            },
            |_value| std::panic::panic_any(PanicOnDrop),
        );

        let report = registry::evaluate_report(&1_u8);
        assert!(report.values.is_empty());
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].message, "non-string panic payload");
        assert_eq!(registry::descriptors_for::<u8>().len(), 1);
    }

    #[test]
    fn metric_values_report_non_finite_members() {
        assert!(MetricValue::Scalar(1.0).is_finite());
        assert!(!MetricValue::Scalar(f64::NAN).is_finite());
        assert!(MetricValue::Distribution(vec![1.0, 2.0]).is_finite());
        assert!(!MetricValue::Distribution(vec![1.0, f64::INFINITY]).is_finite());
    }
}
