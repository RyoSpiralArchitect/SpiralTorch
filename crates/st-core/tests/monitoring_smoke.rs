// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::HashMap;

use bandit::SoftBanditMode;
use st_core::runtime::blackcat::zmeta::ZMetaParams;
use st_core::runtime::blackcat::{BlackCatRuntime, ChoiceGroups, StepMetrics};
use st_core::telemetry::monitoring::{AlertKind, MonitoringHub};

fn sample_runtime() -> BlackCatRuntime {
    let mut groups = HashMap::new();
    groups.insert("tile".to_string(), vec!["a".to_string(), "b".to_string()]);
    let groups = ChoiceGroups { groups };
    let mut runtime =
        BlackCatRuntime::new(ZMetaParams::default(), groups, 4, SoftBanditMode::TS, None);
    runtime.monitoring_mut().reconfigure(2.5, 4, 4);
    runtime
}

#[test]
fn smoke_runtime_emits_drift_alerts() {
    let mut runtime = sample_runtime();
    for _ in 0..8 {
        let mut metrics = StepMetrics::default();
        metrics.extra.insert("band_here".into(), 0.1);
        runtime.post_step(&metrics);
    }
    for _ in 0..5 {
        let mut metrics = StepMetrics::default();
        metrics.extra.insert("band_here".into(), 5.0);
        runtime.post_step(&metrics);
    }
    let alerts = runtime.monitoring().latest_alerts();
    assert!(
        alerts
            .iter()
            .any(|a| matches!(a.kind, AlertKind::FeatureDrift { .. })),
        "expected drift alert"
    );
}

#[test]
fn smoke_runtime_emits_performance_regression() {
    let mut runtime = sample_runtime();
    for _ in 0..10 {
        let mut metrics = StepMetrics::default();
        metrics.extra.insert("step_loss".into(), 0.1);
        runtime.post_step(&metrics);
    }
    for _ in 0..5 {
        let mut metrics = StepMetrics::default();
        metrics.extra.insert("step_loss".into(), 0.8);
        runtime.post_step(&metrics);
    }
    let alerts = runtime.monitoring().latest_alerts();
    assert!(
        alerts
            .iter()
            .any(|a| matches!(a.kind, AlertKind::PerformanceRegression { .. })),
        "expected performance regression alert"
    );
}

#[test]
fn monitoring_hub_reconfigure_resets_alert_history() {
    let mut hub = MonitoringHub::new(2.5, 4, 4);
    let mut metrics = StepMetrics::default();
    metrics.extra.insert("band_here".into(), 0.2);
    for _ in 0..6 {
        hub.observe(&metrics, 0.0);
    }
    hub.reconfigure(3.0, 8, 8);
    assert!(hub.latest_alerts().is_empty());
}
