// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use crate::runtime::blackcat::StepMetrics;

const EPS: f64 = 1e-9;
const DEFAULT_WINDOW: usize = 32;
const DEFAULT_ALERT_CAP: usize = 128;

/// Severity level for emitted monitoring alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    /// Observation worth surfacing but not paging.
    Info,
    /// Actionable drift or regression requiring inspection.
    Warning,
    /// Critical signal — automation should route to paging flows.
    Critical,
}

/// Alert categories produced by the monitoring hub.
#[derive(Debug, Clone)]
pub enum AlertKind {
    /// Feature drift detected when comparing the current feature window to the baseline.
    FeatureDrift {
        feature: String,
        z_score: f64,
        threshold: f64,
    },
    /// Model performance regressed relative to historical baseline.
    PerformanceRegression {
        metric: String,
        delta_ratio: f64,
        tolerance: f64,
    },
}

/// Materialised alert instance with timestamp and formatted message.
#[derive(Debug, Clone)]
pub struct AlertRecord {
    pub timestamp: Instant,
    pub severity: AlertSeverity,
    pub kind: AlertKind,
    pub message: String,
}

impl AlertRecord {
    fn new(severity: AlertSeverity, kind: AlertKind, message: String) -> Self {
        Self {
            timestamp: Instant::now(),
            severity,
            kind,
            message,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count as f64 - 1.0)
        }
    }

    fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }
}

#[derive(Debug, Clone)]
struct FeatureWindow {
    history: VecDeque<f64>,
    capacity: usize,
    baseline: RunningStats,
    window_stats: RunningStats,
    warmup: usize,
}

impl FeatureWindow {
    fn new(capacity: usize, warmup: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            capacity,
            baseline: RunningStats::default(),
            window_stats: RunningStats::default(),
            warmup,
        }
    }

    fn update(&mut self, value: f64) {
        if self.history.len() == self.capacity {
            self.history.pop_front();
            self.window_stats = recompute_stats(&self.history);
        }
        self.history.push_back(value);
        self.baseline.update(value);
        if self.history.len() == 1 {
            self.window_stats = RunningStats::default();
        }
        self.window_stats.update(value);
    }

    fn ready(&self) -> bool {
        self.baseline.count as usize >= self.warmup && self.history.len() >= self.capacity / 2
    }

    fn z_score(&self) -> f64 {
        if !self.ready() {
            return 0.0;
        }
        let baseline_std = self.baseline.stddev().max(EPS);
        (self.window_stats.mean - self.baseline.mean).abs() / baseline_std
    }
}

fn recompute_stats(window: &VecDeque<f64>) -> RunningStats {
    if window.is_empty() {
        return RunningStats::default();
    }
    let mut stats = RunningStats::default();
    for value in window.iter().copied() {
        stats.update(value);
    }
    stats
}

/// Drift detector responsible for tracking feature level changes over time.
#[derive(Debug)]
pub struct DriftDetector {
    features: HashMap<String, FeatureWindow>,
    z_threshold: f64,
    warmup: usize,
}

impl DriftDetector {
    pub fn new(z_threshold: f64, warmup: usize) -> Self {
        Self {
            features: HashMap::new(),
            z_threshold,
            warmup,
        }
    }

    pub fn observe(&mut self, features: &HashMap<String, f64>) -> Vec<AlertRecord> {
        let mut alerts = Vec::new();
        for (name, value) in features {
            let window = self
                .features
                .entry(name.clone())
                .or_insert_with(|| FeatureWindow::new(DEFAULT_WINDOW, self.warmup));
            window.update(*value);
            let z = window.z_score();
            if z.is_finite() && z >= self.z_threshold {
                let message = format!(
                    "Feature drift detected for `{}` (z-score {:.2} ≥ {:.2})",
                    name, z, self.z_threshold
                );
                alerts.push(AlertRecord::new(
                    AlertSeverity::Warning,
                    AlertKind::FeatureDrift {
                        feature: name.clone(),
                        z_score: z,
                        threshold: self.z_threshold,
                    },
                    message,
                ));
            }
        }
        alerts
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceDirection {
    /// Higher values are better (e.g. accuracy, reward).
    HigherIsBetter,
    /// Lower values are better (e.g. loss, latency).
    LowerIsBetter,
}

#[derive(Debug)]
struct PerformanceTarget {
    name: String,
    direction: PerformanceDirection,
    tolerance: f64,
    warmup: usize,
    stats: RunningStats,
    latest: Option<f64>,
    consecutive_failures: usize,
    required_failures: usize,
}

impl PerformanceTarget {
    fn new(name: String, direction: PerformanceDirection, tolerance: f64, warmup: usize) -> Self {
        Self {
            name,
            direction,
            tolerance,
            warmup,
            stats: RunningStats::default(),
            latest: None,
            consecutive_failures: 0,
            required_failures: 3,
        }
    }

    fn update(&mut self, value: f64) -> Option<AlertRecord> {
        self.latest = Some(value);
        self.stats.update(value);
        if self.stats.count < self.warmup as u64 {
            return None;
        }
        let baseline = self.stats.mean;
        let drift = match self.direction {
            PerformanceDirection::HigherIsBetter => {
                if baseline.abs() <= EPS {
                    0.0
                } else {
                    (baseline - value) / baseline.max(EPS)
                }
            }
            PerformanceDirection::LowerIsBetter => {
                if baseline.abs() <= EPS {
                    0.0
                } else {
                    (value - baseline) / baseline.max(EPS)
                }
            }
        };
        let degrade = match self.direction {
            PerformanceDirection::HigherIsBetter => drift > self.tolerance,
            PerformanceDirection::LowerIsBetter => drift > self.tolerance,
        };
        if degrade {
            self.consecutive_failures += 1;
        } else {
            self.consecutive_failures = 0;
        }
        if self.consecutive_failures >= self.required_failures {
            self.consecutive_failures = 0;
            let message = match self.direction {
                PerformanceDirection::HigherIsBetter => format!(
                    "Performance regression on `{}` — value {:.4} is {:.1}% below baseline {:.4}",
                    self.name,
                    value,
                    drift * 100.0,
                    baseline
                ),
                PerformanceDirection::LowerIsBetter => format!(
                    "Performance regression on `{}` — value {:.4} is {:.1}% above baseline {:.4}",
                    self.name,
                    value,
                    drift * 100.0,
                    baseline
                ),
            };
            return Some(AlertRecord::new(
                AlertSeverity::Critical,
                AlertKind::PerformanceRegression {
                    metric: self.name.clone(),
                    delta_ratio: drift,
                    tolerance: self.tolerance,
                },
                message,
            ));
        }
        None
    }
}

/// Observes performance metrics (loss, accuracy, reward) and emits alerts when
/// regressions exceed configured tolerances.
#[derive(Debug)]
pub struct PerformanceMonitor {
    targets: HashMap<String, PerformanceTarget>,
}

impl PerformanceMonitor {
    pub fn new(default_warmup: usize) -> Self {
        let mut targets = HashMap::new();
        targets.insert(
            "step_loss".to_string(),
            PerformanceTarget::new(
                "step_loss".to_string(),
                PerformanceDirection::LowerIsBetter,
                0.25,
                default_warmup,
            ),
        );
        targets.insert(
            "loss_weighted".to_string(),
            PerformanceTarget::new(
                "loss_weighted".to_string(),
                PerformanceDirection::LowerIsBetter,
                0.25,
                default_warmup,
            ),
        );
        targets.insert(
            "reward".to_string(),
            PerformanceTarget::new(
                "reward".to_string(),
                PerformanceDirection::HigherIsBetter,
                0.2,
                default_warmup,
            ),
        );
        Self { targets }
    }

    pub fn with_target(
        mut self,
        name: impl Into<String>,
        direction: PerformanceDirection,
        tolerance: f64,
        warmup: usize,
    ) -> Self {
        let name = name.into();
        self.targets.insert(
            name.clone(),
            PerformanceTarget::new(name, direction, tolerance, warmup),
        );
        self
    }

    pub fn observe(&mut self, metrics: &HashMap<String, f64>, reward: f64) -> Vec<AlertRecord> {
        let mut alerts = Vec::new();
        for (name, target) in self.targets.iter_mut() {
            let value = if name == "reward" {
                reward
            } else if let Some(v) = metrics.get(name) {
                *v
            } else {
                continue;
            };
            if let Some(alert) = target.update(value) {
                alerts.push(alert);
            }
        }
        alerts
    }

    pub fn latest_values(&self) -> HashMap<String, f64> {
        self.targets
            .iter()
            .filter_map(|(name, target)| target.latest.map(|value| (name.clone(), value)))
            .collect()
    }
}

/// Trait implemented by exporter backends for propagating metrics to external systems.
pub trait MetricsExporter: Send + Sync {
    fn record_gauge(&self, name: &str, value: f64, labels: &[(&str, &str)]);
    fn record_counter(&self, name: &str, value: f64, labels: &[(&str, &str)]);
}

#[cfg(feature = "observability-prometheus")]
mod prometheus_exporter {
    use super::{MetricsExporter, EPS};
    use prometheus::{Counter, Gauge, Registry};
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub struct PrometheusExporter {
        registry: Registry,
        gauges: Mutex<HashMap<String, Gauge>>,
        counters: Mutex<HashMap<String, Counter>>,
    }

    impl PrometheusExporter {
        pub fn new() -> Self {
            Self {
                registry: Registry::new(),
                gauges: Mutex::new(HashMap::new()),
                counters: Mutex::new(HashMap::new()),
            }
        }

        pub fn registry(&self) -> &Registry {
            &self.registry
        }
    }

    impl MetricsExporter for PrometheusExporter {
        fn record_gauge(&self, name: &str, value: f64, _labels: &[(&str, &str)]) {
            let mut gauges = self.gauges.lock().expect("gauge mutex poisoned");
            let gauge = gauges.entry(name.to_string()).or_insert_with(|| {
                let gauge = Gauge::new(name, format!("Gauge for {}", name)).expect("gauge");
                self.registry
                    .register(Box::new(gauge.clone()))
                    .expect("register gauge");
                gauge
            });
            gauge.set(value);
        }

        fn record_counter(&self, name: &str, value: f64, _labels: &[(&str, &str)]) {
            let mut counters = self.counters.lock().expect("counter mutex poisoned");
            let counter = counters.entry(name.to_string()).or_insert_with(|| {
                let counter = Counter::new(name, format!("Counter for {}", name)).expect("counter");
                self.registry
                    .register(Box::new(counter.clone()))
                    .expect("register counter");
                counter
            });
            if value >= 0.0 {
                counter.inc_by(value);
            } else {
                counter.inc_by(EPS);
            }
        }
    }

    pub use prometheus::Encoder;
    pub use prometheus::TextEncoder;
}

#[cfg(feature = "observability-prometheus")]
pub use prometheus_exporter::PrometheusExporter;

#[cfg(feature = "observability-otlp")]
mod otel_exporter {
    use super::MetricsExporter;
    use opentelemetry::metrics::{Counter, Gauge, Meter};
    use opentelemetry::Context;
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub struct OtelExporter {
        meter: Meter,
        gauges: Mutex<HashMap<String, Gauge<f64>>>,
        counters: Mutex<HashMap<String, Counter<f64>>>,
    }

    impl OtelExporter {
        pub fn new(meter: Meter) -> Self {
            Self {
                meter,
                gauges: Mutex::new(HashMap::new()),
                counters: Mutex::new(HashMap::new()),
            }
        }
    }

    impl MetricsExporter for OtelExporter {
        fn record_gauge(&self, name: &str, value: f64, _labels: &[(&str, &str)]) {
            let mut gauges = self.gauges.lock().expect("gauge mutex poisoned");
            let gauge = gauges.entry(name.to_string()).or_insert_with(|| {
                self.meter
                    .f64_gauge(name)
                    .with_description(format!("Gauge for {}", name))
                    .init()
            });
            gauge.record(&Context::current(), value, &[]);
        }

        fn record_counter(&self, name: &str, value: f64, _labels: &[(&str, &str)]) {
            let mut counters = self.counters.lock().expect("counter mutex poisoned");
            let counter = counters.entry(name.to_string()).or_insert_with(|| {
                self.meter
                    .f64_counter(name)
                    .with_description(format!("Counter for {}", name))
                    .init()
            });
            counter.add(&Context::current(), value, &[]);
        }
    }

    pub use opentelemetry::global;
}

#[cfg(feature = "observability-otlp")]
pub use otel_exporter::OtelExporter;

/// Central hub orchestrating drift detectors, performance monitors, and exporters.
pub struct MonitoringHub {
    drift: DriftDetector,
    performance: PerformanceMonitor,
    exporters: Vec<Arc<dyn MetricsExporter>>,
    alert_log: VecDeque<AlertRecord>,
    alert_capacity: usize,
    last_observation: Option<Instant>,
}

impl MonitoringHub {
    pub fn new(drift_threshold: f64, drift_warmup: usize, perf_warmup: usize) -> Self {
        Self {
            drift: DriftDetector::new(drift_threshold, drift_warmup),
            performance: PerformanceMonitor::new(perf_warmup),
            exporters: Vec::new(),
            alert_log: VecDeque::with_capacity(DEFAULT_ALERT_CAP),
            alert_capacity: DEFAULT_ALERT_CAP,
            last_observation: None,
        }
    }

    pub fn with_exporter(mut self, exporter: Arc<dyn MetricsExporter>) -> Self {
        self.exporters.push(exporter);
        self
    }

    pub fn register_exporter(&mut self, exporter: Arc<dyn MetricsExporter>) {
        self.exporters.push(exporter);
    }

    /// Reset the hub with new drift and performance parameters.
    pub fn reconfigure(&mut self, drift_threshold: f64, drift_warmup: usize, perf_warmup: usize) {
        *self = MonitoringHub::new(drift_threshold, drift_warmup, perf_warmup);
    }

    pub fn observe(&mut self, metrics: &StepMetrics, reward: f64) -> Vec<AlertRecord> {
        self.last_observation = Some(Instant::now());
        let mut alerts = self.drift.observe(&metrics.extra);
        alerts.extend(self.performance.observe(&metrics.extra, reward));
        if !alerts.is_empty() {
            for alert in alerts.iter().cloned() {
                self.record_alert(alert);
            }
        }
        self.export(metrics, reward);
        alerts
    }

    fn export(&self, metrics: &StepMetrics, reward: f64) {
        let latest_perf = self.performance.latest_values();
        let sanitize = |name: &str| {
            name.chars()
                .map(|c| match c {
                    'a'..='z' | 'A'..='Z' | '0'..='9' => c,
                    _ => '_',
                })
                .collect::<String>()
        };
        for exporter in &self.exporters {
            exporter.record_gauge("runtime_step_time_ms", metrics.step_time_ms, &[]);
            exporter.record_gauge("runtime_mem_peak_mb", metrics.mem_peak_mb, &[]);
            exporter.record_gauge("runtime_retry_rate", metrics.retry_rate, &[]);
            exporter.record_gauge("runtime_reward", reward, &[]);
            for (key, value) in &metrics.extra {
                let metric_name = format!("extra_{}", sanitize(key));
                exporter.record_gauge(&metric_name, *value, &[]);
            }
            for (key, value) in &latest_perf {
                let metric_name = format!("performance_{}", sanitize(key));
                exporter.record_gauge(&metric_name, *value, &[]);
            }
        }
    }

    fn record_alert(&mut self, alert: AlertRecord) {
        if self.alert_log.len() == self.alert_capacity {
            self.alert_log.pop_front();
        }
        self.alert_log.push_back(alert);
    }

    pub fn latest_alerts(&self) -> Vec<AlertRecord> {
        self.alert_log.iter().cloned().collect()
    }

    pub fn last_observation(&self) -> Option<Instant> {
        self.last_observation
    }
}

impl fmt::Debug for MonitoringHub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MonitoringHub")
            .field("drift", &self.drift)
            .field("performance", &self.performance)
            .field("exporter_count", &self.exporters.len())
            .field("alert_log_len", &self.alert_log.len())
            .field("alert_capacity", &self.alert_capacity)
            .field("last_observation", &self.last_observation)
            .finish()
    }
}

impl Default for MonitoringHub {
    fn default() -> Self {
        Self::new(3.0, 16, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_metrics(value: f64) -> StepMetrics {
        let mut metrics = StepMetrics::default();
        metrics.step_time_ms = 10.0;
        metrics.extra.insert("step_loss".into(), value);
        metrics.extra.insert("band_here".into(), value * 0.1);
        metrics
    }

    #[test]
    fn drift_detector_emits_alerts_after_warmup() {
        let mut hub = MonitoringHub::new(2.5, 8, 8);
        let mut reward = 1.0;
        for i in 0..16 {
            let mut metrics = StepMetrics::default();
            metrics
                .extra
                .insert("band_here".into(), if i < 12 { 0.5 } else { 5.0 });
            hub.observe(&metrics, reward);
            reward += 0.1;
        }
        let alerts = hub.latest_alerts();
        assert!(
            alerts
                .iter()
                .any(|a| matches!(a.kind, AlertKind::FeatureDrift { .. })),
            "expected drift alert"
        );
    }

    #[test]
    fn performance_monitor_detects_regression() {
        let mut hub = MonitoringHub::new(3.0, 10, 5);
        for _ in 0..10 {
            let metrics = build_metrics(0.1);
            hub.observe(&metrics, 1.0);
        }
        for _ in 0..5 {
            let metrics = build_metrics(0.4);
            hub.observe(&metrics, 1.0);
        }
        let alerts = hub.latest_alerts();
        assert!(
            alerts
                .iter()
                .any(|a| matches!(a.kind, AlertKind::PerformanceRegression { .. })),
            "expected performance regression alert"
        );
    }
}
