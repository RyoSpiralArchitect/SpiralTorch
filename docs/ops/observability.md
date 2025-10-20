# Observability & Monitoring

This document captures the current observability surface for SpiralTorch runtimes, outlines the newly added drift/performance detectors, and provides guidance for operators to deploy dashboards or automated smoke checks.

## Runtime telemetry at a glance

SpiralTorch exposes training loop statistics through the `StepMetrics` structure inside `crates/st-core/src/runtime/blackcat.rs`. Core dimensions are:

- Latency: `step_time_ms`
- Memory: `mem_peak_mb`
- Stability: `retry_rate`
- Rich contextual signals inserted into the `extra` hash map (e.g. gradient norms, band energies, step losses)

These metrics power internal components under `crates/st-core/src/telemetry/`, including:

- `telemetry::atlas`, `telemetry::dashboard`, and `telemetry::hub` for dashboard frames and operator-facing narratives.
- `telemetry::psi` / `telemetry::psychoid` (optional features) for psychoid feedback loops.
- `telemetry::monitoring` (new) for automated drift and performance alerting.

## Drift & performance detectors

`telemetry::monitoring::MonitoringHub` combines two subsystems:

1. **Feature drift detection** – The `DriftDetector` maintains rolling windows per feature (populated from `StepMetrics::extra`) and compares the current window mean against a warm-up baseline. Alerts are emitted when the z-score exceeds the configured threshold.
2. **Model performance monitoring** – The `PerformanceMonitor` tracks directional metrics (losses, weighted losses, runtime reward) and raises critical alerts when consecutive samples exceed tolerance bands relative to the baseline.

Alerts are retained in a bounded log accessible via `BlackCatRuntime::monitoring()` for debugging or CI validation. Operators can tune thresholds at runtime by calling `MonitoringHub::reconfigure`.

### Exporters

The monitoring hub supports optional exporters:

- **Prometheus** – Enable the `observability-prometheus` feature and register a `PrometheusExporter` instance with `MonitoringHub::register_exporter` to expose gauges/counters.
- **OpenTelemetry Metrics** – Enable the `observability-otlp` feature to push metrics via an `OtelExporter` built from any OTLP meter.

Exporters stream the base runtime metrics, feature extras, and derived performance gauges, allowing downstream alerting or dashboards.

## Operational deployment

A reference deployment is provided under `tools/ops/observability/docker-compose.yml`. The stack includes:

- Prometheus scraping the runtime exporter endpoint.
- Grafana wired to Prometheus with starter dashboards (import `tools/monitoring/grafana.json`).
- An optional OpenTelemetry Collector forwarding metrics to Prometheus.

Run the stack locally:

```bash
cd tools/ops/observability
docker compose up
```

Point the runtime at the Prometheus push endpoint or expose the exporter via HTTP and configure Prometheus to scrape it.

## CI smoke tests

Automated smoke tests live in `crates/st-core/tests/monitoring_smoke.rs`. They validate:

- Drift detectors fire after a distribution shift.
- Performance regressions trigger critical alerts.
- Reconfiguring the hub resets alert history (ensuring clean baselines per run).

These tests run during `cargo test` to prevent regressions in alerting and logging coverage.

## Rollout checklist

1. Enable desired exporters via Cargo features and register them during runtime boot.
2. Configure Prometheus/Grafana using the provided compose file or integrate with your existing observability stack.
3. Monitor the `MonitoringHub::latest_alerts` feed for drift/performance signals and route high-severity alerts into incident workflows.
4. Keep CI smoke tests green to guarantee the alerting pipeline remains intact.
