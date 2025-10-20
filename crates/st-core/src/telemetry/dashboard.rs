// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Lightweight metric captured by the dashboard runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DashboardMetric {
    pub name: String,
    pub value: f64,
    pub unit: Option<String>,
    pub trend: Option<f64>,
}

impl DashboardMetric {
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            unit: None,
            trend: None,
        }
    }

    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    pub fn with_trend(mut self, trend: f64) -> Self {
        self.trend = Some(trend);
        self
    }
}

/// Narrative note surfaced alongside metrics for operators.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DashboardEvent {
    pub message: String,
    pub severity: EventSeverity,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventSeverity {
    Info,
    Warning,
    Critical,
}

/// Snapshot emitted at a given point in time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DashboardFrame {
    pub timestamp: SystemTime,
    pub metrics: Vec<DashboardMetric>,
    pub events: Vec<DashboardEvent>,
}

impl DashboardFrame {
    pub fn new(timestamp: SystemTime) -> Self {
        Self {
            timestamp,
            metrics: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn push_metric(&mut self, metric: DashboardMetric) {
        self.metrics.push(metric);
    }

    pub fn push_event(&mut self, event: DashboardEvent) {
        self.events.push(event);
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MetricAggregate {
    pub name: String,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub latest: f64,
    pub samples: usize,
}

impl MetricAggregate {
    fn from_accumulator(name: String, acc: MetricAccumulator) -> Self {
        let mean = if acc.count == 0 {
            0.0
        } else {
            acc.sum / acc.count as f64
        };
        Self {
            name,
            mean,
            min: acc.min,
            max: acc.max,
            latest: acc.latest,
            samples: acc.count,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DashboardSummary {
    pub frame_count: usize,
    /// Time span between the oldest and newest frame included in the summary (seconds).
    pub span_seconds: f64,
    pub metrics: Vec<MetricAggregate>,
    pub events: BTreeMap<EventSeverity, usize>,
}

impl DashboardSummary {
    pub fn is_empty(&self) -> bool {
        self.frame_count == 0
    }
}

#[derive(Clone, Debug)]
struct MetricAccumulator {
    latest: f64,
    sum: f64,
    min: f64,
    max: f64,
    count: usize,
}

/// Rolling dashboard state that retains recent frames for quick inspection.
#[derive(Debug)]
pub struct DashboardRing {
    capacity: usize,
    frames: VecDeque<DashboardFrame>,
}

impl DashboardRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            frames: VecDeque::new(),
        }
    }

    pub fn push(&mut self, frame: DashboardFrame) {
        if self.frames.len() == self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    pub fn latest(&self) -> Option<&DashboardFrame> {
        self.frames.back()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &DashboardFrame> + ExactSizeIterator {
        self.frames.iter()
    }

    /// Builds an aggregate summary across the most recent `limit` frames. When `limit` is `None`
    /// the entire ring is considered. Returns `None` if the ring is empty.
    pub fn summarize(&self, limit: Option<usize>) -> Option<DashboardSummary> {
        if self.frames.is_empty() {
            return None;
        }

        let limit = limit.unwrap_or(usize::MAX).max(1);
        let mut accumulators: BTreeMap<String, MetricAccumulator> = BTreeMap::new();
        let mut events: BTreeMap<EventSeverity, usize> = BTreeMap::new();
        let mut frame_count = 0usize;
        let mut latest_ts: Option<SystemTime> = None;
        let mut earliest_ts: Option<SystemTime> = None;

        for frame in self.frames.iter().rev().take(limit) {
            frame_count += 1;
            let ts = frame.timestamp;
            if latest_ts.is_none() {
                latest_ts = Some(ts);
            }
            earliest_ts = Some(ts);

            for metric in &frame.metrics {
                let entry =
                    accumulators
                        .entry(metric.name.clone())
                        .or_insert_with(|| MetricAccumulator {
                            latest: metric.value,
                            sum: 0.0,
                            min: metric.value,
                            max: metric.value,
                            count: 0,
                        });
                entry.sum += metric.value;
                entry.count += 1;
                entry.min = entry.min.min(metric.value);
                entry.max = entry.max.max(metric.value);
            }

            for event in &frame.events {
                *events.entry(event.severity).or_insert(0) += 1;
            }
        }

        if frame_count == 0 {
            return None;
        }

        let span_seconds = match (earliest_ts, latest_ts) {
            (Some(start), Some(end)) => end
                .duration_since(start)
                .or_else(|_| start.duration_since(end))
                .map(|duration| duration.as_secs_f64())
                .unwrap_or(0.0),
            _ => 0.0,
        };

        let metrics = accumulators
            .into_iter()
            .map(|(name, acc)| MetricAggregate::from_accumulator(name, acc))
            .collect();

        Some(DashboardSummary {
            frame_count,
            span_seconds,
            metrics,
            events,
        })
    }
}

/// Helper that materialises frames from streaming metric updates.
#[derive(Debug)]
pub struct DashboardBuilder {
    metrics: Vec<DashboardMetric>,
    events: Vec<DashboardEvent>,
    start: SystemTime,
}

impl Default for DashboardBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DashboardBuilder {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            events: Vec::new(),
            start: SystemTime::now(),
        }
    }

    pub fn with_start(mut self, start: SystemTime) -> Self {
        self.start = start;
        self
    }

    pub fn record_metric(&mut self, metric: DashboardMetric) {
        self.metrics.push(metric);
    }

    pub fn record_event(&mut self, message: impl Into<String>, severity: EventSeverity) {
        self.events.push(DashboardEvent {
            message: message.into(),
            severity,
        });
    }

    pub fn finish(mut self) -> DashboardFrame {
        let mut frame = DashboardFrame::new(self.start);
        frame.metrics.append(&mut self.metrics);
        frame.events.append(&mut self.events);
        frame
    }

    pub fn finish_with_latency(self) -> DashboardFrame {
        let elapsed = self
            .start
            .elapsed()
            .unwrap_or_else(|_| Duration::from_secs(0));
        let mut frame = self.finish();
        frame.metrics.push(
            DashboardMetric::new("latency_ms", elapsed.as_secs_f64() * 1000.0).with_unit("ms"),
        );
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_retains_latest_frame() {
        let mut ring = DashboardRing::new(2);
        let mut frame = DashboardFrame::new(SystemTime::now());
        frame.push_metric(DashboardMetric::new("energy", 1.0));
        ring.push(frame);
        assert!(ring.latest().is_some());
    }

    #[test]
    fn ring_summarises_recent_frames() {
        let mut ring = DashboardRing::new(8);
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1000);

        let mut first = DashboardFrame::new(base);
        first.push_metric(DashboardMetric::new("energy", 1.0));
        first.push_metric(DashboardMetric::new("latency_ms", 10.0));
        first.push_event(DashboardEvent {
            message: "warmup".to_string(),
            severity: EventSeverity::Info,
        });
        ring.push(first);

        let mut second = DashboardFrame::new(base + Duration::from_millis(500));
        second.push_metric(DashboardMetric::new("energy", 3.0));
        second.push_metric(DashboardMetric::new("latency_ms", 30.0));
        second.push_event(DashboardEvent {
            message: "lag".to_string(),
            severity: EventSeverity::Warning,
        });
        ring.push(second);

        let summary = ring.summarize(None).expect("summary should exist");
        assert_eq!(summary.frame_count, 2);
        assert!(summary.span_seconds >= 0.5);
        assert_eq!(summary.events.get(&EventSeverity::Info), Some(&1));
        assert_eq!(summary.events.get(&EventSeverity::Warning), Some(&1));

        let energy = summary
            .metrics
            .iter()
            .find(|metric| metric.name == "energy")
            .expect("energy metric present");
        assert_eq!(energy.samples, 2);
        assert!((energy.mean - 2.0).abs() < 1e-6);
        assert_eq!(energy.latest, 3.0);

        let latency_latest_only = ring
            .summarize(Some(1))
            .expect("summary should exist")
            .metrics
            .into_iter()
            .find(|metric| metric.name == "latency_ms")
            .expect("latency metric present");
        assert_eq!(latency_latest_only.samples, 1);
        assert_eq!(latency_latest_only.latest, 30.0);
        assert_eq!(latency_latest_only.min, 30.0);
        assert_eq!(latency_latest_only.max, 30.0);
    }

    #[test]
    fn builder_materialises_latency_metric() {
        let builder = DashboardBuilder::new();
        let frame = builder.finish_with_latency();
        assert!(frame
            .metrics
            .iter()
            .any(|metric| metric.name == "latency_ms"));
    }
}
