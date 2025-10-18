// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
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

    pub fn iter(&self) -> impl Iterator<Item = &DashboardFrame> {
        self.frames.iter()
    }
}

/// Helper that materialises frames from streaming metric updates.
#[derive(Debug)]
pub struct DashboardBuilder {
    metrics: Vec<DashboardMetric>,
    events: Vec<DashboardEvent>,
    start: SystemTime,
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
    fn builder_materialises_latency_metric() {
        let builder = DashboardBuilder::new();
        let frame = builder.finish_with_latency();
        assert!(frame
            .metrics
            .iter()
            .any(|metric| metric.name == "latency_ms"));
    }
}
