// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// =============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// =============================================================================

//! Atlas projections stitching timeline telemetry with maintainer diagnostics.

use super::chrono::{ChronoHarmonics, ChronoSummary};
use super::maintainer::MaintainerStatus;
use std::collections::BTreeMap;

/// Named scalar surfaced through the atlas projection.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasMetric {
    /// Human-readable identifier for the metric.
    pub name: String,
    /// Scalar value carried by the metric.
    pub value: f32,
    /// Optional district categorisation supplied by the producer.
    pub district: Option<String>,
}

impl AtlasMetric {
    /// Creates a new metric when the value is finite.
    pub fn new(name: impl Into<String>, value: f32) -> Option<Self> {
        if value.is_finite() {
            Some(Self {
                name: name.into(),
                value,
                district: None,
            })
        } else {
            None
        }
    }

    /// Creates a new metric with an explicit district when the value is finite.
    pub fn with_district(
        name: impl Into<String>,
        value: f32,
        district: impl Into<String>,
    ) -> Option<Self> {
        if value.is_finite() {
            Some(Self {
                name: name.into(),
                value,
                district: Some(district.into()),
            })
        } else {
            None
        }
    }

    /// Returns the district when supplied by the producer.
    pub fn district(&self) -> Option<&str> {
        self.district.as_deref()
    }
}

/// Fragment of atlas state produced by one subsystem.
#[derive(Clone, Debug, Default)]
pub struct AtlasFragment {
    /// Optional timestamp associated with the fragment.
    pub timestamp: Option<f32>,
    /// Optional chrono summary captured by the fragment.
    pub summary: Option<ChronoSummary>,
    /// Optional harmonic analysis paired with the summary.
    pub harmonics: Option<ChronoHarmonics>,
    /// Support contributed by the fragment's producer.
    pub loop_support: Option<f32>,
    /// Optional collapse total conveyed by the fragment.
    pub collapse_total: Option<f32>,
    /// Optional Z-space control signal routed through the fragment.
    pub z_signal: Option<f32>,
    /// Optional SpiralK script hint attached to the fragment.
    pub script_hint: Option<String>,
    /// Maintainer status, if the fragment carried one.
    pub maintainer_status: Option<MaintainerStatus>,
    /// Human-readable maintainer diagnostic, if present.
    pub maintainer_diagnostic: Option<String>,
    /// Suggested geometry clamp from maintainer analysis, if any.
    pub suggested_max_scale: Option<f32>,
    /// Suggested Leech pressure bump from maintainer analysis, if any.
    pub suggested_pressure: Option<f32>,
    /// Additional scalar metrics emitted alongside the fragment.
    pub metrics: Vec<AtlasMetric>,
    /// Free-form notes describing the fragment provenance.
    pub notes: Vec<String>,
}

impl AtlasFragment {
    /// Creates a new, empty fragment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when the fragment does not contain any payload.
    pub fn is_empty(&self) -> bool {
        self.timestamp.is_none()
            && self.summary.is_none()
            && self.harmonics.is_none()
            && self.loop_support.is_none()
            && self.collapse_total.is_none()
            && self.z_signal.is_none()
            && self.script_hint.is_none()
            && self.maintainer_status.is_none()
            && self.maintainer_diagnostic.is_none()
            && self.suggested_max_scale.is_none()
            && self.suggested_pressure.is_none()
            && self.metrics.is_empty()
            && self.notes.is_empty()
    }

    /// Pushes a metric onto the fragment when the value is finite.
    pub fn push_metric(&mut self, name: impl Into<String>, value: f32) {
        if let Some(metric) = AtlasMetric::new(name, value) {
            self.metrics.push(metric);
        }
    }

    /// Pushes a metric with an explicit district when the value is finite.
    pub fn push_metric_with_district(
        &mut self,
        name: impl Into<String>,
        value: f32,
        district: impl Into<String>,
    ) {
        if let Some(metric) = AtlasMetric::with_district(name, value, district) {
            self.metrics.push(metric);
        }
    }

    /// Appends a note to the fragment provenance trail.
    pub fn push_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }
}

/// Aggregated atlas state that can be replayed by clients.
#[derive(Clone, Debug, Default)]
pub struct AtlasFrame {
    /// Latest timestamp captured across all fragments.
    pub timestamp: f32,
    /// Latest chrono summary applied to the frame.
    pub chrono_summary: Option<ChronoSummary>,
    /// Latest harmonic snapshot applied to the frame.
    pub harmonics: Option<ChronoHarmonics>,
    /// Accumulated loop support collected from fragments.
    pub loop_support: f32,
    /// Last collapse total reported into the atlas.
    pub collapse_total: Option<f32>,
    /// Last Z-space control signal reported into the atlas.
    pub z_signal: Option<f32>,
    /// Latest SpiralK script hint stitched into the atlas.
    pub script_hint: Option<String>,
    /// Latest maintainer status routed into the frame.
    pub maintainer_status: Option<MaintainerStatus>,
    /// Latest maintainer diagnostic carried with the frame.
    pub maintainer_diagnostic: Option<String>,
    /// Most recent geometry clamp recommendation.
    pub suggested_max_scale: Option<f32>,
    /// Most recent Leech pressure recommendation.
    pub suggested_pressure: Option<f32>,
    /// Scalar metrics accumulated across fragments.
    pub metrics: Vec<AtlasMetric>,
    /// Provenance notes collected from fragments.
    pub notes: Vec<String>,
}

impl AtlasFrame {
    /// Creates a frame initialised at the provided timestamp.
    pub fn new(timestamp: f32) -> Self {
        let timestamp = if timestamp.is_finite() {
            timestamp
        } else {
            0.0
        };
        Self {
            timestamp,
            ..Default::default()
        }
    }

    /// Creates a frame from a fragment when it carries any payload.
    pub fn from_fragment(fragment: AtlasFragment) -> Option<Self> {
        if fragment.is_empty() {
            return None;
        }
        let mut frame = Self::default();
        frame.merge_fragment(fragment);
        if frame.timestamp <= 0.0 {
            None
        } else {
            Some(frame)
        }
    }

    /// Merges a fragment into the frame.
    pub fn merge_fragment(&mut self, fragment: AtlasFragment) {
        if fragment.is_empty() {
            return;
        }
        let timestamp = fragment.timestamp.or_else(|| {
            fragment
                .summary
                .as_ref()
                .map(|summary| summary.latest_timestamp)
        });
        if let Some(ts) = timestamp {
            if ts.is_finite() {
                if self.timestamp <= 0.0 {
                    self.timestamp = ts.max(f32::EPSILON);
                } else {
                    self.timestamp = self.timestamp.max(ts);
                }
            }
        }
        if let Some(summary) = fragment.summary {
            self.chrono_summary = Some(summary);
        }
        if let Some(harmonics) = fragment.harmonics {
            self.harmonics = Some(harmonics);
        }
        if let Some(support) = fragment.loop_support {
            if support.is_finite() {
                self.loop_support += support.max(0.0);
            }
        }
        if let Some(total) = fragment.collapse_total {
            if total.is_finite() {
                self.collapse_total = Some(total);
            }
        }
        if let Some(z) = fragment.z_signal {
            if z.is_finite() {
                self.z_signal = Some(z);
            }
        }
        if let Some(script) = fragment.script_hint {
            if !script.is_empty() {
                self.script_hint = Some(script);
            }
        }
        if let Some(status) = fragment.maintainer_status {
            self.maintainer_status = Some(status);
        }
        if let Some(diagnostic) = fragment.maintainer_diagnostic {
            if !diagnostic.is_empty() {
                self.maintainer_diagnostic = Some(diagnostic);
            }
        }
        if let Some(clamp) = fragment.suggested_max_scale {
            if clamp.is_finite() {
                self.suggested_max_scale = Some(clamp);
            }
        }
        if let Some(pressure) = fragment.suggested_pressure {
            if pressure.is_finite() {
                self.suggested_pressure = Some(pressure);
            }
        }
        if !fragment.metrics.is_empty() {
            self.metrics.extend(fragment.metrics.into_iter());
        }
        if !fragment.notes.is_empty() {
            self.notes.extend(fragment.notes.into_iter());
        }
    }

    /// Groups metrics into named districts following the SpiralTorch atlas map.
    pub fn districts(&self) -> Vec<AtlasDistrict> {
        if self.metrics.is_empty() {
            return Vec::new();
        }
        let mut buckets: BTreeMap<String, Vec<AtlasMetric>> = BTreeMap::new();
        for metric in &self.metrics {
            let district = metric
                .district()
                .map(|name| name.to_string())
                .unwrap_or_else(|| infer_district(&metric.name).to_string());
            buckets.entry(district).or_default().push(metric.clone());
        }
        buckets
            .into_iter()
            .map(|(name, metrics)| AtlasDistrict::from_metrics(name, metrics))
            .filter(|district| !district.metrics.is_empty())
            .collect()
    }
}

/// Aggregated atlas route storing chronological frames for replay.
#[derive(Clone, Debug, Default)]
pub struct AtlasRoute {
    /// Chronological atlas frames retained in the route.
    pub frames: Vec<AtlasFrame>,
}

impl AtlasRoute {
    /// Creates a new, empty route.
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Returns true when the route does not contain any frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the number of frames stored in the route.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns the latest frame when available.
    pub fn latest(&self) -> Option<&AtlasFrame> {
        self.frames.last()
    }

    /// Appends a frame while keeping the total number of frames bounded.
    pub fn push_bounded(&mut self, frame: AtlasFrame, capacity: usize) {
        if frame.timestamp <= 0.0 {
            return;
        }
        self.frames.push(frame);
        if capacity > 0 && self.frames.len() > capacity {
            let overflow = self.frames.len() - capacity;
            self.frames.drain(0..overflow);
        }
    }

    /// Summarises the stored frames into district-level statistics and loopback hints.
    pub fn summary(&self) -> AtlasRouteSummary {
        let mut summary = AtlasRouteSummary::default();
        summary.frames = self.frames.len();
        if self.frames.is_empty() {
            return summary;
        }
        summary.latest_timestamp = self
            .frames
            .last()
            .map(|frame| frame.timestamp)
            .unwrap_or(0.0);
        let mut loop_total = 0.0;
        let mut loop_sq_total = 0.0;
        let mut loop_samples = 0usize;
        let mut district_map: BTreeMap<String, DistrictAccumulator> = BTreeMap::new();
        let mut first_collapse = None;
        let mut first_z_signal = None;
        for frame in &self.frames {
            let support = frame.loop_support.max(0.0);
            loop_total += support;
            loop_sq_total += support * support;
            loop_samples += 1;
            if let Some(total) = frame.collapse_total {
                if total.is_finite() {
                    if first_collapse.is_none() {
                        first_collapse = Some(total);
                    }
        let mut district_map: BTreeMap<String, DistrictAccumulator> = BTreeMap::new();
        for frame in &self.frames {
            loop_total += frame.loop_support.max(0.0);
            if let Some(total) = frame.collapse_total {
                if total.is_finite() {
                    summary.latest_collapse_total = Some(total);
                }
            }
            if let Some(z) = frame.z_signal {
                if z.is_finite() {
                    if first_z_signal.is_none() {
                        first_z_signal = Some(z);
                    }
                    summary.latest_z_signal = Some(z);
                }
            }
            if let Some(status) = frame.maintainer_status {
                summary.maintainer_status = Some(status);
            }
            if let Some(diagnostic) = frame.maintainer_diagnostic.as_ref() {
                if !diagnostic.is_empty() {
                    summary.maintainer_diagnostic = Some(diagnostic.clone());
                }
            }
            if let Some(clamp) = frame.suggested_max_scale {
                if clamp.is_finite() {
                    summary.suggested_max_scale = Some(clamp);
                }
            }
            if let Some(pressure) = frame.suggested_pressure {
                if pressure.is_finite() {
                    summary.suggested_pressure = Some(pressure);
                }
            }
            if let Some(script) = frame.script_hint.as_ref() {
                if !script.is_empty() {
                    summary.script_hint = Some(script.clone());
                }
            }
            for district in frame.districts() {
                let entry = district_map
                    .entry(district.name.clone())
                    .or_insert_with(DistrictAccumulator::new);
                entry.push(district.mean);
            }
        }
        if summary.frames > 0 {
            summary.mean_loop_support = loop_total / summary.frames as f32;
        }
        if loop_samples > 0 {
            let mean = loop_total / loop_samples as f32;
            let variance = (loop_sq_total / loop_samples as f32) - mean * mean;
            summary.loop_std = variance.max(0.0).sqrt();
        }
        if let (Some(first), Some(latest)) = (first_collapse, summary.latest_collapse_total) {
            summary.collapse_trend = Some(latest - first);
        }
        if let (Some(first), Some(latest)) = (first_z_signal, summary.latest_z_signal) {
            summary.z_signal_trend = Some(latest - first);
        }
        summary.districts = district_map
            .into_iter()
            .map(|(name, accumulator)| accumulator.into_summary(name))
            .filter(|district| district.coverage > 0)
            .collect();
        summary.districts.sort_by(|a, b| {
            b.coverage
                .cmp(&a.coverage)
                .then_with(|| a.name.cmp(&b.name))
        });
        summary
    }
}

#[derive(Clone, Debug, Default)]
pub struct AtlasRouteSummary {
    /// Number of frames retained in the route at the time of the summary.
    pub frames: usize,
    /// Timestamp of the latest frame contributing to the summary.
    pub latest_timestamp: f32,
    /// Mean loop support accumulated across the route.
    pub mean_loop_support: f32,
    /// Standard deviation of loop support across the retained frames.
    pub loop_std: f32,
    /// Latest collapse total observed within the route.
    pub latest_collapse_total: Option<f32>,
    /// Signed drift between the first and latest collapse totals.
    pub collapse_trend: Option<f32>,
    /// Latest Z-space control signal observed within the route.
    pub latest_z_signal: Option<f32>,
    /// Signed drift between the first and latest Z-space control signals.
    pub z_signal_trend: Option<f32>,
    /// Last maintainer status routed through the atlas route.
    pub maintainer_status: Option<MaintainerStatus>,
    /// Last maintainer diagnostic routed through the atlas route.
    pub maintainer_diagnostic: Option<String>,
    /// Most recent clamp recommendation encountered in the route.
    pub suggested_max_scale: Option<f32>,
    /// Most recent Leech pressure recommendation encountered in the route.
    pub suggested_pressure: Option<f32>,
    /// Latest SpiralK script hint emitted by the route.
    pub script_hint: Option<String>,
    /// District activity summaries accumulated across the route.
    pub districts: Vec<AtlasDistrictSummary>,
}

impl AtlasRouteSummary {
    /// Returns true when the summary did not gather any frames.
    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct AtlasDistrictSummary {
    /// Human readable name of the district.
    pub name: String,
    /// Number of frames contributing to the district statistics.
    pub coverage: usize,
    /// Average district activity across the covered frames.
    pub mean: f32,
    /// Latest district activity value encountered in the route.
    pub latest: f32,
    /// Minimum district activity observed in the route.
    pub min: f32,
    /// Maximum district activity observed in the route.
    pub max: f32,
    /// Signed change between the latest and first observation.
    pub delta: f32,
    /// Standard deviation of the district activity across observations.
    pub std_dev: f32,
}

#[derive(Clone, Debug, Default)]
struct DistrictAccumulator {
    sum: f32,
    sum_sq: f32,
    min: f32,
    max: f32,
    first: Option<f32>,
    last: f32,
    count: usize,
}

impl DistrictAccumulator {
    fn new() -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            first: None,
            last: 0.0,
            count: 0,
        }
    }

    fn push(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        if self.first.is_none() {
            self.first = Some(value);
        }
        self.last = value;
    }

    fn into_summary(self, name: String) -> AtlasDistrictSummary {
        if self.count == 0 {
            return AtlasDistrictSummary {
                name,
                ..Default::default()
            };
        }
        let mean = self.sum / self.count as f32;
        let variance = (self.sum_sq / self.count as f32) - mean * mean;
        let first = self.first.unwrap_or(self.last);
        AtlasDistrictSummary {
            name,
            coverage: self.count,
            mean,
            latest: self.last,
            min: if self.min.is_finite() { self.min } else { mean },
            max: if self.max.is_finite() { self.max } else { mean },
            delta: self.last - first,
            std_dev: variance.max(0.0).sqrt(),
        }
    }
}

/// Atlas district representing a logical SpiralTorch layer.
#[derive(Clone, Debug, Default)]
pub struct AtlasDistrict {
    /// Human readable name of the district.
    pub name: String,
    /// Metrics that contributed to the district activity.
    pub metrics: Vec<AtlasMetric>,
    /// Mean activity value across the district metrics.
    pub mean: f32,
    /// Range spanned by the district metrics.
    pub span: f32,
}

impl AtlasDistrict {
    /// Builds a district from aggregated metrics.
    fn from_metrics(name: String, metrics: Vec<AtlasMetric>) -> Self {
        let mut mean = 0.0;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let len = metrics.len() as f32;
        if len > 0.0 {
            for metric in &metrics {
                mean += metric.value;
                min = min.min(metric.value);
                max = max.max(metric.value);
            }
            mean /= len;
        }
        let span = if metrics.len() > 1 { max - min } else { 0.0 };
        Self {
            name,
            metrics,
            mean,
            span,
        }
    }
}

fn infer_district(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    let token = lower
        .split(|c| c == '.' || c == ':' || c == '/' || c == '-')
        .next()
        .unwrap_or("");
    match token {
        "py" | "python" | "bindings" | "session" | "timeline" => "Surface",
        "trainer" | "maintainer" | "atlas" | "loop" | "chrono" | "policy" | "resonator" => {
            "Concourse"
        }
        "tensor" | "backend" | "core" | "z" | "collapse" | "geometry" | "kdsl" => "Substrate",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_frame_groups_metrics_into_districts() {
        let mut fragment = AtlasFragment::new();
        fragment.timestamp = Some(1.0);
        fragment.push_metric("py.bridge.latency", 0.2);
        fragment.push_metric_with_district("trainer.loop.energy", 0.8, "Concourse");
        fragment.push_metric("tensor.backend.util", 0.6);
        let mut frame = AtlasFrame::new(1.0);
        frame.merge_fragment(fragment);
        let districts = frame.districts();
        assert_eq!(districts.len(), 3);
        let surface = districts
            .iter()
            .find(|district| district.name == "Surface")
            .expect("surface district");
        assert!((surface.mean - 0.2).abs() <= f32::EPSILON);
        let concourse = districts
            .iter()
            .find(|district| district.name == "Concourse")
            .expect("concourse district");
        assert_eq!(concourse.metrics.len(), 1);
        let substrate = districts
            .iter()
            .find(|district| district.name == "Substrate")
            .expect("substrate district");
        assert!(substrate.span <= f32::EPSILON);
    }

    #[test]
    fn atlas_route_trims_capacity() {
        let mut route = AtlasRoute::new();
        for idx in 0..5 {
            let mut frame = AtlasFrame::new((idx + 1) as f32);
            frame.loop_support = idx as f32;
            route.push_bounded(frame, 3);
        }
        assert_eq!(route.len(), 3);
        assert_eq!(route.frames[0].timestamp, 3.0);
        assert_eq!(route.latest().unwrap().timestamp, 5.0);
    }

    #[test]
    fn atlas_route_summary_tracks_district_trends() {
        let mut route = AtlasRoute::new();
        for idx in 0..4 {
            let mut fragment = AtlasFragment::new();
            fragment.timestamp = Some((idx + 1) as f32);
            fragment.push_metric_with_district("session.surface.latency", idx as f32, "Surface");
            fragment.push_metric_with_district(
                "trainer.loop.energy",
                idx as f32 + 1.0,
                "Concourse",
            );
            fragment.push_metric_with_district(
                "tensor.backend.util",
                0.5 + idx as f32 * 0.1,
                "Substrate",
            );
            let mut frame = AtlasFrame::new((idx + 1) as f32);
            frame.loop_support = (idx as f32) * 0.5;
            frame.collapse_total = Some(0.5 + idx as f32 * 0.1);
            frame.z_signal = Some(0.2 + idx as f32 * 0.05);
            frame.merge_fragment(fragment);
            route.push_bounded(frame, usize::MAX);
        }
        let summary = route.summary();
        assert_eq!(summary.frames, 4);
        assert!(summary.latest_timestamp >= 4.0 - f32::EPSILON);
        assert!(summary.mean_loop_support > 0.0);
        assert!(summary.loop_std > 0.0);
        assert!(!summary.districts.is_empty());
        assert!(summary
            .collapse_trend
            .expect("collapse trend")
            .is_sign_positive());
        assert!(summary.z_signal_trend.expect("z trend").is_sign_positive());
        let surface = summary
            .districts
            .iter()
            .find(|district| district.name == "Surface")
            .expect("surface summary");
        assert_eq!(surface.coverage, 4);
        assert!((surface.latest - 3.0).abs() <= f32::EPSILON);
        assert!((surface.delta - 3.0).abs() <= f32::EPSILON);
        assert!(surface.std_dev > 0.0);
    }
}
