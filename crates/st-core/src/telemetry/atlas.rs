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

/// Named scalar surfaced through the atlas projection.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasMetric {
    /// Human-readable identifier for the metric.
    pub name: String,
    /// Scalar value carried by the metric.
    pub value: f32,
}

impl AtlasMetric {
    /// Creates a new metric when the value is finite.
    pub fn new(name: impl Into<String>, value: f32) -> Option<Self> {
        if value.is_finite() {
            Some(Self {
                name: name.into(),
                value,
            })
        } else {
            None
        }
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
}
