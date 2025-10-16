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

//! Chronological telemetry stream that captures how resonance geometry evolves in time.
//!
//! The [`ChronoTimeline`] collects [`ChronoFrame`] samples so higher level tooling can
//! reason about curvature drift, energy decay, and other temporal dynamics emitted by the
//! Z-space resonance pipeline. Frames are stored in insertion order and behave like a
//! fixed-size ring buffer so callers can stream long-lived sessions without unbounded
//! memory growth.

use core::f32;
use std::collections::VecDeque;

/// Aggregated summary computed over a slice of chrono frames.
#[derive(Clone, Debug)]
pub struct ChronoSummary {
    /// Number of frames that participated in the summary.
    pub frames: usize,
    /// Total duration covered by the analysed frames.
    pub duration: f32,
    /// Timestamp of the most recent frame.
    pub latest_timestamp: f32,
    /// Mean signed curvature drift across the window.
    pub mean_drift: f32,
    /// Mean absolute curvature drift across the window.
    pub mean_abs_drift: f32,
    /// Standard deviation of the signed curvature drift.
    pub drift_std: f32,
    /// Mean total resonance energy across the window.
    pub mean_energy: f32,
    /// Standard deviation of the resonance energy.
    pub energy_std: f32,
    /// Mean energy decay rate (positive = decaying, negative = growing).
    pub mean_decay: f32,
    /// Minimum total energy observed within the window.
    pub min_energy: f32,
    /// Maximum total energy observed within the window.
    pub max_energy: f32,
}

impl ChronoSummary {
    /// Computes a summary from the provided frames, returning `None` when empty.
    pub fn from_frames(frames: &[ChronoFrame]) -> Option<Self> {
        Self::from_iter(frames.iter())
    }

    /// Computes a summary from the provided iterator of frames.
    pub fn from_iter<'a, I>(frames: I) -> Option<Self>
    where
        I: IntoIterator<Item = &'a ChronoFrame>,
    {
        let mut duration = 0.0f32;
        let mut sum_abs_drift = 0.0f32;
        let mut sum_drift = 0.0f32;
        let mut sum_drift_sq = 0.0f32;
        let mut sum_energy = 0.0f32;
        let mut sum_energy_sq = 0.0f32;
        let mut sum_decay = 0.0f32;
        let mut min_energy = f32::INFINITY;
        let mut max_energy = f32::NEG_INFINITY;
        let mut count = 0usize;
        let mut latest_timestamp = 0.0f32;
        for frame in frames.into_iter() {
            let dt = if frame.dt.is_finite() && frame.dt > 0.0 {
                frame.dt
            } else {
                f32::EPSILON
            };
            duration += dt;
            let energy = frame.total_energy.max(0.0);
            min_energy = min_energy.min(energy);
            max_energy = max_energy.max(energy);
            sum_energy += energy;
            sum_energy_sq += energy * energy;
            let drift = frame.curvature_drift;
            sum_abs_drift += drift.abs();
            sum_drift += drift;
            sum_drift_sq += drift * drift;
            sum_decay += frame.energy_decay;
            latest_timestamp = frame.timestamp;
            count += 1;
        }
        if count == 0 {
            return None;
        }
        let count_f32 = count as f32;
        let mean_energy = sum_energy / count_f32;
        let mean_drift = sum_drift / count_f32;
        let mean_abs_drift = sum_abs_drift / count_f32;
        let energy_var = (sum_energy_sq / count_f32) - mean_energy.powi(2);
        let drift_var = (sum_drift_sq / count_f32) - mean_drift.powi(2);
        Some(Self {
            frames: count,
            duration,
            latest_timestamp,
            mean_drift,
            mean_abs_drift,
            drift_std: drift_var.max(0.0).sqrt(),
            mean_energy,
            energy_std: energy_var.max(0.0).sqrt(),
            mean_decay: sum_decay / count_f32,
            min_energy: if min_energy.is_finite() {
                min_energy
            } else {
                0.0
            },
            max_energy: if max_energy.is_finite() {
                max_energy
            } else {
                0.0
            },
        })
    }
}

/// Snapshot capturing temporal diagnostics for a single time step.
#[derive(Clone, Debug)]
pub struct ChronoFrame {
    /// Monotonic index of the frame within the timeline.
    pub step: u64,
    /// Absolute timestamp accumulated by summing the supplied `dt`s.
    pub timestamp: f32,
    /// Duration in seconds (or caller-supplied units) represented by this frame.
    pub dt: f32,
    /// Observed curvature reconstructed from the resonance flow.
    pub observed_curvature: f32,
    /// Difference between this frame's curvature and the previous frame.
    pub curvature_drift: f32,
    /// Aggregate resonance energy captured during the frame.
    pub total_energy: f32,
    /// Estimated decay rate of the aggregate energy relative to the previous frame.
    pub energy_decay: f32,
    /// Energy emitted by the homotopy flow.
    pub homotopy_energy: f32,
    /// Energy emitted by the functor linearisation.
    pub functor_energy: f32,
    /// Energy emitted by the recursive objective curve.
    pub recursive_energy: f32,
    /// Energy emitted by the \(\infty\)-tower projection.
    pub projection_energy: f32,
    /// Energy accumulated in the \(\infty\)-tower hierarchy.
    pub infinity_energy: f32,
}

impl ChronoFrame {
    /// Returns the ratio between homotopy energy and the aggregate energy.
    pub fn homotopy_ratio(&self) -> f32 {
        if self.total_energy <= f32::EPSILON {
            0.0
        } else {
            (self.homotopy_energy / self.total_energy).clamp(0.0, 1.0)
        }
    }

    /// Returns the ratio between functor energy and the aggregate energy.
    pub fn functor_ratio(&self) -> f32 {
        if self.total_energy <= f32::EPSILON {
            0.0
        } else {
            (self.functor_energy / self.total_energy).clamp(0.0, 1.0)
        }
    }

    /// Returns the ratio between recursive energy and the aggregate energy.
    pub fn recursive_ratio(&self) -> f32 {
        if self.total_energy <= f32::EPSILON {
            0.0
        } else {
            (self.recursive_energy / self.total_energy).clamp(0.0, 1.0)
        }
    }

    /// Returns the ratio between \(\infty\) projection energy and the aggregate energy.
    pub fn projection_ratio(&self) -> f32 {
        if self.total_energy <= f32::EPSILON {
            0.0
        } else {
            (self.projection_energy / self.total_energy).clamp(0.0, 1.0)
        }
    }

    /// Returns the ratio between \(\infty\) hierarchy energy and the aggregate energy.
    pub fn infinity_ratio(&self) -> f32 {
        if self.total_energy <= f32::EPSILON {
            0.0
        } else {
            (self.infinity_energy / self.total_energy).clamp(0.0, 1.0)
        }
    }
}

/// Rolling timeline that stores a bounded number of [`ChronoFrame`] samples.
#[derive(Clone, Debug)]
pub struct ChronoTimeline {
    frames: VecDeque<ChronoFrame>,
    capacity: usize,
    elapsed: f32,
    step: u64,
}

impl ChronoTimeline {
    /// Creates a new timeline with the provided maximum capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
            elapsed: 0.0,
            step: 0,
        }
    }

    /// Creates a new timeline with a default capacity of 256 frames.
    pub fn new() -> Self {
        Self::with_capacity(256)
    }

    /// Returns the configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of frames currently stored.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns true when no frames have been recorded.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Clears all recorded frames and resets the accumulated timestamp.
    pub fn reset(&mut self) {
        self.frames.clear();
        self.elapsed = 0.0;
        self.step = 0;
    }

    /// Returns an iterator over the stored frames in chronological order.
    pub fn frames(&self) -> impl Iterator<Item = &ChronoFrame> {
        self.frames.iter()
    }

    /// Returns the most recent frame if one has been recorded.
    pub fn latest(&self) -> Option<&ChronoFrame> {
        self.frames.back()
    }

    /// Records a new frame, computing drift metrics against the previously stored frame.
    pub fn record(&mut self, dt: f32, metrics: ResonanceTemporalMetrics) -> ChronoFrame {
        let safe_dt = if dt.is_finite() && dt > f32::EPSILON {
            dt
        } else {
            f32::EPSILON
        };
        let timestamp = self.elapsed + safe_dt;
        let step = self.step;
        let previous = self.frames.back();
        let curvature_drift = previous
            .map(|frame| metrics.observed_curvature - frame.observed_curvature)
            .unwrap_or(0.0);
        let energy_decay = previous
            .map(|frame| (frame.total_energy - metrics.total_energy) / safe_dt)
            .unwrap_or(0.0);
        let frame = ChronoFrame {
            step,
            timestamp,
            dt: safe_dt,
            observed_curvature: metrics.observed_curvature,
            curvature_drift,
            total_energy: metrics.total_energy,
            energy_decay,
            homotopy_energy: metrics.homotopy_energy,
            functor_energy: metrics.functor_energy,
            recursive_energy: metrics.recursive_energy,
            projection_energy: metrics.projection_energy,
            infinity_energy: metrics.infinity_energy,
        };
        if self.frames.len() == self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(frame.clone());
        self.elapsed = timestamp;
        self.step = step.saturating_add(1);
        frame
    }

    /// Returns a rolling summary computed over the most recent `window` frames.
    pub fn summarise(&self, window: usize) -> Option<ChronoSummary> {
        if self.frames.is_empty() {
            return None;
        }
        let window = window.max(1).min(self.frames.len());
        let skip = self.frames.len() - window;
        ChronoSummary::from_iter(self.frames.iter().skip(skip))
    }
}

impl Default for ChronoTimeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience bundle of resonance metrics used to create a [`ChronoFrame`].
#[derive(Clone, Debug)]
pub struct ResonanceTemporalMetrics {
    /// Observed curvature reconstructed from the resonance.
    pub observed_curvature: f32,
    /// Total resonance energy captured during the time step.
    pub total_energy: f32,
    /// Homotopy contribution to the total energy.
    pub homotopy_energy: f32,
    /// Functor contribution to the total energy.
    pub functor_energy: f32,
    /// Recursive contribution to the total energy.
    pub recursive_energy: f32,
    /// Projection contribution to the total energy.
    pub projection_energy: f32,
    /// \(\infty\)-tower contribution to the total energy.
    pub infinity_energy: f32,
}

impl ResonanceTemporalMetrics {
    /// Normalises any non-finite component to zero so the timeline remains stable.
    pub fn sanitise(mut self) -> Self {
        if !self.observed_curvature.is_finite() {
            self.observed_curvature = 0.0;
        }
        if !self.total_energy.is_finite() {
            self.total_energy = 0.0;
        }
        if !self.homotopy_energy.is_finite() {
            self.homotopy_energy = 0.0;
        }
        if !self.functor_energy.is_finite() {
            self.functor_energy = 0.0;
        }
        if !self.recursive_energy.is_finite() {
            self.recursive_energy = 0.0;
        }
        if !self.projection_energy.is_finite() {
            self.projection_energy = 0.0;
        }
        if !self.infinity_energy.is_finite() {
            self.infinity_energy = 0.0;
        }
        if self.total_energy <= f32::EPSILON {
            self.total_energy = 0.0;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_records_and_limits_capacity() {
        let mut timeline = ChronoTimeline::with_capacity(2);
        let metrics = ResonanceTemporalMetrics {
            observed_curvature: -1.0,
            total_energy: 3.0,
            homotopy_energy: 1.0,
            functor_energy: 1.0,
            recursive_energy: 0.5,
            projection_energy: 0.3,
            infinity_energy: 0.2,
        };
        let first = timeline.record(0.1, metrics.clone().sanitise());
        assert_eq!(first.step, 0);
        assert_eq!(timeline.len(), 1);
        let second = timeline.record(0.1, metrics.clone().sanitise());
        assert_eq!(second.step, 1);
        assert_eq!(timeline.len(), 2);
        let third = timeline.record(0.1, metrics.clone().sanitise());
        assert_eq!(third.step, 2);
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline.frames().next().unwrap().step, 1);
        assert_eq!(timeline.latest().unwrap().step, 2);
    }

    #[test]
    fn ratios_return_zero_when_energy_vanishes() {
        let frame = ChronoFrame {
            step: 0,
            timestamp: 0.1,
            dt: 0.1,
            observed_curvature: -1.0,
            curvature_drift: 0.0,
            total_energy: 0.0,
            energy_decay: 0.0,
            homotopy_energy: 0.0,
            functor_energy: 0.0,
            recursive_energy: 0.0,
            projection_energy: 0.0,
            infinity_energy: 0.0,
        };
        assert_eq!(frame.homotopy_ratio(), 0.0);
        assert_eq!(frame.functor_ratio(), 0.0);
        assert_eq!(frame.recursive_ratio(), 0.0);
        assert_eq!(frame.projection_ratio(), 0.0);
        assert_eq!(frame.infinity_ratio(), 0.0);
    }

    #[test]
    fn summary_computes_expected_statistics() {
        let frames = vec![
            ChronoFrame {
                step: 0,
                timestamp: 0.1,
                dt: 0.1,
                observed_curvature: -1.0,
                curvature_drift: 0.2,
                total_energy: 1.0,
                energy_decay: -0.05,
                homotopy_energy: 0.4,
                functor_energy: 0.2,
                recursive_energy: 0.2,
                projection_energy: 0.1,
                infinity_energy: 0.1,
            },
            ChronoFrame {
                step: 1,
                timestamp: 0.2,
                dt: 0.1,
                observed_curvature: -1.05,
                curvature_drift: -0.1,
                total_energy: 1.2,
                energy_decay: -0.02,
                homotopy_energy: 0.48,
                functor_energy: 0.24,
                recursive_energy: 0.24,
                projection_energy: 0.12,
                infinity_energy: 0.12,
            },
            ChronoFrame {
                step: 2,
                timestamp: 0.3,
                dt: 0.1,
                observed_curvature: -1.1,
                curvature_drift: 0.05,
                total_energy: 0.9,
                energy_decay: 0.04,
                homotopy_energy: 0.36,
                functor_energy: 0.18,
                recursive_energy: 0.18,
                projection_energy: 0.09,
                infinity_energy: 0.09,
            },
        ];
        let summary = ChronoSummary::from_frames(&frames).unwrap();
        assert_eq!(summary.frames, 3);
        assert!((summary.duration - 0.3).abs() < 1e-6);
        assert_eq!(summary.latest_timestamp, 0.3);
        assert!(summary.mean_energy > 0.0);
        assert!(summary.energy_std >= 0.0);
        assert!(summary.mean_abs_drift > 0.0);
        assert!(summary.drift_std >= 0.0);
        assert!(summary.min_energy <= summary.max_energy);
    }

    #[test]
    fn timeline_summarise_limits_window() {
        let mut timeline = ChronoTimeline::with_capacity(8);
        let metrics = ResonanceTemporalMetrics {
            observed_curvature: -1.0,
            total_energy: 1.0,
            homotopy_energy: 0.4,
            functor_energy: 0.2,
            recursive_energy: 0.2,
            projection_energy: 0.1,
            infinity_energy: 0.1,
        };
        for step in 0..5 {
            let mut metrics = metrics.clone();
            metrics.total_energy += step as f32 * 0.1;
            timeline.record(0.1, metrics);
        }
        let summary = timeline.summarise(2).unwrap();
        assert_eq!(summary.frames, 2);
        assert!((summary.duration - 0.2).abs() < 1e-6);
        assert!(summary.latest_timestamp > 0.0);
    }
}
