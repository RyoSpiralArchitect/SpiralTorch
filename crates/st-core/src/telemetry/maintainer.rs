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

//! Maintainer heuristics that monitor temporal resonance and trigger self-rewrite flows.
//!
//! The maintainer inspects [`ChronoFrame`](super::chrono::ChronoFrame) samples to estimate
//! curvature jitter, energy trends, and dormancy so higher level tooling can decide when to
//! tighten geometry clamps or escalate into a full self-rewrite cycle.

#[cfg(feature = "kdsl")]
use super::chrono::ChronoLoopSignal;
use super::chrono::{ChronoFrame, ChronoHarmonics, ChronoPeak, ChronoSummary};

/// Indicates the level of maintenance required to stabilise the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaintainerStatus {
    /// Temporal telemetry looks healthy; no action required.
    Stable,
    /// The resonance is almost quiescent and can be left alone.
    Dormant,
    /// Clamp geometry feedback before jitter grows out of control.
    Clamp,
    /// Energy growth or runaway curvature requires a self-rewrite.
    Rewrite,
}

impl MaintainerStatus {
    /// Returns the canonical string label for the status.
    pub fn as_str(&self) -> &'static str {
        match self {
            MaintainerStatus::Stable => "stable",
            MaintainerStatus::Dormant => "dormant",
            MaintainerStatus::Clamp => "clamp",
            MaintainerStatus::Rewrite => "rewrite",
        }
    }
}

/// Tunable thresholds used to analyse the temporal telemetry stream.
#[derive(Clone, Debug)]
pub struct MaintainerConfig {
    /// Maximum tolerated average absolute curvature drift before clamping is recommended.
    pub jitter_threshold: f32,
    /// Maximum tolerated negative energy decay (i.e. growth) before rewrite escalation.
    pub growth_threshold: f32,
    /// Energy floor below which the timeline is considered dormant.
    pub energy_floor: f32,
    /// Lower bound for the geometry max-scale clamp.
    pub clamp_min: f32,
    /// Upper bound for the geometry max-scale clamp.
    pub clamp_max: f32,
    /// Increment applied to Leech density pressure when instability is detected.
    pub pressure_step: f32,
    /// Number of most recent frames to inspect when generating a report.
    pub window: usize,
}

impl Default for MaintainerConfig {
    fn default() -> Self {
        Self {
            jitter_threshold: 0.35,
            growth_threshold: 0.05,
            energy_floor: 1e-3,
            clamp_min: 2.0,
            clamp_max: 3.0,
            pressure_step: 0.12,
            window: 32,
        }
    }
}

impl MaintainerConfig {
    /// Ensures the clamp bounds are ordered and finite.
    pub fn sanitise(mut self) -> Self {
        if !self.jitter_threshold.is_finite() || self.jitter_threshold <= 0.0 {
            self.jitter_threshold = 0.35;
        }
        if !self.growth_threshold.is_finite() || self.growth_threshold <= 0.0 {
            self.growth_threshold = 0.05;
        }
        if !self.energy_floor.is_finite() || self.energy_floor < 0.0 {
            self.energy_floor = 1e-3;
        }
        if !self.clamp_min.is_finite() {
            self.clamp_min = 2.0;
        }
        if !self.clamp_max.is_finite() {
            self.clamp_max = 3.0;
        }
        if self.clamp_min > self.clamp_max {
            std::mem::swap(&mut self.clamp_min, &mut self.clamp_max);
        }
        if !self.pressure_step.is_finite() || self.pressure_step <= 0.0 {
            self.pressure_step = 0.12;
        }
        if self.window == 0 {
            self.window = 1;
        }
        self
    }
}

/// Maintenance report distilled from a slice of temporal resonance frames.
#[derive(Clone, Debug)]
pub struct MaintainerReport {
    /// Recommended maintenance severity.
    pub status: MaintainerStatus,
    /// Mean absolute curvature drift observed within the analysis window.
    pub average_drift: f32,
    /// Mean energy captured within the analysis window.
    pub mean_energy: f32,
    /// Mean decay rate (positive = energy decaying, negative = energy growing).
    pub mean_decay: f32,
    /// Dominant oscillation of the curvature drift, if any.
    pub drift_peak: Option<ChronoPeak>,
    /// Dominant oscillation of the energy signal, if any.
    pub energy_peak: Option<ChronoPeak>,
    /// Recommended clamp for geometry max_scale, if any.
    pub suggested_max_scale: Option<f32>,
    /// Recommended bump to Leech density pressure, if any.
    pub suggested_pressure: Option<f32>,
    /// Human-readable diagnostics summarising the decision.
    pub diagnostic: String,
    /// Synthesised SpiralK script mirroring the temporal telemetry.
    #[cfg(feature = "kdsl")]
    pub spiralk_script: Option<String>,
}

impl MaintainerReport {
    /// Returns true when a self-rewrite cycle should be triggered.
    pub fn should_rewrite(&self) -> bool {
        self.status == MaintainerStatus::Rewrite
    }
}

/// Maintainer that analyses chrono frames according to a [`MaintainerConfig`].
#[derive(Clone, Debug)]
pub struct Maintainer {
    config: MaintainerConfig,
}

impl Maintainer {
    /// Creates a new maintainer with the provided configuration.
    pub fn new(config: MaintainerConfig) -> Self {
        Self {
            config: config.sanitise(),
        }
    }

    /// Analyses the provided frames and returns a maintenance report.
    pub fn assess(&self, frames: &[ChronoFrame]) -> MaintainerReport {
        if frames.is_empty() {
            return MaintainerReport {
                status: MaintainerStatus::Dormant,
                average_drift: 0.0,
                mean_energy: 0.0,
                mean_decay: 0.0,
                drift_peak: None,
                energy_peak: None,
                suggested_max_scale: None,
                suggested_pressure: None,
                diagnostic: "No temporal frames recorded; timeline is dormant.".to_string(),
                #[cfg(feature = "kdsl")]
                spiralk_script: None,
            };
        }

        let window = self.config.window.min(frames.len());
        let window = window.max(1);
        let start = frames.len() - window;
        let slice = &frames[start..];

        let summary = ChronoSummary::from_frames(slice).unwrap();
        let average_drift = summary.mean_abs_drift;
        let mean_energy = summary.mean_energy;
        let mean_decay = summary.mean_decay;
        let harmonics = ChronoHarmonics::from_frames(slice, 16);
        let drift_peak = harmonics
            .as_ref()
            .and_then(|spec| spec.dominant_drift.clone());
        let energy_peak = harmonics
            .as_ref()
            .and_then(|spec| spec.dominant_energy.clone());
        #[cfg(feature = "kdsl")]
        let spiralk_script =
            ChronoLoopSignal::new(summary.clone(), harmonics.clone()).spiralk_script;

        let growth = (-mean_decay).max(0.0);
        let mut status = if mean_energy <= self.config.energy_floor {
            MaintainerStatus::Dormant
        } else {
            MaintainerStatus::Stable
        };
        let mut reasons = Vec::<String>::new();
        let mut suggested_max_scale = None;
        let mut suggested_pressure = None;

        if average_drift > self.config.jitter_threshold {
            status = MaintainerStatus::Clamp;
            let clamp = (self.config.clamp_min
                + average_drift / (self.config.jitter_threshold + f32::EPSILON))
                .clamp(self.config.clamp_min, self.config.clamp_max);
            suggested_max_scale = Some(clamp);
            reasons.push(format!(
                "curvature jitter {:.3} exceeds threshold {:.3}",
                average_drift, self.config.jitter_threshold
            ));
        }

        if growth > self.config.growth_threshold {
            status = MaintainerStatus::Rewrite;
            let pressure = (self.config.pressure_step
                * (1.0 + growth / self.config.growth_threshold))
                .min(1.0);
            suggested_pressure = Some(pressure);
            reasons.push(format!(
                "energy growth {:.3} exceeds threshold {:.3}",
                growth, self.config.growth_threshold
            ));
        } else if matches!(status, MaintainerStatus::Clamp) {
            let pressure = (self.config.pressure_step
                * (1.0 + average_drift / self.config.jitter_threshold))
                .min(1.0);
            suggested_pressure = Some(pressure);
        }

        if let Some(peak) = drift_peak.as_ref() {
            if peak.magnitude > self.config.jitter_threshold {
                status = MaintainerStatus::Clamp;
                reasons.push(format!(
                    "drift harmonic {:.2}Hz magnitude {:.3}",
                    peak.frequency, peak.magnitude
                ));
                suggested_max_scale = suggested_max_scale.or(Some(self.config.clamp_max));
            }
        }

        if let Some(peak) = energy_peak.as_ref() {
            if peak.magnitude > self.config.growth_threshold {
                status = MaintainerStatus::Rewrite;
                reasons.push(format!(
                    "energy harmonic {:.2}Hz magnitude {:.3}",
                    peak.frequency, peak.magnitude
                ));
                let pressure = (self.config.pressure_step
                    * (1.0 + peak.magnitude / self.config.growth_threshold))
                    .min(1.0);
                suggested_pressure = Some(pressure);
            }
        }

        if reasons.is_empty() {
            if matches!(status, MaintainerStatus::Dormant) {
                reasons.push(format!(
                    "timeline energy below floor; drift {:.3}±{:.3}",
                    summary.mean_abs_drift, summary.drift_std
                ));
            } else {
                reasons.push(format!(
                    "temporal dynamics within expected range (drift {:.3}±{:.3}, energy {:.3}±{:.3})",
                    summary.mean_abs_drift, summary.drift_std, summary.mean_energy, summary.energy_std
                ));
            }
        }

        MaintainerReport {
            status,
            average_drift,
            mean_energy,
            mean_decay,
            drift_peak,
            energy_peak,
            suggested_max_scale,
            suggested_pressure,
            diagnostic: reasons.join("; "),
            #[cfg(feature = "kdsl")]
            spiralk_script,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    fn frame(step: u64, drift: f32, energy: f32, decay: f32) -> ChronoFrame {
        ChronoFrame {
            step,
            timestamp: step as f32,
            dt: 0.1,
            observed_curvature: -1.0,
            curvature_drift: drift,
            total_energy: energy,
            energy_decay: decay,
            homotopy_energy: energy * 0.4,
            functor_energy: energy * 0.2,
            recursive_energy: energy * 0.2,
            projection_energy: energy * 0.1,
            infinity_energy: energy * 0.1,
        }
    }

    #[test]
    fn maintainer_marks_dormant_when_no_frames() {
        let maintainer = Maintainer::new(MaintainerConfig::default());
        let report = maintainer.assess(&[]);
        assert_eq!(report.status, MaintainerStatus::Dormant);
        assert!(report.diagnostic.contains("dormant"));
        assert!(!report.should_rewrite());
    }

    #[test]
    fn maintainer_recommends_clamp_on_jitter() {
        let maintainer = Maintainer::new(MaintainerConfig {
            jitter_threshold: 0.1,
            ..MaintainerConfig::default()
        });
        let frames = vec![frame(0, 0.2, 1.0, 0.1); 6];
        let report = maintainer.assess(&frames);
        assert_eq!(report.status, MaintainerStatus::Clamp);
        assert!(report.suggested_max_scale.is_some());
        assert!(report.suggested_pressure.unwrap() >= 0.0);
        assert!(!report.should_rewrite());
    }

    #[test]
    fn maintainer_escalates_to_rewrite_on_energy_growth() {
        let maintainer = Maintainer::new(MaintainerConfig {
            growth_threshold: 0.02,
            ..MaintainerConfig::default()
        });
        let frames = vec![
            frame(0, 0.05, 1.2, -0.05),
            frame(1, 0.08, 1.4, -0.06),
            frame(2, 0.04, 1.6, -0.07),
        ];
        let report = maintainer.assess(&frames);
        assert_eq!(report.status, MaintainerStatus::Rewrite);
        assert!(report.should_rewrite());
        assert!(report.suggested_pressure.unwrap() > 0.0);
        assert!(report.diagnostic.contains("energy growth"));
    }

    #[test]
    fn maintainer_reports_harmonic_peaks() {
        let maintainer = Maintainer::new(MaintainerConfig {
            jitter_threshold: 0.05,
            ..MaintainerConfig::default()
        });
        let mut frames = Vec::new();
        for step in 0..32 {
            let phase = TAU * step as f32 / 8.0;
            frames.push(frame(
                step,
                phase.sin() * 0.2,
                (phase.cos() + 1.5).max(0.0),
                -0.01,
            ));
        }
        let report = maintainer.assess(&frames);
        assert!(report.drift_peak.is_some());
    }
}
