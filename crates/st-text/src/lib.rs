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

//! Textual and acoustic narrators that turn resonance telemetry into natural language.
//!
//! The [`TextResonator`] stitches together [`DifferentialResonance`] snapshots with
//! [`ChronoFrame`] samples so higher level tooling can surface temporal stories about
//! Z-space activity.

use st_core::telemetry::atlas::AtlasFrame;
use st_core::telemetry::chrono::{
    ChronoFrame, ChronoHarmonics, ChronoSummary, ResonanceTemporalMetrics,
};
#[cfg(test)]
use st_tensor::pure::Tensor;
use st_tensor::pure::{DifferentialResonance, LanguageWaveEncoder, PureResult};

/// Human-readable synopsis describing what a resonance snapshot is doing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResonanceNarrative {
    /// Leading summary sentence.
    pub summary: String,
    /// Secondary highlights that dig into individual bands.
    pub highlights: Vec<String>,
}

impl ResonanceNarrative {
    fn new(summary: String, highlights: Vec<String>) -> Self {
        Self {
            summary,
            highlights,
        }
    }
}

/// Simple audio-friendly representation of a language wave.
#[derive(Clone, Debug, PartialEq)]
pub struct LanguageWave {
    /// Summary used to generate the wave.
    pub summary: String,
    /// Amplitude envelope derived from the encoded complex spectrum.
    pub amplitude: Vec<f32>,
}

/// Narrative generator that casts resonance telemetry into descriptive language.
#[derive(Clone, Debug)]
pub struct TextResonator {
    encoder: LanguageWaveEncoder,
}

impl TextResonator {
    /// Creates a new narrator from curvature and temperature parameters.
    pub fn new(curvature: f32, temperature: f32) -> PureResult<Self> {
        Ok(Self {
            encoder: LanguageWaveEncoder::new(curvature, temperature)?,
        })
    }

    /// Wraps an existing language wave encoder.
    pub fn with_encoder(encoder: LanguageWaveEncoder) -> Self {
        Self { encoder }
    }

    /// Returns the underlying encoder reference.
    pub fn encoder(&self) -> &LanguageWaveEncoder {
        &self.encoder
    }

    /// Produces a narrative describing the instantaneous resonance snapshot.
    pub fn describe_resonance(&self, resonance: &DifferentialResonance) -> ResonanceNarrative {
        let metrics = resonance_metrics(resonance);
        let summary = narrative_summary(&metrics);
        let highlights = narrative_highlights(&metrics);
        ResonanceNarrative::new(summary, highlights)
    }

    /// Produces a narrative describing a temporal frame.
    pub fn describe_frame(&self, frame: &ChronoFrame) -> ResonanceNarrative {
        let summary = if frame.total_energy <= f32::EPSILON {
            format!(
                "At t={:.3} the resonance settled; curvature drift {:+.3} with negligible energy.",
                frame.timestamp, frame.curvature_drift
            )
        } else {
            format!(
                "At t={:.3} the resonance carried {:.3} energy with curvature drift {:+.3}.",
                frame.timestamp, frame.total_energy, frame.curvature_drift
            )
        };
        let highlights = vec![
            format!("homotopy {:.1}%", frame.homotopy_ratio() * 100.0),
            format!("functor {:.1}%", frame.functor_ratio() * 100.0),
            format!("recursive {:.1}%", frame.recursive_ratio() * 100.0),
            format!("projection {:.1}%", frame.projection_ratio() * 100.0),
            format!("infinity {:.1}%", frame.infinity_ratio() * 100.0),
            format!("energy decay {:+.3}", frame.energy_decay),
        ];
        ResonanceNarrative::new(summary, highlights)
    }

    /// Produces a narrative describing a timeline of frames.
    pub fn describe_timeline(&self, frames: &[ChronoFrame]) -> ResonanceNarrative {
        if frames.is_empty() {
            return ResonanceNarrative::new(
                "No resonance history recorded.".to_string(),
                Vec::new(),
            );
        }
        let summary = ChronoSummary::from_frames(frames).unwrap();
        let harmonics = ChronoHarmonics::from_frames(frames, 12);
        let mut text = format!(
            "Timeline span {:.3}s with {:.3} energy ±{:.3} and drift {:+.3}±{:.3}.",
            summary.duration,
            summary.mean_energy,
            summary.energy_std,
            summary.mean_drift,
            summary.drift_std
        );
        if summary.mean_decay < 0.0 {
            text.push_str(&format!(" Energy growing at {:+.3}.", summary.mean_decay));
        } else {
            text.push_str(&format!(" Energy relaxing at {:+.3}.", summary.mean_decay));
        }
        let mut highlights = vec![
            format!("min energy {:.3}", summary.min_energy),
            format!("max energy {:.3}", summary.max_energy),
            format!("frames {}", summary.frames),
        ];
        if let Some(spec) = harmonics {
            if let Some(peak) = spec.dominant_drift {
                highlights.push(format!(
                    "drift harmonic {:.2}Hz magnitude {:.3}",
                    peak.frequency, peak.magnitude
                ));
            }
            if let Some(peak) = spec.dominant_energy {
                highlights.push(format!(
                    "energy harmonic {:.2}Hz magnitude {:.3}",
                    peak.frequency, peak.magnitude
                ));
            }
        }
        ResonanceNarrative::new(text, highlights)
    }

    /// Produces a narrative describing the aggregated atlas frame.
    pub fn describe_atlas(&self, atlas: &AtlasFrame) -> ResonanceNarrative {
        let mut summary = if let Some(chrono) = &atlas.chrono_summary {
            format!(
                "Atlas at t={:.3} spans {:.3}s with energy {:.3} and drift {:+.3}.",
                atlas.timestamp, chrono.duration, chrono.mean_energy, chrono.mean_drift
            )
        } else {
            format!("Atlas snapshot at t={:.3}.", atlas.timestamp)
        };
        if let Some(status) = atlas.maintainer_status {
            summary.push(' ');
            summary.push_str("Maintainer ");
            summary.push_str(status.as_str());
            summary.push('.');
        }
        if let Some(diagnostic) = &atlas.maintainer_diagnostic {
            if !diagnostic.is_empty() {
                summary.push(' ');
                summary.push_str(diagnostic);
            }
        }
        let mut highlights = Vec::new();
        if let Some(clamp) = atlas.suggested_max_scale {
            highlights.push(format!("clamp {:.3}", clamp));
        }
        if let Some(pressure) = atlas.suggested_pressure {
            highlights.push(format!("pressure {:.3}", pressure));
        }
        if atlas.loop_support > 0.0 {
            highlights.push(format!("loop support {:.3}", atlas.loop_support));
        }
        if let Some(total) = atlas.collapse_total {
            highlights.push(format!("collapse total {:.3}", total));
        }
        if let Some(z) = atlas.z_signal {
            highlights.push(format!("z bias {:+.3}", z));
        }
        if let Some(harmonics) = &atlas.harmonics {
            if let Some(peak) = &harmonics.dominant_drift {
                highlights.push(format!(
                    "drift harmonic {:.2}Hz magnitude {:.3}",
                    peak.frequency, peak.magnitude
                ));
            }
            if let Some(peak) = &harmonics.dominant_energy {
                highlights.push(format!(
                    "energy harmonic {:.2}Hz magnitude {:.3}",
                    peak.frequency, peak.magnitude
                ));
            }
        }
        for metric in &atlas.metrics {
            highlights.push(format!("{} {:.3}", metric.name, metric.value));
        }
        for note in &atlas.notes {
            highlights.push(note.clone());
        }
        if let Some(script) = &atlas.script_hint {
            highlights.push(format!("script {}", script));
        }
        ResonanceNarrative::new(summary, highlights)
    }

    /// Encodes the narrative into a wave amplitude for visualisation or audio playback.
    pub fn language_wave(&self, resonance: &DifferentialResonance) -> PureResult<LanguageWave> {
        let narrative = self.describe_resonance(resonance);
        let wave = self.encoder.encode_wave(&narrative.summary)?;
        let amplitude = wave
            .data()
            .iter()
            .map(|complex| complex.modulus())
            .collect();
        Ok(LanguageWave {
            summary: narrative.summary,
            amplitude,
        })
    }

    /// Generates an amplitude envelope describing an entire temporal trace.
    pub fn speak(&self, frames: &[ChronoFrame]) -> PureResult<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let narrative = self.describe_timeline(frames);
        let mut story = narrative.summary.clone();
        for highlight in &narrative.highlights {
            story.push(' ');
            story.push_str(highlight);
            story.push('.');
        }
        let wave = self.encoder.encode_wave(&story)?;
        Ok(wave
            .data()
            .iter()
            .map(|complex| complex.modulus())
            .collect())
    }
}

/// Convenience helper that returns a narrative summary using default narrator settings.
pub fn describe_resonance(resonance: &DifferentialResonance) -> PureResult<String> {
    let narrator = TextResonator::new(-1.0, 0.6)?;
    Ok(narrator.describe_resonance(resonance).summary)
}

/// Convenience helper that returns a narrative for a temporal frame using default settings.
pub fn describe_frame(frame: &ChronoFrame) -> String {
    TextResonator::new(-1.0, 0.6)
        .map(|narrator| narrator.describe_frame(frame).summary)
        .unwrap_or_else(|_| {
            format!(
                "At t={:.3} curvature drift registered {:+.3} with energy {:.3}.",
                frame.timestamp, frame.curvature_drift, frame.total_energy
            )
        })
}

/// Convenience helper that summarises a timeline with default narrator settings.
pub fn describe_timeline(frames: &[ChronoFrame]) -> PureResult<String> {
    TextResonator::new(-1.0, 0.6).map(|narrator| narrator.describe_timeline(frames).summary)
}

/// Convenience helper that summarises an atlas frame with default narrator settings.
pub fn describe_atlas(atlas: &AtlasFrame) -> PureResult<String> {
    TextResonator::new(-1.0, 0.6).map(|narrator| narrator.describe_atlas(atlas).summary)
}

fn resonance_metrics(resonance: &DifferentialResonance) -> ResonanceTemporalMetrics {
    let homotopy = resonance.homotopy_flow.squared_l2_norm();
    let functor = resonance.functor_linearisation.squared_l2_norm();
    let recursive = resonance.recursive_objective.squared_l2_norm();
    let projection = resonance.infinity_projection.squared_l2_norm();
    let infinity = resonance.infinity_energy.squared_l2_norm();
    let total = homotopy + functor + recursive + projection + infinity;
    let observed_curvature = if homotopy > 0.0 {
        -homotopy.sqrt()
    } else {
        -1.0
    };
    ResonanceTemporalMetrics {
        observed_curvature,
        total_energy: total,
        homotopy_energy: homotopy,
        functor_energy: functor,
        recursive_energy: recursive,
        projection_energy: projection,
        infinity_energy: infinity,
    }
    .sanitise()
}

fn narrative_summary(metrics: &ResonanceTemporalMetrics) -> String {
    if metrics.total_energy <= f32::EPSILON {
        return "Resonance geometry is quiescent; no meaningful curvature pulses detected."
            .to_string();
    }
    let mut bands = vec![
        (metrics.homotopy_energy, "Above band accelerating curvature"),
        (
            metrics.functor_energy,
            "Here band weaving functor linearisation",
        ),
        (
            metrics.recursive_energy,
            "Beneath band anchoring recursive objective",
        ),
        (
            metrics.projection_energy,
            "∞ projection tightening geodesics",
        ),
        (metrics.infinity_energy, "∞ hierarchy storing latent energy"),
    ];
    bands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    let dominant = bands
        .first()
        .map(|(_, label)| *label)
        .unwrap_or("Resonance energy evenly distributed across bands");
    format!(
        "{dominant} while total energy sits at {:.3}.",
        metrics.total_energy
    )
}

fn narrative_highlights(metrics: &ResonanceTemporalMetrics) -> Vec<String> {
    if metrics.total_energy <= f32::EPSILON {
        return vec!["all bands below activation threshold".to_string()];
    }
    let total = metrics.total_energy;
    vec![
        format!("Above {:.1}%", metrics.homotopy_energy / total * 100.0),
        format!("Here {:.1}%", metrics.functor_energy / total * 100.0),
        format!("Beneath {:.1}%", metrics.recursive_energy / total * 100.0),
        format!("∞ proj {:.1}%", metrics.projection_energy / total * 100.0),
        format!("∞ tower {:.1}%", metrics.infinity_energy / total * 100.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::telemetry::chrono::ChronoTimeline;
    use std::f32::consts::TAU;

    fn demo_tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(1, values.len(), values.to_vec()).unwrap()
    }

    fn demo_resonance() -> DifferentialResonance {
        let tensor = demo_tensor(&[0.5, -0.3]);
        DifferentialResonance {
            homotopy_flow: tensor.clone(),
            functor_linearisation: tensor.clone(),
            recursive_objective: tensor.clone(),
            infinity_projection: tensor.clone(),
            infinity_energy: tensor,
        }
    }

    #[test]
    fn describes_resonance() {
        let narrator = TextResonator::new(-1.0, 0.5).unwrap();
        let narrative = narrator.describe_resonance(&demo_resonance());
        assert!(!narrative.summary.is_empty());
        assert_eq!(narrative.highlights.len(), 5);
    }

    #[test]
    fn encodes_language_wave() {
        let narrator = TextResonator::new(-1.0, 0.5).unwrap();
        let wave = narrator.language_wave(&demo_resonance()).unwrap();
        assert!(!wave.amplitude.is_empty());
    }

    #[test]
    fn speak_generates_amplitude() {
        let narrator = TextResonator::new(-1.0, 0.5).unwrap();
        let mut timeline = ChronoTimeline::new();
        let metrics = resonance_metrics(&demo_resonance());
        let frame = timeline.record(0.1, metrics);
        let amplitude = narrator.speak(&[frame]).unwrap();
        assert!(!amplitude.is_empty());
    }

    #[test]
    fn timeline_narrative_mentions_harmonics() {
        let narrator = TextResonator::new(-1.0, 0.5).unwrap();
        let mut timeline = ChronoTimeline::with_capacity(32);
        for step in 0..24 {
            let phase = TAU * step as f32 / 6.0;
            let metrics = ResonanceTemporalMetrics {
                observed_curvature: phase.cos(),
                total_energy: phase.sin().abs() + 0.2,
                homotopy_energy: 0.1,
                functor_energy: 0.05,
                recursive_energy: 0.04,
                projection_energy: 0.03,
                infinity_energy: 0.02,
            };
            timeline.record(0.1, metrics);
        }
        let frames: Vec<_> = timeline.frames().cloned().collect();
        let narrative = narrator.describe_timeline(&frames);
        assert!(narrative.summary.contains("Timeline span"));
        assert!(narrative
            .highlights
            .iter()
            .any(|line| line.contains("harmonic")));
    }
}
