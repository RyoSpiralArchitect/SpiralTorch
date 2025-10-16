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

use st_core::telemetry::chrono::{ChronoFrame, ResonanceTemporalMetrics};
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
        let mut story = String::new();
        for frame in frames {
            let snippet = if frame.total_energy <= f32::EPSILON {
                format!(
                    "t={:.3}: drift {:+.3} with dormant energy. ",
                    frame.timestamp, frame.curvature_drift
                )
            } else {
                format!(
                    "t={:.3}: energy {:.3}, drift {:+.3}. ",
                    frame.timestamp, frame.total_energy, frame.curvature_drift
                )
            };
            story.push_str(&snippet);
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
}
