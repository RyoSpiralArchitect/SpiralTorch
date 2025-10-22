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
use st_tensor::Tensor;
use st_tensor::{DifferentialResonance, LanguageWaveEncoder, PureResult};

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

impl LanguageWave {
    /// Synthesises mono PCM samples from the amplitude envelope.
    pub fn to_audio_samples(&self, sample_rate: usize) -> Vec<f32> {
        if self.amplitude.is_empty() || sample_rate == 0 {
            return Vec::new();
        }
        let mut samples = Vec::with_capacity(self.amplitude.len() * sample_rate);
        for (idx, amp) in self.amplitude.iter().enumerate() {
            let phase = idx as f32 / self.amplitude.len() as f32;
            let carrier = (phase * std::f32::consts::TAU * 220.0).sin();
            let frame = carrier * *amp;
            for _ in 0..sample_rate {
                samples.push(frame);
            }
        }
        samples
    }
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
        if let Some((min_curvature, max_curvature)) = curvature_envelope(frames) {
            text.push_str(&format!(
                " Curvature envelope spans {:+.3}→{:+.3}.",
                min_curvature, max_curvature
            ));
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
        if let Some(percentages) = aggregate_band_percentages(frames) {
            highlights.push(format!("avg Above {:.1}%", percentages[0] * 100.0));
            highlights.push(format!("avg Here {:.1}%", percentages[1] * 100.0));
            highlights.push(format!("avg Beneath {:.1}%", percentages[2] * 100.0));
            highlights.push(format!("avg ∞ proj {:.1}%", percentages[3] * 100.0));
            highlights.push(format!("avg ∞ tower {:.1}%", percentages[4] * 100.0));
            let entropy = band_entropy(&percentages);
            if entropy > 0.0 {
                highlights.push(format!("band entropy {:.2}", entropy));
            }
            let contrast = band_contrast(&percentages);
            if contrast > 0.0 {
                highlights.push(format!("dominant contrast {:.1}%", contrast * 100.0));
            }
            let spread = band_spread(&percentages);
            if spread > 0.0 {
                highlights.push(format!("band spread σ {:.1}%", spread * 100.0));
            }
            let evenness = band_evenness(&percentages);
            if evenness > 0.0 {
                highlights.push(format!("band evenness {:.1}%", evenness * 100.0));
            }
            let active = active_band_count(&percentages);
            if active > 0 {
                highlights.push(format!("active bands {active}"));
            }
        }
        if let Some(mean_curvature) = mean_observed_curvature(frames) {
            highlights.push(format!("avg curvature {:+.3}", mean_curvature));
        }
        if let Some((min_curvature, max_curvature)) = curvature_envelope(frames) {
            highlights.push(format!(
                "curvature envelope {:+.3}→{:+.3}",
                min_curvature, max_curvature
            ));
        }
        if let Some((min_drift, max_drift)) = drift_envelope(frames) {
            highlights.push(format!("drift span {:+.3}→{:+.3}", min_drift, max_drift));
        }
        if let Some(velocity) = energy_velocity(frames) {
            highlights.push(format!("energy velocity {:+.3}/s", velocity));
        }
        if let Some(churn) = band_churn(frames) {
            highlights.push(format!("band churn {:.1}%/frame", churn * 100.0));
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

    /// Produces a language wave from a pre-baked narrative.
    pub fn synthesize_wave(&self, narrative: &ResonanceNarrative) -> PureResult<LanguageWave> {
        let mut story = narrative.summary.clone();
        for highlight in &narrative.highlights {
            story.push(' ');
            story.push_str(highlight);
        }
        let wave = self.encoder.encode_wave(&story)?;
        let amplitude = wave
            .data()
            .iter()
            .map(|complex| complex.modulus())
            .collect();
        Ok(LanguageWave {
            summary: narrative.summary.clone(),
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

/// Realtime narrator that emits audio-ready samples.
#[derive(Clone, Debug)]
pub struct RealtimeNarrator {
    inner: TextResonator,
    sample_rate: usize,
}

impl RealtimeNarrator {
    pub fn new(curvature: f32, temperature: f32, sample_rate: usize) -> PureResult<Self> {
        Ok(Self {
            inner: TextResonator::new(curvature, temperature)?,
            sample_rate: sample_rate.max(1),
        })
    }

    pub fn from_resonator(resonator: TextResonator, sample_rate: usize) -> Self {
        Self {
            inner: resonator,
            sample_rate: sample_rate.max(1),
        }
    }

    pub fn narrate_resonance(&self, resonance: &DifferentialResonance) -> PureResult<Vec<f32>> {
        let narrative = self.inner.describe_resonance(resonance);
        let wave = self.inner.synthesize_wave(&narrative)?;
        Ok(wave.to_audio_samples(self.sample_rate))
    }

    pub fn narrate_timeline(&self, frames: &[ChronoFrame]) -> PureResult<Vec<f32>> {
        let narrative = self.inner.describe_timeline(frames);
        let wave = self.inner.synthesize_wave(&narrative)?;
        Ok(wave.to_audio_samples(self.sample_rate))
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
    let dominant_share = bands
        .first()
        .map(|(energy, _)| {
            if metrics.total_energy <= f32::EPSILON {
                0.0
            } else {
                (energy / metrics.total_energy).clamp(0.0, 1.0)
            }
        })
        .unwrap_or(0.0);
    format!(
        "{dominant} commanding {:.1}% of the field while total energy sits at {:.3} with curvature {:+.3}.",
        dominant_share * 100.0,
        metrics.total_energy,
        metrics.observed_curvature
    )
}

fn narrative_highlights(metrics: &ResonanceTemporalMetrics) -> Vec<String> {
    if metrics.total_energy <= f32::EPSILON {
        return vec!["all bands below activation threshold".to_string()];
    }
    let percentages = band_percentages(metrics);
    let mut highlights = vec![
        format!("Above {:.1}%", percentages[0] * 100.0),
        format!("Here {:.1}%", percentages[1] * 100.0),
        format!("Beneath {:.1}%", percentages[2] * 100.0),
        format!("∞ proj {:.1}%", percentages[3] * 100.0),
        format!("∞ tower {:.1}%", percentages[4] * 100.0),
        format!("curvature {:+.3}", metrics.observed_curvature),
    ];
    let entropy = band_entropy(&percentages);
    if entropy > 0.0 {
        highlights.push(format!("band entropy {:.2}", entropy));
    }
    let contrast = band_contrast(&percentages);
    if contrast > 0.0 {
        highlights.push(format!("dominant contrast {:.1}%", contrast * 100.0));
    }
    let spread = band_spread(&percentages);
    if spread > 0.0 {
        highlights.push(format!("band spread σ {:.1}%", spread * 100.0));
    }
    let evenness = band_evenness(&percentages);
    if evenness > 0.0 {
        highlights.push(format!("band evenness {:.1}%", evenness * 100.0));
    }
    let active = active_band_count(&percentages);
    if active > 0 {
        highlights.push(format!("active bands {active}"));
    }
    highlights
}

fn band_percentages(metrics: &ResonanceTemporalMetrics) -> [f32; 5] {
    if metrics.total_energy <= f32::EPSILON {
        return [0.0; 5];
    }
    let total = metrics.total_energy.max(f32::EPSILON);
    [
        (metrics.homotopy_energy / total).clamp(0.0, 1.0),
        (metrics.functor_energy / total).clamp(0.0, 1.0),
        (metrics.recursive_energy / total).clamp(0.0, 1.0),
        (metrics.projection_energy / total).clamp(0.0, 1.0),
        (metrics.infinity_energy / total).clamp(0.0, 1.0),
    ]
}

fn band_entropy(percentages: &[f32; 5]) -> f32 {
    const EPS: f32 = 1e-6;
    percentages
        .iter()
        .filter(|p| **p > EPS)
        .map(|p| {
            let logp = p.ln();
            -p * logp
        })
        .sum::<f32>()
        / std::f32::consts::LN_2
}

fn band_contrast(percentages: &[f32; 5]) -> f32 {
    let mut sorted = *percentages;
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    if sorted[0] <= 0.0 {
        0.0
    } else {
        (sorted[0] - sorted[1].max(0.0)).max(0.0)
    }
}

fn band_spread(percentages: &[f32; 5]) -> f32 {
    let mean = percentages.iter().sum::<f32>() / percentages.len() as f32;
    let variance = percentages
        .iter()
        .map(|p| {
            let diff = p - mean;
            diff * diff
        })
        .sum::<f32>()
        / percentages.len() as f32;
    variance.sqrt()
}

fn band_evenness(percentages: &[f32; 5]) -> f32 {
    const EPS: f32 = 1e-6;
    let total: f32 = percentages.iter().sum();
    if total <= EPS {
        return 0.0;
    }
    let uniform = 1.0 / percentages.len() as f32;
    let l1 = percentages
        .iter()
        .map(|p| (p - uniform).abs())
        .sum::<f32>();
    let max_l1 = 2.0 * (1.0 - uniform);
    if max_l1 <= EPS {
        1.0
    } else {
        (1.0 - (l1 / max_l1).min(1.0)).max(0.0)
    }
}

fn active_band_count(percentages: &[f32; 5]) -> usize {
    const ACTIVATION_THRESHOLD: f32 = 0.05;
    percentages
        .iter()
        .filter(|&&value| value > ACTIVATION_THRESHOLD)
        .count()
}

fn aggregate_band_percentages(frames: &[ChronoFrame]) -> Option<[f32; 5]> {
    let mut totals = [0.0f32; 5];
    let mut energy_total = 0.0f32;
    for frame in frames {
        if frame.total_energy <= f32::EPSILON {
            continue;
        }
        let energy = frame.total_energy.max(0.0);
        totals[0] += frame.homotopy_energy.max(0.0);
        totals[1] += frame.functor_energy.max(0.0);
        totals[2] += frame.recursive_energy.max(0.0);
        totals[3] += frame.projection_energy.max(0.0);
        totals[4] += frame.infinity_energy.max(0.0);
        energy_total += energy;
    }
    if energy_total <= f32::EPSILON {
        return None;
    }
    Some([
        (totals[0] / energy_total).clamp(0.0, 1.0),
        (totals[1] / energy_total).clamp(0.0, 1.0),
        (totals[2] / energy_total).clamp(0.0, 1.0),
        (totals[3] / energy_total).clamp(0.0, 1.0),
        (totals[4] / energy_total).clamp(0.0, 1.0),
    ])
}

fn mean_observed_curvature(frames: &[ChronoFrame]) -> Option<f32> {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for frame in frames {
        if !frame.observed_curvature.is_finite() {
            continue;
        }
        sum += frame.observed_curvature;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f32)
    }
}

fn curvature_envelope(frames: &[ChronoFrame]) -> Option<(f32, f32)> {
    let mut min_curvature = f32::INFINITY;
    let mut max_curvature = f32::NEG_INFINITY;
    for frame in frames {
        if !frame.observed_curvature.is_finite() {
            continue;
        }
        min_curvature = min_curvature.min(frame.observed_curvature);
        max_curvature = max_curvature.max(frame.observed_curvature);
    }
    if min_curvature.is_finite() && max_curvature.is_finite() {
        Some((min_curvature, max_curvature))
    } else {
        None
    }
}

fn drift_envelope(frames: &[ChronoFrame]) -> Option<(f32, f32)> {
    let mut min_drift = f32::INFINITY;
    let mut max_drift = f32::NEG_INFINITY;
    for frame in frames {
        if !frame.curvature_drift.is_finite() {
            continue;
        }
        min_drift = min_drift.min(frame.curvature_drift);
        max_drift = max_drift.max(frame.curvature_drift);
    }
    if min_drift.is_finite() && max_drift.is_finite() {
        Some((min_drift, max_drift))
    } else {
        None
    }
}

fn energy_velocity(frames: &[ChronoFrame]) -> Option<f32> {
    if frames.len() < 2 {
        return None;
    }
    let first = frames.first()?;
    let last = frames.last()?;
    if !first.total_energy.is_finite() || !last.total_energy.is_finite() {
        return None;
    }
    let duration = (last.timestamp - first.timestamp).abs().max(f32::EPSILON);
    Some((last.total_energy - first.total_energy) / duration)
}

fn band_churn(frames: &[ChronoFrame]) -> Option<f32> {
    if frames.len() < 2 {
        return None;
    }
    let mut total = 0.0f32;
    let mut transitions = 0usize;
    for window in frames.windows(2) {
        let a = frame_band_percentages(&window[0]);
        let b = frame_band_percentages(&window[1]);
        let variation = a
            .iter()
            .zip(b.iter())
            .map(|(left, right)| (left - right).abs())
            .sum::<f32>()
            * 0.5;
        total += variation;
        transitions += 1;
    }
    if transitions == 0 {
        None
    } else {
        Some(total / transitions as f32)
    }
}

fn frame_band_percentages(frame: &ChronoFrame) -> [f32; 5] {
    if frame.total_energy <= f32::EPSILON {
        return [0.0; 5];
    }
    let total = frame.total_energy.max(f32::EPSILON);
    [
        (frame.homotopy_energy / total).clamp(0.0, 1.0),
        (frame.functor_energy / total).clamp(0.0, 1.0),
        (frame.recursive_energy / total).clamp(0.0, 1.0),
        (frame.projection_energy / total).clamp(0.0, 1.0),
        (frame.infinity_energy / total).clamp(0.0, 1.0),
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
        assert!(narrative.summary.contains("curvature"));
        assert!(narrative.highlights.len() >= 6);
        assert!(narrative
            .highlights
            .iter()
            .any(|line| line.contains("curvature")));
        assert!(narrative
            .highlights
            .iter()
            .any(|line| line.contains("band entropy")));
        assert!(narrative
            .highlights
            .iter()
            .any(|line| line.contains("band evenness")));
        assert!(narrative
            .highlights
            .iter()
            .any(|line| line.contains("active bands")));
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
        assert!(narrative
            .highlights
            .iter()
            .any(|line| line.contains("band entropy")));
        assert!(narrative.summary.contains("Curvature envelope"));
        assert!(narrative
            .highlights
            .iter()
            .any(|line| line.contains("energy velocity")));
        assert!(narrative
            .highlights
            .iter()
            .any(|line| line.contains("band churn")));
    }
}
