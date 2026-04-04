// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::rank_entry::RankPlan;
use std::cmp::Ordering;

/// Compact summary of spectral statistics extracted from a Z-space tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpectralFeatureSample {
    /// Sheet index with the highest accumulated magnitude.
    pub sheet_index: usize,
    /// Confidence score for the dominant sheet (ratio of its energy to the total energy).
    pub sheet_confidence: f32,
    /// Normalised curvature obtained from the second discrete derivative.
    pub curvature: f32,
    /// Spin alignment describing how often consecutive samples agree in sign.
    pub spin: f32,
    /// Mean absolute magnitude across the analysed samples.
    pub energy: f32,
}

impl SpectralFeatureSample {
    /// Extracts spectral statistics from the provided slice using `sheet_hint` buckets.
    pub fn from_slice(samples: &[f32], sheet_hint: usize) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let (sheet_index, sheet_confidence) =
            estimate_sheet_index(samples, sheet_hint).unwrap_or((0, 0.0));
        let curvature = estimate_curvature(samples);
        let spin = estimate_spin_alignment(samples);
        let energy = samples.iter().map(|value| value.abs()).sum::<f32>() / samples.len() as f32;
        Some(Self {
            sheet_index,
            sheet_confidence,
            curvature,
            spin,
            energy,
        })
    }
}

/// Estimates the dominant sheet index alongside its confidence score.
pub fn estimate_sheet_index(samples: &[f32], sheet_hint: usize) -> Option<(usize, f32)> {
    if samples.is_empty() {
        return None;
    }
    let sheet_count = sheet_hint.max(1).min(samples.len());
    let window = samples.len().div_ceil(sheet_count);
    let total_energy = samples.iter().map(|value| value.abs()).sum::<f32>();
    if total_energy <= f32::EPSILON {
        return Some((0, 0.0));
    }

    let mut best_index = 0usize;
    let mut best_energy = -f32::INFINITY;
    for sheet in 0..sheet_count {
        let start = sheet * window;
        if start >= samples.len() {
            break;
        }
        let end = ((sheet + 1) * window).min(samples.len());
        let energy = samples[start..end]
            .iter()
            .map(|value| value.abs())
            .sum::<f32>();
        if energy > best_energy {
            best_energy = energy;
            best_index = sheet;
        }
    }

    let confidence = if best_energy <= 0.0 {
        0.0
    } else {
        (best_energy / total_energy).clamp(0.0, 1.0)
    };
    Some((best_index, confidence))
}

/// Estimates the discrete curvature by comparing first and second order differences.
pub fn estimate_curvature(samples: &[f32]) -> f32 {
    if samples.len() < 3 {
        return 0.0;
    }
    let mut first_order = 0.0f32;
    let mut second_order = 0.0f32;
    for window in samples.windows(3) {
        let d1_a = window[1] - window[0];
        let d1_b = window[2] - window[1];
        first_order += d1_a.abs() + d1_b.abs();
        second_order += (d1_b - d1_a).abs();
    }
    if first_order <= f32::EPSILON {
        return 0.0;
    }
    (second_order / (first_order + 1e-6)).min(4.0)
}

/// Estimates the spin alignment between consecutive samples.
pub fn estimate_spin_alignment(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for window in samples.windows(2) {
        let a = window[0];
        let b = window[1];
        numerator += a * b;
        denominator += (a.abs() * b.abs()).max(0.0);
    }
    if denominator <= f32::EPSILON {
        return 0.0;
    }
    (numerator / (denominator + 1e-6)).clamp(-1.0, 1.0)
}

/// Convenience wrapper that extracts all spectral features in a single call.
pub fn spectral_features(samples: &[f32], sheet_hint: usize) -> Option<SpectralFeatureSample> {
    SpectralFeatureSample::from_slice(samples, sheet_hint)
}

pub fn roundtable_spectral_features(
    samples: &[f32],
    above: &RankPlan,
    here: &RankPlan,
    beneath: &RankPlan,
) -> SpectralFeatureSample {
    let len = samples.len().max(1);
    let sheet_hint = RoundtableChoiceProfile::from_plan(above, len)
        .sheet_hint
        .max(RoundtableChoiceProfile::from_plan(here, len).sheet_hint)
        .max(RoundtableChoiceProfile::from_plan(beneath, len).sheet_hint)
        .max(1);
    SpectralFeatureSample::from_slice(samples, sheet_hint).unwrap_or_default()
}

/// Roundtable band classification used to map gradients into Above/Here/Beneath
/// streams without requiring tensor dependencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundtableBand {
    Above,
    Here,
    Beneath,
}

impl RoundtableBand {
    pub fn as_str(&self) -> &'static str {
        match self {
            RoundtableBand::Above => "above",
            RoundtableBand::Here => "here",
            RoundtableBand::Beneath => "beneath",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct RoundtableChoiceProfile {
    sheet_hint: usize,
    spectral_gain: f32,
    coherence_gain: f32,
    neutrality_gain: f32,
}

impl RoundtableChoiceProfile {
    fn from_plan(plan: &RankPlan, len: usize) -> Self {
        let choice = &plan.choice;
        let safe_len = len.max(1) as u32;
        let sheet_hint = choice
            .fft_segments
            .max(1)
            .saturating_mul(choice.fft_radix.clamp(2, 4))
            .div_ceil(2)
            .clamp(1, safe_len) as usize;
        let segment_drive = (choice.fft_segments.saturating_sub(1).min(3) as f32) / 3.0;
        let radix_drive = (choice.fft_radix.clamp(2, 4).saturating_sub(2) as f32) / 2.0;
        let lane_density = (choice.kl.max(1) as f32 / choice.wg.max(1) as f32).clamp(0.0, 1.0);
        let merge_drive = match choice.mk {
            2 => 1.0,
            1 => 0.55,
            _ => 0.2,
        };
        let compaction_drive = if choice.tile > 0 {
            (choice.ctile as f32 / choice.tile.max(1) as f32).clamp(0.0, 1.0)
        } else if choice.ctile > 0 {
            0.5
        } else {
            0.0
        };
        let latency_drive = choice
            .latency_window
            .map(|window| (window.slack as f32 / window.target.max(1) as f32).clamp(0.0, 1.0))
            .unwrap_or(0.0);

        Self {
            sheet_hint,
            spectral_gain: (0.02
                + 0.05 * segment_drive
                + 0.03 * radix_drive
                + if choice.subgroup { 0.015 } else { 0.0 })
            .clamp(0.0, 0.14),
            coherence_gain: (0.015
                + 0.055 * merge_drive
                + 0.04 * lane_density
                + if choice.subgroup { 0.03 } else { 0.0 })
            .clamp(0.0, 0.16),
            neutrality_gain: (0.01 + 0.07 * compaction_drive + 0.08 * latency_drive)
                .clamp(0.0, 0.16),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct RoundtableLaneStats {
    rel_mag: f32,
    coherence: f32,
    instability: f32,
    sheet_affinity: f32,
}

#[derive(Clone, Copy, Debug)]
struct ScoredLane {
    idx: usize,
    mag: f32,
    score: f32,
}

fn lane_coherence(samples: &[f32], idx: usize) -> f32 {
    let current = samples[idx];
    let mut signed = 0.0f32;
    let mut total = 0.0f32;
    for offset in [-1isize, 1] {
        let neighbor = idx as isize + offset;
        if !(0..samples.len() as isize).contains(&neighbor) {
            continue;
        }
        let neighbor = neighbor as usize;
        let weight = current.abs().min(samples[neighbor].abs()).max(1e-6);
        total += weight;
        signed += if current * samples[neighbor] >= 0.0 {
            weight
        } else {
            -weight
        };
    }
    if total <= f32::EPSILON {
        0.5
    } else {
        ((signed / total) + 1.0).mul_add(0.5, 0.0).clamp(0.0, 1.0)
    }
}

fn lane_instability(samples: &[f32], idx: usize, global_curvature: f32) -> f32 {
    let left = idx
        .checked_sub(1)
        .map(|pos| samples[pos])
        .unwrap_or(samples[idx]);
    let center = samples[idx];
    let right = samples.get(idx + 1).copied().unwrap_or(center);
    let second = (right - 2.0 * center + left).abs();
    let scale = left.abs() + center.abs() + right.abs() + 1e-6;
    let local = (second / scale).clamp(0.0, 1.0);
    if global_curvature <= f32::EPSILON {
        local
    } else {
        (0.5 * local + 0.5 * (local / global_curvature.max(1e-6)).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    }
}

fn sheet_affinity(samples: &[f32], sheet_hint: usize) -> (Vec<f32>, f32) {
    if samples.is_empty() {
        return (Vec::new(), 0.0);
    }
    let sheet_count = sheet_hint.max(1).min(samples.len());
    let window = samples.len().div_ceil(sheet_count);
    let mut sheet_energy = vec![0.0f32; sheet_count];
    for (idx, sample) in samples.iter().enumerate() {
        let segment = (idx / window).min(sheet_count - 1);
        sheet_energy[segment] += sample.abs();
    }
    let total_energy = sheet_energy.iter().copied().sum::<f32>().max(1e-6);
    let (dominant_idx, dominant_energy) = sheet_energy
        .iter()
        .copied()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal))
        .unwrap_or((0, 0.0));
    let max_energy = dominant_energy.max(1e-6);
    let dominant_confidence = (dominant_energy / total_energy).clamp(0.0, 1.0);
    let affinity = (0..samples.len())
        .map(|idx| {
            let segment = (idx / window).min(sheet_count - 1);
            let energy = sheet_energy[segment];
            if segment == dominant_idx {
                1.0
            } else {
                (energy / max_energy).clamp(0.0, 1.0)
            }
        })
        .collect();
    (affinity, dominant_confidence)
}

fn top_score(
    lane: RoundtableLaneStats,
    above: RoundtableChoiceProfile,
    here: RoundtableChoiceProfile,
    dominant_confidence: f32,
) -> f32 {
    let stability = lane.coherence * (1.0 - lane.instability);
    lane.rel_mag
        + above.spectral_gain * lane.sheet_affinity * dominant_confidence
        + above.coherence_gain * lane.coherence
        - here.neutrality_gain * stability * 0.45
}

fn bottom_score(
    lane: RoundtableLaneStats,
    beneath: RoundtableChoiceProfile,
    here: RoundtableChoiceProfile,
    dominant_confidence: f32,
) -> f32 {
    let stability = lane.coherence * (1.0 - lane.instability);
    (1.0 - lane.rel_mag)
        + beneath.spectral_gain * (1.0 - lane.sheet_affinity) * dominant_confidence
        + beneath.coherence_gain * (1.0 - lane.coherence)
        - here.neutrality_gain * stability * 0.3
}

/// Error returned when a gradient cannot be split into bands.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RoundtableError {
    #[error("gradient was empty")]
    EmptyGradient,
}

/// Assignment of every gradient lane to a specific roundtable band.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundtableAssignment {
    bands: Vec<RoundtableBand>,
}

impl RoundtableAssignment {
    pub fn new(bands: Vec<RoundtableBand>) -> Self {
        Self { bands }
    }

    pub fn len(&self) -> usize {
        self.bands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bands.is_empty()
    }

    pub fn band(&self, index: usize) -> RoundtableBand {
        self.bands[index]
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, RoundtableBand)> + '_ {
        self.bands.iter().copied().enumerate()
    }

    pub fn counts(&self) -> (usize, usize, usize) {
        let mut above = 0usize;
        let mut here = 0usize;
        let mut beneath = 0usize;
        for band in &self.bands {
            match band {
                RoundtableBand::Above => above += 1,
                RoundtableBand::Here => here += 1,
                RoundtableBand::Beneath => beneath += 1,
            }
        }
        (above, here, beneath)
    }

    pub fn energy(&self, gradient: &[f32]) -> (f32, f32, f32) {
        let mut above = 0.0f32;
        let mut here = 0.0f32;
        let mut beneath = 0.0f32;
        for (idx, value) in gradient.iter().enumerate() {
            let mag = value.abs();
            match self.bands[idx] {
                RoundtableBand::Above => above += mag,
                RoundtableBand::Here => here += mag,
                RoundtableBand::Beneath => beneath += mag,
            }
        }
        (above, here, beneath)
    }
}

/// Classifies every entry of the gradient into Above/Here/Beneath bands.
pub fn classify_roundtable(
    gradient: &[f32],
    above: &RankPlan,
    here: &RankPlan,
    beneath: &RankPlan,
    tolerance: f32,
) -> Result<RoundtableAssignment, RoundtableError> {
    if gradient.is_empty() {
        return Err(RoundtableError::EmptyGradient);
    }

    let len = gradient.len();
    let top = usize::min(above.k as usize, len);
    let bottom = usize::min(beneath.k as usize, len.saturating_sub(top));

    let above_profile = RoundtableChoiceProfile::from_plan(above, len);
    let here_profile = RoundtableChoiceProfile::from_plan(here, len);
    let beneath_profile = RoundtableChoiceProfile::from_plan(beneath, len);
    let sheet_hint = above_profile
        .sheet_hint
        .max(here_profile.sheet_hint)
        .max(beneath_profile.sheet_hint)
        .max(1);
    let (sheet_affinity, dominant_confidence) = sheet_affinity(gradient, sheet_hint);
    let max_mag = gradient
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let global_curvature = estimate_curvature(gradient).clamp(0.0, 1.0);
    let lane_stats = gradient
        .iter()
        .enumerate()
        .map(|(idx, value)| RoundtableLaneStats {
            rel_mag: (value.abs() / max_mag).clamp(0.0, 1.0),
            coherence: lane_coherence(gradient, idx),
            instability: lane_instability(gradient, idx, global_curvature),
            sheet_affinity: sheet_affinity.get(idx).copied().unwrap_or(1.0),
        })
        .collect::<Vec<_>>();

    let mut bands = vec![RoundtableBand::Here; len];
    let mut top_ranked = gradient
        .iter()
        .enumerate()
        .map(|(idx, value)| ScoredLane {
            idx,
            mag: value.abs(),
            score: top_score(
                lane_stats[idx],
                above_profile,
                here_profile,
                dominant_confidence,
            ),
        })
        .collect::<Vec<_>>();
    top_ranked.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| rhs.mag.partial_cmp(&lhs.mag).unwrap_or(Ordering::Equal))
            .then_with(|| lhs.idx.cmp(&rhs.idx))
    });
    for entry in top_ranked.iter().take(top) {
        let idx = entry.idx;
        bands[idx] = RoundtableBand::Above;
    }

    let mut bottom_ranked = gradient
        .iter()
        .enumerate()
        .filter(|(idx, _)| bands[*idx] != RoundtableBand::Above)
        .map(|(idx, value)| ScoredLane {
            idx,
            mag: value.abs(),
            score: bottom_score(
                lane_stats[idx],
                beneath_profile,
                here_profile,
                dominant_confidence,
            ),
        })
        .collect::<Vec<_>>();
    bottom_ranked.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.mag.partial_cmp(&rhs.mag).unwrap_or(Ordering::Equal))
            .then_with(|| lhs.idx.cmp(&rhs.idx))
    });
    for entry in bottom_ranked.iter().take(bottom) {
        let idx = entry.idx;
        if bands[idx] != RoundtableBand::Above {
            bands[idx] = RoundtableBand::Beneath;
        }
    }

    let tol = tolerance.max(0.0);
    for (idx, value) in gradient.iter().enumerate() {
        let magnitude = value.abs();
        if magnitude <= tol {
            bands[idx] = RoundtableBand::Here;
        }
    }

    Ok(RoundtableAssignment::new(bands))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::unison_heuristics::{Choice, RankKind};

    fn plan(kind: RankKind, k: u32) -> RankPlan {
        RankPlan {
            kind,
            rows: 1,
            cols: 1,
            k,
            choice: Choice {
                use_2ce: false,
                wg: 32,
                kl: 32,
                ch: 0,
                mk: 0,
                mkd: 0,
                tile: 0,
                ctile: 0,
                subgroup: false,
                fft_tile: 0,
                fft_radix: 2,
                fft_segments: 1,
                latency_window: None,
            },
        }
    }

    #[test]
    fn assignment_splits_indices() {
        let gradient = [0.9, 0.1, 0.5, 0.05, 0.2, 0.8, 0.4, 0.02];
        let assign = classify_roundtable(
            &gradient,
            &plan(RankKind::TopK, 2),
            &plan(RankKind::MidK, 3),
            &plan(RankKind::BottomK, 2),
            0.01,
        )
        .unwrap();
        assert_eq!(assign.len(), gradient.len());
        let (a, h, b) = assign.counts();
        assert_eq!(a, 2);
        assert_eq!(b, 2);
        assert_eq!(h, gradient.len() - a - b);
    }

    #[test]
    fn energy_tracks_band_mass() {
        let gradient = [1.0, -2.0, 0.5, -0.25];
        let assign = classify_roundtable(
            &gradient,
            &plan(RankKind::TopK, 1),
            &plan(RankKind::MidK, 1),
            &plan(RankKind::BottomK, 1),
            0.0,
        )
        .unwrap();
        let (above, here, beneath) = assign.energy(&gradient);
        assert!(above >= 1.0);
        assert!(beneath >= 0.25);
        assert!((above + here + beneath) - 3.75 < 1e-4);
    }

    #[test]
    fn rich_choice_can_shift_above_membership() {
        let gradient = [0.93, 0.88, 0.10, 0.10, 0.89, 0.20, 0.10, 0.10];
        let mut rich_above = plan(RankKind::TopK, 2);
        rich_above.choice.subgroup = true;
        rich_above.choice.mk = 2;
        rich_above.choice.fft_segments = 2;
        rich_above.choice.fft_radix = 4;
        rich_above.choice.wg = 128;
        rich_above.choice.kl = 64;

        let baseline = classify_roundtable(
            &gradient,
            &plan(RankKind::TopK, 2),
            &plan(RankKind::MidK, 4),
            &plan(RankKind::BottomK, 1),
            0.0,
        )
        .unwrap();
        let enriched = classify_roundtable(
            &gradient,
            &rich_above,
            &plan(RankKind::MidK, 4),
            &plan(RankKind::BottomK, 1),
            0.0,
        )
        .unwrap();

        assert_eq!(baseline.band(4), RoundtableBand::Above);
        assert_eq!(baseline.band(1), RoundtableBand::Here);
        assert_eq!(enriched.band(1), RoundtableBand::Above);
        assert_eq!(enriched.band(4), RoundtableBand::Here);
    }

    #[test]
    fn empty_gradient_errors() {
        let err = classify_roundtable(
            &[],
            &plan(RankKind::TopK, 1),
            &plan(RankKind::MidK, 1),
            &plan(RankKind::BottomK, 1),
            0.0,
        )
        .unwrap_err();
        assert_eq!(err, RoundtableError::EmptyGradient);
    }

    #[test]
    fn spectral_features_capture_sheet_bias() {
        let samples = [0.8f32, 0.9, 0.1, 0.05, -0.2, -0.1];
        let (sheet, confidence) = estimate_sheet_index(&samples, 3).unwrap();
        assert_eq!(sheet, 0);
        assert!(confidence > 0.45);
        let features = spectral_features(&samples, 3).unwrap();
        assert_eq!(features.sheet_index, sheet);
        assert!(features.curvature >= 0.0);
        assert!(features.energy > 0.0);
    }

    #[test]
    fn spin_alignment_detects_alternation() {
        let samples = [1.0f32, -1.0, 1.0, -1.0, 1.0];
        let spin = estimate_spin_alignment(&samples);
        assert!(spin < -0.5);
        let curvature = estimate_curvature(&samples);
        assert!(curvature > 0.0);
    }
}
