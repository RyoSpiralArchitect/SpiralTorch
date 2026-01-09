// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::rank_entry::RankPlan;

/// Compact summary of spectral statistics extracted from a Z-space tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
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

    let mut indexed: Vec<(usize, f32)> = gradient
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, value.abs()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let len = indexed.len();
    let top = usize::min(above.k as usize, len);
    let bottom = usize::min(beneath.k as usize, len.saturating_sub(top));
    let leftover = len.saturating_sub(top + bottom);
    let here_target = usize::min(here.k as usize, leftover);

    let mut bands = vec![RoundtableBand::Here; len];

    for &(idx, _) in indexed.iter().take(top) {
        bands[idx] = RoundtableBand::Above;
    }

    for &(idx, _) in indexed.iter().rev().take(bottom) {
        bands[idx] = RoundtableBand::Beneath;
    }

    if here_target > 0 {
        let mut assigned = 0usize;
        let mid_end = len.saturating_sub(bottom);
        for &(idx, _) in indexed.iter().skip(top).take(mid_end.saturating_sub(top)) {
            if bands[idx] == RoundtableBand::Here {
                assigned += 1;
                if assigned >= here_target {
                    break;
                }
            }
        }
    }

    let tol = tolerance.max(0.0);
    for &(idx, magnitude) in &indexed {
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
