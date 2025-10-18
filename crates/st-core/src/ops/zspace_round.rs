// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::rank_entry::RankPlan;

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
}
