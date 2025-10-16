// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::schedule::BandEnergy;

/// Adaptive low-rank injector that can raise or lower the effective rank based
/// on runtime sensitivity signals.
#[derive(Clone, Debug)]
pub struct Injector {
    rank: u32,
    alpha: f32,
    r_min: u32,
    r_max: u32,
}

impl Injector {
    /// Creates a new injector with the provided base rank and learning factor.
    pub fn with_rank(rank: u32, alpha: f32) -> Self {
        let r = rank.max(1);
        Self {
            rank: r,
            alpha: alpha.max(0.0),
            r_min: 1,
            r_max: r.max(2),
        }
    }

    /// Updates the admissible rank bounds.
    pub fn with_bounds(mut self, r_min: u32, r_max: u32) -> Self {
        self.r_min = r_min.max(1);
        self.r_max = r_max.max(self.r_min);
        self.rank = self.rank.clamp(self.r_min, self.r_max);
        self
    }

    /// Returns the current rank.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Adjusts the rank based on the band energy distribution and an external
    /// sensitivity score.
    pub fn rank_gate(&mut self, band: BandEnergy, sensitivity: f32) {
        let norm = band.norm();
        let novelty = norm.above - norm.beneath;
        let stability = 1.0 - norm.here;
        let sens = sensitivity.clamp(-1.0, 1.0);
        let delta = self.alpha * (0.5 * novelty - 0.3 * stability + 0.4 * sens);
        let step = delta.round() as i32;
        let new_rank = (self.rank as i32 + step).clamp(self.r_min as i32, self.r_max as i32);
        self.rank = new_rank as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injector_respects_bounds() {
        let mut inj = Injector::with_rank(4, 1.0).with_bounds(2, 8);
        let band = BandEnergy {
            above: 0.9,
            here: 0.05,
            beneath: 0.05,
            drift: 0.0,
        };
        inj.rank_gate(band, 0.5);
        assert!(inj.rank() >= 4);
        let suppress = BandEnergy {
            above: 0.1,
            here: 0.7,
            beneath: 0.2,
            drift: 0.0,
        };
        inj.rank_gate(suppress, -0.8);
        assert!(inj.rank() >= 2);
    }
}
