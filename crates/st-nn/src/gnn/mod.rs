// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "psi")]
pub mod coherence;
use crate::schedule::{BandEnergy, RoundtableSchedule};
use std::time::SystemTime;

pub mod context;
pub mod handoff;
pub mod layer;
pub mod spiralk;
pub mod stack;

#[cfg(feature = "psi")]
pub use coherence::{PsiCoherenceAdaptor, PsiCoherenceDiagnostics};
pub use context::{GraphContext, GraphContextBuilder, GraphNormalization};
pub use handoff::{
    embed_into_biome, flows_to_canvas_tensor, flows_to_canvas_tensor_with_shape,
    fold_into_roundtable, fold_with_band_energy, GraphMonadExport, QuadBandEnergy,
};
pub use layer::{AggregationReducer, NeighborhoodAggregation, ZSpaceGraphConvolution};
pub use spiralk::{GraphConsensusBridge, GraphConsensusDigest};
pub use stack::{GraphActivation, GraphLayerSpec, ZSpaceGraphNetwork, ZSpaceGraphNetworkBuilder};

/// Snapshot of the roundtable energy split and corresponding schedule that should
/// influence graph message passing.
#[derive(Clone, Debug)]
pub struct RoundtableBandSignal {
    energy: BandEnergy,
    band_sizes: (u32, u32, u32),
    issued_at: SystemTime,
}

impl RoundtableBandSignal {
    /// Builds a signal from explicit energy magnitudes and band sizes.
    pub fn new(energy: BandEnergy, band_sizes: (u32, u32, u32)) -> Self {
        Self {
            energy,
            band_sizes,
            issued_at: SystemTime::now(),
        }
    }

    /// Builds a signal directly from the active roundtable schedule.
    pub fn from_schedule(schedule: &RoundtableSchedule, energy: BandEnergy) -> Self {
        Self {
            energy,
            band_sizes: (schedule.above().k, schedule.here().k, schedule.beneath().k),
            issued_at: SystemTime::now(),
        }
    }

    /// Overrides the timestamp associated with the signal.
    pub fn with_issued_at(mut self, issued_at: SystemTime) -> Self {
        self.issued_at = issued_at;
        self
    }

    /// Returns the raw band energy magnitudes.
    pub fn energy(&self) -> BandEnergy {
        self.energy
    }

    /// Returns the `(TopK, MidK, BottomK)` schedule sizes.
    pub fn band_sizes(&self) -> (u32, u32, u32) {
        self.band_sizes
    }

    /// Returns when the signal was emitted.
    pub fn issued_at(&self) -> SystemTime {
        self.issued_at
    }

    /// Returns the combined depth of the schedule.
    pub fn depth(&self) -> u32 {
        let (above, here, beneath) = self.band_sizes;
        above + here + beneath
    }
}

/// Derived multipliers computed from a [`RoundtableBandSignal`] that can be
/// applied to each message passing step.
#[derive(Clone, Debug)]
pub struct RoundtableBandInfluence {
    signal: RoundtableBandSignal,
    multipliers: (f32, f32, f32),
    drift_bias: f32,
}

impl RoundtableBandInfluence {
    /// Builds a new influence from the provided signal.
    pub fn from_signal(signal: &RoundtableBandSignal) -> Self {
        let energy = signal.energy();
        let total = (energy.above.abs() + energy.here.abs() + energy.beneath.abs()).max(1e-6);
        let bary = [
            (energy.above.abs() / total).clamp(0.0, 1.0),
            (energy.here.abs() / total).clamp(0.0, 1.0),
            (energy.beneath.abs() / total).clamp(0.0, 1.0),
        ];
        let depth = signal.depth().max(1);
        let (k_above, k_here, k_beneath) = signal.band_sizes();
        let share = [
            k_above as f32 / depth as f32,
            k_here as f32 / depth as f32,
            k_beneath as f32 / depth as f32,
        ];
        let spectral_focus = energy.spectral_focus();
        let spectral_curvature = energy.spectral_curvature();
        let spectral_stability = energy.spectral_stability();
        let spectral_spin = energy.spectral.spin.clamp(-1.0, 1.0);
        let bary_asymmetry = (bary[0] - bary[2]).abs().clamp(0.0, 1.0);
        let energy_reliance =
            (0.5 + 0.35 * spectral_focus + 0.25 * bary_asymmetry).clamp(0.55, 0.9);
        let schedule_reliance = 1.0 - energy_reliance;
        let occupancy = [
            energy_reliance * bary[0] + schedule_reliance * share[0],
            energy_reliance * bary[1] + schedule_reliance * share[1],
            energy_reliance * bary[2] + schedule_reliance * share[2],
        ];
        let schedule_asymmetry = (occupancy[0] - occupancy[2]).clamp(-1.0, 1.0);
        let direction =
            (0.6 * schedule_asymmetry + 0.4 * spectral_focus * spectral_spin).clamp(-1.0, 1.0);
        let curvature_pull = (spectral_focus * spectral_curvature).clamp(0.0, 1.0);
        let edge_bias = (bary[0].max(bary[2]) - bary[1]).max(0.0);
        let edge_emphasis =
            (0.5 * edge_bias + 0.5 * direction.abs() * spectral_focus).clamp(0.0, 1.0);
        let edge_lift = (1.0 + 0.18 * edge_emphasis).clamp(1.0, 1.35);
        let center_damping = (1.0 - 0.24 * edge_emphasis).clamp(0.72, 1.0);
        let stability_lift = (0.92 + 0.16 * spectral_stability + 0.10 * edge_bias).clamp(0.85, 1.2);
        let neutrality =
            ((occupancy[1] + spectral_stability + curvature_pull) / 3.0).clamp(0.0, 1.0);
        let directional_relaxation = (1.0 - curvature_pull).clamp(0.65, 1.0);
        let mut multipliers = (
            (0.72 + 0.62 * occupancy[0])
                * (1.0 + 0.26 * direction.max(0.0) + 0.12 * edge_bias)
                * directional_relaxation,
            (0.78 + 0.52 * occupancy[1])
                * (1.0 + 0.18 * curvature_pull + 0.12 * neutrality)
                * (1.0 + 0.07 * (1.0 - spectral_focus))
                * center_damping,
            (0.72 + 0.62 * occupancy[2])
                * (1.0 + 0.26 * (-direction).max(0.0) + 0.12 * edge_bias)
                * directional_relaxation,
        );
        multipliers.0 *= edge_lift;
        multipliers.2 *= edge_lift;
        if direction > 0.0 {
            multipliers.0 *= stability_lift;
        } else if direction < 0.0 {
            multipliers.2 *= stability_lift;
        } else {
            multipliers.1 *= stability_lift;
        }
        multipliers.0 = multipliers.0.clamp(0.35, 1.75);
        multipliers.1 = multipliers.1.clamp(0.35, 1.75);
        multipliers.2 = multipliers.2.clamp(0.35, 1.75);
        let directional_bias = (0.55 * energy.drift.tanh() + 0.45 * direction).clamp(-1.0, 1.0);
        let drift_bias = ((1.0 + 0.22 * directional_bias)
            * (1.0 + 0.08 * edge_emphasis)
            * (1.0 - 0.12 * curvature_pull)
            * (1.0 - 0.06 * neutrality * spectral_stability))
            .clamp(0.55, 1.45);
        Self {
            signal: signal.clone(),
            multipliers,
            drift_bias,
        }
    }

    /// Returns the raw signal that produced this influence.
    pub fn signal(&self) -> &RoundtableBandSignal {
        &self.signal
    }

    /// Returns the per-band multipliers derived from the signal.
    pub fn multipliers(&self) -> (f32, f32, f32) {
        self.multipliers
    }

    /// Returns the drift bias used to emphasise the Above/Beneath steps.
    pub fn drift_bias(&self) -> f32 {
        self.drift_bias
    }

    /// Returns the scaling applied to the provided message passing step.
    pub fn scale_for_step(&self, step_index: usize) -> f32 {
        let mut scale = match step_index {
            0 => self.multipliers.1,
            1 => self.multipliers.0 * self.drift_bias,
            _ => {
                let mut base = self.multipliers.2 / self.drift_bias.max(0.5);
                if step_index > 2 {
                    let decay = 1.0 - ((step_index - 2) as f32 * 0.05);
                    base *= decay.max(0.6);
                }
                base
            }
        };
        scale = scale.clamp(0.25, 2.0);
        scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::ops::zspace_round::SpectralFeatureSample;

    #[test]
    fn directional_signal_biases_above_step() {
        let signal = RoundtableBandSignal::new(
            BandEnergy::new(1.4, 0.45, 0.2)
                .with_drift(0.35)
                .with_spectral(SpectralFeatureSample {
                    sheet_index: 1,
                    sheet_confidence: 0.9,
                    curvature: 0.6,
                    spin: 0.85,
                    energy: 0.72,
                }),
            (4, 2, 2),
        );
        let influence = RoundtableBandInfluence::from_signal(&signal);
        let (above, here, beneath) = influence.multipliers();
        assert!(above > beneath);
        assert!(influence.scale_for_step(1) > influence.scale_for_step(2));
        assert!(above > here * 0.85);
    }

    #[test]
    fn curvature_pull_reinforces_here_band() {
        let signal = RoundtableBandSignal::new(
            BandEnergy::new(0.3, 1.2, 0.3).with_spectral(SpectralFeatureSample {
                sheet_index: 0,
                sheet_confidence: 0.88,
                curvature: 3.4,
                spin: 0.1,
                energy: 0.58,
            }),
            (2, 5, 2),
        );
        let influence = RoundtableBandInfluence::from_signal(&signal);
        let (above, here, beneath) = influence.multipliers();
        assert!(here > above);
        assert!(here > beneath);
        assert!(influence.scale_for_step(0) >= influence.scale_for_step(1));
    }

    #[test]
    fn edge_skewed_signal_can_promote_above_step() {
        let signal = RoundtableBandSignal::new(
            BandEnergy::new(0.19, 0.14, 0.01).with_spectral(SpectralFeatureSample {
                sheet_index: 0,
                sheet_confidence: 0.62,
                curvature: 0.48,
                spin: 0.45,
                energy: 0.08,
            }),
            (1, 2, 1),
        );
        let influence = RoundtableBandInfluence::from_signal(&signal);
        let (above, here, beneath) = influence.multipliers();
        assert!(above > beneath);
        assert!(influence.scale_for_step(1) > influence.scale_for_step(0));
        assert!(above >= here * 0.95);
    }
}
