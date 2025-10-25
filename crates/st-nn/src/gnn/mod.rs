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
        let mut multipliers = (
            (0.7 + 0.6 * bary[0]) * (0.55 + 0.45 * share[0]),
            (0.7 + 0.6 * bary[1]) * (0.55 + 0.45 * share[1]),
            (0.7 + 0.6 * bary[2]) * (0.55 + 0.45 * share[2]),
        );
        multipliers.0 = multipliers.0.clamp(0.35, 1.75);
        multipliers.1 = multipliers.1.clamp(0.35, 1.75);
        multipliers.2 = multipliers.2.clamp(0.35, 1.75);
        let drift_bias = (1.0 + energy.drift.tanh() * 0.2).clamp(0.5, 1.5);
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
