// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "psi")]
pub mod coherence;
use crate::schedule::{BandEnergy, RoundtableSchedule};
use crate::{PureResult, TensorError};
use st_core::inference::gnn_roundtable::{
    derive_gnn_roundtable_influence, validate_gnn_roundtable_signal,
    GnnRoundtableBandEnergyObservation, GnnRoundtableBandSizes, GnnRoundtableInfluencePayload,
    GnnRoundtableSignalObservation, GnnRoundtableSpectralObservation,
};
use st_core::ops::zspace_round::RoundtableBand;
use st_core::runtime::trainer_external::TrainerTimestampCheckpoint;
use std::time::SystemTime;

pub mod context;
pub mod handoff;
pub mod layer;
pub mod readout;
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
pub use readout::{
    GraphBatch, GraphBatchReadoutEntry, GraphBatchReadoutErrorEntry, GraphBatchReadoutErrorTrace,
    GraphBatchReadoutTrace, GraphReadout, ZSpaceGraphBatchRegressor, ZSpaceGraphRegressor,
};
pub use spiralk::{GraphConsensusBridge, GraphConsensusDigest};
pub use stack::{GraphActivation, GraphLayerSpec, ZSpaceGraphNetwork, ZSpaceGraphNetworkBuilder};

/// Snapshot of the roundtable energy split and corresponding schedule that should
/// influence graph message passing.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundtableBandSignal {
    energy: BandEnergy,
    band_sizes: (u32, u32, u32),
    issued_at: SystemTime,
    issued_at_checkpoint: TrainerTimestampCheckpoint,
}

impl RoundtableBandSignal {
    /// Builds a signal from explicit energy magnitudes and band sizes.
    pub fn new(energy: BandEnergy, band_sizes: (u32, u32, u32)) -> PureResult<Self> {
        let issued_at = SystemTime::now();
        let issued_at_checkpoint = Self::checkpoint_for_system_time(issued_at)?;
        let signal = Self {
            energy,
            band_sizes,
            issued_at,
            issued_at_checkpoint,
        };
        signal.validate()?;
        Ok(signal)
    }

    /// Builds a signal directly from the active roundtable schedule.
    pub fn from_schedule(schedule: &RoundtableSchedule, energy: BandEnergy) -> PureResult<Self> {
        Self::new(
            energy,
            (schedule.above().k, schedule.here().k, schedule.beneath().k),
        )
    }

    /// Overrides the timestamp associated with the signal.
    pub fn with_issued_at(mut self, issued_at: SystemTime) -> PureResult<Self> {
        let issued_at_checkpoint = Self::checkpoint_for_system_time(issued_at)?;
        self.issued_at = issued_at;
        self.issued_at_checkpoint = issued_at_checkpoint;
        self.validate()?;
        Ok(self)
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

    /// Returns the exact cross-runtime timestamp retained by the signal.
    pub fn issued_at_checkpoint(&self) -> TrainerTimestampCheckpoint {
        self.issued_at_checkpoint
    }

    /// Returns the combined depth of the schedule.
    pub fn depth(&self) -> u64 {
        let (above, here, beneath) = self.band_sizes;
        u64::from(above) + u64::from(here) + u64::from(beneath)
    }

    /// Returns the portable observation validated by the canonical Rust contract.
    pub fn observation(&self) -> PureResult<GnnRoundtableSignalObservation> {
        let sheet_index = u64::try_from(self.energy.spectral.sheet_index).map_err(|_| {
            TensorError::InvalidValue {
                label: "GNN roundtable sheet index",
            }
        })?;
        Ok(GnnRoundtableSignalObservation {
            band_energy: GnnRoundtableBandEnergyObservation {
                above: f64::from(self.energy.above),
                here: f64::from(self.energy.here),
                beneath: f64::from(self.energy.beneath),
                drift: f64::from(self.energy.drift),
            },
            band_sizes: GnnRoundtableBandSizes {
                above: self.band_sizes.0,
                here: self.band_sizes.1,
                beneath: self.band_sizes.2,
            },
            spectral: GnnRoundtableSpectralObservation {
                sheet_index,
                sheet_confidence: f64::from(self.energy.spectral.sheet_confidence),
                curvature: f64::from(self.energy.spectral.curvature),
                spin: f64::from(self.energy.spectral.spin),
                energy: f64::from(self.energy.spectral.energy),
            },
        })
    }

    /// Rebuilds a native signal from a Rust-validated portable observation.
    pub fn from_observation(
        observation: GnnRoundtableSignalObservation,
        issued_at: SystemTime,
    ) -> PureResult<Self> {
        let issued_at_checkpoint = Self::checkpoint_for_system_time(issued_at)?;
        Self::from_observation_checkpoint(observation, issued_at_checkpoint)
    }

    /// Rebuilds a signal while preserving an exact portable checkpoint timestamp.
    pub fn from_observation_checkpoint(
        observation: GnnRoundtableSignalObservation,
        issued_at_checkpoint: TrainerTimestampCheckpoint,
    ) -> PureResult<Self> {
        validate_gnn_roundtable_signal(observation).map_err(|_| TensorError::InvalidValue {
            label: "GNN roundtable signal observation",
        })?;
        issued_at_checkpoint
            .validate("gnn_roundtable_signal.issued_at")
            .map_err(|_| TensorError::InvalidValue {
                label: "GNN roundtable signal timestamp",
            })?;
        let issued_at = issued_at_checkpoint
            .try_to_system_time("gnn_roundtable_signal.issued_at")
            .map_err(|_| TensorError::InvalidValue {
                label: "GNN roundtable signal timestamp",
            })?;
        let sheet_index = usize::try_from(observation.spectral.sheet_index).map_err(|_| {
            TensorError::InvalidValue {
                label: "GNN roundtable sheet index",
            }
        })?;
        let signal = Self {
            energy: BandEnergy::new(
                observation.band_energy.above as f32,
                observation.band_energy.here as f32,
                observation.band_energy.beneath as f32,
            )
            .with_drift(observation.band_energy.drift as f32)
            .with_spectral(st_core::ops::zspace_round::SpectralFeatureSample {
                sheet_index,
                sheet_confidence: observation.spectral.sheet_confidence as f32,
                curvature: observation.spectral.curvature as f32,
                spin: observation.spectral.spin as f32,
                energy: observation.spectral.energy as f32,
            }),
            band_sizes: (
                observation.band_sizes.above,
                observation.band_sizes.here,
                observation.band_sizes.beneath,
            ),
            issued_at,
            issued_at_checkpoint,
        };
        signal.validate()?;
        Ok(signal)
    }

    /// Validates all signal evidence without deriving graph coefficients.
    pub fn validate(&self) -> PureResult<()> {
        validate_gnn_roundtable_signal(self.observation()?).map_err(|_| {
            TensorError::InvalidValue {
                label: "GNN roundtable signal",
            }
        })?;
        self.issued_at_checkpoint
            .validate("gnn_roundtable_signal.issued_at")
            .map_err(|_| TensorError::InvalidValue {
                label: "GNN roundtable signal timestamp",
            })?;
        let projected = self
            .issued_at_checkpoint
            .try_to_system_time("gnn_roundtable_signal.issued_at")
            .map_err(|_| TensorError::InvalidValue {
                label: "GNN roundtable signal timestamp",
            })?;
        if projected != self.issued_at {
            return Err(TensorError::InvalidValue {
                label: "GNN roundtable signal timestamp projection",
            });
        }
        Ok(())
    }

    fn checkpoint_for_system_time(issued_at: SystemTime) -> PureResult<TrainerTimestampCheckpoint> {
        TrainerTimestampCheckpoint::try_from_system_time(
            "gnn_roundtable_signal.issued_at",
            issued_at,
        )
        .map_err(|_| TensorError::InvalidValue {
            label: "GNN roundtable signal timestamp",
        })
    }
}

/// Derived multipliers computed from a [`RoundtableBandSignal`] that can be
/// applied to each message passing step.
#[derive(Clone, Debug)]
pub struct RoundtableBandInfluence {
    signal: RoundtableBandSignal,
    projection: GnnRoundtableInfluencePayload,
}

impl RoundtableBandInfluence {
    /// Builds a new influence from the provided signal.
    pub fn from_signal(signal: &RoundtableBandSignal) -> PureResult<Self> {
        let projection = derive_gnn_roundtable_influence(signal.observation()?).map_err(|_| {
            TensorError::InvalidValue {
                label: "GNN roundtable influence",
            }
        })?;
        Ok(Self {
            signal: signal.clone(),
            projection,
        })
    }

    /// Returns the raw signal that produced this influence.
    pub fn signal(&self) -> &RoundtableBandSignal {
        &self.signal
    }

    /// Returns the per-band multipliers derived from the signal.
    pub fn multipliers(&self) -> (f32, f32, f32) {
        (
            self.projection.multipliers[0] as f32,
            self.projection.multipliers[1] as f32,
            self.projection.multipliers[2] as f32,
        )
    }

    /// Returns the drift bias used to emphasise the Above/Beneath steps.
    pub fn drift_bias(&self) -> f32 {
        self.projection.drift_bias as f32
    }

    /// Returns the scaling applied to the provided message passing step.
    pub fn scale_for_step(&self, step_index: usize) -> f32 {
        self.projection.scale_for_step(step_index as u64) as f32
    }

    /// Returns the replay-specific bias for a message passing step while a
    /// particular gradient band is being replayed through `backward_bands()`.
    pub fn band_pass_scale_for_step(
        &self,
        band: RoundtableBand,
        step_index: usize,
        intensity: f32,
    ) -> PureResult<f32> {
        self.projection
            .band_pass_scale_for_step(band, step_index as u64, f64::from(intensity))
            .map(|scale| scale as f32)
            .map_err(|_| TensorError::InvalidValue {
                label: "GNN roundtable band-pass influence",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::ops::zspace_round::SpectralFeatureSample;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn signal_rejects_timestamp_that_checkpoint_cannot_represent() {
        let issued_at = UNIX_EPOCH.checked_sub(Duration::from_secs(1)).unwrap();
        let signal = RoundtableBandSignal::new(BandEnergy::new(0.8, 0.4, 0.2), (1, 1, 1)).unwrap();

        assert!(signal.with_issued_at(issued_at).is_err());
    }

    #[test]
    fn signal_preserves_exact_portable_timestamp_beyond_system_time_resolution() {
        let source = RoundtableBandSignal::new(BandEnergy::new(0.8, 0.4, 0.2), (1, 1, 1)).unwrap();
        let timestamp = TrainerTimestampCheckpoint {
            unix_seconds: 17,
            subsec_nanos: 42,
        };
        let restored = RoundtableBandSignal::from_observation_checkpoint(
            source.observation().unwrap(),
            timestamp,
        )
        .unwrap();

        assert_eq!(restored.issued_at_checkpoint(), timestamp);
    }

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
        )
        .unwrap();
        let influence = RoundtableBandInfluence::from_signal(&signal).unwrap();
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
        )
        .unwrap();
        let influence = RoundtableBandInfluence::from_signal(&signal).unwrap();
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
        )
        .unwrap();
        let influence = RoundtableBandInfluence::from_signal(&signal).unwrap();
        let (above, here, beneath) = influence.multipliers();
        assert!(above > beneath);
        assert!(influence.scale_for_step(1) > influence.scale_for_step(0));
        assert!(above >= here * 0.95);
    }

    #[test]
    fn band_pass_scaling_biases_matching_steps() {
        let signal =
            RoundtableBandSignal::new(BandEnergy::new(1.1, 0.5, 0.25).with_drift(0.3), (2, 1, 1))
                .unwrap();
        let influence = RoundtableBandInfluence::from_signal(&signal).unwrap();
        let intensity = 0.75;
        let above = influence
            .band_pass_scale_for_step(RoundtableBand::Above, 1, intensity)
            .unwrap();
        let here = influence
            .band_pass_scale_for_step(RoundtableBand::Here, 0, intensity)
            .unwrap();
        let beneath = influence
            .band_pass_scale_for_step(RoundtableBand::Beneath, 2, intensity)
            .unwrap();
        assert!(above > 1.0);
        assert!(here > 1.0);
        assert!(beneath > 1.0);
        assert!(
            influence
                .band_pass_scale_for_step(RoundtableBand::Above, 2, intensity)
                .unwrap()
                < above
        );
        assert!(
            influence
                .band_pass_scale_for_step(RoundtableBand::Beneath, 0, intensity)
                .unwrap()
                < beneath
        );
    }
}
