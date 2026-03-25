// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::theory::zpulse::ZPulse;
use serde::{Deserialize, Serialize};
use st_tensor::{AmegaHypergrad, HypergradTelemetry};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonCollapsePhase {
    Observation,
    Injection,
    Integration,
}

impl NonCollapsePhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observation => "observation",
            Self::Injection => "injection",
            Self::Integration => "integration",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NonCollapseSnapshot {
    pub coherence_entropy: Option<f32>,
    pub preserved_channels: Option<usize>,
    pub discarded_channels: Option<usize>,
    pub z_bias: Option<f32>,
    pub hypergrad_penalty: Option<f32>,
    pub phase: Option<NonCollapsePhase>,
    pub band_energy: Option<(f32, f32, f32)>,
    pub dominant_channel: Option<usize>,
    pub energy_ratio: Option<f32>,
    pub mean_coherence: Option<f32>,
    pub hypergrad_l2: Option<f32>,
    pub hypergrad_linf: Option<f32>,
    pub hypergrad_non_finite_ratio: Option<f32>,
    pub pre_discard_preserved_ratio: Option<f32>,
    pub pre_discard_survivor_energy_ratio: Option<f32>,
}

impl NonCollapseSnapshot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    pub fn merge_from(&mut self, other: &Self) {
        merge_option(&mut self.coherence_entropy, other.coherence_entropy);
        merge_option(&mut self.preserved_channels, other.preserved_channels);
        merge_option(&mut self.discarded_channels, other.discarded_channels);
        merge_option(&mut self.z_bias, other.z_bias);
        merge_option(&mut self.hypergrad_penalty, other.hypergrad_penalty);
        merge_option(&mut self.phase, other.phase);
        merge_option(&mut self.band_energy, other.band_energy);
        merge_option(&mut self.dominant_channel, other.dominant_channel);
        merge_option(&mut self.energy_ratio, other.energy_ratio);
        merge_option(&mut self.mean_coherence, other.mean_coherence);
        merge_option(&mut self.hypergrad_l2, other.hypergrad_l2);
        merge_option(&mut self.hypergrad_linf, other.hypergrad_linf);
        merge_option(
            &mut self.hypergrad_non_finite_ratio,
            other.hypergrad_non_finite_ratio,
        );
        merge_option(
            &mut self.pre_discard_preserved_ratio,
            other.pre_discard_preserved_ratio,
        );
        merge_option(
            &mut self.pre_discard_survivor_energy_ratio,
            other.pre_discard_survivor_energy_ratio,
        );
    }

    pub fn merged(&self, other: &Self) -> Self {
        let mut merged = self.clone();
        merged.merge_from(other);
        merged
    }

    pub fn with_coherence_metrics(
        mut self,
        entropy: f32,
        preserved: usize,
        discarded: usize,
    ) -> Self {
        self.coherence_entropy = finite_some(entropy);
        self.preserved_channels = Some(preserved);
        self.discarded_channels = Some(discarded);
        self
    }

    pub fn with_z_bias(mut self, z_bias: f32) -> Self {
        self.z_bias = finite_some(z_bias);
        self
    }

    pub fn with_hypergrad_penalty(mut self, penalty: f32) -> Self {
        self.hypergrad_penalty = finite_some(penalty);
        self
    }

    pub fn with_phase(mut self, phase: NonCollapsePhase) -> Self {
        self.phase = Some(phase);
        self
    }

    pub fn with_band_energy(mut self, band_energy: (f32, f32, f32)) -> Self {
        self.band_energy = Some((
            sanitise_non_negative(band_energy.0),
            sanitise_non_negative(band_energy.1),
            sanitise_non_negative(band_energy.2),
        ));
        self
    }

    pub fn with_coherence_profile(
        mut self,
        dominant_channel: Option<usize>,
        energy_ratio: f32,
        mean_coherence: f32,
    ) -> Self {
        self.dominant_channel = dominant_channel;
        self.energy_ratio = finite_some(energy_ratio);
        self.mean_coherence = finite_some(mean_coherence);
        self
    }

    pub fn with_pre_discard_ratios(
        mut self,
        preserved_ratio: f32,
        survivor_energy_ratio: f32,
    ) -> Self {
        self.pre_discard_preserved_ratio = finite_some(preserved_ratio);
        self.pre_discard_survivor_energy_ratio = finite_some(survivor_energy_ratio);
        self
    }
}

impl From<HypergradTelemetry> for NonCollapseSnapshot {
    fn from(telemetry: HypergradTelemetry) -> Self {
        let summary = telemetry.summary();
        Self {
            hypergrad_l2: finite_some(summary.l2()),
            hypergrad_linf: finite_some(summary.linf()),
            hypergrad_non_finite_ratio: finite_some(telemetry.non_finite_ratio()),
            ..Self::default()
        }
    }
}

impl From<&HypergradTelemetry> for NonCollapseSnapshot {
    fn from(telemetry: &HypergradTelemetry) -> Self {
        (*telemetry).into()
    }
}

impl From<&AmegaHypergrad> for NonCollapseSnapshot {
    fn from(tape: &AmegaHypergrad) -> Self {
        tape.telemetry().into()
    }
}

impl From<&ZPulse> for NonCollapseSnapshot {
    fn from(pulse: &ZPulse) -> Self {
        Self::new()
            .with_z_bias(pulse.z_bias)
            .with_band_energy(pulse.band_energy)
    }
}

fn merge_option<T: Copy>(slot: &mut Option<T>, value: Option<T>) {
    if let Some(value) = value {
        *slot = Some(value);
    }
}

fn finite_some(value: f32) -> Option<f32> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

fn sanitise_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::zpulse::{ZScale, ZSource, ZSupport};
    use st_tensor::Tensor;

    #[test]
    fn snapshot_merges_sparse_sources() {
        let left = NonCollapseSnapshot::new()
            .with_coherence_metrics(0.6, 5, 2)
            .with_z_bias(0.2);
        let right = NonCollapseSnapshot::new()
            .with_hypergrad_penalty(0.4)
            .with_phase(NonCollapsePhase::Integration);

        let merged = left.merged(&right);
        assert_eq!(merged.coherence_entropy, Some(0.6));
        assert_eq!(merged.preserved_channels, Some(5));
        assert_eq!(merged.discarded_channels, Some(2));
        assert_eq!(merged.z_bias, Some(0.2));
        assert_eq!(merged.hypergrad_penalty, Some(0.4));
        assert_eq!(merged.phase, Some(NonCollapsePhase::Integration));
    }

    #[test]
    fn snapshot_from_hypergrad_tracks_norms() {
        let mut tape = AmegaHypergrad::new(-0.9, 0.05, 1, 3).unwrap();
        let tensor = Tensor::from_vec(1, 3, vec![0.5, -0.25, 0.1]).unwrap();
        tape.accumulate_wave(&tensor).unwrap();

        let snapshot = NonCollapseSnapshot::from(&tape);
        assert!(snapshot.hypergrad_l2.unwrap_or_default() > 0.0);
        assert!(snapshot.hypergrad_linf.unwrap_or_default() > 0.0);
        assert_eq!(snapshot.hypergrad_non_finite_ratio, Some(0.0));
    }

    #[test]
    fn snapshot_from_zpulse_tracks_bias_and_band_energy() {
        let pulse = ZPulse {
            source: ZSource::Maxwell,
            ts: 1,
            tempo: 0.5,
            band_energy: (0.6, 0.2, 0.1),
            density_fluctuation: 0.1,
            drift: 0.3,
            z_bias: 0.25,
            support: ZSupport::from((0.6, 0.2, 0.1)),
            scale: ZScale::new(2.0),
            quality: 0.7,
            stderr: 0.1,
            latency_ms: 4.0,
        };

        let snapshot = NonCollapseSnapshot::from(&pulse);
        assert_eq!(snapshot.z_bias, Some(0.25));
        assert_eq!(snapshot.band_energy, Some((0.6, 0.2, 0.1)));
    }
}
