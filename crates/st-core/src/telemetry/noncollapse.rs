// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::theory::zpulse::ZPulse;
use serde::{Deserialize, Serialize};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, AmegaHypergrad, HypergradTelemetry};

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
        let before_fields = self.observed_fields();
        let incoming_fields = other.observed_fields();
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
        emit_noncollapse_snapshot_meta(
            "noncollapse_snapshot_merge",
            "st_core_noncollapse_snapshot_merge",
            self,
            &[before_fields, incoming_fields],
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

    fn observed_fields(&self) -> usize {
        usize::from(self.coherence_entropy.is_some())
            + usize::from(self.preserved_channels.is_some())
            + usize::from(self.discarded_channels.is_some())
            + usize::from(self.z_bias.is_some())
            + usize::from(self.hypergrad_penalty.is_some())
            + usize::from(self.phase.is_some())
            + usize::from(self.band_energy.is_some())
            + usize::from(self.dominant_channel.is_some())
            + usize::from(self.energy_ratio.is_some())
            + usize::from(self.mean_coherence.is_some())
            + usize::from(self.hypergrad_l2.is_some())
            + usize::from(self.hypergrad_linf.is_some())
            + usize::from(self.hypergrad_non_finite_ratio.is_some())
            + usize::from(self.pre_discard_preserved_ratio.is_some())
            + usize::from(self.pre_discard_survivor_energy_ratio.is_some())
    }
}

impl From<HypergradTelemetry> for NonCollapseSnapshot {
    fn from(telemetry: HypergradTelemetry) -> Self {
        let summary = telemetry.summary();
        let snapshot = Self {
            hypergrad_l2: finite_some(summary.l2()),
            hypergrad_linf: finite_some(summary.linf()),
            hypergrad_non_finite_ratio: finite_some(telemetry.non_finite_ratio()),
            ..Self::default()
        };
        emit_noncollapse_snapshot_meta(
            "noncollapse_hypergrad_snapshot",
            "st_core_noncollapse_hypergrad_snapshot",
            &snapshot,
            &[snapshot.observed_fields()],
        );
        snapshot
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
        let snapshot = Self::new()
            .with_z_bias(pulse.z_bias)
            .with_band_energy(pulse.band_energy);
        emit_noncollapse_snapshot_meta(
            "noncollapse_zpulse_snapshot",
            "st_core_noncollapse_zpulse_snapshot",
            &snapshot,
            &[snapshot.observed_fields()],
        );
        snapshot
    }
}

fn emit_noncollapse_snapshot_meta(
    op_name: &'static str,
    kind: &'static str,
    snapshot: &NonCollapseSnapshot,
    input_shape: &[usize],
) {
    let fields = snapshot.observed_fields();
    let band_total = snapshot
        .band_energy
        .map(|(above, here, beneath)| above + here + beneath)
        .unwrap_or(0.0);
    emit_tensor_op(op_name, input_shape, &[1, fields.max(1)]);
    emit_tensor_op_meta(op_name, || {
        serde_json::json!({
            "backend": "cpu",
            "requested_backend": "auto",
            "kind": kind,
            "observed_fields": fields,
            "empty": snapshot.is_empty(),
            "phase": snapshot.phase.map(|phase| phase.as_str()).unwrap_or("none"),
            "has_coherence": snapshot.coherence_entropy.is_some()
                || snapshot.mean_coherence.is_some(),
            "coherence_entropy": snapshot.coherence_entropy.unwrap_or(0.0),
            "mean_coherence": snapshot.mean_coherence.unwrap_or(0.0),
            "preserved_channels": snapshot.preserved_channels.unwrap_or(0),
            "discarded_channels": snapshot.discarded_channels.unwrap_or(0),
            "has_z_bias": snapshot.z_bias.is_some(),
            "z_bias": snapshot.z_bias.unwrap_or(0.0),
            "has_band_energy": snapshot.band_energy.is_some(),
            "band_energy_total": band_total,
            "dominant_channel": snapshot.dominant_channel.unwrap_or(usize::MAX),
            "has_hypergrad": snapshot.hypergrad_l2.is_some()
                || snapshot.hypergrad_linf.is_some()
                || snapshot.hypergrad_non_finite_ratio.is_some(),
            "hypergrad_l2": snapshot.hypergrad_l2.unwrap_or(0.0),
            "hypergrad_linf": snapshot.hypergrad_linf.unwrap_or(0.0),
            "hypergrad_non_finite_ratio": snapshot.hypergrad_non_finite_ratio.unwrap_or(0.0),
            "pre_discard_preserved_ratio": snapshot
                .pre_discard_preserved_ratio
                .unwrap_or(0.0),
            "pre_discard_survivor_energy_ratio": snapshot
                .pre_discard_survivor_energy_ratio
                .unwrap_or(0.0),
        })
    });
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
    use std::sync::{Arc, Mutex};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::telemetry::tensor_observer_lock()
    }

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

    #[test]
    fn snapshots_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let left = NonCollapseSnapshot::new()
            .with_coherence_metrics(0.6, 5, 2)
            .with_z_bias(0.2);
        let right = NonCollapseSnapshot::new()
            .with_band_energy((0.6, 0.2, 0.1))
            .with_phase(NonCollapsePhase::Integration);
        let merged = left.merged(&right);

        let mut tape = AmegaHypergrad::new(-0.9, 0.05, 1, 3).unwrap();
        let tensor = Tensor::from_vec(1, 3, vec![0.5, -0.25, 0.1]).unwrap();
        tape.accumulate_wave(&tensor).unwrap();
        let hypergrad = NonCollapseSnapshot::from(&tape);

        let pulse = ZPulse {
            source: ZSource::Maxwell,
            ts: 1,
            tempo: 0.5,
            band_energy: (0.4, 0.3, 0.2),
            density_fluctuation: 0.1,
            drift: 0.3,
            z_bias: 0.35,
            support: ZSupport::from((0.4, 0.3, 0.2)),
            scale: ZScale::new(2.0),
            quality: 0.7,
            stderr: 0.1,
            latency_ms: 4.0,
        };
        let z_snapshot = NonCollapseSnapshot::from(&pulse);
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(merged.phase, Some(NonCollapsePhase::Integration));
        assert!(hypergrad.hypergrad_l2.unwrap_or_default() > 0.0);
        assert_eq!(z_snapshot.z_bias, Some(0.35));

        let events = events.lock().unwrap();
        let merge = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "noncollapse_snapshot_merge"
                    && data["phase"] == "integration"
                    && data["has_band_energy"] == true
            })
            .expect("noncollapse_snapshot_merge metadata event");
        assert_eq!(merge.1["backend"], "cpu");
        assert_eq!(merge.1["kind"], "st_core_noncollapse_snapshot_merge");

        let hypergrad_event = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "noncollapse_hypergrad_snapshot" && data["has_hypergrad"] == true
            })
            .expect("noncollapse_hypergrad_snapshot metadata event");
        assert_eq!(
            hypergrad_event.1["kind"],
            "st_core_noncollapse_hypergrad_snapshot"
        );

        let zpulse_event = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "noncollapse_zpulse_snapshot"
                    && data["has_z_bias"] == true
                    && data["has_band_energy"] == true
            })
            .expect("noncollapse_zpulse_snapshot metadata event");
        assert_eq!(
            zpulse_event.1["kind"],
            "st_core_noncollapse_zpulse_snapshot"
        );
        assert!(zpulse_event.1["band_energy_total"].as_f64().unwrap_or(0.0) > 0.0);
    }
}
