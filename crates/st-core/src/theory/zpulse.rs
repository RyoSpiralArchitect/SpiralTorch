// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Minimal pulse fusion scaffolding used by the microlocal and Maxwell
//! interfaces when projecting runtime telemetry into Z-space.  The goal here is
//! to provide a compact, dependency-free representation that downstream crates
//! can rely on while the full adaptive conductor is iterated on elsewhere.

use std::time::Duration;

/// Identifies the producer that emitted a [`ZPulse`].
#[derive(Clone, Debug, PartialEq)]
pub enum ZSource {
    /// Pulse emitted from the Maxwell softlogic pipeline.
    Maxwell,
    /// Pulse emitted from the microlocal interface gauges.
    Microlocal,
    /// Catch-all variant used by callers that want to tag the pulse with a
    /// custom label (for example RealGrad).
    Other(String),
}

impl ZSource {
    /// Convenience helper that builds the catch-all [`ZSource::Other`] variant.
    pub fn other(label: impl Into<String>) -> Self {
        ZSource::Other(label.into())
    }
}

impl Default for ZSource {
    fn default() -> Self {
        ZSource::Microlocal
    }
}

/// Snapshot of a single gauge pulse expressed in Z-space.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZPulse {
    pub source: ZSource,
    pub ts: u64,
    pub band_energy: (f32, f32, f32),
    pub quality: f32,
    pub stderr: f32,
    pub latency_ms: f32,
    pub tempo: f32,
}

impl ZPulse {
    /// Returns a pulse tagged with the provided source and tempo.  All other
    /// attributes default to zero and can be filled in by the caller as
    /// required.
    pub fn new(source: ZSource, tempo: f32) -> Self {
        ZPulse {
            source,
            tempo,
            ..ZPulse::default()
        }
    }
}

/// Parameters controlling how Z pulses should be fused across time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZFrequencyConfig {
    pub sample_hz: f32,
    pub smoothing: f32,
}

/// Parameters controlling adaptive gain adjustments when combining pulses.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZAdaptiveGainCfg {
    pub gain: f32,
    pub clamp: (f32, f32),
}

/// Parameters for the optional latency aligner.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZLatencyConfig {
    pub target: Duration,
    pub window: Duration,
}

/// Aggregated pulse produced by [`ZConductor::step`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZFused {
    pub pulse: Option<ZPulse>,
    pub ts: u64,
}

/// Configuration for a [`ZConductor`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZConductorCfg {
    pub freq: Option<ZFrequencyConfig>,
    pub adaptive_gain: Option<ZAdaptiveGainCfg>,
    pub latency: Option<ZLatencyConfig>,
}

/// Minimal conductor implementation that keeps track of the last pulse and
/// exposes helpers that mirror the API used by the runtime telemetry modules.
#[derive(Clone, Debug, Default)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    last: Option<ZPulse>,
}

impl ZConductor {
    /// Creates a new conductor from the provided configuration.
    pub fn new(cfg: ZConductorCfg) -> Self {
        ZConductor { cfg, last: None }
    }

    /// Returns a mutable handle to the underlying configuration so callers can
    /// tweak fields in place.
    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }

    /// Updates the frequency fusion settings.
    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.cfg.freq = cfg;
    }

    /// Updates the adaptive gain settings.
    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
        self.cfg.adaptive_gain = cfg;
    }

    /// Updates the latency alignment settings.
    pub fn set_latency_config(&mut self, cfg: Option<ZLatencyConfig>) {
        self.cfg.latency = cfg;
    }

    /// Ingests the latest pulse.  The pulse is stored verbatim so callers can
    /// inspect it prior to calling [`Self::step`].
    pub fn ingest(&mut self, pulse: ZPulse) {
        self.last = Some(pulse);
    }

    /// Produces the current fused pulse.  The implementation is intentionally
    /// simple: it returns the last ingested pulse and clears the storage slot so
    /// repeated calls do not emit duplicates.  The returned [`ZFused`] record is
    /// still future-proof should more elaborate fusion logic be introduced
    /// later.
    pub fn step(&mut self, now: u64) -> ZFused {
        ZFused {
            pulse: self.last.take(),
            ts: now,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conductor_emits_last_pulse() {
        let mut conductor = ZConductor::default();
        assert!(conductor.step(0).pulse.is_none());

        conductor.ingest(ZPulse::new(ZSource::Microlocal, 1.5));
        let fused = conductor.step(10);
        assert_eq!(fused.ts, 10);
        assert!(fused.pulse.is_some());
        assert!(conductor.step(11).pulse.is_none());
    }

    #[test]
    fn config_helpers_update_fields() {
        let mut conductor = ZConductor::new(ZConductorCfg::default());
        conductor.set_frequency_config(Some(ZFrequencyConfig {
            sample_hz: 2.0,
            smoothing: 0.5,
        }));
        conductor.set_adaptive_gain_config(Some(ZAdaptiveGainCfg {
            gain: 1.5,
            clamp: (0.0, 2.0),
        }));
        conductor.set_latency_config(Some(ZLatencyConfig {
            target: Duration::from_millis(5),
            window: Duration::from_millis(20),
        }));

        assert!(conductor.cfg.freq.is_some());
        assert!(conductor.cfg.adaptive_gain.is_some());
        assert!(conductor.cfg.latency.is_some());

        let cfg = conductor.cfg_mut();
        cfg.freq = None;
        assert!(conductor.cfg.freq.is_none());
    }
}
