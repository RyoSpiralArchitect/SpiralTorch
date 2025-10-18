// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Microlocal pulse fusion utilities shared between the observability
//! backends and telemetry exporters.

use std::collections::VecDeque;

/// Origin marker for a captured pulse.
#[derive(Clone, Debug, PartialEq)]
pub enum ZSource {
    Microlocal,
    Maxwell,
    Other(String),
}

impl Default for ZSource {
    fn default() -> Self {
        Self::Other("unspecified".to_string())
    }
}

/// Envelope information for the leading/central/trailing bands.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZSupport {
    pub leading: f32,
    pub central: f32,
    pub trailing: f32,
}

/// Snapshot of a single microlocal pulse observation.
#[derive(Clone, Debug, PartialEq)]
pub struct ZPulse {
    pub source: ZSource,
    pub ts: u64,
    pub tempo: f32,
    pub drift: f32,
    pub z_bias: f32,
    pub support: ZSupport,
    pub band_energy: (f32, f32, f32),
    pub quality: f32,
    pub stderr: f32,
    pub latency_ms: f32,
}

impl Default for ZPulse {
    fn default() -> Self {
        Self {
            source: ZSource::default(),
            ts: 0,
            tempo: 0.0,
            drift: 0.0,
            z_bias: 0.0,
            support: ZSupport::default(),
            band_energy: (0.0, 0.0, 0.0),
            quality: 0.0,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }
}

/// Configuration for the conductor frequency tracker.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZFrequencyConfig {
    pub target_hz: f32,
    pub window_hz: f32,
}

/// Configuration for adaptive gain smoothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZAdaptiveGainCfg {
    pub min_gain: f32,
    pub max_gain: f32,
    pub smoothing: f32,
}

/// Configuration for latency alignment smoothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZLatencyConfig {
    pub target_ms: f32,
    pub clamp_ms: f32,
    pub smoothing: f32,
}

impl ZLatencyConfig {
    pub fn new(target_ms: f32, clamp_ms: f32, smoothing: f32) -> Self {
        Self {
            target_ms,
            clamp_ms,
            smoothing,
        }
    }

    pub fn update(&mut self, target_ms: f32, clamp_ms: f32, smoothing: f32) {
        self.target_ms = target_ms;
        self.clamp_ms = clamp_ms;
        self.smoothing = smoothing;
    }
}

/// Latency aligner state used by the conductor to keep a rolling window.
#[derive(Clone, Debug, PartialEq)]
pub struct LatencyAligner {
    pub config: ZLatencyConfig,
    pub last_latency: f32,
}

impl LatencyAligner {
    pub fn new(config: ZLatencyConfig) -> Self {
        Self {
            config,
            last_latency: 0.0,
        }
    }

    pub fn update_config(&mut self, config: ZLatencyConfig) {
        self.config = config;
    }

    pub fn observe(&mut self, latency_ms: f32) {
        let blend = self.config.smoothing.clamp(0.0, 1.0);
        self.last_latency = blend * latency_ms + (1.0 - blend) * self.last_latency;
        self.last_latency = self.last_latency.clamp(0.0, self.config.clamp_ms.max(0.0));
    }
}

/// Global configuration for the conductor.
#[derive(Clone, Debug, PartialEq)]
pub struct ZConductorCfg {
    pub gain: f32,
    pub alpha_fast: f32,
    pub alpha_slow: f32,
    pub slew_max: f32,
    pub flip_hold: u32,
    pub freq: Option<ZFrequencyConfig>,
    pub adaptive_gain: Option<ZAdaptiveGainCfg>,
    pub latency: Option<ZLatencyConfig>,
}

impl Default for ZConductorCfg {
    fn default() -> Self {
        Self {
            gain: 1.0,
            alpha_fast: 0.5,
            alpha_slow: 0.05,
            slew_max: 1.0,
            flip_hold: 0,
            freq: None,
            adaptive_gain: None,
            latency: None,
        }
    }
}

/// Aggregated fused pulse after running through the conductor.
#[derive(Clone, Debug, PartialEq)]
pub struct ZFused {
    pub ts: u64,
    pub tempo: f32,
    pub drift: f32,
    pub z: f32,
    pub pulse: ZPulse,
}

impl Default for ZFused {
    fn default() -> Self {
        Self {
            ts: 0,
            tempo: 0.0,
            drift: 0.0,
            z: 0.0,
            pulse: ZPulse::default(),
        }
    }
}

/// Rolling conductor that fuses incoming pulses with simple exponential
/// smoothing.
#[derive(Clone, Debug)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    latency: Option<LatencyAligner>,
    queue: VecDeque<ZPulse>,
    fused: ZFused,
}

impl ZConductor {
    pub fn new(cfg: ZConductorCfg) -> Self {
        let latency = cfg.latency.clone().map(LatencyAligner::new);
        Self {
            cfg,
            latency,
            queue: VecDeque::new(),
            fused: ZFused::default(),
        }
    }

    pub fn cfg(&self) -> &ZConductorCfg {
        &self.cfg
    }

    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }

    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.cfg.freq = cfg;
    }

    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
        self.cfg.adaptive_gain = cfg;
    }

    pub fn set_latency_config(&mut self, cfg: Option<ZLatencyConfig>) {
        self.cfg.latency = cfg.clone();
        self.latency = cfg.map(LatencyAligner::new);
    }

    pub fn ingest(&mut self, pulse: ZPulse) {
        if let Some(latency) = &mut self.latency {
            latency.observe(pulse.latency_ms);
        }
        self.queue.push_back(pulse);
    }

    pub fn step(&mut self, now: u64) -> ZFused {
        if let Some(mut latest) = self.queue.pop_back() {
            // retain only the latest observation for the simple smoother.
            self.queue.clear();
            let alpha = self.cfg.alpha_fast.clamp(0.0, 1.0);
            self.fused.tempo = alpha * latest.tempo + (1.0 - alpha) * self.fused.tempo;
            self.fused.drift = alpha * latest.drift + (1.0 - alpha) * self.fused.drift;
            self.fused.z = alpha * latest.z_bias + (1.0 - alpha) * self.fused.z;
            latest.ts = now;
            self.fused.ts = now;
            self.fused.pulse = latest;
        } else {
            self.fused.ts = now;
        }
        self.fused.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_aligner_clamps() {
        let mut aligner = LatencyAligner::new(ZLatencyConfig::new(8.0, 10.0, 0.5));
        aligner.observe(12.0);
        assert!(aligner.last_latency <= 10.0);
        aligner.observe(2.0);
        assert!(aligner.last_latency >= 0.0);
    }

    #[test]
    fn conductor_blends_pulses() {
        let cfg = ZConductorCfg {
            alpha_fast: 1.0,
            ..Default::default()
        };
        let mut conductor = ZConductor::new(cfg);
        conductor.ingest(ZPulse {
            tempo: 42.0,
            drift: 0.5,
            z_bias: 0.25,
            ..ZPulse::default()
        });
        let fused = conductor.step(10);
        assert_eq!(fused.ts, 10);
        assert_eq!(fused.pulse.ts, 10);
        assert!((fused.tempo - 42.0).abs() < f32::EPSILON);
    }
}
