// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight runtime helpers for fusing microlocal Z pulses into telemetry
//! suitable for downstream control loops.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, Default)]
pub struct ZFrequencyConfig {
    pub smoothing: f32,
    pub minimum_energy: f32,
}

impl ZFrequencyConfig {
    pub fn new(smoothing: f32, minimum_energy: f32) -> Self {
        Self {
            smoothing: smoothing.clamp(0.0, 1.0),
            minimum_energy: minimum_energy.max(0.0),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ZAdaptiveGainCfg {
    pub gain_floor: f32,
    pub gain_ceil: f32,
    pub responsiveness: f32,
}

impl ZAdaptiveGainCfg {
    pub fn new(gain_floor: f32, gain_ceil: f32, responsiveness: f32) -> Self {
        Self {
            gain_floor: gain_floor.max(0.0),
            gain_ceil: gain_ceil.max(gain_floor),
            responsiveness: responsiveness.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ZLatencyConfig {
    pub history: usize,
    pub hysteresis: f32,
}

impl ZLatencyConfig {
    pub fn new(history: usize, hysteresis: f32) -> Self {
        Self {
            history: history.max(1),
            hysteresis: hysteresis.max(0.0),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ZTempo {
    pub target_latency: f32,
    pub smoothing: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ZPulse {
    pub support: f32,
    pub drift: f32,
    pub z_bias: f32,
    pub tempo: Option<ZTempo>,
}

impl ZPulse {
    pub fn into_fused(self) -> ZFused {
        ZFused {
            support: self.support,
            drift: self.drift,
            z_bias: self.z_bias,
            tempo: self.tempo,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ZFused {
    pub support: f32,
    pub drift: f32,
    pub z_bias: f32,
    pub tempo: Option<ZTempo>,
}

#[derive(Debug, Clone)]
pub struct ZSource {
    pub name: String,
    pub pulse: ZPulse,
}

impl ZSource {
    pub fn new(name: impl Into<String>, pulse: ZPulse) -> Self {
        Self {
            name: name.into(),
            pulse,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrequencyFusionState {
    cfg: ZFrequencyConfig,
    fused: ZFused,
}

impl FrequencyFusionState {
    pub fn new(cfg: ZFrequencyConfig) -> Self {
        Self {
            cfg,
            fused: ZFused::default(),
        }
    }

    pub fn update_config(&mut self, cfg: ZFrequencyConfig) {
        self.cfg = cfg;
    }

    pub fn fuse(&mut self, pulse: ZPulse) -> ZFused {
        let alpha = self.cfg.smoothing;
        self.fused.support = lerp(self.fused.support, pulse.support, alpha);
        self.fused.drift = lerp(self.fused.drift, pulse.drift, alpha);
        self.fused.z_bias = lerp(self.fused.z_bias, pulse.z_bias, alpha);
        self.fused.tempo = pulse.tempo.or(self.fused.tempo);
        self.fused
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveGainState {
    cfg: ZAdaptiveGainCfg,
    pub gain: f32,
}

impl AdaptiveGainState {
    pub fn new(cfg: ZAdaptiveGainCfg) -> Self {
        Self {
            gain: cfg.gain_floor,
            cfg,
        }
    }

    pub fn update_config(&mut self, cfg: ZAdaptiveGainCfg) {
        self.cfg = cfg;
        self.gain = self.gain.clamp(self.cfg.gain_floor, self.cfg.gain_ceil);
    }

    pub fn adapt(&mut self, fused: &ZFused) {
        let response = fused.support.abs() + fused.drift.abs() + fused.z_bias.abs();
        let alpha = self.cfg.responsiveness;
        let target = (self.cfg.gain_floor + response).min(self.cfg.gain_ceil);
        self.gain = lerp(self.gain, target, alpha);
    }
}

#[derive(Debug, Clone)]
pub struct LatencyAlignerCfg {
    pub history: usize,
}

impl LatencyAlignerCfg {
    pub fn new(history: usize) -> Self {
        Self {
            history: history.max(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LatencyAlignerState {
    cfg: LatencyAlignerCfg,
    events: VecDeque<String>,
}

impl LatencyAlignerState {
    pub fn new(cfg: ZLatencyConfig) -> Self {
        Self {
            cfg: LatencyAlignerCfg::new(cfg.history),
            events: VecDeque::new(),
        }
    }

    pub fn update_config(&mut self, cfg: ZLatencyConfig) {
        self.cfg = LatencyAlignerCfg::new(cfg.history);
        while self.events.len() > self.cfg.history {
            self.events.pop_front();
        }
    }

    pub fn record(&mut self, event: impl Into<String>) {
        self.events.push_back(event.into());
        while self.events.len() > self.cfg.history {
            self.events.pop_front();
        }
    }

    pub fn drain(&mut self) -> Vec<String> {
        self.events.drain(..).collect()
    }
}

#[derive(Debug, Clone)]
pub struct LatencyAligner {
    cfg: ZLatencyConfig,
}

impl LatencyAligner {
    fn new(cfg: ZLatencyConfig) -> Self {
        Self { cfg }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ZConductorCfg {
    pub freq: Option<ZFrequencyConfig>,
    pub adaptive_gain: Option<ZAdaptiveGainCfg>,
    pub latency: Option<ZLatencyConfig>,
}

#[derive(Debug, Clone)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    freq: Option<FrequencyFusionState>,
    adaptive: Option<AdaptiveGainState>,
    latency: Option<LatencyAlignerState>,
    latency_events: VecDeque<String>,
    fused: ZFused,
}

impl ZConductor {
    pub fn new(cfg: ZConductorCfg) -> Self {
        let freq = cfg.freq.map(FrequencyFusionState::new);
        let adaptive = cfg.adaptive_gain.map(AdaptiveGainState::new);
        let latency = cfg
            .latency
            .map(|latency_cfg| LatencyAlignerState::new(latency_cfg));
        Self {
            cfg,
            freq,
            adaptive,
            latency,
            latency_events: VecDeque::new(),
            fused: ZFused::default(),
        }
    }

    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.cfg.freq = cfg;
        match (&mut self.freq, cfg) {
            (Some(state), Some(new_cfg)) => state.update_config(new_cfg),
            (slot @ None, Some(new_cfg)) => {
                *slot = Some(FrequencyFusionState::new(new_cfg));
            }
            (state @ Some(_), None) => {
                *state = None;
            }
            (None, None) => {}
        }
    }

    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
        self.cfg.adaptive_gain = cfg;
        match (&mut self.adaptive, cfg) {
            (Some(state), Some(new_cfg)) => state.update_config(new_cfg),
            (slot @ None, Some(new_cfg)) => {
                *slot = Some(AdaptiveGainState::new(new_cfg));
            }
            (state @ Some(_), None) => {
                *state = None;
            }
            (None, None) => {}
        }
    }

    pub fn set_latency_config(&mut self, cfg: Option<ZLatencyConfig>) {
        self.cfg.latency = cfg;
        match (&mut self.latency, cfg) {
            (Some(state), Some(new_cfg)) => state.update_config(new_cfg),
            (slot @ None, Some(new_cfg)) => {
                *slot = Some(LatencyAlignerState::new(new_cfg));
            }
            (state @ Some(_), None) => {
                *state = None;
            }
            (None, None) => {}
        }
    }

    pub fn fuse(&mut self, sources: &[ZSource]) -> ZFused {
        let mut fused = ZFused::default();
        for source in sources {
            let pulse = source.pulse;
            fused.support += pulse.support;
            fused.drift += pulse.drift;
            fused.z_bias += pulse.z_bias;
            if fused.tempo.is_none() {
                fused.tempo = pulse.tempo;
            }
        }

        if let Some(state) = &mut self.freq {
            fused = state.fuse(ZPulse {
                support: fused.support,
                drift: fused.drift,
                z_bias: fused.z_bias,
                tempo: fused.tempo,
            });
        }

        if let Some(state) = &mut self.adaptive {
            state.adapt(&fused);
        }

        if let Some(latency) = &mut self.latency {
            latency.record("fused");
            self.latency_events.extend(latency.drain());
        }

        self.fused = fused;
        fused
    }

    pub fn latest(&self) -> ZFused {
        self.fused
    }

    pub fn drain_latency_events(&mut self) -> Vec<String> {
        self.latency_events.drain(..).collect()
    }
}

fn lerp(current: f32, target: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(0.0, 1.0);
    (1.0 - alpha) * current + alpha * target
}
