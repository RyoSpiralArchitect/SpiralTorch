// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Canonical representation of Z pulses together with a lightweight
//! conductor that fuses multiple sources into a single control signal.

use std::collections::VecDeque;

/// Identifies the origin of a [`ZPulse`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZSource {
    Microlocal,
    Maxwell,
    RealGrad,
    Desire,
    External(&'static str),
    Other(&'static str),
}

/// Discrete Z pulse emitted by an upstream source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZPulse {
    pub source: ZSource,
    pub ts: u64,
    pub band_energy: (f32, f32, f32),
    pub drift: f32,
    pub z_bias: f32,
    pub support: f32,
    pub quality: f32,
    pub stderr: f32,
    pub latency_ms: f32,
}

impl Default for ZPulse {
    fn default() -> Self {
        Self {
            source: ZSource::Microlocal,
            ts: 0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_bias: 0.0,
            support: 0.0,
            quality: 0.0,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }
}

/// Result of fusing multiple [`ZPulse`] records.
#[derive(Clone, Debug, PartialEq)]
pub struct ZFused {
    pub ts: u64,
    pub support: f32,
    pub drift: f32,
    pub z: f32,
    pub quality: f32,
    pub events: Vec<String>,
    pub attributions: Vec<(ZSource, f32)>,
}

impl Default for ZFused {
    fn default() -> Self {
        Self {
            ts: 0,
            support: 0.0,
            drift: 0.0,
            z: 0.0,
            quality: 0.0,
            events: Vec::new(),
            attributions: Vec::new(),
        }
    }
}

/// Optional smoothing applied to the fused support/energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZFrequencyConfig {
    pub smoothing: f32,
}

impl ZFrequencyConfig {
    pub fn new(smoothing: f32) -> Self {
        Self {
            smoothing: smoothing.clamp(0.0, 1.0),
        }
    }
}

/// Optional adaptive gain configuration for downstream consumers.
#[derive(Clone, Copy, Debug, PartialEq)]
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

/// Optional latency alignment configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZLatencyConfig {
    pub history: usize,
}

impl ZLatencyConfig {
    pub fn new(history: usize) -> Self {
        Self { history: history.max(1) }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ZConductorCfg {
    pub alpha_fast: f32,
    pub alpha_slow: f32,
    pub flip_hold: u64,
    pub slew_max: f32,
    pub z_budget: f32,
    pub robust_delta: f32,
    pub latency_align: bool,
}

impl Default for ZConductorCfg {
    fn default() -> Self {
        Self {
            alpha_fast: 0.6,
            alpha_slow: 0.12,
            flip_hold: 5,
            slew_max: 0.08,
            z_budget: 0.9,
            robust_delta: 0.2,
            latency_align: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    pending: Vec<ZPulse>,
    fused: ZFused,
    last_step: Option<u64>,
    hold_until: Option<u64>,
    pending_flip_sign: Option<f32>,
    frequency_cfg: Option<ZFrequencyConfig>,
    adaptive_cfg: Option<ZAdaptiveGainCfg>,
    latency_cfg: Option<ZLatencyConfig>,
    latency_events: VecDeque<String>,
}

impl Default for ZConductor {
    fn default() -> Self {
        Self::new(ZConductorCfg::default())
    }
}

impl ZConductor {
    pub fn new(cfg: ZConductorCfg) -> Self {
        Self {
            cfg,
            pending: Vec::new(),
            fused: ZFused::default(),
            last_step: None,
            hold_until: None,
            pending_flip_sign: None,
            frequency_cfg: None,
            adaptive_cfg: None,
            latency_cfg: None,
            latency_events: VecDeque::new(),
        }
    }

    pub fn cfg(&self) -> &ZConductorCfg {
        &self.cfg
    }

    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }

    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.frequency_cfg = cfg;
    }

    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
        self.adaptive_cfg = cfg;
    }

    pub fn set_latency_config(&mut self, cfg: Option<ZLatencyConfig>) {
        self.latency_cfg = cfg;
        if let Some(cfg) = cfg {
            self.latency_events.truncate(cfg.history);
        }
    }

    pub fn ingest(&mut self, pulse: ZPulse) {
        self.pending.push(pulse);
    }

    pub fn step(&mut self, now: u64) -> ZFused {
        if self.pending.is_empty() {
            self.fused.ts = now;
            self.last_step = Some(now);
            return self.fused.clone();
        }

        let pulses = std::mem::take(&mut self.pending);
        let mut fused = ZFused::default();
        fused.ts = now;

        let mut total_support = 0.0f32;
        let mut weighted_z = 0.0f32;
        let mut weighted_quality = 0.0f32;
        let mut total_quality_weight = 0.0f32;

        for pulse in pulses {
            total_support += pulse.support.max(0.0);
            fused.drift += pulse.drift;
            weighted_z += pulse.z_bias * pulse.support.max(0.0);
            let weight = pulse.support.max(1e-6);
            weighted_quality += pulse.quality * weight;
            total_quality_weight += weight;
            fused.attributions.push((pulse.source, pulse.support.max(0.0)));
        }

        fused.support = total_support;
        if total_quality_weight > 0.0 {
            fused.quality = (weighted_quality / total_quality_weight).clamp(0.0, 1.0);
        }

        let mut raw_z = if total_support > 0.0 {
            weighted_z / total_support.max(1e-6)
        } else {
            weighted_z
        };

        let desired_raw = raw_z;
        let previous_z = self.fused.z;
        let mut events = Vec::new();
        let mut flip_armed = false;

        if let Some(hold) = self.hold_until {
            if now < hold {
                events.push("flip-held".to_string());
                raw_z = 0.0;
            } else {
                self.hold_until = None;
                flip_armed = true;
            }
        }

        let incoming_sign = if desired_raw == 0.0 {
            0.0
        } else {
            desired_raw.signum()
        };
        let previous_sign = if previous_z == 0.0 { 0.0 } else { previous_z.signum() };

        if self.hold_until.is_none() && !flip_armed {
            if previous_sign != 0.0 && incoming_sign != 0.0 && incoming_sign != previous_sign {
                self.hold_until = Some(now + self.cfg.flip_hold);
                self.pending_flip_sign = Some(incoming_sign);
                events.push("flip-held".to_string());
                raw_z = 0.0;
            }
        }

        if self.hold_until.is_none() {
            if flip_armed {
                let desired = self
                    .pending_flip_sign
                    .take()
                    .unwrap_or_else(|| if incoming_sign != 0.0 { incoming_sign } else { -previous_sign });
                let sign = if desired == 0.0 {
                    if previous_sign >= 0.0 { -1.0 } else { 1.0 }
                } else {
                    desired.signum()
                };
                events.push("sign-flip".to_string());
                let magnitude = desired_raw.abs().max(self.cfg.robust_delta.max(1e-6));
                raw_z = magnitude * sign;
            }
        }

        let alpha = if previous_sign == 0.0 || previous_sign == incoming_sign {
            self.cfg.alpha_fast
        } else {
            self.cfg.alpha_slow
        };
        let mut target = lerp(previous_z, raw_z, alpha);
        let delta = target - previous_z;
        if delta.abs() > self.cfg.slew_max {
            target = previous_z + self.cfg.slew_max.copysign(delta);
        }
        target = target.clamp(-self.cfg.z_budget, self.cfg.z_budget);

        fused.z = target;
        fused.events = events;

        if let Some(latency_cfg) = self.latency_cfg {
            if fused.z != previous_z {
                let label = format!(
                    "slew:{}->{}",
                    (previous_z * 100.0).round() / 100.0,
                    (fused.z * 100.0).round() / 100.0
                );
                self.latency_events.push_back(label);
                while self.latency_events.len() > latency_cfg.history {
                    self.latency_events.pop_front();
                }
            }
        }

        self.fused = fused.clone();
        self.last_step = Some(now);
        fused
    }

    pub fn latest(&self) -> ZFused {
        self.fused.clone()
    }

    pub fn drain_latency_events(&mut self) -> Vec<String> {
        self.latency_events.drain(..).collect()
    }
}

fn lerp(current: f32, target: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(0.0, 1.0);
    (1.0 - alpha) * current + alpha * target
}
