// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Canonical representation of Z pulses together with a lightweight
//! conductor that fuses multiple sources into a single control signal.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Identifies a source capable of emitting [`ZPulse`] records.
pub trait ZEmitter {
    /// Returns the canonical source identifier for pulses emitted by this
    /// implementation.
    fn name(&self) -> ZSource;

    /// Advances the emitter one step and returns the next available pulse, if
    /// any. Implementations may return more than one pulse per call by keeping
    /// an internal queue; [`ZRegistry::gather`] will keep polling the emitter
    /// until it reports `None`.
    fn tick(&mut self, now: u64) -> Option<ZPulse>;

    /// Optional quality hint describing the reliability of upcoming pulses.
    /// When absent the conductor infers quality from the pulses themselves.
    fn quality_hint(&self) -> Option<f32> {
        None
    }
}

/// Central registry that owns a fleet of [`ZEmitter`] implementations.
#[derive(Default)]
pub struct ZRegistry {
    emitters: Vec<Box<dyn ZEmitter + Send>>,
}

impl ZRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            emitters: Vec::new(),
        }
    }

    /// Creates a registry with pre-allocated storage for the provided number of
    /// emitters.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            emitters: Vec::with_capacity(capacity),
        }
    }

    /// Registers a new emitter so it participates in subsequent gather calls.
    pub fn register<E>(&mut self, emitter: E)
    where
        E: ZEmitter + Send + 'static,
    {
        self.emitters.push(Box::new(emitter));
    }

    /// Polls each registered emitter and aggregates all pulses emitted at the
    /// supplied timestamp.
    pub fn gather(&mut self, now: u64) -> Vec<ZPulse> {
        let mut pulses = Vec::new();
        for emitter in &mut self.emitters {
            while let Some(pulse) = emitter.tick(now) {
                pulses.push(pulse);
            }
        }
        pulses
    }

    /// Returns `true` when no emitters are registered.
    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty()
    }
}

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

fn source_lookup_key(source: &ZSource) -> Cow<'static, str> {
    match source {
        ZSource::Microlocal => Cow::Borrowed("microlocal"),
        ZSource::Maxwell => Cow::Borrowed("maxwell"),
        ZSource::RealGrad => Cow::Borrowed("realgrad"),
        ZSource::Desire => Cow::Borrowed("desire"),
        ZSource::External(label) | ZSource::Other(label) => Cow::Borrowed(*label),
    }
}

/// Per-band support accounting used by [`ZPulse`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZSupport {
    pub leading: f32,
    pub central: f32,
    pub trailing: f32,
}

impl ZSupport {
    /// Returns the total support mass contributed by the pulse.
    pub fn total(&self) -> f32 {
        self.leading.max(0.0) + self.central.max(0.0) + self.trailing.max(0.0)
    }

    /// Scales all support components by the provided gain.
    pub fn scaled(self, gain: f32) -> Self {
        Self {
            leading: self.leading * gain,
            central: self.central * gain,
            trailing: self.trailing * gain,
        }
    }
}

impl Default for ZSupport {
    fn default() -> Self {
        Self {
            leading: 0.0,
            central: 0.0,
            trailing: 0.0,
        }
    }
}

impl From<(f32, f32, f32)> for ZSupport {
    fn from(bands: (f32, f32, f32)) -> Self {
        Self {
            leading: bands.0,
            central: bands.1,
            trailing: bands.2,
        }
    }
}

/// Discrete Z pulse emitted by an upstream source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZPulse {
    pub source: ZSource,
    pub ts: u64,
    pub tempo: f32,
    pub band_energy: (f32, f32, f32),
    pub drift: f32,
    pub z_bias: f32,
    pub support: ZSupport,
    pub quality: f32,
    pub stderr: f32,
    pub latency_ms: f32,
}

impl Default for ZPulse {
    fn default() -> Self {
        Self {
            source: ZSource::Microlocal,
            ts: 0,
            tempo: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_bias: 0.0,
            support: ZSupport::default(),
            quality: 0.0,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
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

impl Default for ZAdaptiveGainCfg {
    fn default() -> Self {
        Self {
            gain_floor: 0.0,
            gain_ceil: 1.0,
            responsiveness: 0.0,
        }
    }
}

impl ZAdaptiveGainCfg {
    pub fn new(gain_floor: f32, gain_ceil: f32, responsiveness: f32) -> Self {
        let floor = gain_floor.max(0.0);
        let ceil = gain_ceil.max(floor);
        Self {
            gain_floor: floor,
            gain_ceil: ceil,
            responsiveness: responsiveness.clamp(0.0, 1.0),
        }
    }

    pub fn update(&mut self, gain_floor: f32, gain_ceil: f32, responsiveness: f32) {
        *self = Self::new(gain_floor, gain_ceil, responsiveness);
    }
}

/// Optional latency alignment configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZLatencyConfig {
    pub history: usize,
}

impl ZLatencyConfig {
    pub fn new(history: usize) -> Self {
        Self {
            history: history.max(1),
        }
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
    fused: ZFused,
    last_step: Option<u64>,
    hold_until: Option<u64>,
    pending_flip_sign: Option<f32>,
    frequency_cfg: Option<ZFrequencyConfig>,
    adaptive_cfg: Option<ZAdaptiveGainCfg>,
    latency_cfg: Option<ZLatencyConfig>,
    latency_events: VecDeque<String>,
    source_gains: HashMap<String, f32>,
    source_limits: HashMap<String, f32>,
    config_events: VecDeque<String>,
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
            fused: ZFused::default(),
            last_step: None,
            hold_until: None,
            pending_flip_sign: None,
            frequency_cfg: None,
            adaptive_cfg: None,
            latency_cfg: None,
            latency_events: VecDeque::new(),
            source_gains: HashMap::new(),
            source_limits: HashMap::new(),
            config_events: VecDeque::new(),
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

    pub fn step<I>(&mut self, pulses: I, now: u64) -> ZFused
    where
        I: IntoIterator<Item = ZPulse>,
    {
        let mut fused = ZFused::default();
        fused.ts = now;

        let mut total_support = 0.0f32;
        let mut weighted_z = 0.0f32;
        let mut weighted_quality = 0.0f32;
        let mut total_quality_weight = 0.0f32;

        let mut had_pulse = false;

        for pulse in pulses {
            had_pulse = true;
            let support = pulse.support.total().max(0.0);
            total_support += support;
            fused.drift += pulse.drift;
            let key = source_lookup_key(&pulse.source);
            let gain = self
                .source_gains
                .get(key.as_ref())
                .copied()
                .unwrap_or(1.0)
                .max(0.0);
            let limit = self
                .source_limits
                .get(key.as_ref())
                .copied()
                .unwrap_or(f32::INFINITY)
                .max(0.0);
            let mut z_bias = pulse.z_bias * gain;
            if limit.is_finite() {
                if limit == 0.0 {
                    z_bias = 0.0;
                } else {
                    z_bias = z_bias.clamp(-limit, limit);
                }
            }
            weighted_z += z_bias * support;
            let weight = support.max(1e-6);
            weighted_quality += pulse.quality * weight;
            total_quality_weight += weight;
            fused.attributions.push((pulse.source, support));
        }

        if !had_pulse {
            self.fused.ts = now;
            self.last_step = Some(now);
            return self.fused.clone();
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
        let previous_sign = if previous_z == 0.0 {
            0.0
        } else {
            previous_z.signum()
        };

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
                let desired = self.pending_flip_sign.take().unwrap_or_else(|| {
                    if incoming_sign != 0.0 {
                        incoming_sign
                    } else {
                        -previous_sign
                    }
                });
                let sign = if desired == 0.0 {
                    if previous_sign >= 0.0 {
                        -1.0
                    } else {
                        1.0
                    }
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

    /// Convenience helper that gathers pulses from the provided registry prior
    /// to fusing them.
    pub fn step_from_registry(&mut self, registry: &mut ZRegistry, now: u64) -> ZFused {
        let pulses = registry.gather(now);
        self.step(pulses, now)
    }

    pub fn latest(&self) -> ZFused {
        self.fused.clone()
    }
}

#[derive(Clone, Default, Debug)]
pub struct DesireEmitter {
    queue: Arc<Mutex<VecDeque<ZPulse>>>,
}

impl DesireEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, mut pulse: ZPulse) {
        pulse.source = ZSource::Desire;
        let mut queue = self.queue.lock().expect("desire emitter queue poisoned");
        queue.push_back(pulse);
    }

    pub fn extend<I>(&self, pulses: I)
    where
        I: IntoIterator<Item = ZPulse>,
    {
        let mut queue = self.queue.lock().expect("desire emitter queue poisoned");
        queue.extend(pulses.into_iter().map(|mut pulse| {
            pulse.source = ZSource::Desire;
            pulse
        }));
    }
}

impl ZEmitter for DesireEmitter {
    fn name(&self) -> ZSource {
        ZSource::Desire
    }

    fn tick(&mut self, _now: u64) -> Option<ZPulse> {
        self.queue
            .lock()
            .expect("desire emitter queue poisoned")
            .pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conductor_allows_optional_configs() {
        let mut conductor = ZConductor::new(ZConductorCfg::default());
        assert!(conductor.frequency_cfg.is_none());
        assert!(conductor.adaptive_cfg.is_none());
        assert!(conductor.latency_cfg.is_none());

        conductor.set_frequency_config(Some(ZFrequencyConfig::new(0.5)));
        conductor.set_adaptive_gain_config(Some(ZAdaptiveGainCfg::new(0.1, 1.0, 0.8)));
        conductor.set_latency_config(Some(ZLatencyConfig::new(4)));

        assert!(conductor.frequency_cfg.is_some());
        assert!(conductor.adaptive_cfg.is_some());
        assert!(conductor.latency_cfg.is_some());

        conductor.set_frequency_config(None);
        conductor.set_adaptive_gain_config(None);
        conductor.set_latency_config(None);

        assert!(conductor.frequency_cfg.is_none());
        assert!(conductor.adaptive_cfg.is_none());
        assert!(conductor.latency_cfg.is_none());
    }

    #[test]
    fn desire_emitter_retags_pulses() {
        let emitter = DesireEmitter::new();
        let mut registry = ZRegistry::new();
        registry.register(emitter.clone());

        let mut pulse = ZPulse {
            source: ZSource::Microlocal,
            support: ZSupport::from((0.2, 0.2, 0.0)),
            drift: 0.1,
            ..ZPulse::default()
        };
        emitter.enqueue(pulse);

        let pulses = registry.gather(42);
        assert_eq!(pulses.len(), 1);
        assert_eq!(pulses[0].source, ZSource::Desire);

        pulse.source = ZSource::Maxwell;
        emitter.extend([pulse]);
        let pulses = registry.gather(43);
        assert_eq!(pulses.len(), 1);
        assert_eq!(pulses[0].source, ZSource::Desire);
    }
}
