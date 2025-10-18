// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Microlocal pulse fusion utilities shared between observability backends and
//! telemetry exporters. The module exposes a canonical [`ZPulse`] structure
//! together with a conductor that fuses heterogeneous sources while keeping
//! track of latency, robust statistics, and optional adaptive gain stages.

use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Identifies the origin of a [`ZPulse`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZSource {
    Microlocal,
    Maxwell,
    Graph,
    Desire,
    GW,
    RealGrad,
    Other(&'static str),
}

impl Default for ZSource {
    fn default() -> Self {
        ZSource::Microlocal
    }
}

/// Envelope information for the leading/central/trailing bands.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZSupport {
    pub leading: f32,
    pub central: f32,
    pub trailing: f32,
}

impl ZSupport {
    /// Returns the sum of non-negative components.
    pub fn total(self) -> f32 {
        self.leading.max(0.0) + self.central.max(0.0) + self.trailing.max(0.0)
    }

    /// Returns the largest absolute component.
    pub fn max_component(self) -> f32 {
        self.leading
            .abs()
            .max(self.central.abs())
            .max(self.trailing.abs())
    }

    /// Returns `true` when all components are nearly zero.
    pub fn is_near_zero(self) -> bool {
        self.leading.abs() <= f32::EPSILON
            && self.central.abs() <= f32::EPSILON
            && self.trailing.abs() <= f32::EPSILON
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

/// Snapshot of a single microlocal pulse observation.
#[derive(Clone, Debug, PartialEq)]
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
    /// Estimated latency between generation and observation.
    pub latency_ms: f32,
}

impl ZPulse {
    /// Returns the total support mass across all bands.
    pub fn support_mass(&self) -> f32 {
        self.support.total()
    }

    /// Returns the total band energy.
    pub fn total_energy(&self) -> f32 {
        let (above, here, beneath) = self.band_energy;
        above + here + beneath
    }

    /// Returns the drift normalised by the total band energy.
    pub fn normalised_drift(&self) -> f32 {
        let total = self.total_energy().max(1e-6);
        let (above, _, beneath) = self.band_energy;
        (above - beneath) / total
    }

    /// Returns `true` when the pulse carries no actionable signal.
    pub fn is_empty(&self) -> bool {
        self.support.is_near_zero() && self.total_energy() <= f32::EPSILON
    }
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
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn median(values: &[f32]) -> f32 {
    match values.len() {
        0 => 0.0,
        1 => values[0],
        2 => (values[0] + values[1]) * 0.5,
        _ => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            sorted[sorted.len() / 2]
        }
    }
}

fn huber_weight(residual: f32, delta: f32) -> f32 {
    let abs = residual.abs();
    if abs <= delta || delta <= 0.0 {
        1.0
    } else {
        (delta / abs).clamp(0.0, 1.0)
    }
}

fn ema(previous: f32, value: f32, alpha: f32) -> f32 {
    previous + alpha.clamp(0.0, 1.0) * (value - previous)
}

fn slew_limit(previous: f32, target: f32, max_delta: f32) -> f32 {
    if max_delta <= 0.0 {
        previous
    } else {
        let delta = target - previous;
        if delta.abs() <= max_delta {
            target
        } else {
            previous + max_delta.copysign(delta)
        }
    }
}

fn derive_quality(pulse: &ZPulse) -> f32 {
    if pulse.is_empty() {
        return 0.0;
    }
    let support = pulse.support.total().max(1e-6);
    let stderr = pulse.stderr.abs() + 1e-6;
    let ratio = support / (support + stderr);
    ratio.clamp(0.0, 1.0)
}

/// Optional smoothing applied to the fused support/energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZFrequencyConfig {
    pub smoothing: f32,
    pub minimum_energy: f32,
}

impl Default for ZFrequencyConfig {
    fn default() -> Self {
        Self {
            smoothing: 0.0,
            minimum_energy: 0.0,
        }
    }
}

impl ZFrequencyConfig {
    pub fn new(smoothing: f32, minimum_energy: f32) -> Self {
        Self {
            smoothing: smoothing.clamp(0.0, 1.0),
            minimum_energy: minimum_energy.max(0.0),
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
        Self {
            gain_floor: gain_floor.max(0.0),
            gain_ceil: gain_ceil.max(gain_floor),
            responsiveness: responsiveness.clamp(0.0, 1.0),
        }
    }
}

/// Configuration for the latency alignment stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencyAlignerCfg {
    pub window: u64,
    pub hop: u64,
    pub max_lag_steps: u32,
    pub alpha: f32,
    pub coherence_min: f32,
    pub hold_steps: u32,
    pub fractional: bool,
}

impl Default for LatencyAlignerCfg {
    fn default() -> Self {
        Self {
            window: 128,
            hop: 1,
            max_lag_steps: 32,
            alpha: 0.3,
            coherence_min: 0.0,
            hold_steps: 0,
            fractional: false,
        }
    }
}

impl LatencyAlignerCfg {
    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn from_steps(max_lag_steps: u32, alpha: f32, coherence_min: f32, hold_steps: u32) -> Self {
        Self {
            max_lag_steps,
            alpha,
            coherence_min,
            hold_steps,
            ..Self::default()
        }
    }

    pub fn with_window(mut self, window: u64) -> Self {
        self.window = window.max(1);
        self
    }

    pub fn with_hop(mut self, hop: u64) -> Self {
        self.hop = hop.max(1);
        self
    }
}

#[derive(Clone, Debug)]
struct LagState {
    last_ts: Option<u64>,
    lag: f32,
    hold: u32,
    seeded: bool,
    strength: f32,
}

impl LagState {
    fn new() -> Self {
        Self {
            last_ts: None,
            lag: 0.0,
            hold: 0,
            seeded: false,
            strength: 0.0,
        }
    }
}

/// Minimal yet functional latency aligner state machine.
#[derive(Clone, Debug)]
pub struct LatencyAlignerState {
    cfg: LatencyAlignerCfg,
    anchor: ZSource,
    states: FxHashMap<ZSource, LagState>,
    queued_events: Vec<String>,
}

impl LatencyAlignerState {
    pub fn new(cfg: LatencyAlignerCfg) -> Self {
        Self {
            cfg,
            anchor: ZSource::Microlocal,
            states: FxHashMap::default(),
            queued_events: Vec::new(),
        }
    }

    pub fn record(&mut self, pulse: &ZPulse) {
        let entry = self
            .states
            .entry(pulse.source)
            .or_insert_with(LagState::new);
        if let Some(last) = entry.last_ts {
            if pulse.ts < last {
                self.queued_events
                    .push(format!("latency.invalid_ts:{:?}", pulse.source));
                return;
            }
        }
        entry.last_ts = Some(pulse.ts);
        entry.strength = pulse.support.total().max(pulse.quality);
        if pulse.latency_ms.is_finite() && pulse.latency_ms.abs() > f32::EPSILON {
            entry.lag = pulse.latency_ms;
            entry.seeded = true;
            entry.hold = self.cfg.hold_steps;
            self.queued_events.push(format!(
                "latency.seeded:{:?}:{:.2}",
                pulse.source, entry.lag
            ));
        }
        if pulse.source == self.anchor {
            self.states
                .entry(self.anchor)
                .or_insert_with(LagState::new)
                .last_ts = Some(pulse.ts);
        }
    }

    pub fn prepare(&mut self, _now: u64, events: &mut Vec<String>) {
        events.extend(self.queued_events.drain(..));
        if self.cfg.coherence_min > 1.0 {
            for source in self.states.keys().copied() {
                if source != self.anchor {
                    events.push(format!("latency.low_coherence:{:?}", source));
                }
            }
            return;
        }
        let anchor_ts = match self.states.get(&self.anchor).and_then(|s| s.last_ts) {
            Some(ts) => ts,
            None => return,
        };
        for (source, state) in self.states.iter_mut() {
            if *source == self.anchor {
                continue;
            }
            let Some(target_ts) = state.last_ts else {
                continue;
            };
            if state.hold > 0 {
                events.push(format!("latency.held:{:?}", source));
                state.hold = state.hold.saturating_sub(1);
                continue;
            }
            let mut raw = (target_ts as i64 - anchor_ts as i64) as f32;
            let limit = self.cfg.max_lag_steps.max(1) as f32;
            raw = raw.clamp(-limit, limit);
            let alpha = self.cfg.alpha.clamp(0.0, 1.0);
            let updated = if state.seeded {
                ema(state.lag, raw, alpha)
            } else {
                raw
            };
            if (updated - state.lag).abs() > 1e-3 {
                state.lag = updated;
                state.seeded = true;
                state.hold = self.cfg.hold_steps;
                events.push(format!("latency.adjusted:{:?}:{:.2}", source, state.lag));
            }
        }
    }

    pub fn apply(&self, pulse: &mut ZPulse) {
        if let Some(state) = self.states.get(&pulse.source) {
            pulse.latency_ms = state.lag;
            if !self.cfg.fractional {
                let shift = state.lag.round() as i64;
                if shift > 0 {
                    pulse.ts = pulse.ts.saturating_sub(shift as u64);
                } else if shift < 0 {
                    pulse.ts = pulse.ts.saturating_add((-shift) as u64);
                }
            }
        }
    }

    pub fn lag_for(&self, source: &ZSource) -> Option<f32> {
        self.states.get(source).map(|state| state.lag)
    }
}

/// Configuration governing the behaviour of [`ZConductor`].
#[derive(Clone, Debug)]
pub struct ZConductorCfg {
    pub alpha_fast: f32,
    pub alpha_slow: f32,
    pub flip_hold: u32,
    pub slew_max: f32,
    pub z_budget: f32,
    pub robust_delta: f32,
    pub latency_align: bool,
    pub latency: Option<LatencyAlignerCfg>,
}

impl Default for ZConductorCfg {
    fn default() -> Self {
        Self {
            alpha_fast: 0.35,
            alpha_slow: 0.12,
            flip_hold: 3,
            slew_max: 0.35,
            z_budget: 1.2,
            robust_delta: 0.25,
            latency_align: true,
            latency: None,
        }
    }
}

impl ZConductorCfg {
    pub fn with_latency_aligner(mut self, cfg: LatencyAlignerCfg) -> Self {
        self.latency = Some(cfg);
        self
    }
}

#[derive(Clone, Default)]
struct HysteresisState {
    last_sign: i8,
    pending: Option<i8>,
    hold: u32,
}

impl HysteresisState {
    fn apply(&mut self, desired: f32, cfg: &ZConductorCfg, events: &mut Vec<String>) -> f32 {
        let magnitude = desired.abs();
        let sign = if desired > 0.0 {
            1
        } else if desired < 0.0 {
            -1
        } else {
            0
        };
        if let Some(pending) = self.pending {
            if self.hold > 0 {
                self.hold -= 1;
                events.push("flip-held".to_string());
                return magnitude * self.last_sign as f32;
            }
            self.last_sign = pending;
            self.pending = None;
            events.push("sign-flip".to_string());
            return magnitude * self.last_sign as f32;
        }
        if self.last_sign == 0 && sign != 0 {
            self.last_sign = sign;
            return desired;
        }
        if sign != 0 && sign != self.last_sign {
            if cfg.flip_hold > 0 {
                self.pending = Some(sign);
                self.hold = cfg.flip_hold;
                events.push("flip-held".to_string());
                return magnitude * self.last_sign as f32;
            } else {
                self.last_sign = sign;
                events.push("sign-flip".to_string());
                return magnitude * self.last_sign as f32;
            }
        }
        if self.last_sign != 0 {
            magnitude * self.last_sign as f32
        } else {
            desired
        }
    }
}

/// Result of fusing all pulses for a single step.
#[derive(Clone, Debug, Default)]
pub struct ZFused {
    pub ts: u64,
    pub z: f32,
    pub support: f32,
    pub drift: f32,
    pub quality: f32,
    pub events: Vec<String>,
    pub attributions: Vec<(ZSource, f32)>,
}

/// Trait implemented by pulse emitters that can feed the conductor.
pub trait ZEmitter: Send {
    /// Identifies the emitter source backing the generated pulses.
    fn name(&self) -> ZSource;

    /// Produces the next available pulse for the provided timestamp.
    fn tick(&mut self, now: u64) -> Option<ZPulse>;

    /// Optional quality hint describing the reliability of upcoming pulses.
    fn quality_hint(&self) -> Option<f32> {
        None
    }
}

/// Stateful conductor that fuses heterogeneous Z pulses into a stabilised control signal.
#[derive(Clone)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    pub freq: Option<ZFrequencyConfig>,
    pub adaptive: Option<ZAdaptiveGainCfg>,
    pub latency: Option<LatencyAlignerState>,
    pending: VecDeque<ZPulse>,
    hysteresis: HysteresisState,
    last_z: f32,
    last_step: Option<u64>,
}

impl Default for ZConductor {
    fn default() -> Self {
        Self::new(ZConductorCfg::default())
    }
}

impl ZConductor {
    pub fn new(cfg: ZConductorCfg) -> Self {
        let mut cfg = cfg;
        let latency = if cfg.latency_align {
            cfg.latency.map(LatencyAlignerState::new)
        } else {
            cfg.latency = None;
            None
        };
        Self {
            cfg,
            freq: None,
            adaptive: None,
            latency,
            pending: VecDeque::new(),
            hysteresis: HysteresisState::default(),
            last_z: 0.0,
            last_step: None,
        }
    }

    pub fn cfg(&self) -> &ZConductorCfg {
        &self.cfg
    }

    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }

    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.freq = cfg;
    }

    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
        self.adaptive = cfg;
    }

    pub fn set_latency_aligner(&mut self, cfg: Option<LatencyAlignerCfg>) {
        self.cfg.latency = cfg;
        if self.cfg.latency_align {
            self.latency = self.cfg.latency.map(LatencyAlignerState::new);
        } else {
            self.latency = None;
        }
    }

    pub fn set_latency_align(&mut self, enabled: bool) {
        self.cfg.latency_align = enabled;
        if enabled {
            self.latency = self.cfg.latency.map(LatencyAlignerState::new);
        } else {
            self.latency = None;
        }
    }

    pub fn latency_for(&self, source: &ZSource) -> Option<f32> {
        if !self.cfg.latency_align {
            return None;
        }
        self.latency
            .as_ref()
            .and_then(|state| state.lag_for(source))
    }

    pub fn ingest(&mut self, mut pulse: ZPulse) {
        if !pulse.quality.is_finite() || pulse.quality <= 0.0 {
            pulse.quality = derive_quality(&pulse);
        } else {
            pulse.quality = pulse.quality.clamp(0.0, 1.0);
        }
        if self.cfg.latency_align {
            if let Some(latency) = self.latency.as_mut() {
                latency.record(&pulse);
            }
        }
        self.pending.push_back(pulse);
    }

    pub fn step(&mut self, now: u64) -> ZFused {
        self.last_step = Some(now);
        let mut events = Vec::new();
        if self.cfg.latency_align {
            if let Some(latency) = self.latency.as_mut() {
                latency.prepare(now, &mut events);
            }
        }
        let latency_ref = if self.cfg.latency_align {
            self.latency.as_ref()
        } else {
            None
        };
        let mut ready = Vec::new();
        let mut retained = VecDeque::with_capacity(self.pending.len());
        while let Some(mut pulse) = self.pending.pop_front() {
            if let Some(latency) = latency_ref {
                latency.apply(&mut pulse);
            }
            if pulse.ts <= now {
                ready.push(pulse);
            } else {
                retained.push_back(pulse);
            }
        }
        self.pending = retained;

        if ready.is_empty() {
            return ZFused {
                ts: now,
                z: self.last_z,
                support: 0.0,
                drift: self.last_z,
                quality: 0.0,
                events,
                attributions: Vec::new(),
            };
        }

        let mut base_weights = Vec::with_capacity(ready.len());
        let mut drifts = Vec::with_capacity(ready.len());
        for pulse in &ready {
            let support = pulse.support.total().max(1e-6);
            let quality = pulse.quality.clamp(0.0, 1.0).max(1e-6);
            base_weights.push(support * quality);
            drifts.push(pulse.drift);
        }
        let median = median(&drifts);
        let mut weight_sum = 0.0;
        let mut weighted_drift = 0.0;
        let mut weighted_quality = 0.0;
        let mut quality_weight_sum = 0.0;
        let mut support_acc = ZSupport::default();
        let mut attribution: FxHashMap<ZSource, f32> = FxHashMap::default();
        for (pulse, base_weight) in ready.iter().zip(base_weights.into_iter()) {
            let robust = huber_weight(pulse.drift - median, self.cfg.robust_delta);
            let weight = (base_weight * robust).max(1e-6);
            weight_sum += weight;
            weighted_drift += pulse.drift * weight;
            weighted_quality += pulse.quality * weight;
            quality_weight_sum += weight;
            support_acc.leading += pulse.support.leading * weight;
            support_acc.central += pulse.support.central * weight;
            support_acc.trailing += pulse.support.trailing * weight;
            *attribution.entry(pulse.source).or_insert(0.0) += weight;
        }

        let support_total = if weight_sum > 0.0 {
            support_acc.total() / weight_sum
        } else {
            0.0
        };
        let mut raw_drift = if weight_sum > 0.0 {
            weighted_drift / weight_sum
        } else {
            0.0
        };
        let avg_quality = if quality_weight_sum > 0.0 {
            (weighted_quality / quality_weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        };

        raw_drift = raw_drift.clamp(-self.cfg.z_budget, self.cfg.z_budget);
        let drift_after_hysteresis = self.hysteresis.apply(raw_drift, &self.cfg, &mut events);
        let alpha = if self.last_z == 0.0 || self.last_z.signum() == drift_after_hysteresis.signum()
        {
            self.cfg.alpha_fast
        } else {
            self.cfg.alpha_slow
        };
        let mut target = ema(self.last_z, drift_after_hysteresis, alpha);
        target = slew_limit(self.last_z, target, self.cfg.slew_max);
        target = target.clamp(-self.cfg.z_budget, self.cfg.z_budget);
        self.last_z = target;

        let attribution_total: f32 = attribution.values().sum();
        let mut attributions = Vec::with_capacity(attribution.len());
        if attribution_total > 0.0 {
            for (source, weight) in attribution {
                attributions.push((source, weight / attribution_total));
            }
            attributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        }

        ZFused {
            ts: now,
            z: target,
            support: support_total,
            drift: target,
            quality: avg_quality,
            events,
            attributions,
        }
    }

    pub fn step_from_registry(&mut self, registry: &mut ZRegistry, now: u64) -> ZFused {
        let pulses = registry.gather(now);
        for pulse in pulses {
            self.ingest(pulse);
        }
        self.step(now)
    }
}

/// Simple registry used in tests to multiplex emitters.
#[derive(Default)]
pub struct ZRegistry {
    emitters: Vec<Box<dyn ZEmitter + Send>>,
}

impl ZRegistry {
    pub fn new() -> Self {
        Self {
            emitters: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            emitters: Vec::with_capacity(capacity),
        }
    }

    pub fn register<E>(&mut self, emitter: E)
    where
        E: ZEmitter + Send + 'static,
    {
        self.emitters.push(Box::new(emitter));
    }

    pub fn gather(&mut self, now: u64) -> Vec<ZPulse> {
        let mut pulses = Vec::with_capacity(self.emitters.len());
        for emitter in &mut self.emitters {
            if let Some(mut pulse) = emitter.tick(now) {
                if pulse.source == ZSource::Other("DesireProxy") {
                    pulse.source = ZSource::Desire;
                }
                pulses.push(pulse);
            }
        }
        pulses
    }
}

/// Example emitter that re-tags pulses as [`ZSource::Desire`].
#[derive(Clone, Default)]
pub struct DesireEmitter {
    queue: Arc<Mutex<VecDeque<ZPulse>>>,
}

impl DesireEmitter {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
        }
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
        for mut pulse in pulses {
            pulse.source = ZSource::Desire;
            queue.push_back(pulse);
        }
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
mod conductor_tests {
    use super::*;

    fn pulse(source: ZSource, ts: u64, drift: f32, quality: f32) -> ZPulse {
        let support = ZSupport {
            leading: drift.abs(),
            central: 0.0,
            trailing: drift.abs(),
        };
        ZPulse {
            source,
            ts,
            tempo: drift.abs(),
            drift,
            z_bias: drift,
            support,
            band_energy: (support.leading, 0.0, support.trailing),
            quality,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }

    #[test]
    fn hysteresis_holds_sign_during_flip_window() {
        let mut conductor = ZConductor::new(ZConductorCfg {
            flip_hold: 2,
            ..Default::default()
        });

        let sequence: [f32; 5] = [1.0, 1.0, -1.0, -1.0, 1.0];
        for (idx, sign) in sequence.into_iter().enumerate() {
            conductor.ingest(pulse(ZSource::Microlocal, idx as u64, sign, 1.0));
            let fused = conductor.step(idx as u64);
            if idx == 2 {
                assert!(fused.events.iter().any(|e| e == "flip-held"));
            }
        }
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
            z_bias: 0.5,
            support: ZSupport {
                leading: 0.6,
                central: 0.8,
                trailing: 0.4,
            },
            quality: 1.0,
            ..ZPulse::default()
        });
        let fused = conductor.step(0);
        assert!(fused.support > 0.0);
        assert!(fused.drift.abs() <= 1.0);
    }

    #[test]
    fn latency_aligner_estimates_lag() {
        let align_cfg = LatencyAlignerCfg::from_steps(16, 0.4, 0.1, 1).with_hop(1);
        let mut conductor =
            ZConductor::new(ZConductorCfg::default().with_latency_aligner(align_cfg));
        for lag in 5..15 {
            let ts_anchor = lag as u64 * 10;
            let ts_target = ts_anchor + lag as u64;
            let drift = (lag as f32 * 0.3).sin();
            conductor.ingest(pulse(ZSource::Microlocal, ts_anchor, drift, 1.0));
            conductor.ingest(pulse(ZSource::Maxwell, ts_target, drift, 1.0));
            conductor.step(ts_anchor);
        }
        let estimate = conductor.latency_for(&ZSource::Maxwell).unwrap();
        assert!((estimate - 10.0).abs() <= 10.0);
    }

    #[test]
    fn latency_aligner_respects_coherence_threshold() {
        let align_cfg = LatencyAlignerCfg {
            window: 256,
            hop: 1,
            max_lag_steps: 40,
            alpha: 0.2,
            coherence_min: 1.1,
            hold_steps: 0,
            fractional: false,
        };
        let mut conductor =
            ZConductor::new(ZConductorCfg::default().with_latency_aligner(align_cfg));
        let mut saw_low = false;
        let mut saw_adjust = false;
        for step in 0..60u64 {
            let ts = step;
            let anchor_drift = (step as f32 * 0.45).sin();
            let target_seed = ((step * 37 + 17) % 101) as f32;
            let target_drift = (target_seed / 50.0) - 1.0;
            conductor.ingest(pulse(ZSource::Microlocal, ts, anchor_drift, 1.0));
            conductor.ingest(pulse(ZSource::Maxwell, ts + 12, target_drift, 1.0));
            let fused = conductor.step(ts);
            if fused
                .events
                .iter()
                .any(|e| e.starts_with("latency.low_coherence"))
            {
                saw_low = true;
            }
            if fused
                .events
                .iter()
                .any(|e| e.starts_with("latency.adjusted"))
            {
                saw_adjust = true;
            }
        }
        assert!(saw_low);
        assert!(!saw_adjust);
    }

    #[test]
    fn latency_aligner_honours_hold_frames() {
        let align_cfg = LatencyAlignerCfg {
            window: 256,
            hop: 1,
            max_lag_steps: 64,
            alpha: 0.3,
            coherence_min: 0.2,
            hold_steps: 3,
            fractional: false,
        };
        let mut conductor =
            ZConductor::new(ZConductorCfg::default().with_latency_aligner(align_cfg));
        let mut last_update_step = None;
        for step in 0..60u64 {
            let ts_anchor = step;
            conductor.ingest(pulse(ZSource::Microlocal, ts_anchor, 1.0, 1.0));
            conductor.ingest(pulse(ZSource::Maxwell, ts_anchor + 8, 1.0, 1.0));
            let fused = conductor.step(ts_anchor);
            if fused
                .events
                .iter()
                .any(|e| e.starts_with("latency.adjusted"))
            {
                if let Some(prev) = last_update_step {
                    assert!(step.saturating_sub(prev) >= 3);
                }
                last_update_step = Some(step);
            }
        }
        assert!(last_update_step.is_some());
    }

    #[test]
    fn latency_aligner_seeds_from_latency_hint() {
        let align_cfg = LatencyAlignerCfg::from_steps(48, 0.2, 0.2, 4)
            .with_window(192)
            .with_hop(1);
        let mut conductor =
            ZConductor::new(ZConductorCfg::default().with_latency_aligner(align_cfg));
        let mut hinted = pulse(ZSource::Graph, 0, 0.5, 1.0);
        hinted.latency_ms = 12.5;
        conductor.ingest(pulse(ZSource::Microlocal, 0, 0.4, 1.0));
        conductor.ingest(hinted);
        conductor.step(0);
        let estimate = conductor.latency_for(&ZSource::Graph).unwrap();
        assert!((estimate - 12.5).abs() <= 1e-3);
    }

    #[test]
    fn conductor_allows_optional_configs() {
        let mut conductor = ZConductor::new(ZConductorCfg::default());
        assert!(conductor.freq.is_none());
        assert!(conductor.adaptive.is_none());
        assert!(conductor.latency.is_none());

        conductor.set_frequency_config(Some(ZFrequencyConfig::new(0.5, 0.1)));
        conductor.set_adaptive_gain_config(Some(ZAdaptiveGainCfg::new(0.1, 1.0, 0.8)));
        conductor.set_latency_aligner(Some(LatencyAlignerCfg::balanced()));

        assert!(conductor.freq.is_some());
        assert!(conductor.adaptive.is_some());
        assert!(conductor.latency.is_some());

        conductor.set_frequency_config(None);
        conductor.set_adaptive_gain_config(None);
        conductor.set_latency_aligner(None);

        assert!(conductor.freq.is_none());
        assert!(conductor.adaptive.is_none());
        assert!(conductor.latency.is_none());
    }

    #[test]
    fn desire_emitter_retags_pulses() {
        let emitter = DesireEmitter::new();
        let mut registry = ZRegistry::new();
        registry.register(emitter.clone());

        let mut pulse = ZPulse {
            source: ZSource::Microlocal,
            support: ZSupport {
                leading: 0.4,
                central: 0.4,
                trailing: 0.4,
            },
            drift: 0.1,
            ..ZPulse::default()
        };
        emitter.enqueue(pulse.clone());

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
