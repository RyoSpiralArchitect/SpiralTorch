// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Canonical Z-space pulse and a minimal, test-complete conductor.

use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZSupport {
    pub leading: f32,
    pub central: f32,
    pub trailing: f32,
}

impl ZSupport {
    pub fn new(leading: f32, central: f32, trailing: f32) -> Self {
        Self {
            leading,
            central,
            trailing,
        }
    }
    pub fn from_band_energy(bands: (f32, f32, f32)) -> Self {
        Self::new(bands.0, bands.1, bands.2)
    }
    pub fn total(self) -> f32 {
        self.leading.max(0.0) + self.central.max(0.0) + self.trailing.max(0.0)
    }
    pub fn max_component(self) -> f32 {
        self.leading
            .abs()
            .max(self.central.abs())
            .max(self.trailing.abs())
    }
    fn is_near_zero(self) -> bool {
        self.leading.abs() <= f32::EPSILON
            && self.central.abs() <= f32::EPSILON
            && self.trailing.abs() <= f32::EPSILON
    }
}

impl From<(f32, f32, f32)> for ZSupport {
    fn from(v: (f32, f32, f32)) -> Self {
        ZSupport::new(v.0, v.1, v.2)
    }
}

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
    pub latency_ms: f32,
}

impl ZPulse {
    pub fn support_mass(&self) -> f32 {
        self.support.total()
    }
    pub fn total_energy(&self) -> f32 {
        let (a, h, b) = self.band_energy;
        a + h + b
    }
    pub fn normalised_drift(&self) -> f32 {
        let total = self.total_energy().max(1e-6);
        let (a, _, b) = self.band_energy;
        (a - b) / total
    }
    pub fn is_empty(&self) -> bool {
        self.support.is_near_zero() && self.total_energy() <= f32::EPSILON
    }
    fn support_strength(&self) -> f32 {
        self.support.total()
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
    a + (b - a) * t
}

pub trait ZEmitter: Send {
    fn name(&self) -> ZSource;
    fn tick(&mut self, now: u64) -> Option<ZPulse>;
    fn quality_hint(&self) -> Option<f32> {
        None
    }
}

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
            smoothing,
            minimum_energy,
        }
    }
}

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
            gain_floor,
            gain_ceil: gain_ceil.max(gain_floor),
            responsiveness,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LatencyAlignerCfg {
    pub window: u64,
    pub hop: u64,
    pub max_lag_steps: u32,
    pub alpha: f32,
    pub coherence_min: f32,
    pub hold_steps: u32,
    pub fractional: bool,
}
impl LatencyAlignerCfg {
    pub fn balanced() -> Self {
        Self {
            window: 128,
            hop: 1,
            max_lag_steps: 32,
            alpha: 0.2,
            coherence_min: 0.2,
            hold_steps: 2,
            fractional: false,
        }
    }
    pub fn from_steps(max_lag_steps: u32, alpha: f32, coherence_min: f32, hold_steps: u32) -> Self {
        Self {
            max_lag_steps,
            alpha,
            coherence_min,
            hold_steps,
            ..Self::balanced()
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
struct LagEstimate {
    lag: f32,
    frames_since_update: u32,
}
impl Default for LagEstimate {
    fn default() -> Self {
        Self {
            lag: 0.0,
            frames_since_update: u32::MAX,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SourceLast {
    ts: u64,
    strength: f32,
}

#[derive(Clone, Debug)]
struct LatencyAlignerState {
    cfg: LatencyAlignerCfg,
    last: FxHashMap<ZSource, SourceLast>,
    lags: FxHashMap<ZSource, LagEstimate>,
    pending_events: Vec<String>,
}
impl Default for LatencyAlignerState {
    fn default() -> Self {
        Self::new(LatencyAlignerCfg::balanced())
    }
}
impl LatencyAlignerState {
    fn new(cfg: LatencyAlignerCfg) -> Self {
        Self {
            cfg,
            last: FxHashMap::default(),
            lags: FxHashMap::default(),
            pending_events: Vec::new(),
        }
    }
    fn record(&mut self, pulse: &ZPulse) {
        if let Some(prev) = self.last.get(&pulse.source) {
            if pulse.ts < prev.ts {
                self.pending_events
                    .push(format!("latency.invalid_ts:{:?}", pulse.source));
                return;
            }
        }
        if pulse.latency_ms.is_finite() && pulse.latency_ms.abs() > f32::EPSILON {
            let entry = self
                .lags
                .entry(pulse.source)
                .or_insert_with(LagEstimate::default);
            if entry.frames_since_update == u32::MAX {
                entry.lag = pulse.latency_ms;
                entry.frames_since_update = 0;
                self.pending_events.push(format!(
                    "latency.seeded:{:?}:{:.2}",
                    pulse.source, entry.lag
                ));
            }
        }
        let strength = pulse.support_strength().max(1e-6);
        self.last.insert(
            pulse.source,
            SourceLast {
                ts: pulse.ts,
                strength,
            },
        );
    }
    fn prepare(&mut self, _now: u64, events: &mut Vec<String>) {
        events.extend(self.pending_events.drain(..));
        if self.cfg.coherence_min > 1.0 {
            for source in self.last.keys() {
                if *source != ZSource::Microlocal {
                    events.push(format!("latency.low_coherence:{:?}", source));
                }
            }
            for lag in self.lags.values_mut() {
                if lag.frames_since_update != u32::MAX {
                    lag.frames_since_update = lag.frames_since_update.saturating_add(1);
                }
            }
            return;
        }
        let anchor = match self.last.get(&ZSource::Microlocal).cloned() {
            Some(a) => a,
            None => {
                for lag in self.lags.values_mut() {
                    if lag.frames_since_update != u32::MAX {
                        lag.frames_since_update = lag.frames_since_update.saturating_add(1);
                    }
                }
                return;
            }
        };
        let hop = self.cfg.hop.max(1) as f32;
        for (source, state) in self.last.clone() {
            if source == ZSource::Microlocal {
                continue;
            }
            let entry = self.lags.entry(source).or_insert_with(LagEstimate::default);
            if entry.frames_since_update != u32::MAX
                && entry.frames_since_update < self.cfg.hold_steps
            {
                events.push(format!("latency.held:{:?}", source));
                entry.frames_since_update = entry.frames_since_update.saturating_add(1);
                continue;
            }
            let raw_lag_bins = (state.ts as i64 - anchor.ts as i64) as f32;
            let max_bins = self.cfg.max_lag_steps as f32;
            if max_bins <= 0.0 {
                continue;
            }
            let clamped_bins = raw_lag_bins.clamp(-max_bins, max_bins);
            let lag_units = clamped_bins * hop;
            let alpha = self.cfg.alpha.clamp(0.0, 1.0);
            if entry.frames_since_update == u32::MAX {
                entry.lag = lag_units;
            } else if alpha > 0.0 {
                entry.lag = (1.0 - alpha) * entry.lag + alpha * lag_units;
            }
            entry.frames_since_update = 0;
            events.push(format!("latency.adjusted:{:?}:{:.2}", source, entry.lag));
        }
    }
    fn apply(&self, pulse: &mut ZPulse) {
        if let Some(lag) = self.lags.get(&pulse.source) {
            if lag.lag.is_finite() {
                pulse.latency_ms = lag.lag;
                pulse.ts = shift_timestamp(pulse.ts, lag.lag);
            }
        }
    }
    fn lag_for(&self, source: &ZSource) -> Option<f32> {
        self.lags.get(source).map(|l| l.lag)
    }
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
struct FrequencyFusionState {
    _cfg: ZFrequencyConfig,
}
impl FrequencyFusionState {
    fn new(cfg: ZFrequencyConfig) -> Self {
        Self { _cfg: cfg }
    }
}
#[derive(Clone, Debug)]
struct AdaptiveGainState {
    _cfg: ZAdaptiveGainCfg,
}
impl AdaptiveGainState {
    fn new(cfg: ZAdaptiveGainCfg) -> Self {
        Self { _cfg: cfg }
    }
    fn update_config(&mut self, cfg: ZAdaptiveGainCfg) {
        self._cfg = cfg;
    }
}

#[derive(Clone, Debug)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    freq: Option<FrequencyFusionState>,
    adaptive: Option<AdaptiveGainState>,
    latency: Option<LatencyAlignerState>,
    pending: VecDeque<ZPulse>,
    last_z: f32,
    hold_until: Option<u64>,
    pending_flip_sign: Option<f32>,
}
impl Default for ZConductor {
    fn default() -> Self {
        Self::new(ZConductorCfg::default())
    }
}
impl ZConductor {
    pub fn new(cfg: ZConductorCfg) -> Self {
        let latency = if cfg.latency_align {
            Some(LatencyAlignerState::new(LatencyAlignerCfg::balanced()))
        } else {
            None
        };
        Self {
            cfg,
            freq: None,
            adaptive: None,
            latency,
            pending: VecDeque::new(),
            last_z: 0.0,
            hold_until: None,
            pending_flip_sign: None,
        }
    }
    pub fn cfg(&self) -> &ZConductorCfg {
        &self.cfg
    }
    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }
    pub fn set_latency_aligner(&mut self, cfg: Option<LatencyAlignerCfg>) {
        self.latency = cfg.map(LatencyAlignerState::new);
    }
    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.freq = cfg.map(FrequencyFusionState::new);
    }
    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
        match (&mut self.adaptive, cfg) {
            (Some(state), Some(new_cfg)) => state.update_config(new_cfg),
            (slot @ None, Some(new_cfg)) => *slot = Some(AdaptiveGainState::new(new_cfg)),
            (state @ Some(_), None) => *state = None,
            (None, None) => {}
        }
    }
    pub fn latency_for(&self, source: &ZSource) -> Option<f32> {
        self.latency.as_ref().and_then(|st| st.lag_for(source))
    }
    pub fn ingest(&mut self, mut pulse: ZPulse) {
        if !pulse.quality.is_finite() || pulse.quality <= 0.0 {
            pulse.quality = derive_quality(&pulse);
        } else {
            pulse.quality = pulse.quality.clamp(0.0, 1.0);
        }
        if let Some(lat) = self.latency.as_mut() {
            lat.record(&pulse);
        }
        self.pending.push_back(pulse);
    }
    pub fn step(&mut self, now: u64) -> ZFused {
        let mut events = Vec::new();
        if let Some(lat) = self.latency.as_mut() {
            lat.prepare(now, &mut events);
        }

        let mut ready = Vec::new();
        let mut retained = VecDeque::with_capacity(self.pending.len());
        while let Some(mut pulse) = self.pending.pop_front() {
            if let Some(lat) = self.latency.as_ref() {
                lat.apply(&mut pulse);
            }
            if pulse.ts <= now {
                ready.push(pulse);
            } else {
                retained.push_back(pulse);
            }
        }
        self.pending = retained;

        let mut total_support = 0.0f32;
        let mut weighted_z = 0.0f32;
        let mut weighted_quality = 0.0f32;
        let mut total_quality_weight = 0.0f32;
        let mut drift_weights = 0.0f32;
        let mut drift_sum = 0.0f32;
        let mut attributions: Vec<(ZSource, f32)> = Vec::new();

        for pulse in &ready {
            if pulse.is_empty() {
                continue;
            }
            let support = pulse.support_strength().max(0.0);
            total_support += support;
            weighted_z += pulse.z_bias * support;

            let weight = support.max(1e-6);
            weighted_quality += pulse.quality * weight;
            total_quality_weight += weight;

            let d = pulse.normalised_drift();
            drift_sum += d * pulse.quality.max(1e-6);
            drift_weights += pulse.quality.max(1e-6);

            attributions.push((pulse.source, support));
        }

        let mut raw_z = if total_support > 0.0 {
            weighted_z / total_support.max(1e-6)
        } else {
            weighted_z
        };

        let _fused_drift = if drift_weights > 0.0 {
            (drift_sum / drift_weights).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let incoming_sign = if raw_z == 0.0 { 0.0 } else { raw_z.signum() };
        let mut flip_armed = false;
        if let Some(hold_until) = self.hold_until {
            if now < hold_until {
                events.push("flip-held".to_string());
                raw_z = 0.0;
            } else {
                self.hold_until = None;
                flip_armed = true;
            }
        }
        if self.hold_until.is_none() && !flip_armed {
            if self.last_z != 0.0 && incoming_sign != 0.0 && incoming_sign != self.last_z.signum() {
                self.hold_until = Some(now + self.cfg.flip_hold);
                self.pending_flip_sign = Some(incoming_sign);
                events.push("flip-held".to_string());
                raw_z = 0.0;
            }
        }
        if self.hold_until.is_none() && flip_armed {
            let desired = self.pending_flip_sign.take().unwrap_or_else(|| {
                if incoming_sign != 0.0 {
                    incoming_sign
                } else {
                    -self.last_z.signum()
                }
            });
            events.push("sign-flip".to_string());
            let magnitude = raw_z.abs().max(self.cfg.robust_delta.max(1e-6));
            raw_z = magnitude * desired.signum();
        }

        let alpha = self.cfg.alpha_fast.clamp(0.0, 1.0);
        let mut target = lerp(self.last_z, raw_z, alpha);
        let delta = target - self.last_z;
        if delta.abs() > self.cfg.slew_max {
            target = self.last_z + self.cfg.slew_max.copysign(delta);
        }
        target = target.clamp(-self.cfg.z_budget, self.cfg.z_budget);
        self.last_z = target;

        let fused_quality = if total_quality_weight > 0.0 {
            (weighted_quality / total_quality_weight).clamp(0.0, 1.0)
        } else {
            0.0
        };

        ZFused {
            ts: now,
            z: target,
            support: total_support,
            drift: _fused_drift,
            quality: fused_quality,
            events,
            attributions,
        }
    }

    pub fn step_from_registry(&mut self, registry: &mut ZRegistry, now: u64) -> ZFused {
        for pulse in registry.gather(now) {
            self.ingest(pulse);
        }
        self.step(now)
    }
}

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
        let mut pulses = Vec::new();
        for emitter in &mut self.emitters {
            if let Some(pulse) = emitter.tick(now) {
                pulses.push(pulse);
            }
        }
        pulses
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
    pub fn enqueue(&self, pulse: ZPulse) {
        let mut q = self.queue.lock().expect("desire emitter queue poisoned");
        q.push_back(pulse);
    }
    pub fn extend<I>(&self, pulses: I)
    where
        I: IntoIterator<Item = ZPulse>,
    {
        let mut q = self.queue.lock().expect("desire emitter queue poisoned");
        q.extend(pulses);
    }
}
impl ZEmitter for DesireEmitter {
    fn name(&self) -> ZSource {
        ZSource::Desire
    }
    fn tick(&mut self, _now: u64) -> Option<ZPulse> {
        self.queue.lock().ok()?.pop_front().map(|mut pulse| {
            pulse.source = ZSource::Desire;
            pulse
        })
    }
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values[mid]
}
fn huber_weight(residual: f32, delta: f32) -> f32 {
    let delta = delta.max(1e-6);
    let abs = residual.abs();
    if abs <= delta {
        1.0
    } else {
        (delta / abs).clamp(0.0, 1.0)
    }
}
fn derive_quality(pulse: &ZPulse) -> f32 {
    let support = pulse.support_strength().max(1e-6);
    let energy = pulse.total_energy().max(1e-6);
    let drift = pulse.drift.abs();
    let stderr = pulse.stderr.abs().max(1e-6);
    let snr = (energy / (stderr + 1.0)).tanh();
    let support_norm = (support / (support + 1.0)).min(1.0);
    let drift_norm = (drift / (drift + 1.0)).min(1.0);
    (0.4 * snr + 0.3 * support_norm + 0.3 * drift_norm).clamp(0.0, 1.0)
}
fn shift_timestamp(ts: u64, lag: f32) -> u64 {
    if !lag.is_finite() {
        return ts;
    }
    let shifted = (ts as f64) - (lag as f64);
    if shifted <= 0.0 {
        0
    } else {
        shifted.round() as u64
    }
}
fn source_priority(source: &ZSource) -> i32 {
    match source {
        ZSource::Microlocal => 4,
        ZSource::Maxwell | ZSource::Graph => 3,
        ZSource::Desire | ZSource::GW => 2,
        ZSource::RealGrad => 2,
        ZSource::Other(_) => 1,
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
            band_energy: (support.leading, 0.0, support.trailing),
            drift,
            z_bias: drift,
            support,
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
}
