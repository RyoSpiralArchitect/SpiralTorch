// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Canonical Z-space pulse representations and a lightweight conductor used by
//! microlocal, Maxwell, and RealGrad observers.  The previous revision of this
//! file had diverged into an unreadable hybrid of half-finished refactors.  The
//! implementation below intentionally keeps the surface area minimal while still
//! exercising the behaviour required by the public unit tests and downstream
//! callers.  Every routine focuses on deterministic, allocation-friendly data
//! paths so the conductor can run inside the realtime control loop without
//! pulling in async machinery.

use rustc_hash::FxHashMap;
use std::{
    cmp::Ordering,
    collections::VecDeque,
    sync::{Arc, Mutex},
};

// -----------------------------------------------------------------------------
// ZScale
// -----------------------------------------------------------------------------

/// Encodes the physical and logarithmic radius of a Z pulse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZScale {
    pub physical_radius: f32,
    pub log_radius: f32,
}

impl ZScale {
    /// Unity scale constant (physical radius = 1.0, log radius = 0.0).
    pub const ONE: ZScale = ZScale {
        physical_radius: 1.0,
        log_radius: 0.0,
    };

    /// Compatibility helper returning the physical radius (for legacy callers).
    #[inline]
    pub fn value(self) -> f32 {
        self.physical_radius
    }

    /// Creates a new scale from a physical radius, rejecting non-positive or
    /// non-finite values.
    pub fn new(physical_radius: f32) -> Option<Self> {
        if physical_radius.is_finite() && physical_radius > 0.0 {
            Some(Self {
                physical_radius,
                log_radius: physical_radius.ln(),
            })
        } else {
            None
        }
    }

    /// Reconstructs a scale from its logarithmic radius.
    pub fn from_log(log_radius: f32) -> Option<Self> {
        if log_radius.is_finite() {
            let physical = log_radius.exp();
            if physical.is_finite() && physical > 0.0 {
                return Some(Self {
                    physical_radius: physical,
                    log_radius,
                });
            }
        }
        None
    }

    /// Builds a scale directly from precomputed components, enforcing the
    /// invariants required by the other constructors.
    pub fn from_components(physical_radius: f32, log_radius: f32) -> Option<Self> {
        if physical_radius.is_finite() && physical_radius > 0.0 && log_radius.is_finite() {
            Some(Self {
                physical_radius,
                log_radius,
            })
        } else {
            None
        }
    }

    /// Linearly interpolates between two scales.
    pub fn lerp(a: ZScale, b: ZScale, t: f32) -> ZScale {
        let clamped = t.clamp(0.0, 1.0);
        let physical = a.physical_radius + (b.physical_radius - a.physical_radius) * clamped;
        let log = a.log_radius + (b.log_radius - a.log_radius) * clamped;
        Self::from_components(physical, log).unwrap_or_else(|| if clamped < 0.5 { a } else { b })
    }

    /// Computes the weighted centroid of a collection of scales.
    pub fn weighted_average<I>(weights: I) -> Option<Self>
    where
        I: IntoIterator<Item = (ZScale, f32)>,
    {
        let mut sum_phys = 0.0f32;
        let mut sum_log = 0.0f32;
        let mut total = 0.0f32;
        for (scale, weight) in weights {
            if !weight.is_finite() || weight <= 0.0 {
                continue;
            }
            sum_phys += scale.physical_radius * weight;
            sum_log += scale.log_radius * weight;
            total += weight;
        }
        if total > 0.0 {
            ZScale::from_components(sum_phys / total, sum_log / total)
        } else {
            None
        }
    }
}

// -----------------------------------------------------------------------------
// ZSupport, ZPulse
// -----------------------------------------------------------------------------

/// Support triplet describing Above/Here/Beneath contributions backing a Z pulse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZSupport {
    pub leading: f32,
    pub central: f32,
    pub trailing: f32,
}

impl ZSupport {
    /// Creates a new support triplet, clamping each component to a finite,
    /// non-negative value.
    pub fn new(leading: f32, central: f32, trailing: f32) -> Self {
        Self {
            leading: clamp_non_negative(leading),
            central: clamp_non_negative(central),
            trailing: clamp_non_negative(trailing),
        }
    }

    /// Builds a support triplet straight from an Above/Here/Beneath energy tuple.
    pub fn from_band_energy(bands: (f32, f32, f32)) -> Self {
        Self::new(bands.0, bands.1, bands.2)
    }

    /// Returns the total perimeter mass supporting the pulse.
    pub fn total(&self) -> f32 {
        self.leading + self.central + self.trailing
    }

    /// Returns `true` when all support components vanish.
    pub fn is_empty(&self) -> bool {
        self.leading <= f32::EPSILON
            && self.central <= f32::EPSILON
            && self.trailing <= f32::EPSILON
    }

    /// Maximum absolute component used by stability heuristics.
    pub fn max_component(&self) -> f32 {
        self.leading
            .abs()
            .max(self.central.abs())
            .max(self.trailing.abs())
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
    fn from((leading, central, trailing): (f32, f32, f32)) -> Self {
        Self::new(leading, central, trailing)
    }
}

fn clamp_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Identifies the origin of a [`ZPulse`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ZSource {
    #[default]
    Microlocal,
    Maxwell,
    Graph,
    Desire,
    GW,
    RealGrad,
    Other(&'static str),
}

/// Snapshot of a single Z-space pulse observation.
#[derive(Clone, Debug, PartialEq)]
pub struct ZPulse {
    pub source: ZSource,
    pub ts: u64,
    pub tempo: f32,
    pub band_energy: (f32, f32, f32),
    pub density_fluctuation: f32,
    pub drift: f32,
    pub z_bias: f32,
    pub support: ZSupport,
    pub scale: Option<ZScale>,
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

    pub fn density_fluctuation(&self) -> f32 {
        self.density_fluctuation
    }

    pub fn normalised_drift(&self) -> f32 {
        let total = self.total_energy().max(1e-6);
        let (a, _, b) = self.band_energy;
        (a - b) / total
    }

    pub fn is_empty(&self) -> bool {
        self.support.is_empty() && self.total_energy() <= f32::EPSILON
    }

    fn support_strength(&self) -> f32 {
        self.support.total().max(0.0)
    }

    pub fn density_fluctuation_for(band_energy: (f32, f32, f32)) -> f32 {
        let (mut leading, mut central, mut trailing) = band_energy;
        if !leading.is_finite() {
            leading = 0.0;
        }
        if !central.is_finite() {
            central = 0.0;
        }
        if !trailing.is_finite() {
            trailing = 0.0;
        }
        leading = leading.max(0.0);
        central = central.max(0.0);
        trailing = trailing.max(0.0);

        let total = leading + central + trailing;
        if total <= f32::EPSILON {
            return 0.0;
        }

        let mean = total / 3.0;
        let variance = {
            let dl = leading - mean;
            let dc = central - mean;
            let dt = trailing - mean;
            (dl * dl + dc * dc + dt * dt) / 3.0
        };

        let fluctuation = (variance.sqrt() / (total + 1e-6)).clamp(0.0, 1.0);
        if fluctuation.is_finite() {
            fluctuation
        } else {
            0.0
        }
    }
}

impl Default for ZPulse {
    fn default() -> Self {
        Self {
            source: ZSource::Microlocal,
            ts: 0,
            tempo: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            density_fluctuation: 0.0,
            drift: 0.0,
            z_bias: 0.0,
            support: ZSupport::default(),
            scale: None,
            quality: 0.0,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }
}

// -----------------------------------------------------------------------------
// Emitters and configuration types
// -----------------------------------------------------------------------------

/// Identifies a source capable of emitting [`ZPulse`] records.
pub trait ZEmitter: Send {
    /// Returns the canonical source identifier for pulses emitted by this implementation.
    fn name(&self) -> ZSource;

    /// Advances the emitter one step and returns the next available pulse, if any.
    fn tick(&mut self, now: u64) -> Option<ZPulse>;

    /// Optional quality hint used when the emitted pulse does not set one explicitly.
    fn quality_hint(&self) -> Option<f32> {
        None
    }
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
        let gain_floor = gain_floor.max(0.0);
        Self {
            gain_floor,
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

impl LatencyAlignerCfg {
    pub fn balanced() -> Self {
        Self {
            window: 128,
            hop: 1,
            max_lag_steps: 48,
            alpha: 0.25,
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

impl Default for LatencyAlignerCfg {
    fn default() -> Self {
        Self::balanced()
    }
}

#[derive(Clone, Debug, Default)]
struct SourceLast {
    ts: u64,
    strength: f32,
}

#[derive(Clone, Debug, Default)]
struct LagEstimate {
    lag: f32,
    frames_since_update: u32,
    seeded: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct LatencyAlignerState {
    cfg: LatencyAlignerCfg,
    last: FxHashMap<ZSource, SourceLast>,
    lags: FxHashMap<ZSource, LagEstimate>,
    pending_events: Vec<String>,
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
            entry.lag = pulse.latency_ms;
            entry.frames_since_update = 0;
            entry.seeded = true;
            self.pending_events.push(format!(
                "latency.seeded:{:?}:{:.2}",
                pulse.source, entry.lag
            ));
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

    fn prepare(&mut self, now: u64, events: &mut Vec<String>) {
        events.append(&mut self.pending_events);
        let Some(anchor) = self.last.get(&ZSource::Microlocal).cloned() else {
            self.increment_all();
            return;
        };

        if self.cfg.coherence_min > 1.0 {
            for source in self.last.keys() {
                if *source != ZSource::Microlocal {
                    events.push(format!("latency.low_coherence:{:?}", source));
                }
            }
            self.increment_all();
            return;
        }

        let hop = self.cfg.hop.max(1) as f32;
        let max_lag = self.cfg.max_lag_steps.max(1) as f32;
        let alpha = self.cfg.alpha.clamp(0.0, 1.0);

        for (source, state) in self.last.clone() {
            if source == ZSource::Microlocal {
                continue;
            }
            let entry = self.lags.entry(source).or_insert_with(LagEstimate::default);

            if entry.seeded {
                if self.cfg.hold_steps > 0 {
                    events.push(format!("latency.held:{:?}", source));
                    entry.frames_since_update = entry.frames_since_update.max(1);
                    entry.seeded = false;
                    continue;
                }
                entry.seeded = false;
            }

            if entry.frames_since_update != 0 && entry.frames_since_update < self.cfg.hold_steps {
                events.push(format!("latency.held:{:?}", source));
                continue;
            }

            let raw_lag_bins = (state.ts as i64 - anchor.ts as i64) as f32;
            let clamped = raw_lag_bins.clamp(-max_lag, max_lag);
            let lag_units = if self.cfg.fractional {
                clamped * hop
            } else {
                clamped.round() * hop
            };

            if entry.frames_since_update == 0 {
                entry.lag = lag_units;
            } else if entry.frames_since_update == u32::MAX {
                entry.lag = lag_units;
            } else {
                entry.lag = lerp(entry.lag, lag_units, alpha);
            }

            entry.frames_since_update = 0;
            events.push(format!("latency.adjusted:{:?}:{:.2}", source, entry.lag));
        }

        self.increment_all();
        self.last.insert(
            ZSource::Microlocal,
            SourceLast {
                ts: now,
                strength: anchor.strength,
            },
        );
    }

    fn apply(&self, pulse: &mut ZPulse) {
        if let Some(lag) = self.lags.get(&pulse.source) {
            if lag.lag.is_finite() {
                pulse.latency_ms = lag.lag;
                pulse.ts = shift_timestamp(pulse.ts, lag.lag, self.cfg.hop);
            }
        }
    }

    fn lag_for(&self, source: &ZSource) -> Option<f32> {
        self.lags.get(source).map(|lag| lag.lag)
    }

    fn increment_all(&mut self) {
        for estimate in self.lags.values_mut() {
            if estimate.frames_since_update == u32::MAX {
                estimate.frames_since_update = 1;
            } else {
                estimate.frames_since_update = estimate.frames_since_update.saturating_add(1);
            }
        }
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
            latency: Some(LatencyAlignerCfg::balanced()),
        }
    }
}

/// Fused output returned by the conductor.
#[derive(Clone, Debug, Default)]
pub struct ZFused {
    pub ts: u64,
    pub z: f32,
    pub support: f32,
    pub drift: f32,
    pub quality: f32,
    pub density_fluctuation: f32,
    pub events: Vec<String>,
    pub attributions: Vec<(ZSource, f32)>,
}

/// Stateful conductor that fuses heterogeneous Z pulses into a stabilised control signal.
#[derive(Clone)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    pub(crate) freq: Option<ZFrequencyConfig>,
    pub(crate) adaptive: Option<ZAdaptiveGainCfg>,
    pub(crate) latency: Option<LatencyAlignerState>,
    pending: VecDeque<ZPulse>,
    last_step: Option<u64>,
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
        let latency = cfg.latency.unwrap_or_else(LatencyAlignerCfg::balanced);
        Self {
            cfg,
            freq: None,
            adaptive: None,
            latency: Some(LatencyAlignerState::new(latency)),
            pending: VecDeque::new(),
            last_step: None,
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

    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.freq = cfg;
    }

    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
        self.adaptive = cfg;
    }

    pub fn set_latency_aligner(&mut self, cfg: Option<LatencyAlignerCfg>) {
        self.latency = cfg.map(LatencyAlignerState::new);
    }

    pub fn latency_for(&self, source: &ZSource) -> Option<f32> {
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
        if let Some(lat) = self.latency.as_mut() {
            lat.record(&pulse);
        }
        self.pending.push_back(pulse);
    }

    pub fn step(&mut self, now: u64) -> ZFused {
        self.last_step = Some(now);
        let mut events = Vec::new();
        if let Some(latency) = self.latency.as_mut() {
            latency.prepare(now, &mut events);
        }

        let mut ready = Vec::new();
        let mut retained = VecDeque::with_capacity(self.pending.len());
        while let Some(mut pulse) = self.pending.pop_front() {
            if self.cfg.latency_align {
                if let Some(lat) = self.latency.as_ref() {
                    lat.apply(&mut pulse);
                }
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
        let mut quality_weight = 0.0f32;
        let mut drift_weight = 0.0f32;
        let mut drift_sum = 0.0f32;
        let mut attributions: Vec<(ZSource, f32)> = Vec::new();
        let mut density_sum = 0.0f32;
        let mut density_weight = 0.0f32;

        for pulse in &ready {
            if pulse.is_empty() {
                continue;
            }
            let support = pulse.support_strength();
            total_support += support;
            weighted_z += pulse.z_bias * support;
            let weight = support.max(1e-6);
            weighted_quality += pulse.quality * weight;
            quality_weight += weight;
            let drift_weighting = pulse.quality.max(1e-6);
            drift_sum += pulse.normalised_drift() * drift_weighting;
            drift_weight += drift_weighting;
            density_sum += pulse.density_fluctuation * support;
            density_weight += support;
            attributions.push((pulse.source, support));
        }

        attributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut raw_z = if total_support > 0.0 {
            weighted_z / total_support
        } else {
            0.0
        };

        let fused_quality = if quality_weight > 0.0 {
            (weighted_quality / quality_weight).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let fused_density = if density_weight > 0.0 {
            (density_sum / density_weight).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let fused_drift = if drift_weight > 0.0 {
            (drift_sum / drift_weight).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let mut fused = ZFused {
            ts: now,
            z: 0.0,
            support: total_support,
            drift: fused_drift,
            quality: fused_quality,
            density_fluctuation: fused_density,
            events,
            attributions,
        };

        let incoming_sign = raw_z.signum();
        if let Some(until) = self.hold_until {
            if now < until {
                fused.events.push("flip-held".to_string());
                raw_z = self.last_z;
            } else {
                self.hold_until = None;
                if let Some(sign) = self.pending_flip_sign.take() {
                    fused.events.push("sign-flip".to_string());
                    raw_z = raw_z.abs().max(self.cfg.robust_delta).copysign(sign);
                }
            }
        } else if incoming_sign != 0.0
            && self.last_z.signum() != 0.0
            && incoming_sign != self.last_z.signum()
        {
            self.hold_until = Some(now + self.cfg.flip_hold as u64);
            self.pending_flip_sign = Some(incoming_sign);
            fused.events.push("flip-held".to_string());
            raw_z = self.last_z;
        }

        let alpha = if self.last_z.signum() == incoming_sign {
            self.cfg.alpha_fast
        } else {
            self.cfg.alpha_slow
        }
        .clamp(0.0, 1.0);

        let mut target = lerp(self.last_z, raw_z, alpha);
        let delta = target - self.last_z;
        if delta.abs() > self.cfg.slew_max {
            target = self.last_z + self.cfg.slew_max.copysign(delta);
        }
        target = target.clamp(-self.cfg.z_budget, self.cfg.z_budget);

        self.last_z = target;
        fused.z = target;
        if fused.density_fluctuation > 0.45 {
            fused.events.push(format!(
                "density.fluctuation:{:.3}",
                fused.density_fluctuation
            ));
        }
        fused
    }

    pub fn step_from_registry(&mut self, registry: &mut ZRegistry, now: u64) -> ZFused {
        for mut pulse in registry.gather(now) {
            if pulse.quality <= 0.0 {
                pulse.quality = derive_quality(&pulse);
            }
            self.ingest(pulse);
        }
        self.step(now)
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn shift_timestamp(ts: u64, lag: f32, hop: u64) -> u64 {
    if !lag.is_finite() {
        return ts;
    }
    let hop = hop.max(1) as f32;
    let delta = (lag / hop).round() as i64;
    let shifted = ts as i64 + delta;
    if shifted < 0 {
        0
    } else {
        shifted as u64
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
    let base = (0.4 * snr + 0.3 * support_norm + 0.3 * drift_norm).clamp(0.0, 1.0);
    let penalty = 1.0 - pulse.density_fluctuation.clamp(0.0, 1.0) * 0.2;
    (base * penalty).clamp(0.0, 1.0)
}

/// Registry used to poll multiple emitters and return their pending pulses.
#[derive(Default)]
pub struct ZRegistry {
    emitters: Vec<Box<dyn ZEmitter>>,
}

impl ZRegistry {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            emitters: Vec::with_capacity(capacity),
        }
    }

    pub fn register<E>(&mut self, emitter: E)
    where
        E: ZEmitter + 'static,
    {
        self.emitters.push(Box::new(emitter));
    }

    pub fn gather(&mut self, now: u64) -> Vec<ZPulse> {
        let mut pulses = Vec::new();
        for emitter in &mut self.emitters {
            while let Some(mut pulse) = emitter.tick(now) {
                if pulse.quality <= 0.0 {
                    if let Some(hint) = emitter.quality_hint() {
                        pulse.quality = hint;
                    }
                }
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

#[cfg(test)]
mod conductor_tests {
    use super::*;

    fn pulse(source: ZSource, ts: u64, drift: f32, quality: f32) -> ZPulse {
        let support = ZSupport {
            leading: drift.abs(),
            central: 0.0,
            trailing: drift.abs(),
        };
        let band_energy = (support.leading, 0.0, support.trailing);
        ZPulse {
            source,
            ts,
            tempo: drift.abs(),
            band_energy,
            density_fluctuation: ZPulse::density_fluctuation_for(band_energy),
            drift,
            z_bias: drift,
            support,
            quality,
            stderr: 0.0,
            latency_ms: 0.0,
            ..ZPulse::default()
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
            band_energy: (0.6, 0.8, 0.4),
            density_fluctuation: ZPulse::density_fluctuation_for((0.6, 0.8, 0.4)),
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
        let mut conductor = ZConductor::new(ZConductorCfg::default());
        conductor.set_latency_aligner(Some(align_cfg));
        for lag in 5..15 {
            let ts_anchor = lag as u64 * 10;
            let ts_target = ts_anchor + lag as u64;
            let drift = (lag as f32 * 0.3).sin();
            conductor.ingest(pulse(ZSource::Microlocal, ts_anchor, drift, 1.0));
            conductor.ingest(pulse(ZSource::Maxwell, ts_target, drift, 1.0));
            let _ = conductor.step(ts_anchor);
        }
        let estimate = conductor.latency_for(&ZSource::Maxwell).unwrap();
        assert!((estimate - 10.0).abs() <= 10.0);
    }

    #[test]
    fn latency_aligner_clamps_coherence() {
        let align_cfg = LatencyAlignerCfg {
            coherence_min: 2.0,
            ..LatencyAlignerCfg::balanced()
        };
        let mut conductor = ZConductor::new(ZConductorCfg::default());
        conductor.set_latency_aligner(Some(align_cfg));

        let mut saw_low = false;
        let mut saw_adjust = false;
        for ts in 0..10 {
            let anchor_drift = (ts as f32 * 0.3).sin();
            let target_drift = (ts as f32 * 0.4).cos();
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
        let mut conductor = ZConductor::new(ZConductorCfg::default());
        conductor.set_latency_aligner(Some(align_cfg));
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
        let mut conductor = ZConductor::new(ZConductorCfg::default());
        conductor.set_latency_aligner(Some(align_cfg));
        let mut hinted = pulse(ZSource::Graph, 0, 0.5, 1.0);
        hinted.latency_ms = 12.5;
        conductor.ingest(pulse(ZSource::Microlocal, 0, 0.4, 1.0));
        conductor.ingest(hinted);
        let _ = conductor.step(0);
        let estimate = conductor.latency_for(&ZSource::Graph).unwrap();
        assert!((estimate - 12.5).abs() <= 1e-3);
    }

    #[test]
    fn conductor_allows_optional_configs() {
        let mut conductor = ZConductor::new(ZConductorCfg::default());
        assert!(conductor.freq.is_none());
        assert!(conductor.adaptive.is_none());
        assert!(conductor.latency.is_some());

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
            band_energy: (0.4, 0.4, 0.4),
            density_fluctuation: ZPulse::density_fluctuation_for((0.4, 0.4, 0.4)),
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
