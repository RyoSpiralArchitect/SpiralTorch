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
use std::cmp::Ordering;
use std::collections::VecDeque;

// [SCALE-TODO] Compatibility shim: ZScale
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ZScale(pub f32);

impl ZScale {
    pub const ONE: ZScale = ZScale(1.0);

    #[inline]
    pub fn new(v: f32) -> Self {
        Self(v)
    }

    #[inline]
    pub fn value(self) -> f32 {
        self.0
    }
}

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

fn clamp_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Encodes the physical and logarithmic radius of a Z pulse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZScale {
    pub physical_radius: f32,
    pub log_radius: f32,
}

impl ZScale {
    /// Creates a new scale from a physical radius, rejecting non-positive or non-finite values.
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

    /// Builds a scale directly from precomputed components.
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
        let mut total = 0.0f32;
        let mut physical = 0.0f32;
        let mut log = 0.0f32;
        for (scale, weight) in weights {
            if weight.is_finite() && weight > 0.0 {
                total += weight;
                physical += scale.physical_radius * weight;
                log += scale.log_radius * weight;
            }
        }
        if total > 0.0 {
            Self::from_components(physical / total, log / total)
        } else {
            None
        }
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
    RealGrad,
    GW,
    Other(&'static str),
}

/// Snapshot of a single Z-space pulse observation.
#[derive(Clone, Debug, PartialEq)]
pub struct ZPulse {
    pub source: ZSource,
    pub ts: u64,
    pub tempo: f32,
    pub band_energy: (f32, f32, f32),
    pub drift: f32,
    pub z_bias: f32,
    pub support: ZSupport,
    pub scale: Option<ZScale>,
    pub quality: f32,
    pub stderr: f32,
    pub latency_ms: f32,
}

impl ZPulse {
    /// Returns the total support mass across all bands.
    pub fn support_mass(&self) -> f32 {
        self.support.leading + self.support.central + self.support.trailing
    }

    /// Returns the total band energy.
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
        self.support.is_empty() && self.total_energy() <= f32::EPSILON
    }

    fn support_strength(&self) -> f32 {
        self.support.total().max(0.0)
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
            scale: None,
            quality: 0.0,
            stderr: 0.0,
            latency_ms: 0.0,
            tempo: 0.0,
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Encodes the physical and logarithmic radius of a Z pulse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZScale {
    pub physical_radius: f32,
    pub log_radius: f32,
}

impl ZScale {
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

/// Encodes the physical and logarithmic radius of a Z pulse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZScale {
    pub physical_radius: f32,
    pub log_radius: f32,
}

impl ZScale {
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

/// Identifies a source capable of emitting [`ZPulse`] records.
pub trait ZEmitter: Send {
    /// Identifies the emitter source backing the generated pulses.
    fn name(&self) -> ZSource;

    /// Produces the next available pulse for the provided timestamp.
    fn tick(&mut self, now: u64) -> Option<ZPulse>;
}

/// Configuration for the conductor frequency tracker.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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

/// Configuration for adaptive gain smoothing.
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

/// Configuration governing the behaviour of [`ZConductor`].
#[derive(Clone, Debug)]
pub struct ZConductorCfg {
    pub gain: FxHashMap<ZSource, f32>,
    pub alpha_fast: f32,
    pub alpha_slow: f32,
    pub slew_max: f32,
    pub flip_hold: u32,
    /// Robustness threshold for the Huber loss.
    pub robust_delta: f32,
    /// Absolute budget allowed for the fused Z output.
    pub z_budget: f32,
    /// Back-calculation coefficient used when the budget engages.
    pub back_calculation: f32,
    /// Optional latency alignment stage applied before fusion.
    pub latency: Option<LatencyAlignerCfg>,
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
        let hop = self.cfg.hop.max(1);
        let bin = pulse.ts / hop;
        let weight = (pulse.quality.max(1e-6) * pulse.support_mass().max(1e-6)).max(1e-6);
        if let Some(history) = self.histories.get(&pulse.source) {
            if let Some(last) = history.last_ts {
                if pulse.ts < last {
                    self.pending_events
                        .push(format!("latency.invalid_ts:{:?}", pulse.source));
                    return;
                }
            }
        }
        if pulse.latency_ms.is_finite() && pulse.latency_ms.abs() > f32::EPSILON {
            let entry = self.lags.entry(pulse.source).or_default();
            entry.lag = pulse.latency_ms;
            entry.frames_since_update = 0;
            self.pending_events.push(format!(
                "latency.seeded:{:?}:{:.2}",
                pulse.source, entry.lag
            ));
        }
    }

    fn increment_all(&mut self) {
        for lag in self.lags.values_mut() {
            if lag.frames_since_update != u32::MAX {
                lag.frames_since_update = lag.frames_since_update.saturating_add(1);
            }
        }
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
        for (source, state) in self.last.clone() {
            if source == ZSource::Microlocal {
                continue;
            }
            let entry = self.lags.entry(source).or_default();
            if entry.frames_since_update != u32::MAX
                && entry.frames_since_update < self.cfg.hold_steps
            {
                events.push(format!("latency.held:{:?}", source));
                continue;
            }
            let raw_lag_bins = (state.ts as i64 - anchor.ts as i64) as f32;
            let clamped = raw_lag_bins.clamp(-max_lag, max_lag);
            let lag_units = clamped * hop;
            if entry.frames_since_update == u32::MAX {
                entry.lag = lag_units;
            } else {
                let alpha = self.cfg.alpha.clamp(0.0, 1.0);
                entry.lag = lerp(entry.lag, lag_units, alpha);
            }
            entry.frames_since_update = 0;
            events.push(format!("latency.adjusted:{:?}:{:.2}", source, entry.lag));
        }

        self.increment_all();
        // Reset the anchor timestamp so future stale pulses do not inherit it.
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

fn source_priority(source: &ZSource) -> i32 {
    match source {
        ZSource::Microlocal => 4,
        ZSource::Maxwell => 3,
        ZSource::Graph => 3,
        ZSource::RealGrad => 3,
        ZSource::Desire => 2,
        ZSource::GW => 2,
        ZSource::Other(_) => 1,
    }
}

fn derive_quality(pulse: &ZPulse) -> f32 {
    let support_score = pulse.support_mass().max(0.0) / (pulse.support_mass().max(0.0) + 1.0);
    let energy = pulse.total_energy().max(0.0);
    let energy_score = energy / (energy + 1.0);
    let drift_score = pulse.drift.abs().tanh();
    let stderr_score = if pulse.stderr.is_finite() && pulse.stderr > 0.0 {
        (1.0 / (1.0 + pulse.stderr)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let score = 0.4 * support_score + 0.3 * energy_score + 0.3 * drift_score;
    (score * stderr_score).clamp(0.0, 1.0)
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| {
        match (a.is_finite(), b.is_finite()) {
            (true, true) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => Ordering::Equal,
        }
    });
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
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

fn slew_limit(previous: f32, target: f32, max_delta: f32) -> f32 {
    let limit = max_delta.abs();
    if limit <= f32::EPSILON {
        return target;
    }
    let delta = target - previous;
    if delta > limit {
        previous + limit
    } else if delta < -limit {
        previous - limit
    } else {
        target
    }
}

fn ema(previous: f32, target: f32, alpha: f32) -> f32 {
    previous + alpha * (target - previous)
}

/// Attribution assigned to a specific source during fusion.
#[derive(Clone, Debug, PartialEq)]
pub struct ZAttribution {
    pub source: ZSource,
    pub weight: f32,
}

/// Result of a [`ZConductor::step`] call.
#[derive(Clone, Debug, PartialEq)]
pub struct ZFused {
    pub ts: u64,
    pub z: f32,
    pub support: f32,
    pub drift: f32,
    pub quality: f32,
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

        let mut drift = 0.0;
        let mut attributions = Vec::new();
        let mut total_support = 0.0;

        if !ready.is_empty() {
            total_support = ready
                .iter()
                .filter(|p| !p.is_empty())
                .map(|pulse| pulse.support_mass().max(0.0))
                .sum();
            let mut contributions: Vec<(ZSource, f32, f32)> = ready
                .iter()
                .filter(|p| !p.is_empty())
                .map(|pulse| {
                    let gain = *self.cfg.gain.get(&pulse.source).unwrap_or(&1.0);
                    let base_w = (pulse.quality * gain).max(1e-6);
                    (pulse.source.clone(), base_w, pulse.normalised_drift())
                })
                .collect();

            if !contributions.is_empty() {
                let mut drifts: Vec<f32> = contributions.iter().map(|(_, _, d)| *d).collect();
                let median = median(&mut drifts);
                let mut weight_sum = 0.0f32;
                let mut numerator = 0.0f32;
                for (source, weight, drift_norm) in contributions.iter_mut() {
                    let robust = huber_weight(*drift_norm - median, self.cfg.robust_delta);
                    *weight *= robust;
                    weight_sum += *weight;
                    numerator += *weight * *drift_norm;
                    attributions.push((source.clone(), *weight));
                }
                if weight_sum > 0.0 {
                    drift = numerator / weight_sum;
                    let inv = 1.0 / weight_sum;
                    for attrib in &mut attributions {
                        attrib.1 *= inv;
                    }
                }
            }
        }

        if attributions.is_empty() {
            attributions.push((ZSource::Microlocal, 0.0));
        }

        let filtered = self.apply_temporal_filters(drift, &mut events);
        let mut z = filtered * self.mag_hat.abs().max(1e-6);
        if filtered.abs() <= f32::EPSILON {
            z = 0.0;
        }

        let z_before_limits = z;
        let limited = slew_limit(self.last_z, z, self.cfg.slew_max);
        if (limited - z).abs() > 1e-5 {
            events.push("slew-limited".to_string());
            z = limited;
        }

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
            drift_sum += pulse.normalised_drift() * pulse.quality.max(1e-6);
            drift_weight += pulse.quality.max(1e-6);
            attributions.push((pulse.source, support));
        }

        let mut raw_z = if total_support > 0.0 {
            weighted_z / total_support
        } else {
            0.0
        };

        let mut fused = ZFused {
            ts: now,
            support: total_support,
            drift: if drift_weight > 0.0 {
                (drift_sum / drift_weight).clamp(-1.0, 1.0)
            } else {
                0.0
            },
            quality: if quality_weight > 0.0 {
                (weighted_quality / quality_weight).clamp(0.0, 1.0)
            } else {
                0.0
            },
            events,
            attributions: Vec::new(),
            z: 0.0,
        };

        attributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        fused.attributions = attributions;

        let incoming_sign = raw_z.signum();
        if let Some(hold_until) = self.hold_until {
            if now < hold_until {
                fused.events.push("flip-held".to_string());
                raw_z = self.last_z;
            } else {
                self.hold_until = None;
            }
        }
        if self.hold_until.is_none() {
            if let Some(sign) = self.pending_flip_sign.take() {
                fused.events.push("sign-flip".to_string());
                let magnitude = raw_z.abs().max(self.cfg.robust_delta);
                raw_z = magnitude.copysign(sign);
            } else if self.last_z != 0.0
                && incoming_sign != 0.0
                && incoming_sign != self.last_z.signum()
            {
                self.hold_until = Some(now + self.cfg.flip_hold as u64);
                self.pending_flip_sign = Some(incoming_sign);
                fused.events.push("flip-held".to_string());
                raw_z = self.last_z;
            }
        }

        let alpha = self.cfg.alpha_fast.clamp(0.0, 1.0);
        let mut target = lerp(self.last_z, raw_z, alpha);
        let delta = target - self.last_z;
        if delta.abs() > self.cfg.slew_max {
            target = self.last_z + self.cfg.slew_max.copysign(delta);
        }
        target = target.clamp(-self.cfg.z_budget, self.cfg.z_budget);
        self.last_z = target;
        fused.z = target;

        fused
    }

#[cfg(test)]
mod conductor_tests {
    use super::*;

    fn pulse(source: ZSource, ts: u64, drift: f32, quality: f32) -> ZPulse {
        ZPulse {
            source,
            ts,
            band_energy: (drift.abs(), 0.0, 0.0),
            drift,
            z_bias: drift,
            support: ZSupport {
                leading: drift.abs(),
                central: 0.0,
                trailing: 0.0,
            },
            quality,
            stderr: 0.0,
            latency_ms: 0.0,
            tempo: 0.0,
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
            conductor.ingest(ZPulse {
                source: ZSource::Microlocal,
                ts: idx as u64,
                band_energy: (1.0 + sign.max(0.0), 0.0, 1.0 + (-sign).max(0.0)),
                drift: sign,
                z_bias: sign,
                support: ZSupport {
                    leading: 1.0 + sign.max(0.0),
                    central: 0.0,
                    trailing: 1.0 + (-sign).max(0.0),
                },
                quality: 1.0,
                stderr: 0.0,
                latency_ms: 0.0,
                tempo: 0.0,
            });
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
            z_bias: 0.25,
            ..ZPulse::default()
        });
        let fused = conductor.step(0);
        assert!(fused.z.abs() <= 0.5 + 1e-6);
        assert!((fused.attributions.iter().map(|a| a.weight).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(fused.events.iter().any(|e| e == "saturated"));
    }
}

#[cfg(test)]
mod latency_tests {
    use super::*;

    fn pulse(source: ZSource, ts: u64, drift: f32, quality: f32) -> ZPulse {
        ZPulse {
            source,
            ts,
            band_energy: (drift.abs(), 0.0, 0.0),
            drift,
            z_bias: drift,
            support: ZSupport {
                leading: drift.abs(),
                central: 0.0,
                trailing: 0.0,
            },
            quality,
            stderr: 0.0,
            latency_ms: 0.0,
            tempo: 0.0,
        }
    }

impl ZRegistry {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            emitters: Vec::with_capacity(capacity),
        }
        let estimate = conductor.latency_for(&ZSource::Maxwell).unwrap();
        assert!((estimate - lag as f32).abs() <= 1.0);
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
        assert!(last_update_step.is_some());
    }

    #[test]
    fn latency_aligner_rejects_non_monotonic_timestamps() {
        let align_cfg = LatencyAlignerCfg {
            window: 128,
            hop: 1,
            max_lag_steps: 32,
            alpha: 0.2,
            coherence_min: 0.2,
            hold_steps: 0,
            fractional: false,
        };
        let mut conductor =
            ZConductor::new(ZConductorCfg::default().with_latency_aligner(align_cfg));
        conductor.ingest(pulse(ZSource::Microlocal, 10, 0.8, 1.0));
        conductor.ingest(pulse(ZSource::Microlocal, 8, 0.6, 1.0));
        let fused = conductor.step(10);
        assert!(fused
            .events
            .iter()
            .any(|e| e.starts_with("latency.invalid_ts")));
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
}

#[cfg(test)]
mod conductor_config_tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default, Debug)]
    struct DesireEmitter {
        queue: Arc<Mutex<VecDeque<ZPulse>>>,
    }

    impl DesireEmitter {
        fn new() -> Self {
            Self::default()
        }

        fn enqueue(&self, pulse: ZPulse) {
            self.extend([pulse]);
        }

        fn extend<I>(&self, pulses: I)
        where
            I: IntoIterator<Item = ZPulse>,
        {
            let mut queue = self.queue.lock().unwrap();
            queue.extend(pulses);
        }

        fn tick(&self, now: u64) -> Option<ZPulse> {
            let mut queue = self.queue.lock().unwrap();
            queue.pop_front().map(|mut pulse| {
                pulse.source = ZSource::Desire;
                if pulse.ts == 0 {
                    pulse.ts = now;
                }
                pulse
            })
        }
    }

    #[derive(Default)]
    struct ZRegistry {
        emitters: Vec<DesireEmitter>,
    }

    impl ZRegistry {
        fn new() -> Self {
            Self::default()
        }

        fn register(&mut self, emitter: DesireEmitter) {
            self.emitters.push(emitter);
        }

        fn gather(&self, now: u64) -> Vec<ZPulse> {
            let mut pulses = Vec::new();
            for emitter in &self.emitters {
                if let Some(pulse) = emitter.tick(now) {
                    pulses.push(pulse);
                }
            }
            pulses
        }
    }

    #[test]
    fn conductor_allows_optional_configs() {
        let mut conductor = ZConductor::new(ZConductorCfg::default());
        assert!(conductor.freq.is_none());
        assert!(conductor.adaptive.is_none());
        assert!(conductor.latency.is_none());

        conductor.set_frequency_config(Some(ZFrequencyConfig::new(0.5, 0.1)));
        conductor.set_adaptive_gain_config(Some(ZAdaptiveGainCfg::new(0.1, 1.0, 0.8)));
        conductor.set_latency_aligner(Some(LatencyAlignerCfg::default()));

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
                leading: 0.2,
                central: 0.2,
                trailing: 0.0,
            },
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

#[cfg(test)]
mod latency_tests {
    use super::*;

    fn conductor_with_latency(cfg: LatencyAlignerCfg) -> ZConductor {
        let mut base = ZConductorCfg::default();
        base.latency = Some(cfg);
        base.latency_align = true;
        ZConductor::new(base)
    }

    fn pulse(source: ZSource, ts: u64, drift: f32) -> ZPulse {
        ZPulse {
            source,
            ts,
            tempo: 0.0,
            band_energy: (drift.abs(), 0.0, 0.0),
            drift,
            z_bias: drift,
            support: ZSupport::new(drift.abs(), drift.abs(), 0.0),
            quality: 1.0,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }

    #[test]
    fn latency_aligner_tracks_known_offset() {
        let align_cfg = LatencyAlignerCfg {
            window: 96,
            hop: 1,
            max_lag_steps: 64,
            alpha: 0.35,
            coherence_min: 0.0,
            hold_steps: 0,
            fractional: false,
        };
        let mut conductor = conductor_with_latency(align_cfg);
        let lag = 6u64;

        for step in 0..96u64 {
            conductor.ingest(pulse(ZSource::Microlocal, step, 0.8));
            conductor.ingest(pulse(ZSource::Maxwell, step + lag, 0.8));
            conductor.step(step);
        }

        let estimate = conductor.latency_for(&ZSource::Maxwell).unwrap();
        assert!((estimate - lag as f32).abs() <= 1.5);
    }

    #[test]
    fn latency_aligner_honours_hold_steps() {
        let align_cfg = LatencyAlignerCfg {
            window: 64,
            hop: 1,
            max_lag_steps: 48,
            alpha: 1.0,
            coherence_min: 0.0,
            hold_steps: 3,
            fractional: false,
        };
        let mut conductor = conductor_with_latency(align_cfg);
        let mut last_update = None;

        for step in 0..48u64 {
            conductor.ingest(pulse(ZSource::Microlocal, step, 1.0));
            conductor.ingest(pulse(ZSource::Graph, step + 5, 1.0));
            let fused = conductor.step(step);

            if fused
                .events
                .iter()
                .any(|event| event.starts_with("latency.adjusted"))
            {
                if let Some(prev) = last_update {
                    assert!(step.saturating_sub(prev) >= 3);
                }
                last_update = Some(step);
            }
        }

        assert!(last_update.is_some());
    }

    #[test]
    fn latency_hint_seeds_initial_estimate() {
        let align_cfg = LatencyAlignerCfg::balanced();
        let mut conductor = conductor_with_latency(align_cfg);

        let mut hinted = pulse(ZSource::Graph, 0, 0.6);
        hinted.latency_ms = 12.0;
        conductor.ingest(pulse(ZSource::Microlocal, 0, 0.6));
        conductor.ingest(hinted);
        let fused = conductor.step(0);

        assert!(fused
            .events
            .iter()
            .any(|event| event.starts_with("latency.seeded")));
        let estimate = conductor.latency_for(&ZSource::Graph).unwrap();
        assert!((estimate - 12.0).abs() <= 1e-3);
    }
}
