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
    Graph,
    GW,
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

    fn is_near_zero(self) -> bool {
        self.leading.abs() <= f32::EPSILON
            && self.central.abs() <= f32::EPSILON
            && self.trailing.abs() <= f32::EPSILON
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
        self.support.leading + self.support.central + self.support.trailing
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
            drift: 0.0,
            z_bias: 0.0,
            support: ZSupport::default(),
            band_energy: (0.0, 0.0, 0.0),
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
            smoothing,
            minimum_energy,
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
            gain_floor,
            gain_ceil: gain_ceil.max(gain_floor),
            responsiveness,
        }
    }
}

/// Configuration for the latency alignment stage.
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

impl Default for ZAdaptiveGainCfg {
    fn default() -> Self {
        Self {
            gain_floor: 0.0,
            gain_ceil: 1.0,
            responsiveness: 0.0,
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

/// Configuration governing the behaviour of [`ZConductor`].
#[derive(Clone, Debug)]
pub struct ZConductorCfg {
    pub gain: FxHashMap<ZSource, f32>,
    pub alpha_fast: f32,
    pub alpha_slow: f32,
    pub slew_max: f32,
    pub flip_hold: u32,
    pub robust_delta: f32,
    pub z_budget: f32,
    pub back_calculation: f32,
    pub latency: Option<LatencyAlignerCfg>,
}

impl Default for ZConductorCfg {
    fn default() -> Self {
        Self {
            gain: FxHashMap::default(),
            alpha_fast: 0.35,
            alpha_slow: 0.12,
            slew_max: 0.35,
            flip_hold: 3,
            robust_delta: 0.25,
            z_budget: 1.2,
            back_calculation: 0.5,
            latency: None,
        }
    }
}

impl ZConductorCfg {
    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn with_latency_aligner(mut self, cfg: LatencyAlignerCfg) -> Self {
        self.latency = Some(cfg);
        self
    }
}

#[derive(Clone, Debug)]
struct LagEstimate {
    lag: f32,
    frames_since_update: u32,
}

impl LagEstimate {
    fn new() -> Self {
        Self {
            lag: 0.0,
            frames_since_update: u32::MAX,
        }
    }
}

#[derive(Clone, Debug)]
struct BinAccum {
    index: u64,
    weighted_value: f64,
    weight: f32,
}

impl BinAccum {
    fn mean(&self) -> f32 {
        if self.weight <= f32::EPSILON {
            0.0
        } else {
            (self.weighted_value / self.weight as f64) as f32
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SourceHistory {
    bins: VecDeque<BinAccum>,
    last_ts: Option<u64>,
}

impl SourceHistory {
    fn add_sample(&mut self, ts: u64, index: u64, value: f32, weight: f32) {
        self.last_ts = Some(ts);
        if let Some(back) = self.bins.back_mut() {
            if back.index == index {
                back.weighted_value += f64::from(weight) * f64::from(value);
                back.weight += weight;
                return;
            }
        }
        self.bins.push_back(BinAccum {
            index,
            weighted_value: f64::from(weight) * f64::from(value),
            weight,
        });
    }

    fn truncate(&mut self, min_index: u64) {
        while let Some(front) = self.bins.front() {
            if front.index < min_index {
                self.bins.pop_front();
            } else {
                break;
            }
        }
    }

    fn sequence(&self, end_index: u64, len: usize) -> Vec<f32> {
        if len == 0 {
            return Vec::new();
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
            cfg,
            histories: FxHashMap::default(),
            lags: FxHashMap::default(),
            last_eval_bin: None,
            pending_events: Vec::new(),
        }
    }

    fn record(&mut self, pulse: &ZPulse) {
        let hop = self.cfg.hop.max(1);
        let bin = pulse.ts / hop;
        let support = pulse.support_strength().max(1e-6);
        let weight = (pulse.quality.max(1e-6) * support).max(1e-6);
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
            self.seed_from_hint(&pulse.source, pulse.latency_ms);
        }
        let history = self
            .histories
            .entry(pulse.source)
            .or_insert_with(SourceHistory::default);
        history.add_sample(pulse.ts, bin, pulse.drift, weight);
    }

    fn seed_from_hint(&mut self, source: &ZSource, hint: f32) {
        let clamped = hint.clamp(-1_000_000.0, 1_000_000.0);
        let estimate = self.lags.entry(*source).or_insert_with(LagEstimate::new);
        if estimate.frames_since_update != u32::MAX {
            return;
        }
        estimate.frames_since_update = self.cfg.hold_steps;
        estimate.lag = clamped;
        self.pending_events
            .push(format!("latency.seeded:{:?}:{:.2}", source, clamped));
    }

    fn prepare(&mut self, now: u64, events: &mut Vec<String>) {
        events.extend(self.pending_events.drain(..));
        let hop = self.cfg.hop.max(1);
        let current_bin = now / hop;
        if let Some(last) = self.last_eval_bin {
            if current_bin == last {
                return;
            }
        }
        self.last_eval_bin = Some(current_bin);
        let mut window_bins = self.cfg.window / hop;
        if window_bins == 0 {
            window_bins = 1;
        }
        let min_index = current_bin.saturating_sub(window_bins.saturating_sub(1));
        for history in self.histories.values_mut() {
            history.truncate(min_index);
        }
        for lag in self.lags.values_mut() {
            lag.frames_since_update = lag.frames_since_update.saturating_add(1);
        }
        self.update_lags(current_bin, window_bins as usize, events);
    }

    fn lag_for(&self, source: &ZSource) -> Option<f32> {
        self.lags.get(source).map(|l| l.lag)
    }

    fn apply(&self, pulse: &mut ZPulse) {
        if let Some(lag) = self.lags.get(&pulse.source) {
            if !lag.lag.is_finite() {
                return;
            }
            let adjusted = shift_timestamp(pulse.ts, lag.lag);
            pulse.latency_ms = lag.lag;
            pulse.ts = adjusted;
        }
    }

    fn update_lags(&mut self, current_bin: u64, window_len: usize, events: &mut Vec<String>) {
        if self.histories.is_empty() {
            return;
        }
        let hop = self.cfg.hop.max(1) as f32;
        let max_lag_bins = self.cfg.max_lag_steps as isize;
        let max_lag_bins = max_lag_bins.max(0);
        if max_lag_bins == 0 {
            return;
        }

        let mut anchor: Option<(ZSource, f32)> = None;
        if let Some(history) = self.histories.get(&ZSource::Microlocal) {
            let strength = history.strength_up_to(current_bin);
            if strength > f32::EPSILON {
                anchor = Some((ZSource::Microlocal, strength));
            }
        }
        for (source, history) in &self.histories {
            let strength = history.strength_up_to(current_bin);
            if strength <= f32::EPSILON {
                continue;
            }
            let update_anchor = match &anchor {
                Some((current, best_strength)) => {
                    if strength > *best_strength + 1e-6 {
                        true
                    } else if (strength - *best_strength).abs() <= 1e-6 {
                        source_priority(source) > source_priority(current)
                    } else {
                        false
                    }
                }
                None => true,
            };
            if update_anchor {
                anchor = Some((*source, strength));
            }
        }

        let (anchor_source, _) = match anchor {
            Some(v) => v,
            None => return,
        };

        let anchor_history = match self.histories.get(&anchor_source) {
            Some(h) => h,
            None => return,
        };

        let anchor_seq = anchor_history.sequence(current_bin, window_len);
        let anchor_power: f32 = anchor_seq.iter().map(|v| v * v).sum();
        if anchor_power <= 1e-9 {
            return;
        }

        self.lags
            .entry(anchor_source)
            .or_insert_with(LagEstimate::new)
            .lag = 0.0;

        for (source, history) in &self.histories {
            if *source == anchor_source {
                continue;
            }
            let target_seq = history.sequence(current_bin, window_len);
            let target_power: f32 = target_seq.iter().map(|v| v * v).sum();
            if target_power <= 1e-9 {
                continue;
            }

            if let Some((coherence, lag_bins, frac_shift)) =
                best_lag(&anchor_seq, &target_seq, max_lag_bins)
            {
                let passes_coherence = coherence >= self.cfg.coherence_min;
                let estimate = self.lags.entry(*source).or_insert_with(LagEstimate::new);
                if !passes_coherence {
                    events.push(format!("latency.low_coherence:{:?}", source));
                    continue;
                }
                if estimate.frames_since_update < self.cfg.hold_steps {
                    events.push(format!("latency.held:{:?}", source));
                    continue;
                }
                let lag_bins = lag_bins as f32 + frac_shift;
                let lag_units = lag_bins * hop;
                let alpha = self.cfg.alpha.clamp(0.0, 1.0);
                if estimate.frames_since_update == u32::MAX || alpha <= 0.0 {
                    estimate.lag = lag_units;
                } else {
                    estimate.lag = (1.0 - alpha) * estimate.lag + alpha * lag_units;
                }
                estimate.frames_since_update = 0;
                events.push(format!("latency.adjusted:{:?}:{:.2}", source, lag_units));
            }
        }
    }
}

fn best_lag(anchor: &[f32], target: &[f32], max_lag: isize) -> Option<(f32, isize, f32)> {
    if anchor.is_empty() || target.is_empty() {
        return None;
    }
    let len = anchor.len().min(target.len());
    if len == 0 {
        return None;
    }
    let mut scores = Vec::new();
    for lag in -max_lag..=max_lag {
        let mut num = 0.0f32;
        let mut denom_a = 0.0f32;
        let mut denom_b = 0.0f32;
        let mut overlap = 0usize;
        for idx in 0..len {
            let j = idx as isize + lag;
            if j < 0 || j >= len as isize {
                continue;
            }
            let a = anchor[idx];
            let b = target[j as usize];
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            num += a * b;
            denom_a += a * a;
            denom_b += b * b;
            overlap += 1;
        }
        if overlap < 2 || denom_a <= 1e-9 || denom_b <= 1e-9 {
            continue;
        }
        let denom = (denom_a * denom_b).sqrt().max(1e-9);
        let corr = num / denom;
        scores.push((lag, corr));
    }
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

/// Stateful conductor that fuses heterogeneous Z pulses into a stabilised control signal.
#[derive(Clone, Debug)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    freq: Option<FrequencyFusionState>,
    adaptive: Option<AdaptiveGainState>,
    latency: Option<LatencyAlignerState>,
    pending: VecDeque<ZPulse>,
    sign_hat: f32,
    mag_hat: f32,
    last_sign: f32,
    flip_age: u32,
    last_z: f32,
    last_step_ts: Option<u64>,
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
            freq: None,
            adaptive: None,
            latency: latency_cfg.map(LatencyAlignerState::new),
            pending: VecDeque::new(),
            sign_hat: 0.0,
            mag_hat: 0.0,
            last_sign: 0.0,
            flip_age: u32::MAX,
            last_z: 0.0,
            last_step_ts: None,
        }
    }

    pub fn cfg(&self) -> &ZConductorCfg {
        &self.cfg
    }

    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }

    pub fn set_latency_aligner(&mut self, cfg: Option<LatencyAlignerCfg>) {
        self.cfg.latency = cfg.clone();
        self.latency = cfg.map(LatencyAlignerState::new);
    }

    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.frequency_cfg = cfg;
    }

    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
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
        if let Some(latency) = self.latency.as_mut() {
            latency.record(&pulse);
        }
        self.pending.push_back(pulse);
    }

    pub fn step(&mut self, now: u64) -> ZFused {
        let mut events = Vec::new();
        if let Some(latency) = self.latency.as_mut() {
            latency.prepare(now, &mut events);
        }
        let mut ready = Vec::new();
        let mut retained = VecDeque::with_capacity(self.pending.len());
        while let Some(mut pulse) = self.pending.pop_front() {
            if let Some(latency) = self.latency.as_ref() {
                latency.apply(&mut pulse);
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
                .map(|pulse| pulse.support_strength().max(0.0))
                .sum();
            let mut contributions: Vec<(ZSource, f32, f32)> = ready
                .iter()
                .filter(|p| !p.is_empty())
                .map(|pulse| {
                    let gain = *self.cfg.gain.get(&pulse.source).unwrap_or(&1.0);
                    let base_w = (pulse.quality * gain).max(1e-6);
                    (pulse.source, base_w, pulse.normalised_drift())
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
                    attributions.push((*source, *weight));
                }
                if weight_sum > 0.0 {
                    drift = numerator / weight_sum;
                    let inv = 1.0 / weight_sum;
                    for attrib in &mut attributions {
                        attrib.1 *= inv;
                    }
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

    pub fn register<E>(&mut self, emitter: E)
    where
        E: ZEmitter + Send + 'static,
    {
        self.emitters.push(Box::new(emitter));
    }

    pub fn gather(&mut self, now: u64) -> Vec<ZPulse> {
        let mut pulses = Vec::new();
        for emitter in &mut self.emitters {
            while let Some(pulse) = emitter.tick(now) {
                pulses.push(pulse);
            }
        }
        pulses
    }
}

/// Synthetic emitter used in tests to re-tag pulses with the Desire source.
#[derive(Clone, Default)]
pub struct DesireEmitter {
    queue: Arc<Mutex<VecDeque<ZPulse>>>,
}

impl DesireEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, pulse: ZPulse) {
        if let Ok(mut guard) = self.queue.lock() {
            guard.push_back(pulse);
        }
    }

    pub fn extend<I>(&self, pulses: I)
    where
        I: IntoIterator<Item = ZPulse>,
    {
        if let Ok(mut guard) = self.queue.lock() {
            guard.extend(pulses);
        }
    }
}

impl ZEmitter for DesireEmitter {
    fn name(&self) -> ZSource {
        ZSource::Desire
    }

    fn tick(&mut self, _now: u64) -> Option<ZPulse> {
        let mut guard = self.queue.lock().ok()?;
        guard.pop_front().map(|mut pulse| {
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
        delta / abs
    }
}

fn ema(prev: f32, value: f32, alpha: f32) -> f32 {
    (1.0 - alpha) * prev + alpha * value
}

fn slew_limit(previous: f32, target: f32, max_delta: f32) -> f32 {
    if max_delta <= f32::EPSILON {
        return target;
    }
    let delta = target - previous;
    if delta.abs() <= max_delta {
        target
    } else {
        previous + max_delta.copysign(delta)
    }
}

fn derive_quality(pulse: &ZPulse) -> f32 {
    let support = pulse.support_strength().max(1e-6);
    let energy = pulse.total_energy().max(1e-6);
    let drift = pulse.drift.abs();
    let stderr = pulse.stderr.abs().max(1e-6);
    let snr = (energy / stderr).tanh();
    let support_norm = (support / (support + 1.0)).min(1.0);
    let drift_norm = (drift / (drift + 1.0)).min(1.0);
    (0.4 * snr + 0.3 * support_norm + 0.3 * drift_norm).clamp(0.0, 1.0)
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

    pub fn enqueue(&self, mut pulse: ZPulse) {
        pulse.source = ZSource::Desire;
        let mut queue = self.queue.lock().expect("desire emitter queue poisoned");
        queue.push_back(pulse);
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
