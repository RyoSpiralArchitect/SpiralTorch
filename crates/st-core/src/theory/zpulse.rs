// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// =============================================================================
//  SpiralReality Proprietary
// Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
// NOTICE: This file contains confidential and proprietary information of
// SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
// OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
// WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
// NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
// SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
// AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// =============================================================================

use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::VecDeque;

/// Canonical pulse exchanged across the Z-space control stack.
#[derive(Clone, Debug)]
//! Microlocal pulse fusion utilities shared between the observability
//! backends and telemetry exporters.

use std::collections::VecDeque;

/// Origin marker for a captured pulse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZSource {
    Microlocal,
    Maxwell,
    RealGrad,
    Desire,
    Custom(u8),
}

impl Default for ZSource {
    fn default() -> Self {
        Self::Custom(0)
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
    /// Estimated latency between generation and observation in the caller's time units.
    /// Callers may populate this field to hint the aligner with a prior estimate; when left
    /// at `0.0` the aligner infers the delay from cross-correlation.
    pub latency_ms: f32,
}

impl ZPulse {
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
        self.support <= f32::EPSILON && self.total_energy() <= f32::EPSILON
    }
}

impl Default for ZPulse {
    fn default() -> Self {
        ZPulse {
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

#[derive(Clone, Copy, Debug, Default)]
pub struct ZFrequencyConfig {
    pub smoothing: f32,
    pub minimum_energy: f32,
}

impl ZFrequencyConfig {
    pub fn new(smoothing: f32, minimum_energy: f32) -> Self {
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

/// Trait implemented by pulse emitters that can feed the conductor.
pub trait ZEmitter: Send {
    /// Identifies the emitter source backing the generated pulses.
    fn name(&self) -> ZSource;

    /// Produces the next available pulse for the provided timestamp.
    fn tick(&mut self, now: u64) -> Option<ZPulse>;
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

/// Configuration governing the behaviour of [`ZConductor`].
#[derive(Clone, Debug)]
pub struct ZConductorCfg {
    pub gain: f32,
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
    /// Returns a configuration tuned for balanced real-time fusion.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Enables the latency aligner with the supplied configuration.
    pub fn with_latency_aligner(mut self, cfg: LatencyAlignerCfg) -> Self {
        self.latency = Some(cfg);
        self
    }
}

/// Configuration for the latency alignment stage.
#[derive(Clone, Debug, PartialEq)]
pub struct LatencyAlignerCfg {
    /// Width of the correlation history window (in the same units as [`ZPulse::ts`]).
    pub window: u64,
    /// Hop used when updating the history window and lag estimate.
    pub hop: u64,
    /// Maximum absolute lag explored during cross-correlation (expressed in hops).
    pub max_lag_steps: u32,
    /// Exponential moving-average factor applied to lag updates.
    pub alpha: f32,
    /// Minimum coherence required before committing a lag update.
    pub coherence_min: f32,
    /// Number of evaluation steps a lag estimate is held before another update is allowed.
    pub hold_steps: u32,
    /// Enables parabolic interpolation for sub-hop lag refinement.
    pub fractional: bool,
}

impl Default for LatencyAlignerCfg {
    fn default() -> Self {
        Self {
            window: 256,
            hop: 1,
            max_lag_steps: 64,
            alpha: 0.2,
            coherence_min: 0.25,
            hold_steps: 8,
            fractional: true,
        }
    }
}

impl LatencyAlignerCfg {
    /// Creates a new configuration with the balanced defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a latency-aligner profile suitable for most interactive workloads.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Helper for constructing configurations when only hop-based limits are needed.
    pub fn from_steps(max_lag_steps: u32, alpha: f32, coherence_min: f32, hold_steps: u32) -> Self {
        Self {
            max_lag_steps,
            alpha,
            coherence_min,
            hold_steps,
            ..Self::default()
        }
    }

    /// Overrides the correlation window length.
    pub fn with_window(mut self, window: u64) -> Self {
        self.window = window.max(1);
        self
    }

    /// Overrides the hop used for correlation updates.
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
                back.weighted_value += (value as f64) * (weight as f64);
                back.weight += weight;
                return;
            }
        }
        self.bins.push_back(BinAccum {
            index,
            weighted_value: (value as f64) * (weight as f64),
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
        let start_index = end_index.saturating_sub(len as u64 - 1);
        let mut seq = Vec::with_capacity(len);
        let mut iter = self.bins.iter();
        let mut current = iter.next();
        for idx in start_index..=end_index {
            while let Some(bin) = current {
                if bin.index < idx {
                    current = iter.next();
                } else {
                    break;
                }
            }
            let value = match current {
                Some(bin) if bin.index == idx => bin.mean(),
                _ => 0.0,
            };
            seq.push(value);
        }
        seq
    }

    fn strength_up_to(&self, end_index: u64) -> f32 {
        self.bins
            .iter()
            .filter(|bin| bin.index <= end_index)
            .map(|bin| bin.weight.abs())
            .sum()
    }
}

#[derive(Clone, Debug)]
struct LatencyAlignerState {
    cfg: LatencyAlignerCfg,
    histories: FxHashMap<ZSource, SourceHistory>,
    lags: FxHashMap<ZSource, LagEstimate>,
    last_eval_bin: Option<u64>,
    pending_events: Vec<String>,
}

impl LatencyAlignerState {
    fn new(cfg: LatencyAlignerCfg) -> Self {
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
        let weight = (pulse.quality.max(1e-6) * pulse.support.max(1e-6)).max(1e-6);
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
            .entry(pulse.source.clone())
            .or_insert_with(SourceHistory::default);
        history.add_sample(pulse.ts, bin, pulse.drift, weight);
    }

    fn seed_from_hint(&mut self, source: &ZSource, hint: f32) {
        let clamped = hint.clamp(-1_000_000.0, 1_000_000.0);
        let estimate = self
            .lags
            .entry(source.clone())
            .or_insert_with(LagEstimate::new);
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
                anchor = Some((source.clone(), strength));
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
            .entry(anchor_source.clone())
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
                let estimate = self
                    .lags
                    .entry(source.clone())
                    .or_insert_with(LagEstimate::new);
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

    if scores.is_empty() {
        return None;
    }

    let mut best = scores[0];
    let mut best_abs = best.1.abs();
    for &(lag, corr) in &scores[1..] {
        let abs = corr.abs();
        if abs > best_abs + 1e-6 {
            best = (lag, corr);
            best_abs = abs;
        } else if (abs - best_abs).abs() <= 1e-6 {
            if lag.abs() < best.0.abs() {
                best = (lag, corr);
                best_abs = abs;
            }
        }
    }

    let (lag, corr) = best;
    let mut frac = 0.0f32;
    if best_abs > 0.0 {
        if let Some(prev) = scores.iter().find(|(l, _)| *l == lag - 1) {
            if let Some(next) = scores.iter().find(|(l, _)| *l == lag + 1) {
                let denom = prev.1 - 2.0 * corr + next.1;
                if denom.abs() > 1e-6 {
                    frac = 0.5 * (prev.1 - next.1) / denom;
                    frac = frac.clamp(-0.5, 0.5);
                }
            }
        }
    }

    Some((best_abs, lag, frac))
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
        ZSource::Maxwell => 3,
        ZSource::Graph => 3,
        ZSource::Desire => 2,
        ZSource::GW => 2,
        ZSource::Other(_) => 1,
    }
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
    pub z: f32,
    pub drift: f32,
    pub support: f32,
    pub z_bias: f32,
    pub attributions: Vec<ZAttribution>,
    pub events: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
struct FrequencyFusionSnapshot {
    support: f32,
    drift: f32,
    z_bias: f32,
    z: f32,
}

impl FrequencyFusionSnapshot {
    fn from_fused(fused: &ZFused) -> Self {
        Self {
            support: fused.support,
            drift: fused.drift,
            z_bias: fused.z_bias,
            z: fused.z,
        }
    }
}

#[derive(Clone, Debug)]
struct FrequencyFusionState {
    cfg: ZFrequencyConfig,
    snapshot: Option<FrequencyFusionSnapshot>,
}

impl FrequencyFusionState {
    fn new(cfg: ZFrequencyConfig) -> Self {
        Self {
            cfg,
            snapshot: None,
        }
    }

    fn update_config(&mut self, cfg: ZFrequencyConfig) {
        self.cfg = cfg;
    }

    fn fuse(&mut self, fused: &mut ZFused) {
        if fused.support < self.cfg.minimum_energy {
            self.snapshot = None;
            return;
        }

        if let Some(prev) = self.snapshot {
            let alpha = self.cfg.smoothing.clamp(0.0, 1.0);
            if alpha > 0.0 {
                fused.support = lerp(prev.support, fused.support, alpha);
                fused.drift = lerp(prev.drift, fused.drift, alpha);
                fused.z_bias = lerp(prev.z_bias, fused.z_bias, alpha);
                fused.z = lerp(prev.z, fused.z, alpha);
                fused.events.push("freq.smoothed".to_string());
            }
        }

        self.snapshot = Some(FrequencyFusionSnapshot::from_fused(fused));
    }
}

#[derive(Clone, Debug)]
struct AdaptiveGainState {
    cfg: ZAdaptiveGainCfg,
    gain: f32,
}

impl AdaptiveGainState {
    fn new(cfg: ZAdaptiveGainCfg) -> Self {
        let baseline = cfg.gain_floor.max(1e-6);
        Self {
            cfg,
            gain: baseline,
        }
    }

    fn update_config(&mut self, cfg: ZAdaptiveGainCfg) {
        self.cfg = cfg;
        let floor = self.cfg.gain_floor.max(1e-6);
        let ceil = self.cfg.gain_ceil.max(floor);
        self.gain = self.gain.clamp(floor, ceil);
    }

    fn adapt(&mut self, fused: &mut ZFused) {
        let response = fused.support.abs() + fused.drift.abs() + fused.z_bias.abs();
        let target =
            (self.cfg.gain_floor + response).min(self.cfg.gain_ceil.max(self.cfg.gain_floor));
        let alpha = self.cfg.responsiveness.clamp(0.0, 1.0);
        let previous = self.gain;
        self.gain = lerp(self.gain, target.max(self.cfg.gain_floor.max(1e-6)), alpha);

        fused.z *= self.gain;
        fused.z_bias *= self.gain;
        if (self.gain - previous).abs() > 1e-6 {
            fused.events.push(format!("adaptive.gain:{:.3}", self.gain));
        }
    }
}

/// Stateful conductor that fuses heterogeneous Z pulses into a stabilised control
/// signal while applying anti-windup, hysteresis and slew protections.
#[derive(Clone, Debug)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    freq: Option<FrequencyFusionState>,
    adaptive: Option<AdaptiveGainState>,
    latency: Option<LatencyAlignerState>,
    pending: Vec<ZPulse>,
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
        let latency_cfg = cfg.latency.clone();
        ZConductor {
            cfg,
            freq: None,
            adaptive: None,
            latency: latency_cfg.map(LatencyAlignerState::new),
            pending: Vec::new(),
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

    /// Returns a mutable reference to the configuration, enabling on-line tuning.
    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }

    /// Installs or removes the latency aligner at runtime.
    pub fn set_latency_aligner(&mut self, cfg: Option<LatencyAlignerCfg>) {
        self.cfg.latency = cfg.clone();
        self.latency = cfg.map(LatencyAlignerState::new);
    }

    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
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

    /// Returns the currently estimated lag for the supplied source in milliseconds.
    pub fn latency_for(&self, source: &ZSource) -> Option<f32> {
        self.latency
            .as_ref()
            .and_then(|state| state.lag_for(source))
    }

    /// Enqueues a pulse to be considered during the next [`step`](Self::step).
    pub fn ingest(&mut self, mut pulse: ZPulse) {
        if !pulse.quality.is_finite() || pulse.quality <= 0.0 {
            pulse.quality = derive_quality(&pulse);
        } else {
            pulse.quality = pulse.quality.clamp(0.0, 1.0);
        }
        if let Some(latency) = self.latency.as_mut() {
            latency.record(&pulse);
        }
    }

    /// Executes one fusion step at the provided timestamp.
    pub fn step(&mut self, now: u64) -> ZFused {
        let mut events = Vec::new();
        if let Some(latency) = self.latency.as_mut() {
            latency.prepare(now, &mut events);
        }
        let mut ready = Vec::new();
        let mut retained = Vec::with_capacity(self.pending.len());
        for mut pulse in self.pending.drain(..) {
            if let Some(latency) = self.latency.as_ref() {
                latency.apply(&mut pulse);
            }
            if pulse.ts <= now {
                ready.push(pulse);
            } else {
                retained.push(pulse);
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
                .map(|pulse| pulse.support.max(0.0))
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

        let budget = self.cfg.z_budget.max(f32::EPSILON);
        if z.abs() > budget {
            let clamped = z.signum() * budget;
            if self.cfg.back_calculation > 0.0 {
                let correction = self.cfg.back_calculation * (clamped - z_before_limits);
                self.mag_hat = (self.mag_hat + correction).max(0.0);
            }
            z = clamped;
            events.push("saturated".to_string());
        }

        let attributions: Vec<ZAttribution> = attributions
            .into_iter()
            .map(|(source, weight)| ZAttribution { source, weight })
            .collect();

        let mut fused = ZFused {
            z,
            drift,
            support: total_support,
            z_bias: z_before_limits,
            attributions,
            events,
        };

        if let Some(state) = self.freq.as_mut() {
            state.fuse(&mut fused);
        }
        if let Some(state) = self.adaptive.as_mut() {
            state.adapt(&mut fused);
        }

        self.last_z = fused.z;
        self.last_step_ts = Some(now);

        fused
    }

    fn apply_temporal_filters(&mut self, drift: f32, events: &mut Vec<String>) -> f32 {
        let sign = if drift.abs() > f32::EPSILON {
            drift.signum()
        } else {
            self.last_sign
        };

        let mut target_sign = self.last_sign;
        if sign != 0.0 {
            if self.last_sign == 0.0 {
                target_sign = sign;
                self.flip_age = 0;
                events.push("sign-init".to_string());
            } else if (sign - self.last_sign).abs() > f32::EPSILON {
                if self.flip_age <= self.cfg.flip_hold {
                    events.push("flip-held".to_string());
                } else {
                    target_sign = sign;
                    self.flip_age = 0;
                    events.push("sign-flip".to_string());
                }
            }
        }

        self.flip_age = self.flip_age.saturating_add(1);
        self.last_sign = target_sign;

        let alpha_fast = self.cfg.alpha_fast.clamp(0.0, 1.0);
        if alpha_fast > 0.0 {
            self.sign_hat = ema(self.sign_hat, target_sign, alpha_fast);
        }

        let magnitude_target = drift.abs();
        let alpha_slow = self.cfg.alpha_slow.clamp(0.0, 1.0);
        if alpha_slow > 0.0 {
            self.mag_hat = ema(self.mag_hat, magnitude_target, alpha_slow);
        } else {
            self.mag_hat = magnitude_target;
        }

        if self.sign_hat.abs() <= f32::EPSILON {
            0.0
        } else {
            self.sign_hat.signum()
        }
    }
}

impl Default for ZConductor {
    fn default() -> Self {
        ZConductor::new(ZConductorCfg::default())
    }
}

impl Default for ZConductor {
    fn default() -> Self {
        ZConductor::new(ZConductorCfg::default())
    }
}

impl Default for ZConductor {
    fn default() -> Self {
        ZConductor::new(ZConductorCfg::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pulse(source: ZSource, ts: u64, drift: f32, quality: f32) -> ZPulse {
        ZPulse {
            source,
            ts,
            band_energy: (drift.abs(), 0.0, 0.0),
            drift,
            z_bias: drift,
            support: 1.0,
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
            conductor.ingest(ZPulse {
                source: ZSource::Microlocal,
                ts: idx as u64,
                band_energy: (1.0 + sign.max(0.0), 0.0, 1.0 + (-sign).max(0.0)),
                drift: sign,
                z_bias: sign,
                support: 1.0,
                quality: 1.0,
                stderr: 0.0,
                latency_ms: 0.0,
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
#[derive(Clone, Default, Debug)]
pub struct DesireEmitter {
    queue: Arc<Mutex<VecDeque<ZPulse>>>,
}

    #[test]
    fn latency_aligner_tracks_known_offset() {
        let align_cfg = LatencyAlignerCfg {
            window: 256,
            hop: 1,
            max_lag_steps: 80,
            alpha: 0.25,
            coherence_min: 0.2,
            hold_steps: 0,
            fractional: true,
        };
        let mut conductor =
            ZConductor::new(ZConductorCfg::default().with_latency_aligner(align_cfg));
        let lag = 6u64;
        for step in 0..80u64 {
            let ts_anchor = step;
            let ts_target = ts_anchor + lag;
            let drift = (step as f32 * 0.35).sin();
            conductor.ingest(pulse(ZSource::Microlocal, ts_anchor, drift, 1.0));
            conductor.ingest(pulse(ZSource::Maxwell, ts_target, drift, 1.0));
            conductor.step(ts_anchor);
        }
        let estimate = conductor.latency_for(&ZSource::Maxwell).unwrap();
        assert!(
            (estimate - lag as f32).abs() <= 1.0,
            "estimate={} lag={}",
            estimate,
            lag
        );
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
        assert!(saw_low, "expected coherence guard to trigger");
        assert!(
            !saw_adjust,
            "alignment should be suppressed under low coherence"
        );
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
        let fused = conductor.step(0);
        let estimate = conductor.latency_for(&ZSource::Graph).unwrap();
        assert!((estimate - 12.5).abs() <= 1e-3);
        assert!(fused.events.iter().any(|e| e.starts_with("latency.seeded")));
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
            support: 0.4,
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
            support: 0.4,
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
