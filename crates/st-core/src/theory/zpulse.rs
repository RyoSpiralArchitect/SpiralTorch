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
pub struct ZPulse {
    /// Which layer originated the pulse.
    pub source: ZSource,
    /// Timestamp of the pulse in the caller's clock domain.
    pub ts: u64,
    /// Above/Here/Beneath energy split.
    pub band_energy: (f32, f32, f32),
    /// Signed drift between Above and Beneath energy prior to normalisation.
    pub drift: f32,
    /// Signed Z bias produced after enrichment.
    pub z_bias: f32,
    /// Scalar support describing how much mass or evidence backs the pulse.
    pub support: f32,
    /// Optional quality score provided by the emitter. When `0` the conductor
    /// derives a surrogate based on the pulse statistics.
    pub quality: f32,
    /// Optional estimated standard error backing the pulse. `0` means unknown.
    pub stderr: f32,
    /// Estimated latency between generation and observation in the caller's time units.
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

/// Origin of a [`ZPulse`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ZSource {
    Microlocal,
    Maxwell,
    Desire,
    Graph,
    GW,
    Other(&'static str),
}

/// Configuration governing the behaviour of [`ZConductor`].
#[derive(Clone, Debug)]
pub struct ZConductorCfg {
    /// Per-source gains applied on top of the quality weighting.
    pub gain: FxHashMap<ZSource, f32>,
    /// Fast EMA used to track the fused sign.
    pub alpha_fast: f32,
    /// Slow EMA used to track the fused magnitude.
    pub alpha_slow: f32,
    /// Maximum delta allowed between subsequent fused Z outputs.
    pub slew_max: f32,
    /// Number of steps a sign flip is held before allowing the reversal.
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
    /// Returns a latency-aligner profile suitable for most interactive workloads.
    pub fn balanced() -> Self {
        Self::default()
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
        let history = self
            .histories
            .entry(pulse.source.clone())
            .or_insert_with(SourceHistory::default);
        if let Some(last) = history.last_ts {
            if pulse.ts < last {
                self.pending_events
                    .push(format!("latency.invalid_ts:{:?}", pulse.source));
                return;
            }
        }
        history.add_sample(pulse.ts, bin, pulse.drift, weight);
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
    pub attributions: Vec<ZAttribution>,
    pub events: Vec<String>,
}

/// Stateful conductor that fuses heterogeneous Z pulses into a stabilised control
/// signal while applying anti-windup, hysteresis and slew protections.
#[derive(Clone, Debug)]
pub struct ZConductor {
    cfg: ZConductorCfg,
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
    /// Creates a new conductor with the supplied configuration.
    pub fn new(cfg: ZConductorCfg) -> Self {
        let latency_cfg = cfg.latency.clone();
        ZConductor {
            cfg,
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

    /// Returns a mutable reference to the configuration, enabling on-line tuning.
    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }

    /// Installs or removes the latency aligner at runtime.
    pub fn set_latency_aligner(&mut self, cfg: Option<LatencyAlignerCfg>) {
        self.cfg.latency = cfg.clone();
        self.latency = cfg.map(LatencyAlignerState::new);
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
        self.pending.push(pulse);
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

        if !ready.is_empty() {
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

        self.last_z = z;
        self.last_step_ts = Some(now);

        let attributions = attributions
            .into_iter()
            .map(|(source, weight)| ZAttribution { source, weight })
            .collect();

        ZFused {
            z,
            drift,
            attributions,
            events,
        }
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

fn derive_quality(pulse: &ZPulse) -> f32 {
    match pulse.source {
        ZSource::Microlocal | ZSource::Graph => {
            let total = pulse.total_energy().max(1e-6);
            let support_norm = (pulse.support / total).clamp(0.0, 8.0);
            sigmoid(1.75 * support_norm)
        }
        ZSource::Maxwell | ZSource::GW => {
            let stderr = pulse.stderr.max(1e-6);
            let snr = (1.0 / stderr).min(1.0);
            let z = pulse.z_bias.abs().max(pulse.drift.abs());
            z.tanh() * snr
        }
        ZSource::Desire => {
            if pulse.quality > 0.0 {
                pulse.quality.clamp(0.0, 1.0)
            } else {
                0.5
            }
        }
        ZSource::Other(_) => {
            if pulse.quality > 0.0 {
                pulse.quality.clamp(0.0, 1.0)
            } else {
                0.5
            }
        }
    }
}

fn huber_weight(residual: f32, delta: f32) -> f32 {
    if delta <= 0.0 {
        return 1.0;
    }
    if residual.abs() <= delta {
        1.0
    } else {
        delta / residual.abs()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn ema(prev: f32, value: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(0.0, 1.0);
    (1.0 - alpha) * prev + alpha * value
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    let (_, median, _) =
        values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    *median
}

fn slew_limit(prev: f32, next: f32, slew: f32) -> f32 {
    if slew <= f32::EPSILON {
        return next;
    }
    let delta = next - prev;
    let clamped = delta.clamp(-slew, slew);
    prev + clamped
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
    fn budget_clamps_output_and_normalises_attributions() {
        let mut cfg = ZConductorCfg::default();
        cfg.z_budget = 0.5;
        cfg.slew_max = 10.0;
        cfg.alpha_fast = 1.0;
        cfg.alpha_slow = 1.0;
        let mut conductor = ZConductor::new(cfg);

        conductor.ingest(ZPulse {
            source: ZSource::Microlocal,
            ts: 0,
            band_energy: (10.0, 0.0, 0.0),
            drift: 10.0,
            z_bias: 10.0,
            support: 10.0,
            quality: 1.0,
            stderr: 0.0,
            latency_ms: 0.0,
        });
        let fused = conductor.step(0);
        assert!(fused.z.abs() <= 0.5 + 1e-6);
        assert!((fused.attributions.iter().map(|a| a.weight).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(fused.events.iter().any(|e| e == "saturated"));
    }

    #[test]
    fn slew_limit_bounds_delta_z() {
        let mut cfg = ZConductorCfg::default();
        cfg.slew_max = 0.1;
        cfg.alpha_fast = 1.0;
        cfg.alpha_slow = 1.0;
        let mut conductor = ZConductor::new(cfg);

        let mut prev_z = 0.0;
        for step in 0..3 {
            conductor.ingest(ZPulse {
                source: ZSource::Microlocal,
                ts: step,
                band_energy: (5.0, 0.0, 0.0),
                drift: 1.0,
                z_bias: 1.0,
                support: 5.0,
                quality: 1.0,
                stderr: 0.0,
                latency_ms: 0.0,
            });
            let fused = conductor.step(step);
            if step > 0 {
                assert!((fused.z - prev_z).abs() <= 0.1 + 1e-4);
            }
            prev_z = fused.z;
        }
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
}
