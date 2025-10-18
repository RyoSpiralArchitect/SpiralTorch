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

// Microlocal pulse fusion utilities shared between the observability
// backends and telemetry exporters.
//! Canonical representation of Z pulses together with a lightweight
//! conductor that fuses multiple sources into a single control signal.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fs;
use std::io::{ErrorKind, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::Deserialize;

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
    Graph,
    Desire,
    RealGrad,
    GW,
    Other(&'static str),
}

impl Default for ZSource {
    fn default() -> Self {
        Self::Microlocal
    }
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
        self.support_mass() <= f32::EPSILON && self.total_energy() <= f32::EPSILON
    }
}

impl Default for ZPulse {
    fn default() -> Self {
        Self {
            source: ZSource::Microlocal,
            ts: 0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_bias: 0.0,
            support: ZSupport::default(),
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

impl Default for ZConductorCfg {
    fn default() -> Self {
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

    pub fn latest(&self) -> ZFused {
        self.fused.clone()
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
