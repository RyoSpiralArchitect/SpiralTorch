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
use st_frac::fft::{fft_inplace, Complex32};
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
    /// Estimated latency between generation and observation in milliseconds.
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
    /// Optional frequency-domain fusion configuration.
    pub freq: Option<ZFrequencyConfig>,
    /// Optional adaptive gain tuning configuration.
    pub adaptive_gain: Option<ZAdaptiveGainCfg>,
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
            freq: None,
            adaptive_gain: None,
        }
    }
}

/// Configuration describing how frequency-domain fusion should be applied.
#[derive(Clone, Debug)]
pub struct ZFrequencyConfig {
    /// Sliding window length (power-of-two) used for the FFT. Values < 4 disable the frequency pass.
    pub window: usize,
    /// Minimum aggregate spectral power required before scaling weights.
    pub min_power: f32,
    /// Lower bound for the frequency multiplier to avoid starving a source entirely.
    pub floor: f32,
    /// Upper bound for the frequency multiplier to avoid runaway amplification.
    pub ceil: f32,
    /// Per-source spectral gains (one value per positive frequency bin).
    pub source_gains: FxHashMap<ZSource, Vec<f32>>,
}

impl Default for ZFrequencyConfig {
    fn default() -> Self {
        Self {
            window: 0,
            min_power: 1e-3,
            floor: 0.5,
            ceil: 2.5,
            source_gains: FxHashMap::default(),
        }
    }
}

/// Configuration controlling the adaptive gain tuning loop.
#[derive(Clone, Debug)]
pub struct ZAdaptiveGainCfg {
    /// Smoothing applied when updating the reliability score (0–1).
    pub alpha: f32,
    /// Lower bound for the adaptive gain multiplier.
    pub min_gain: f32,
    /// Upper bound for the adaptive gain multiplier.
    pub max_gain: f32,
    /// Step size used when nudging the adaptive gain towards the reliability target.
    pub learning_rate: f32,
    /// Desired reliability level (0–1) that keeps the gain steady.
    pub target: f32,
}

impl Default for ZAdaptiveGainCfg {
    fn default() -> Self {
        Self {
            alpha: 0.25,
            min_gain: 0.25,
            max_gain: 4.0,
            learning_rate: 0.25,
            target: 0.65,
        }
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
    pub events: Vec<&'static str>,
}

/// Stateful conductor that fuses heterogeneous Z pulses into a stabilised control
/// signal while applying anti-windup, hysteresis and slew protections.
#[derive(Clone, Debug)]
pub struct ZConductor {
    cfg: ZConductorCfg,
    pending: Vec<ZPulse>,
    sign_hat: f32,
    mag_hat: f32,
    last_sign: f32,
    flip_age: u32,
    last_z: f32,
    last_step_ts: Option<u64>,
    freq: Option<FrequencyFusionState>,
    adaptive: Option<AdaptiveGainState>,
}

impl Default for ZConductor {
    fn default() -> Self {
        Self::new(ZConductorCfg::default())
    }
}

impl ZConductor {
    /// Creates a new conductor with the supplied configuration.
    pub fn new(cfg: ZConductorCfg) -> Self {
        let freq = cfg
            .freq
            .clone()
            .and_then(|cfg| FrequencyFusionState::new(cfg).ok());
        let adaptive = cfg.adaptive_gain.clone().map(AdaptiveGainState::new);
        ZConductor {
            cfg,
            pending: Vec::new(),
            sign_hat: 0.0,
            mag_hat: 0.0,
            last_sign: 0.0,
            flip_age: u32::MAX,
            last_z: 0.0,
            last_step_ts: None,
            freq,
            adaptive,
        }
    }

    /// Returns a mutable reference to the configuration, enabling on-line tuning.
    pub fn cfg_mut(&mut self) -> &mut ZConductorCfg {
        &mut self.cfg
    }

    /// Replaces the frequency fusion configuration and rebuilds the spectral state.
    pub fn set_frequency_config(&mut self, cfg: Option<ZFrequencyConfig>) {
        self.cfg.freq = cfg;
        self.freq = self
            .cfg
            .freq
            .clone()
            .and_then(|cfg| FrequencyFusionState::new(cfg).ok());
    }

    /// Replaces the adaptive gain configuration and rebuilds the adaptive state.
    pub fn set_adaptive_gain_config(&mut self, cfg: Option<ZAdaptiveGainCfg>) {
        self.cfg.adaptive_gain = cfg;
        self.adaptive = self.cfg.adaptive_gain.clone().map(AdaptiveGainState::new);
    }

    /// Enqueues a pulse to be considered during the next [`step`](Self::step).
    pub fn ingest(&mut self, mut pulse: ZPulse) {
        self.reconcile_states();
        if !pulse.quality.is_finite() || pulse.quality <= 0.0 {
            pulse.quality = derive_quality(&pulse);
        } else {
            pulse.quality = pulse.quality.clamp(0.0, 1.0);
        }
        self.pending.push(pulse);
    }

    /// Executes one fusion step at the provided timestamp.
    pub fn step(&mut self, now: u64) -> ZFused {
        self.reconcile_states();
        let mut ready = Vec::new();
        let mut retained = Vec::with_capacity(self.pending.len());
        for pulse in self.pending.drain(..) {
            if pulse.ts <= now {
                ready.push(pulse);
            } else {
                retained.push(pulse);
            }
        }
        self.pending = retained;

        let mut events = Vec::new();
        let mut drift = 0.0;
        let mut attributions = Vec::new();

        if let Some(freq) = self.freq.as_mut() {
            for pulse in &ready {
                freq.observe(&pulse.source, pulse.normalised_drift());
            }
        }

        if !ready.is_empty() {
            let mut contributions: Vec<Contribution> = ready
                .iter()
                .filter(|p| !p.is_empty())
                .map(|pulse| {
                    let mut weight = (pulse.quality
                        * self.cfg.gain.get(&pulse.source).copied().unwrap_or(1.0))
                    .max(1e-6);
                    if let Some(adaptive) = self.adaptive.as_mut() {
                        weight *= adaptive.gain(&pulse.source);
                    }
                    Contribution {
                        source: pulse.source.clone(),
                        weight,
                        drift_norm: pulse.normalised_drift(),
                        quality: pulse.quality,
                    }
                })
                .collect();

            if let Some(freq) = self.freq.as_mut() {
                for contribution in &mut contributions {
                    contribution.weight *= freq.multiplier(&contribution.source);
                }
            }

            if !contributions.is_empty() {
                let mut drifts: Vec<f32> = contributions.iter().map(|c| c.drift_norm).collect();
                let median = median(&mut drifts);
                let mut weight_sum = 0.0f32;
                let mut numerator = 0.0f32;
                for contribution in contributions.iter_mut() {
                    let robust =
                        huber_weight(contribution.drift_norm - median, self.cfg.robust_delta);
                    contribution.weight *= robust;
                    weight_sum += contribution.weight;
                    numerator += contribution.weight * contribution.drift_norm;
                    attributions.push((contribution.source.clone(), contribution.weight));
                }
                if weight_sum > 0.0 {
                    drift = numerator / weight_sum;
                    let inv = 1.0 / weight_sum;
                    for attrib in &mut attributions {
                        attrib.1 *= inv;
                    }
                }

                if let Some(adaptive) = self.adaptive.as_mut() {
                    adaptive.update(&contributions, drift);
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
            events.push("slew-limited");
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
            events.push("saturated");
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

    /// Returns the current adaptive gain multiplier for a specific source, if adaptation is enabled.
    pub fn adaptive_gain(&self, source: &ZSource) -> Option<f32> {
        self.adaptive
            .as_ref()
            .and_then(|state| state.current_gain(source))
    }

    fn apply_temporal_filters(&mut self, drift: f32, events: &mut Vec<&'static str>) -> f32 {
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
                events.push("sign-init");
            } else if (sign - self.last_sign).abs() > f32::EPSILON {
                if self.flip_age <= self.cfg.flip_hold {
                    events.push("flip-held");
                } else {
                    target_sign = sign;
                    self.flip_age = 0;
                    events.push("sign-flip");
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

    fn reconcile_states(&mut self) {
        if let Some(freq_cfg) = self.cfg.freq.clone() {
            let window = freq_cfg.window;
            match self.freq.as_mut() {
                Some(state) if state.window() == window => state.update_config(freq_cfg),
                _ => {
                    self.freq = FrequencyFusionState::new(freq_cfg).ok();
                }
            }
        } else {
            self.freq = None;
        }

        if let Some(adaptive_cfg) = self.cfg.adaptive_gain.clone() {
            match self.adaptive.as_mut() {
                Some(state) => state.update_config(adaptive_cfg),
                None => {
                    self.adaptive = Some(AdaptiveGainState::new(adaptive_cfg));
                }
            }
        } else {
            self.adaptive = None;
        }
    }
}

struct Contribution {
    source: ZSource,
    weight: f32,
    drift_norm: f32,
    quality: f32,
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

#[derive(Clone, Debug)]
struct FrequencyFusionState {
    cfg: ZFrequencyConfig,
    history: FxHashMap<ZSource, VecDeque<f32>>,
    scratch: Vec<Complex32>,
}

impl FrequencyFusionState {
    fn new(cfg: ZFrequencyConfig) -> Result<Self, ()> {
        if cfg.window < 4 || !cfg.window.is_power_of_two() {
            return Err(());
        }
        let mut scratch = Vec::with_capacity(cfg.window);
        scratch.resize(cfg.window, Complex32::default());
        Ok(Self {
            cfg,
            history: FxHashMap::default(),
            scratch,
        })
    }

    fn window(&self) -> usize {
        self.cfg.window
    }

    fn update_config(&mut self, cfg: ZFrequencyConfig) {
        if cfg.window == self.cfg.window {
            self.cfg = cfg;
        }
    }

    fn observe(&mut self, source: &ZSource, drift: f32) {
        let entry = self
            .history
            .entry(source.clone())
            .or_insert_with(|| VecDeque::with_capacity(self.cfg.window));
        if entry.len() == self.cfg.window {
            entry.pop_front();
        }
        entry.push_back(drift);
    }

    fn multiplier(&mut self, source: &ZSource) -> f32 {
        let Some(history) = self.history.get(source) else {
            return 1.0;
        };
        if history.len() < self.cfg.window {
            return 1.0;
        }
        for (slot, sample) in self.scratch.iter_mut().zip(history.iter()) {
            *slot = Complex32::new(*sample, 0.0);
        }
        if fft_inplace(&mut self.scratch, false).is_err() {
            return 1.0;
        }
        let half = self.cfg.window / 2;
        let spectrum = self
            .cfg
            .source_gains
            .get(source)
            .filter(|profile| profile.len() >= half);
        let mut weighted = 0.0f32;
        let mut total = 0.0f32;
        for bin in 0..half {
            let value = self.scratch[bin];
            let mag = (value.re * value.re + value.im * value.im).sqrt();
            if mag <= f32::EPSILON {
                continue;
            }
            let gain = spectrum
                .and_then(|profile| profile.get(bin))
                .copied()
                .unwrap_or(1.0);
            weighted += mag * gain;
            total += mag;
        }
        if total < self.cfg.min_power {
            return 1.0;
        }
        (weighted / total).clamp(self.cfg.floor, self.cfg.ceil)
    }
}

#[derive(Clone, Debug)]
struct AdaptiveGainState {
    cfg: ZAdaptiveGainCfg,
    reliability: FxHashMap<ZSource, f32>,
    gains: FxHashMap<ZSource, f32>,
}

impl AdaptiveGainState {
    fn new(cfg: ZAdaptiveGainCfg) -> Self {
        Self {
            cfg,
            reliability: FxHashMap::default(),
            gains: FxHashMap::default(),
        }
    }

    fn update_config(&mut self, cfg: ZAdaptiveGainCfg) {
        self.cfg = cfg;
    }

    fn gain(&mut self, source: &ZSource) -> f32 {
        *self.gains.entry(source.clone()).or_insert(1.0)
    }

    fn update(&mut self, contributions: &[Contribution], fused: f32) {
        let alpha = self.cfg.alpha.clamp(0.0, 1.0);
        for contribution in contributions {
            let residual = (contribution.drift_norm - fused).abs().min(2.0);
            let quality = contribution.quality.clamp(0.0, 1.0);
            let alignment = (1.0 - 0.5 * residual).clamp(0.0, 1.0) * quality;
            let entry = self
                .reliability
                .entry(contribution.source.clone())
                .or_insert(0.5);
            *entry = (1.0 - alpha) * *entry + alpha * alignment;
            let gain = self.gains.entry(contribution.source.clone()).or_insert(1.0);
            let delta = (*entry - self.cfg.target).clamp(-1.0, 1.0);
            *gain = (*gain + self.cfg.learning_rate * delta)
                .clamp(self.cfg.min_gain, self.cfg.max_gain);
        }
    }

    fn current_gain(&self, source: &ZSource) -> Option<f32> {
        self.gains.get(source).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

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
                assert!(fused.events.contains(&"flip-held"));
            }
        }
    }

    fn pulse(source: ZSource, ts: u64, drift_norm: f32, quality: f32) -> ZPulse {
        ZPulse {
            source,
            ts,
            band_energy: (1.0 + drift_norm, 0.0, 1.0 - drift_norm),
            drift: drift_norm,
            z_bias: drift_norm,
            support: 1.0,
            quality,
            stderr: 0.0,
            latency_ms: 0.0,
        }
    }

    #[test]
    fn frequency_weighting_prefers_high_frequency_source() {
        let mut cfg = ZConductorCfg::default();
        cfg.alpha_fast = 1.0;
        cfg.alpha_slow = 1.0;
        cfg.flip_hold = 0;
        cfg.slew_max = 5.0;
        cfg.robust_delta = 10.0;
        cfg.z_budget = 5.0;
        cfg.back_calculation = 0.0;

        let mut conductor = ZConductor::new(cfg);
        let mut freq_cfg = ZFrequencyConfig::default();
        freq_cfg.window = 4;
        freq_cfg.floor = 0.5;
        freq_cfg.ceil = 4.0;
        let mut gains = FxHashMap::default();
        gains.insert(ZSource::Microlocal, vec![0.5, 3.0]);
        gains.insert(ZSource::Desire, vec![2.0, 0.5]);
        freq_cfg.source_gains = gains;
        conductor.set_frequency_config(Some(freq_cfg));

        for step in 0..6u64 {
            let mic_drift = if step % 2 == 0 { 0.9 } else { -0.9 };
            conductor.ingest(pulse(ZSource::Microlocal, step, mic_drift, 1.0));
            conductor.ingest(pulse(ZSource::Desire, step, 0.2, 1.0));
            let fused = conductor.step(step);
            if step >= 3 {
                let mic = fused
                    .attributions
                    .iter()
                    .find(|a| matches!(a.source, ZSource::Microlocal))
                    .map(|a| a.weight)
                    .unwrap_or(0.0);
                let desire = fused
                    .attributions
                    .iter()
                    .find(|a| matches!(a.source, ZSource::Desire))
                    .map(|a| a.weight)
                    .unwrap_or(0.0);
                assert!(
                    mic > desire,
                    "microlocal weight should dominate after FFT pass"
                );
            }
        }
    }

    #[test]
    fn adaptive_gain_tracks_consistent_alignment() {
        let mut cfg = ZConductorCfg::default();
        cfg.alpha_fast = 1.0;
        cfg.alpha_slow = 1.0;
        cfg.flip_hold = 0;
        cfg.slew_max = 5.0;
        cfg.robust_delta = 10.0;
        cfg.z_budget = 5.0;
        cfg.back_calculation = 0.0;
        cfg.adaptive_gain = Some(ZAdaptiveGainCfg {
            alpha: 0.4,
            min_gain: 0.2,
            max_gain: 3.0,
            learning_rate: 0.6,
            target: 0.6,
        });

        let mut conductor = ZConductor::new(cfg);
        for step in 0..24u64 {
            conductor.ingest(pulse(ZSource::Microlocal, step, 0.6, 1.0));
            conductor.ingest(pulse(ZSource::Desire, step, 0.0, 0.25));
            let _ = conductor.step(step);
        }

        let mic_gain = conductor
            .adaptive_gain(&ZSource::Microlocal)
            .unwrap_or_default();
        let desire_gain = conductor
            .adaptive_gain(&ZSource::Desire)
            .unwrap_or_default();
        assert!(
            mic_gain > desire_gain,
            "consistent source should acquire larger gain"
        );
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
        assert!(fused.events.contains(&"saturated"));
    }

    #[test]
    fn slew_limit_bounds_delta_z() {
        let mut cfg = ZConductorCfg::default();
        cfg.slew_max = 0.1;
        cfg.alpha_fast = 1.0;
        cfg.alpha_slow = 1.0;
        let mut conductor = ZConductor::new(cfg);

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
            let expected = 0.1 * (step as f32 + 1.0);
            assert!((fused.z - expected).abs() < 1e-4);
        }
    }
}
