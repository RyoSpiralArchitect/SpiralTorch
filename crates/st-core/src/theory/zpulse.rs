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
}

impl Default for ZConductor {
    fn default() -> Self {
        Self::new(ZConductorCfg::default())
    }
}

impl ZConductor {
    /// Creates a new conductor with the supplied configuration.
    pub fn new(cfg: ZConductorCfg) -> Self {
        ZConductor {
            cfg,
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

    /// Enqueues a pulse to be considered during the next [`step`](Self::step).
    pub fn ingest(&mut self, mut pulse: ZPulse) {
        if !pulse.quality.is_finite() || pulse.quality <= 0.0 {
            pulse.quality = derive_quality(&pulse);
        } else {
            pulse.quality = pulse.quality.clamp(0.0, 1.0);
        }
        self.pending.push(pulse);
    }

    /// Executes one fusion step at the provided timestamp.
    pub fn step(&mut self, now: u64) -> ZFused {
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
                if self.flip_age < self.cfg.flip_hold {
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
            if step > 0 {
                assert!((fused.z - 0.1 * step as f32).abs() < 1e-4);
            }
        }
    }
}
