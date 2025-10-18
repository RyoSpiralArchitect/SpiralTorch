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

use std::collections::VecDeque;

/// Canonical pulse exchanged across the Z-space control stack.
#[derive(Clone, Debug)]
pub struct ZPulse {
    /// Which layer originated the pulse.
    pub source: PulseSource,
    /// Scalar support describing how much mass or evidence backs the pulse.
    pub support: f32,
    /// Above/Here/Beneath energy split.
    pub band_energy: (f32, f32, f32),
    /// Signed drift between Above and Beneath energy.
    pub drift: f32,
    /// Signed Z bias produced after enrichment.
    pub z_bias: f32,
    /// Estimated latency between generation and observation in milliseconds.
    pub latency_ms: f32,
    /// Characteristic spatial or physical scale of the pulse.
    pub scale: f32,
}

impl ZPulse {
    /// Returns the total band energy.
    pub fn total_energy(&self) -> f32 {
        let (above, here, beneath) = self.band_energy;
        above + here + beneath
    }

    /// Returns `true` when the pulse carries no actionable signal.
    pub fn is_empty(&self) -> bool {
        self.support <= f32::EPSILON && self.total_energy() <= f32::EPSILON
    }
}

impl Default for ZPulse {
    fn default() -> Self {
        ZPulse {
            source: PulseSource::Microlocal,
            support: 0.0,
            band_energy: (0.0, 0.0, 0.0),
            drift: 0.0,
            z_bias: 0.0,
            latency_ms: 0.0,
            scale: 0.0,
        }
    }
}

/// Origin of a [`ZPulse`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PulseSource {
    Microlocal,
    Maxwell,
    Desire,
    GW,
}

/// Stateful conductor that fuses heterogeneous Z pulses into a stabilised control
/// signal while applying anti-windup and slew protections.
#[derive(Clone, Debug)]
pub struct ZConductor {
    alpha: f32,
    anti_windup: f32,
    slew_rate: f32,
    history_len: usize,
    sign_history: VecDeque<i8>,
    last: Option<ZPulse>,
}

impl Default for ZConductor {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl ZConductor {
    /// Creates a new conductor using the supplied smoothing factor.
    pub fn new(alpha: f32) -> Self {
        ZConductor {
            alpha: alpha.clamp(0.0, 1.0),
            anti_windup: 1.0,
            slew_rate: 1.0,
            history_len: 5,
            sign_history: VecDeque::with_capacity(5),
            last: None,
        }
    }

    /// Configures the exponential smoothing factor.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Configures the anti-windup magnitude limit.
    pub fn with_anti_windup(mut self, limit: f32) -> Self {
        let magnitude = limit.abs();
        self.anti_windup = if magnitude <= f32::EPSILON {
            1.0
        } else {
            magnitude
        };
        if self.slew_rate <= f32::EPSILON {
            self.slew_rate = self.anti_windup;
        }
        self
    }

    /// Configures the slew rate limiter applied to Z bias updates.
    pub fn with_slew_rate(mut self, rate: f32) -> Self {
        self.slew_rate = rate.abs();
        self
    }

    /// Configures the history length used for sign stability voting.
    pub fn with_history_len(mut self, len: usize) -> Self {
        self.history_len = len.max(1);
        self.sign_history = VecDeque::with_capacity(self.history_len);
        self
    }

    /// Updates the smoothing factor in place.
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }

    /// Returns the last fused pulse, if available.
    pub fn last(&self) -> Option<&ZPulse> {
        self.last.as_ref()
    }

    /// Fuses the supplied pulses into a single stabilised pulse.
    pub fn fuse(&mut self, pulses: &[ZPulse]) -> ZPulse {
        if pulses.is_empty() {
            return self.last.clone().unwrap_or_default();
        }

        let mut total_weight = 0.0f32;
        let mut support_total = 0.0f32;
        let mut band_energy = (0.0f32, 0.0f32, 0.0f32);
        let mut drift = 0.0f32;
        let mut z_bias = 0.0f32;
        let mut latency = 0.0f32;
        let mut scale = 0.0f32;
        let mut best_weight = f32::NEG_INFINITY;
        let mut best_pulse: Option<ZPulse> = None;

        for pulse in pulses {
            let weight = self.weight_for(pulse);
            if weight <= f32::EPSILON {
                continue;
            }
            support_total += pulse.support;
            band_energy.0 += pulse.band_energy.0 * weight;
            band_energy.1 += pulse.band_energy.1 * weight;
            band_energy.2 += pulse.band_energy.2 * weight;
            drift += pulse.drift * weight;
            z_bias += pulse.z_bias * weight;
            latency += pulse.latency_ms * weight;
            scale += pulse.scale * weight;
            total_weight += weight;
            if weight > best_weight {
                best_weight = weight;
                best_pulse = Some(pulse.clone());
            }
        }

        let mut fused = if total_weight > f32::EPSILON {
            let inv = 1.0 / total_weight;
            ZPulse {
                source: best_pulse
                    .as_ref()
                    .map(|p| p.source)
                    .unwrap_or(PulseSource::Microlocal),
                support: support_total,
                band_energy: (
                    band_energy.0 * inv,
                    band_energy.1 * inv,
                    band_energy.2 * inv,
                ),
                drift: drift * inv,
                z_bias: z_bias * inv,
                latency_ms: latency * inv,
                scale: scale * inv,
            }
        } else if let Some(best) = best_pulse {
            best
        } else {
            pulses[0].clone()
        };
        let proposed_sign = fused.z_bias.signum();
        let proposed_magnitude = fused.z_bias.abs();

        let alpha = self.alpha;
        if let Some(prev) = &self.last {
            fused.support = prev.support + alpha * (fused.support - prev.support);
            fused.band_energy.0 =
                prev.band_energy.0 + alpha * (fused.band_energy.0 - prev.band_energy.0);
            fused.band_energy.1 =
                prev.band_energy.1 + alpha * (fused.band_energy.1 - prev.band_energy.1);
            fused.band_energy.2 =
                prev.band_energy.2 + alpha * (fused.band_energy.2 - prev.band_energy.2);
            fused.drift = prev.drift + alpha * (fused.drift - prev.drift);
            fused.z_bias = prev.z_bias + alpha * (fused.z_bias - prev.z_bias);
            fused.latency_ms = prev.latency_ms + alpha * (fused.latency_ms - prev.latency_ms);
            fused.scale = prev.scale + alpha * (fused.scale - prev.scale);
        }

        let limit = self.anti_windup.max(f32::EPSILON);
        let scaled = (fused.z_bias / limit).clamp(-10.0, 10.0);
        fused.z_bias = scaled.tanh() * limit;

        if let Some(prev) = &self.last {
            if self.slew_rate > f32::EPSILON {
                let delta = fused.z_bias - prev.z_bias;
                let bounded = delta.clamp(-self.slew_rate, self.slew_rate);
                fused.z_bias = prev.z_bias + bounded;
            }
        }

        let sign = fused.z_bias.signum();
        let candidate_sign = if proposed_sign != 0.0 {
            proposed_sign
        } else {
            sign
        };
        if candidate_sign != 0.0 {
            if self.history_len > 0 {
                let majority: i32 = self.sign_history.iter().copied().map(i32::from).sum();
                let mut final_sign = candidate_sign;
                if majority != 0 && candidate_sign != majority.signum() as f32 {
                    let allow_flip = self
                        .last
                        .as_ref()
                        .map(|prev| proposed_magnitude + f32::EPSILON >= prev.z_bias.abs() * 0.25)
                        .unwrap_or(true);
                    if !allow_flip {
                        let enforced = majority.signum() as f32;
                        fused.z_bias = fused.z_bias.abs() * enforced;
                        fused.drift = fused.drift.abs() * enforced;
                        final_sign = enforced;
                    }
                }
                if self.sign_history.len() == self.history_len {
                    self.sign_history.pop_front();
                }
                self.sign_history.push_back(final_sign as i8);
            }
        } else {
            self.sign_history.clear();
        }

        self.last = Some(fused.clone());
        fused
    }

    fn weight_for(&self, pulse: &ZPulse) -> f32 {
        let stability = 1.0 / (1.0 + pulse.latency_ms.abs());
        let energy = pulse.total_energy().max(f32::EPSILON);
        let agreement = self
            .last
            .as_ref()
            .map(|prev| {
                if prev.z_bias.abs() <= f32::EPSILON {
                    1.0
                } else if pulse.z_bias.signum() == prev.z_bias.signum() {
                    1.0
                } else {
                    0.25
                }
            })
            .unwrap_or(1.0);
        let drift_gain = 1.0 + pulse.drift.abs();
        let base = (pulse.support + energy).max(f32::EPSILON);
        base * stability * agreement * drift_gain
    }
}
