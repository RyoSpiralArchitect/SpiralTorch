// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::cmp::Ordering;

/// Rolling trigger that activates whenever the observed loss standard deviation
/// exceeds the configured threshold. The trigger keeps an exponential moving
/// estimate so transient spikes do not cause unnecessary oscillations.
#[derive(Clone, Debug)]
pub struct LossStdTrigger {
    std_threshold: f32,
    max_ratio: f32,
    decay: f32,
    ema: f32,
    warmup: usize,
    seen: usize,
    geometry_eta: f32,
    geometry_curvature: f32,
}

impl LossStdTrigger {
    /// Creates a trigger with the provided standard deviation threshold.
    pub fn new(std_threshold: f32) -> Self {
        Self {
            std_threshold: std_threshold.max(1e-5),
            max_ratio: 3.0,
            decay: 0.8,
            ema: 0.0,
            warmup: 4,
            seen: 0,
            geometry_eta: 0.0,
            geometry_curvature: -1.0,
        }
    }

    /// Sets the exponential decay applied to the internal rolling estimate.
    pub fn with_decay(mut self, decay: f32) -> Self {
        if decay.is_finite() && (0.0..=1.0).contains(&decay) {
            self.decay = decay.max(1e-3);
        }
        self
    }

    /// Caps the injected surprise ratio.
    pub fn with_max_ratio(mut self, ratio: f32) -> Self {
        if ratio.is_finite() && ratio > 0.0 {
            self.max_ratio = ratio;
        }
        self
    }

    /// Number of observations collected before the trigger starts emitting
    /// surprise pulses.
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    pub fn with_geometry_injection(mut self, eta: f32, curvature: f32) -> Self {
        if eta.is_finite() && eta >= 0.0 {
            self.geometry_eta = eta;
        }
        if curvature.is_finite() {
            self.geometry_curvature = curvature;
        }
        self
    }

    /// Observes the latest loss standard deviation and returns a suggested
    /// injection ratio when the measurement exceeds the configured threshold.
    pub fn observe(&mut self, std: f32) -> Option<f32> {
        if !std.is_finite() {
            return None;
        }
        self.seen = self.seen.saturating_add(1);
        if self.seen == 1 {
            self.ema = std.max(0.0);
        } else {
            self.ema = self.decay * self.ema + (1.0 - self.decay) * std.max(0.0);
        }
        if self.seen <= self.warmup {
            return None;
        }
        if self.ema <= self.std_threshold {
            return None;
        }
        let mut ratio = (self.ema / self.std_threshold) - 1.0;
        if self.geometry_eta > 0.0 {
            let curvature_boost = self.geometry_curvature.tanh().abs() as f32;
            ratio *= 1.0 + self.geometry_eta * 0.5 + curvature_boost;
        }
        Some(ratio.clamp(0.0, self.max_ratio))
    }
}

/// Configuration for the HyperSurprise controller.
#[derive(Clone, Copy, Debug)]
pub struct HyperSurpriseConfig {
    pub eta_bar: f32,
    pub curvature: f32,
    pub max_gauge: f32,
}

impl Default for HyperSurpriseConfig {
    fn default() -> Self {
        Self {
            eta_bar: 0.5,
            curvature: -1.0,
            max_gauge: 4.0,
        }
    }
}

/// Signal emitted after fusing the geometry feedback with the loss surprise
/// trigger. Consumers can log the packet to correlate emergent behaviour with
/// η̄ modulation.
#[derive(Clone, Debug)]
pub struct HyperSurpriseSignal {
    pub loss_std: f32,
    pub inject_ratio: f32,
    pub eta_bar: f32,
    pub curvature: f32,
}

#[derive(Clone, Debug)]
pub struct HyperSurpriseOutcome {
    pub learning_rate: f32,
    pub gauge: f32,
    pub signal: Option<HyperSurpriseSignal>,
}

#[derive(Clone, Debug)]
pub struct HyperSurpriseController {
    trigger: LossStdTrigger,
    config: HyperSurpriseConfig,
    last_signal: Option<HyperSurpriseSignal>,
}

impl HyperSurpriseController {
    pub fn new(trigger: LossStdTrigger, config: HyperSurpriseConfig) -> Self {
        Self {
            trigger,
            config,
            last_signal: None,
        }
    }

    pub fn into_trigger(self) -> LossStdTrigger {
        self.trigger
    }

    pub fn last_signal(&self) -> Option<&HyperSurpriseSignal> {
        self.last_signal.as_ref()
    }

    pub fn update(
        &mut self,
        returns: &[f32],
        baseline: f32,
        learning_rate: f32,
    ) -> HyperSurpriseOutcome {
        let std = loss_std(returns, baseline);
        if let Some(ratio) = self.trigger.observe(std) {
            let clamped_ratio = ratio.clamp(0.0, self.config.max_gauge);
            let gauge = 1.0 + clamped_ratio * self.config.curvature.tanh().abs();
            let scaled_lr = match learning_rate.partial_cmp(&0.0) {
                Some(Ordering::Greater) => learning_rate * (1.0 + clamped_ratio),
                _ => learning_rate,
            };
            let eta_bar = self.config.eta_bar * (1.0 + clamped_ratio);
            let signal = HyperSurpriseSignal {
                loss_std: std,
                inject_ratio: clamped_ratio,
                eta_bar,
                curvature: self.config.curvature,
            };
            self.last_signal = Some(signal.clone());
            HyperSurpriseOutcome {
                learning_rate: scaled_lr.max(1e-6),
                gauge,
                signal: Some(signal),
            }
        } else {
            self.last_signal = None;
            HyperSurpriseOutcome {
                learning_rate,
                gauge: 1.0,
                signal: None,
            }
        }
    }
}

fn loss_std(values: &[f32], baseline: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut variance = 0.0f32;
    for &value in values {
        let delta = value - baseline;
        variance += delta * delta;
    }
    (variance / values.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_injection_scales_ratio() {
        let mut trigger = LossStdTrigger::new(0.1)
            .with_warmup(1)
            .with_geometry_injection(0.6, -1.5);
        assert!(trigger.observe(0.05).is_none());
        let boosted = trigger.observe(0.4).expect("boosted ratio");
        assert!(boosted > 0.0);
    }
}
