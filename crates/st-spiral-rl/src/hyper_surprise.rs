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
    /// Minimum ratio margin that must be exceeded before a pulse is emitted.
    deadband: f32,
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
            deadband: 0.0,
            geometry_eta: 0.0,
            geometry_curvature: -1.0,
        }
    }

    /// Returns the current exponential moving standard deviation estimate.
    pub fn ema(&self) -> f32 {
        self.ema
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

    /// Sets a deadband that suppresses small shocks before emitting pulses.
    ///
    /// Ratios below the configured deadband are ignored so the controller only
    /// reacts once the observed deviation clears the margin. This keeps minor
    /// fluctuations from chattering the gauge.
    pub fn with_deadband(mut self, deadband: f32) -> Self {
        if deadband.is_finite() && deadband >= 0.0 {
            self.deadband = deadband;
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
        if ratio <= self.deadband {
            return None;
        }
        ratio = (ratio - self.deadband).max(0.0);
        Some(ratio.clamp(0.0, self.max_ratio))
    }
}

/// Configuration for the HyperSurprise controller.
#[derive(Clone, Copy, Debug)]
pub struct HyperSurpriseConfig {
    pub eta_bar: f32,
    pub curvature: f32,
    pub max_gauge: f32,
    pub min_gauge: f32,
    pub eta_floor: f32,
    pub lr_floor: f32,
    pub smoothing: f32,
    pub reversion: f32,
    pub cooldown_steps: usize,
    pub ratio_smoothing: f32,
}

impl HyperSurpriseConfig {
    pub fn with_eta_bar(mut self, eta_bar: f32) -> Self {
        if eta_bar.is_finite() && eta_bar > 0.0 {
            self.eta_bar = eta_bar;
        }
        self
    }

    pub fn with_curvature(mut self, curvature: f32) -> Self {
        if curvature.is_finite() {
            self.curvature = curvature;
        }
        self
    }

    pub fn with_max_gauge(mut self, max_gauge: f32) -> Self {
        if max_gauge.is_finite() && max_gauge > 0.0 {
            self.max_gauge = max_gauge;
        }
        self
    }

    pub fn with_min_gauge(mut self, min_gauge: f32) -> Self {
        if min_gauge.is_finite() && min_gauge > 0.0 {
            self.min_gauge = min_gauge;
        }
        self
    }

    pub fn with_eta_floor(mut self, eta_floor: f32) -> Self {
        if eta_floor.is_finite() && eta_floor >= 0.0 {
            self.eta_floor = eta_floor;
        }
        self
    }

    pub fn with_lr_floor(mut self, lr_floor: f32) -> Self {
        if lr_floor.is_finite() && lr_floor > 0.0 {
            self.lr_floor = lr_floor;
        }
        self
    }

    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        if smoothing.is_finite() {
            self.smoothing = smoothing.clamp(0.0, 1.0);
        }
        self
    }

    pub fn with_reversion(mut self, reversion: f32) -> Self {
        if reversion.is_finite() {
            self.reversion = reversion.clamp(0.0, 1.0);
        }
        self
    }

    pub fn with_cooldown_steps(mut self, steps: usize) -> Self {
        self.cooldown_steps = steps;
        self
    }

    pub fn with_ratio_smoothing(mut self, smoothing: f32) -> Self {
        if smoothing.is_finite() {
            self.ratio_smoothing = smoothing.clamp(0.0, 1.0);
        }
        self
    }
}

impl Default for HyperSurpriseConfig {
    fn default() -> Self {
        Self {
            eta_bar: 0.5,
            curvature: -1.0,
            max_gauge: 4.0,
            min_gauge: 1.0,
            eta_floor: 0.0,
            lr_floor: 1e-6,
            smoothing: 0.0,
            reversion: 0.35,
            cooldown_steps: 0,
            ratio_smoothing: 0.0,
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
    pub gauge: f32,
    pub learning_rate: f32,
    pub rolling_std: f32,
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
    last_gauge: f32,
    cooldown: usize,
    smoothed_ratio: f32,
}

impl HyperSurpriseController {
    pub fn new(trigger: LossStdTrigger, config: HyperSurpriseConfig) -> Self {
        Self {
            trigger,
            config,
            last_signal: None,
            last_gauge: 1.0,
            cooldown: 0,
            smoothed_ratio: 0.0,
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
        let candidate = self.trigger.observe(std);
        if let Some(ratio) = self.pop_ratio(candidate) {
            let ratio = ratio.max(0.0);
            let curvature_gain = self.config.curvature.tanh().abs().max(1e-3);
            let raw_gauge = 1.0 + ratio * curvature_gain;
            let gauge_floor = self.config.min_gauge.max(1e-3);
            let gauge_ceiling = self
                .config
                .max_gauge
                .max(self.config.min_gauge)
                .max(gauge_floor);
            let bounded_gauge = raw_gauge.clamp(gauge_floor, gauge_ceiling);
            let smoothing = self.config.smoothing.clamp(0.0, 1.0);
            let gauge = if smoothing > 0.0 {
                let relaxed = smoothing * self.last_gauge + (1.0 - smoothing) * bounded_gauge;
                self.last_gauge = relaxed.clamp(gauge_floor, gauge_ceiling);
                self.last_gauge
            } else {
                self.last_gauge = bounded_gauge;
                bounded_gauge
            };
            let effective_ratio = ((gauge - 1.0) / curvature_gain).max(0.0);
            let scaled_lr = match learning_rate.partial_cmp(&0.0) {
                Some(Ordering::Greater) => learning_rate * (1.0 + effective_ratio),
                _ => learning_rate,
            }
            .max(self.config.lr_floor.max(1e-9));
            let eta_bar =
                (self.config.eta_bar * (1.0 + effective_ratio)).max(self.config.eta_floor);
            let signal = HyperSurpriseSignal {
                loss_std: std,
                inject_ratio: effective_ratio,
                eta_bar,
                curvature: self.config.curvature,
                gauge,
                learning_rate: scaled_lr,
                rolling_std: self.trigger.ema(),
            };
            self.last_signal = Some(signal.clone());
            HyperSurpriseOutcome {
                learning_rate: scaled_lr.max(1e-6),
                gauge,
                signal: Some(signal),
            }
        } else {
            let reversion = self.config.reversion.clamp(0.0, 1.0);
            let baseline = self.config.min_gauge.max(1e-3);
            let ceiling = self.config.max_gauge.max(baseline);
            let target = baseline + (self.last_gauge - baseline) * (1.0 - reversion);
            let smoothing = self.config.smoothing.clamp(0.0, 1.0);
            let relaxed = if smoothing > 0.0 {
                smoothing * self.last_gauge + (1.0 - smoothing) * target
            } else {
                target
            };
            self.last_gauge = relaxed.clamp(baseline, ceiling);
            let gauge = self.last_gauge;
            self.last_signal = None;
            HyperSurpriseOutcome {
                learning_rate: learning_rate.max(self.config.lr_floor.max(1e-9)),
                gauge,
                signal: None,
            }
        }
    }

    fn pop_ratio(&mut self, candidate: Option<f32>) -> Option<f32> {
        let accepted = if let Some(ratio) = candidate {
            if self.cooldown == 0 {
                self.cooldown = self.config.cooldown_steps.saturating_add(1);
                ratio.max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        if self.cooldown > 0 {
            self.cooldown = self.cooldown.saturating_sub(1);
        }

        let smoothing = self.config.ratio_smoothing.clamp(0.0, 1.0);
        if smoothing > 0.0 {
            self.smoothed_ratio = smoothing * self.smoothed_ratio + (1.0 - smoothing) * accepted;
        } else {
            self.smoothed_ratio = accepted;
        }

        if self.smoothed_ratio > 1e-6 {
            Some(self.smoothed_ratio)
        } else {
            self.smoothed_ratio = 0.0;
            None
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

    #[test]
    fn deadband_suppresses_small_shocks() {
        let mut trigger = LossStdTrigger::new(0.1).with_warmup(1).with_deadband(0.5);
        // Warmup observation initializes the EMA without emitting a signal.
        assert!(trigger.observe(0.08).is_none());
        // Ratio is below the configured deadband so no pulse should be emitted.
        assert!(trigger.observe(0.14).is_none());
        // Once the ratio clears the deadband margin a pulse is emitted.
        assert!(trigger.observe(0.5).is_some());
    }
}
