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
        let ratio = (self.ema / self.std_threshold) - 1.0;
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
    pub relaxation: f32,
    pub ratio_smoothing: f32,
    pub cooldown_steps: usize,
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

    pub fn with_relaxation(mut self, relaxation: f32) -> Self {
        if relaxation.is_finite() {
            self.relaxation = relaxation.clamp(0.0, 1.0);
        }
        self
    }

    pub fn with_ratio_smoothing(mut self, ratio_smoothing: f32) -> Self {
        if ratio_smoothing.is_finite() {
            self.ratio_smoothing = ratio_smoothing.clamp(0.0, 1.0);
        }
        self
    }

    pub fn with_cooldown_steps(mut self, cooldown_steps: usize) -> Self {
        self.cooldown_steps = cooldown_steps;
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
            relaxation: 0.5,
            ratio_smoothing: 0.0,
            cooldown_steps: 0,
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
    smoothed_ratio: f32,
    cooldown: usize,
}

impl HyperSurpriseController {
    pub fn new(trigger: LossStdTrigger, config: HyperSurpriseConfig) -> Self {
        Self {
            trigger,
            config,
            last_signal: None,
            last_gauge: 1.0,
            smoothed_ratio: 0.0,
            cooldown: 0,
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
        let candidate_ratio = self.trigger.observe(std);
        let ratio = self.pop_ratio(candidate_ratio);
        if let Some(ratio) = ratio {
            let clamped_ratio = ratio.clamp(0.0, self.config.max_gauge);
            let gauge_ceiling = self.config.max_gauge.max(self.config.min_gauge);
            let raw_gauge = 1.0 + clamped_ratio * self.config.curvature.tanh().abs();
            let bounded_gauge = raw_gauge.clamp(self.config.min_gauge, gauge_ceiling);
            let smoothing = self.config.smoothing.clamp(0.0, 1.0);
            let gauge = if smoothing > 0.0 {
                let relaxed = smoothing * self.last_gauge + (1.0 - smoothing) * bounded_gauge;
                self.last_gauge = relaxed;
                relaxed
            } else {
                self.last_gauge = bounded_gauge;
                bounded_gauge
            };
            let scaled_lr = match learning_rate.partial_cmp(&0.0) {
                Some(Ordering::Greater) => learning_rate * (1.0 + clamped_ratio),
                _ => learning_rate,
            }
            .max(self.config.lr_floor.max(1e-9));
            let eta_bar = (self.config.eta_bar * (1.0 + clamped_ratio)).max(self.config.eta_floor);
            let signal = HyperSurpriseSignal {
                loss_std: std,
                inject_ratio: clamped_ratio,
                eta_bar,
                curvature: self.config.curvature,
                gauge,
                learning_rate: scaled_lr,
            };
            self.last_signal = Some(signal.clone());
            HyperSurpriseOutcome {
                learning_rate: scaled_lr.max(1e-6),
                gauge,
                signal: Some(signal),
            }
        } else {
            let relaxation = self.config.relaxation.clamp(0.0, 1.0);
            let gauge = if relaxation > 0.0 {
                let relaxed = relaxation * self.last_gauge + (1.0 - relaxation);
                self.last_gauge = relaxed;
                relaxed
            } else {
                self.last_gauge = 1.0;
                1.0
            };
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
            self.smoothed_ratio =
                smoothing * self.smoothed_ratio + (1.0 - smoothing) * accepted;
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

    fn constant_returns(std: f32) -> Vec<f32> {
        vec![std, -std, std, -std]
    }

    #[test]
    fn smoothing_relaxes_gauge_towards_baseline() {
        let trigger = LossStdTrigger::new(0.4).with_warmup(0);
        let config = HyperSurpriseConfig::default()
            .with_max_gauge(2.0)
            .with_smoothing(0.5);
        let mut controller = HyperSurpriseController::new(trigger, config);

        let outcome = controller.update(&constant_returns(0.5), 0.0, 0.1);
        assert!(outcome.gauge > 1.0);

        // Drop below the threshold and confirm the gauge relaxes instead of snapping.
        let relaxed = controller.update(&constant_returns(0.0), 0.0, 0.1);
        assert!(relaxed.gauge < outcome.gauge);
        assert!(relaxed.gauge > 1.0);

        let baseline = controller.update(&constant_returns(0.0), 0.0, 0.1);
        assert!(baseline.gauge <= relaxed.gauge);
    }

    #[test]
    fn lr_floor_and_eta_floor_are_respected() {
        let trigger = LossStdTrigger::new(0.01).with_warmup(0).with_max_ratio(5.0);
        let config = HyperSurpriseConfig::default()
            .with_lr_floor(1e-3)
            .with_eta_floor(0.2);
        let mut controller = HyperSurpriseController::new(trigger, config);

        let outcome = controller.update(&constant_returns(0.5), 0.0, 1e-4);
        assert!(outcome.learning_rate >= 1e-3);
        let signal = controller.last_signal().unwrap();
        assert!(signal.learning_rate >= 1e-3);
        assert!(signal.eta_bar >= 0.2);
    }

    #[test]
    fn ratio_smoothing_decays_residual_pulse() {
        let trigger = LossStdTrigger::new(0.2)
            .with_warmup(0)
            .with_max_ratio(4.0)
            .with_decay(0.1);
        let config = HyperSurpriseConfig::default()
            .with_ratio_smoothing(0.6)
            .with_relaxation(0.4)
            .with_smoothing(0.0);
        let mut controller = HyperSurpriseController::new(trigger, config);

        let spike = controller.update(&constant_returns(0.5), 0.0, 0.1);
        let first_ratio = spike.signal.unwrap().inject_ratio;
        assert!(first_ratio > 0.0);

        let trailing = controller.update(&constant_returns(0.0), 0.0, 0.1);
        let mut last = trailing.signal.unwrap().inject_ratio;
        assert!(last < first_ratio);
        assert!(last > 0.0);

        let mut steps = 0;
        while steps < 48 {
            let decay = controller.update(&constant_returns(0.0), 0.0, 0.1);
            if let Some(signal) = decay.signal {
                assert!(signal.inject_ratio < last);
                last = signal.inject_ratio;
            } else {
                last = 0.0;
                break;
            }
            steps += 1;
        }
        assert!(last <= 1e-6, "ratio smoothing failed to decay to baseline");
    }

    #[test]
    fn cooldown_blocks_back_to_back_spikes() {
        let trigger = LossStdTrigger::new(0.05).with_warmup(0).with_max_ratio(4.0);
        let config = HyperSurpriseConfig::default()
            .with_cooldown_steps(2)
            .with_ratio_smoothing(0.0)
            .with_smoothing(0.0);
        let mut controller = HyperSurpriseController::new(trigger, config);

        let burst_one = controller.update(&constant_returns(0.5), 0.0, 0.05);
        assert!(burst_one.signal.is_some());

        let burst_two = controller.update(&constant_returns(0.5), 0.0, 0.05);
        assert!(burst_two.signal.is_none());

        let idle = controller.update(&constant_returns(0.0), 0.0, 0.05);
        assert!(idle.signal.is_none());

        let rebound = controller.update(&constant_returns(0.5), 0.0, 0.05);
        assert!(rebound.signal.is_some());
    }
}
