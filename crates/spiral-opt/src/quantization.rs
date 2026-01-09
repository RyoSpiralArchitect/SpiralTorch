use crate::report::QuantizationReport;
use std::f32;

/// Configuration controlling quantization-aware training behaviour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QatConfig {
    /// Number of bits used for the symmetric uniform quantiser.
    pub bit_width: u8,
    /// Exponential moving average decay for running range tracking.
    pub ema_decay: f32,
    /// Optional clamp applied before quantisation to stabilise extremes.
    pub clamp_value: Option<f32>,
    /// Small epsilon used to avoid divide-by-zero when computing scales.
    pub epsilon: f32,
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            bit_width: 8,
            ema_decay: 0.9,
            clamp_value: Some(6.0),
            epsilon: 1e-6,
        }
    }
}

/// Strategy used to compute the effective quantisation levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizationLeveling {
    /// Use symmetric ranges around zero.
    #[default]
    Symmetric,
    /// Use an asymmetric range by aligning zero to the observed minimum.
    Asymmetric,
}

/// Tracks running statistics needed to perform QAT without allocating.
#[derive(Debug, Clone)]
pub struct QatObserver {
    config: QatConfig,
    leveling: QuantizationLeveling,
    running_min: f32,
    running_max: f32,
    steps: u64,
}

impl QatObserver {
    /// Create a new observer with the provided configuration.
    pub fn new(config: QatConfig, leveling: QuantizationLeveling) -> Self {
        Self {
            config,
            leveling,
            running_min: f32::INFINITY,
            running_max: f32::NEG_INFINITY,
            steps: 0,
        }
    }

    /// Update running statistics with a fresh batch of weights or activations.
    pub fn observe(&mut self, weights: &[f32]) {
        if weights.is_empty() {
            return;
        }
        let (mut batch_min, mut batch_max) = (f32::INFINITY, f32::NEG_INFINITY);
        for &w in weights {
            batch_min = batch_min.min(w);
            batch_max = batch_max.max(w);
        }
        let decay = if self.steps == 0 {
            0.0
        } else {
            self.config.ema_decay.clamp(0.0, 1.0)
        };
        self.running_min = self.running_min.min(batch_min);
        self.running_max = self.running_max.max(batch_max);
        if self.steps > 0 {
            self.running_min = decay * self.running_min + (1.0 - decay) * batch_min;
            self.running_max = decay * self.running_max + (1.0 - decay) * batch_max;
        } else {
            self.running_min = batch_min;
            self.running_max = batch_max;
        }
        self.steps = self.steps.saturating_add(1);
    }

    /// Quantise the provided mutable slice in-place and return an audit report.
    pub fn quantize(&mut self, weights: &mut [f32]) -> QuantizationReport {
        if weights.is_empty() {
            return QuantizationReport::empty(self.config.bit_width);
        }

        if let Some(clamp) = self.config.clamp_value {
            for w in weights.iter_mut() {
                *w = w.clamp(-clamp, clamp);
            }
        }

        let (min, max) = match self.leveling {
            QuantizationLeveling::Symmetric => {
                let abs_max = self
                    .running_min
                    .abs()
                    .max(self.running_max.abs())
                    .max(self.config.epsilon);
                (-abs_max, abs_max)
            }
            QuantizationLeveling::Asymmetric => {
                let min = self.running_min.min(-self.config.epsilon);
                let max = self.running_max.max(self.config.epsilon);
                (min, max)
            }
        };

        let levels = 2_i32.pow(self.config.bit_width as u32).saturating_sub(1) as f32;
        let scale = (max - min).max(self.config.epsilon) / levels.max(1.0);
        let zero_point = if matches!(self.leveling, QuantizationLeveling::Asymmetric) {
            (-min / scale).round()
        } else {
            0.0
        };

        let mut squared_error = 0.0;
        for w in weights.iter_mut() {
            let clamped = w.clamp(min, max);
            let quant_level = ((clamped - min) / scale).round();
            let dequantised = quant_level * scale + min;
            squared_error += (*w - dequantised).powi(2);
            *w = dequantised;
        }

        let mse = squared_error / weights.len() as f32;
        QuantizationReport::new(
            self.config.bit_width,
            min,
            max,
            scale,
            zero_point,
            mse,
            self.steps,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_tracks_ranges() {
        let mut observer = QatObserver::new(QatConfig::default(), QuantizationLeveling::Symmetric);
        observer.observe(&[1.0, 2.0, 3.0]);
        observer.observe(&[-4.0, 5.0]);
        let report = observer.quantize(&mut [0.5, -0.5, 1.5, -1.5]);
        assert_eq!(report.bit_width, 8);
        assert!(report.scale > 0.0);
        assert!(report.quant_error <= 1.0);
    }
}
