// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use std::collections::HashMap;

/// Feature-wise scaling layer that learns a multiplicative gain per channel.
#[derive(Debug)]
pub struct Scaler {
    gain: Parameter,
    baseline: Tensor,
}

impl Scaler {
    /// Creates a new scaler layer with unit gain for every feature.
    pub fn new(name: impl Into<String>, features: usize) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        let gains = Tensor::from_vec(1, features, vec![1.0; features])?;
        Self::from_gain(name, gains)
    }

    /// Constructs a scaler from an explicit gain tensor.
    pub fn from_gain(name: impl Into<String>, gain: Tensor) -> PureResult<Self> {
        let name = name.into();
        let (rows, cols) = gain.shape();
        if cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        if rows != 1 {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (1, cols),
            });
        }
        let baseline = gain.clone();
        Ok(Self {
            gain: Parameter::new(format!("{name}::gain"), gain),
            baseline,
        })
    }

    /// Returns an immutable view of the gain parameter.
    pub fn gain(&self) -> &Parameter {
        &self.gain
    }

    /// Returns the reference gain captured during the most recent calibration.
    pub fn baseline(&self) -> &Tensor {
        &self.baseline
    }

    /// Calibrates the scaler gains against the provided samples.
    ///
    /// The layer computes the per-feature standard deviation and updates the
    /// multiplicative gain to become the inverse deviation, optionally padded
    /// by `epsilon` to avoid exploding updates. After calibration the new gain
    /// becomes the reference baseline for subsequent ψ probes.
    pub fn calibrate(&mut self, samples: &Tensor, epsilon: f32) -> PureResult<()> {
        if epsilon < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "scaler_calibrate_epsilon",
            });
        }

        let (rows, cols) = samples.shape();
        let gain_cols = self.gain.value().shape().1;
        if cols != gain_cols {
            return Err(TensorError::ShapeMismatch {
                left: samples.shape(),
                right: (rows, gain_cols),
            });
        }

        if rows == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }

        let mut mean = vec![0.0f32; cols];
        let data = samples.data();
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                mean[c] += data[offset + c];
            }
        }
        let inv_rows = 1.0 / rows as f32;
        for value in &mut mean {
            *value *= inv_rows;
        }

        let mut variance = vec![0.0f32; cols];
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                let diff = data[offset + c] - mean[c];
                variance[c] += diff * diff;
            }
        }

        {
            let gain_tensor = self.gain.value_mut();
            let gain_data = gain_tensor.data_mut();
            for (idx, var) in variance.iter().enumerate() {
                let std = (var / rows as f32).sqrt();
                let denom = std + epsilon;
                gain_data[idx] = if denom <= f32::MIN_POSITIVE {
                    1.0
                } else {
                    1.0 / denom
                };
            }
        }

        self.gain.zero_gradient();
        self.baseline = self.gain.value().clone();
        Ok(())
    }
}

impl Module for Scaler {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let gain = self.gain.value();
        if gain.shape().1 != cols {
            return Err(TensorError::ShapeMismatch {
                left: gain.shape(),
                right: (1, cols),
            });
        }
        let mut output = input.clone();
        let gain_values = gain.data();
        let out_data = output.data_mut();
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                out_data[offset + c] *= gain_values[c];
            }
        }
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = input.shape();
        let gain_values = self.gain.value().data().to_vec();
        let mut grad_input = grad_output.clone();
        {
            let grad_input_data = grad_input.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    grad_input_data[offset + c] *= gain_values[c];
                }
            }
        }

        let mut grad_gain = vec![0.0f32; cols];
        let input_data = input.data();
        let grad_out_data = grad_output.data();
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                grad_gain[c] += input_data[offset + c] * grad_out_data[offset + c];
            }
        }
        let batch = rows as f32;
        let grad_gain_tensor = Tensor::from_vec(1, cols, grad_gain)?.scale(1.0 / batch)?;
        self.gain.accumulate_euclidean(&grad_gain_tensor)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.gain)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.gain)?;
        Ok(())
    }

    fn psi_probe(&self) -> Option<f32> {
        let gains = self.gain.value().data();
        if gains.is_empty() {
            return None;
        }
        let baseline = self.baseline.data();
        let drift = gains
            .iter()
            .zip(baseline.iter())
            .map(|(gain, reference)| (gain - reference).abs())
            .sum::<f32>()
            / gains.len() as f32;
        Some(drift)
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> PureResult<()> {
        self.visit_parameters_mut(&mut |param| {
            let Some(value) = state.get(param.name()) else {
                return Err(TensorError::MissingParameter {
                    name: param.name().to_string(),
                });
            };
            param.load_value(value)
        })?;
        self.baseline = self.gain.value().clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn scaler_forward_scales_each_feature() {
        let layer = Scaler::new("scale", 3).unwrap();
        let input = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn scaler_backward_accumulates_gain_gradient() {
        let mut layer = Scaler::new("scale", 2).unwrap();
        let input = Tensor::from_vec(2, 2, vec![1.0, 0.5, -2.0, 1.5]).unwrap();
        let gain = layer.gain().value();
        assert_eq!(gain.data(), &[1.0, 1.0]);

        let grad_out = Tensor::from_vec(2, 2, vec![0.2, -0.4, 0.5, 0.1]).unwrap();
        let grad_input = layer.backward(&input, &grad_out).unwrap();
        // With unit gain the gradient should flow unchanged.
        assert_eq!(grad_input, grad_out);

        let grads = layer.gain().gradient().unwrap();
        let expected = Tensor::from_vec(
            1,
            2,
            vec![
                (1.0 * 0.2 + -2.0 * 0.5) / 2.0,
                (0.5 * -0.4 + 1.5 * 0.1) / 2.0,
            ],
        )
        .unwrap();
        assert_eq!(grads, &expected);
    }

    #[test]
    fn scaler_from_gain_validates_shape() {
        let gain = Tensor::from_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let layer = Scaler::from_gain("scale", gain.clone()).unwrap();
        assert_eq!(layer.gain().value(), &gain);
        assert_eq!(layer.baseline(), &gain);

        let err = Scaler::from_gain("scale", Tensor::from_vec(2, 3, vec![0.0; 6]).unwrap());
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn scaler_psi_probe_reflects_gain_drift() {
        let mut layer = Scaler::new("scale", 3).unwrap();
        let samples = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 0.0, -1.0, -2.0]).unwrap();
        layer.calibrate(&samples, 1e-3).unwrap();
        assert_eq!(layer.psi_probe(), Some(0.0));

        let baseline = layer.baseline().data().to_vec();
        {
            let values = layer.gain.value_mut();
            let data = values.data_mut();
            data[0] = baseline[0] + 0.1;
            data[1] = baseline[1] - 0.2;
            data[2] = baseline[2] + 0.3;
        }

        let drift = layer.psi_probe().unwrap();
        let expected = (0.1f32.abs() + 0.2f32.abs() + 0.3f32.abs()) / 3.0;
        assert!((drift - expected).abs() < 1e-6);
    }

    #[test]
    fn scaler_state_dict_round_trips_gain() {
        let mut source = Scaler::new("scale", 2).unwrap();
        {
            let values = source.gain.value_mut();
            let data = values.data_mut();
            data.copy_from_slice(&[1.25, 0.75]);
        }
        let state = source.state_dict().unwrap();

        let mut target = Scaler::new("scale", 2).unwrap();
        target.load_state_dict(&state).unwrap();
        assert_eq!(target.gain().value().data(), &[1.25, 0.75]);
        assert_eq!(target.baseline().data(), &[1.25, 0.75]);
    }

    #[test]
    fn scaler_calibrate_updates_gain_and_baseline() {
        let mut layer = Scaler::new("scale", 2).unwrap();
        let samples = Tensor::from_vec(3, 2, vec![1.0, 0.0, 2.0, 2.0, 3.0, 4.0]).unwrap();
        layer.calibrate(&samples, 1e-3).unwrap();

        let gains = layer.gain().value();
        let baseline = layer.baseline();
        assert_eq!(gains, baseline);

        let expected_first = 1.0 / ((2.0f32 / 3.0).sqrt() + 1e-3);
        let expected_second = 1.0 / ((8.0f32 / 3.0).sqrt() + 1e-3);
        let data = gains.data();
        assert!((data[0] - expected_first).abs() < 1e-4);
        assert!((data[1] - expected_second).abs() < 1e-4);
    }

    #[test]
    fn scaler_load_state_updates_baseline() {
        let mut layer = Scaler::new("scale", 2).unwrap();
        let mut state = HashMap::new();
        let gain = Tensor::from_vec(1, 2, vec![0.75, 1.5]).unwrap();
        state.insert("scale::gain".to_string(), gain.clone());
        layer.load_state_dict(&state).unwrap();
        assert_eq!(layer.gain().value(), &gain);
        assert_eq!(layer.baseline(), &gain);
    }
}
