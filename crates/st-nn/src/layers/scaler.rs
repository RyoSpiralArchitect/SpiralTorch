// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};

/// Feature-wise scaling layer that learns a multiplicative gain per channel.
#[derive(Debug)]
pub struct Scaler {
    gain: Parameter,
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
        Ok(Self {
            gain: Parameter::new(format!("{name}::gain"), gain),
        })
    }

    /// Returns an immutable view of the gain parameter.
    pub fn gain(&self) -> &Parameter {
        &self.gain
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
        let drift = gains.iter().map(|g| (g - 1.0).abs()).sum::<f32>() / gains.len() as f32;
        Some(drift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let err = Scaler::from_gain("scale", Tensor::from_vec(2, 3, vec![0.0; 6]).unwrap());
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn scaler_psi_probe_reflects_gain_drift() {
        let mut layer = Scaler::new("scale", 3).unwrap();
        assert_eq!(layer.psi_probe(), Some(0.0));

        {
            let values = layer.gain.value_mut();
            let data = values.data_mut();
            data.copy_from_slice(&[1.0, 0.5, 1.5]);
        }

        let drift = layer.psi_probe().unwrap();
        assert!((drift - 0.333_333_34).abs() < 1e-6);
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
    }
}
