// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use std::cell::Cell;

/// Supported activation families for [`NonLiner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonLinerActivation {
    /// Hyperbolic tangent non-linearity.
    Tanh,
    /// Logistic sigmoid non-linearity.
    Sigmoid,
    /// Smooth alternative to ReLU with bounded outputs.
    Softsign,
}

impl NonLinerActivation {
    fn activate(self, pre_activation: f32) -> f32 {
        match self {
            Self::Tanh => pre_activation.tanh(),
            Self::Sigmoid => 1.0 / (1.0 + (-pre_activation).exp()),
            Self::Softsign => pre_activation / (1.0 + pre_activation.abs()),
        }
    }

    fn derivative(self, activated: f32, pre_activation: f32) -> f32 {
        match self {
            Self::Tanh => 1.0 - activated * activated,
            Self::Sigmoid => activated * (1.0 - activated),
            Self::Softsign => {
                let denom = 1.0 + pre_activation.abs();
                1.0 / (denom * denom)
            }
        }
    }
}

/// Trainable smooth non-linearity with learnable gain, slope, and bias terms.
///
/// The module performs the following computation for every feature `i`:
///
/// ```text
/// y_i = gain_i * activation(slope_i * x_i + bias_i)
/// ```
///
/// All affine terms are parameterised which allows the layer to learn feature
/// specific gating while still integrating with SpiralTorch's hypergrad
/// pipelines.
#[derive(Debug)]
pub struct NonLiner {
    gain: Parameter,
    slope: Parameter,
    bias: Parameter,
    activation: NonLinerActivation,
    last_drift: Cell<Option<f32>>,
}

impl NonLiner {
    /// Creates a new non-linear layer using the default tanh activation and
    /// unit initialisation.
    pub fn new(name: impl Into<String>, features: usize) -> PureResult<Self> {
        Self::with_activation(name, features, NonLinerActivation::Tanh)
    }

    /// Creates a new non-linear layer with the provided activation.
    pub fn with_activation(
        name: impl Into<String>,
        features: usize,
        activation: NonLinerActivation,
    ) -> PureResult<Self> {
        Self::with_init(name, features, activation, 1.0, 1.0, 0.0)
    }

    /// Creates a new non-linear layer with caller supplied initial values.
    pub fn with_init(
        name: impl Into<String>,
        features: usize,
        activation: NonLinerActivation,
        slope: f32,
        gain: f32,
        bias: f32,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions { rows: 0, cols: 0 });
        }

        let name = name.into();
        let slope = Tensor::from_vec(1, features, vec![slope; features])?;
        let gain = Tensor::from_vec(1, features, vec![gain; features])?;
        let bias = Tensor::from_vec(1, features, vec![bias; features])?;

        Ok(Self {
            gain: Parameter::new(format!("{name}::gain"), gain),
            slope: Parameter::new(format!("{name}::slope"), slope),
            bias: Parameter::new(format!("{name}::bias"), bias),
            activation,
            last_drift: Cell::new(None),
        })
    }

    /// Returns the learnable gain parameter.
    pub fn gain(&self) -> &Parameter {
        &self.gain
    }

    /// Returns the learnable slope parameter.
    pub fn slope(&self) -> &Parameter {
        &self.slope
    }

    /// Returns the learnable bias parameter.
    pub fn bias(&self) -> &Parameter {
        &self.bias
    }

    /// Returns the activation family powering this layer.
    pub fn activation(&self) -> NonLinerActivation {
        self.activation
    }

    fn ensure_parameter_shapes(&self, features: usize) -> PureResult<()> {
        let expected = (1, features);
        for parameter in [&self.gain, &self.slope, &self.bias] {
            if parameter.value().shape() != expected {
                return Err(TensorError::ShapeMismatch {
                    left: expected,
                    right: parameter.value().shape(),
                });
            }
        }
        Ok(())
    }
}

impl Module for NonLiner {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols == 0 {
            self.last_drift.set(Some(0.0));
            return Tensor::zeros(rows, cols);
        }

        self.ensure_parameter_shapes(cols)?;

        let gain = self.gain.value().data().to_vec();
        let slope = self.slope.value().data().to_vec();
        let bias = self.bias.value().data().to_vec();

        let mut output = Vec::with_capacity(rows * cols);
        let mut drift_sum = 0.0;
        let total = rows * cols;

        for chunk in input.data().chunks(cols) {
            for (col, value) in chunk.iter().enumerate() {
                let pre = slope[col] * *value + bias[col];
                let activated = self.activation.activate(pre);
                let out = gain[col] * activated;
                output.push(out);
                drift_sum += out.abs();
            }
        }

        let drift = if total == 0 {
            0.0
        } else {
            drift_sum / total as f32
        };
        self.last_drift.set(Some(drift));

        Tensor::from_vec(rows, cols, output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }

        let (rows, cols) = input.shape();
        if cols == 0 {
            return Tensor::zeros(rows, cols);
        }

        self.ensure_parameter_shapes(cols)?;

        let gain = self.gain.value().data().to_vec();
        let slope = self.slope.value().data().to_vec();
        let bias = self.bias.value().data().to_vec();

        let mut grad_input = vec![0.0; rows * cols];
        let mut grad_gain = vec![0.0; cols];
        let mut grad_slope = vec![0.0; cols];
        let mut grad_bias = vec![0.0; cols];

        for row in 0..rows {
            let base = row * cols;
            for col in 0..cols {
                let idx = base + col;
                let input_value = input.data()[idx];
                let grad_out = grad_output.data()[idx];

                let pre = slope[col] * input_value + bias[col];
                let activated = self.activation.activate(pre);
                let derivative = self.activation.derivative(activated, pre);
                let chain = grad_out * gain[col];

                grad_gain[col] += grad_out * activated;
                let delta = chain * derivative;
                grad_bias[col] += delta;
                grad_slope[col] += delta * input_value;
                grad_input[idx] = delta * slope[col];
            }
        }

        if rows > 0 {
            let inv_batch = 1.0 / rows as f32;
            for value in grad_input.iter_mut() {
                *value *= inv_batch;
            }
            for grad in [&mut grad_gain, &mut grad_slope, &mut grad_bias] {
                for value in grad.iter_mut() {
                    *value *= inv_batch;
                }
            }
        }

        let gain_grad = Tensor::from_vec(1, cols, grad_gain)?;
        let slope_grad = Tensor::from_vec(1, cols, grad_slope)?;
        let bias_grad = Tensor::from_vec(1, cols, grad_bias)?;

        self.gain.accumulate_euclidean(&gain_grad)?;
        self.slope.accumulate_euclidean(&slope_grad)?;
        self.bias.accumulate_euclidean(&bias_grad)?;

        Tensor::from_vec(rows, cols, grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.gain)?;
        visitor(&self.slope)?;
        visitor(&self.bias)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.gain)?;
        visitor(&mut self.slope)?;
        visitor(&mut self.bias)?;
        Ok(())
    }

    fn psi_probe(&self) -> Option<f32> {
        self.last_drift.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (l, r) in left.iter().zip(right.iter()) {
            let diff = (l - r).abs();
            assert!(diff < 1e-5, "expected {l} ≈ {r} (diff={diff})");
        }
    }

    #[test]
    fn forward_applies_affine_and_activation() {
        let layer = NonLiner::with_init("nl", 3, NonLinerActivation::Tanh, 0.5, 1.2, -0.1).unwrap();
        let input = Tensor::from_vec(2, 3, vec![-1.0, 0.2, 0.5, 1.3, -0.7, 0.0]).unwrap();
        let output = layer.forward(&input).unwrap();

        let expected: Vec<f32> = input
            .data()
            .iter()
            .map(|x| 1.2 * (0.5 * *x - 0.1).tanh())
            .collect();
        approx_eq(output.data(), &expected);

        let drift = layer.psi_probe().unwrap();
        let expected_drift = expected.iter().map(|v| v.abs()).sum::<f32>() / expected.len() as f32;
        assert!((drift - expected_drift).abs() < 1e-6);
    }

    #[test]
    fn backward_accumulates_parameter_gradients() {
        let mut layer =
            NonLiner::with_init("nl", 2, NonLinerActivation::Sigmoid, 0.7, 1.1, 0.05).unwrap();
        let input = Tensor::from_vec(3, 2, vec![0.2, -0.3, 0.5, 0.8, -0.4, 0.1]).unwrap();
        let grad_output = Tensor::from_vec(3, 2, vec![0.6, -0.2, -0.4, 0.9, 0.3, -0.7]).unwrap();

        let grad_input = layer.backward(&input, &grad_output).unwrap();

        let mut expected_grad_input = vec![0.0; 6];
        let mut expected_gain = vec![0.0; 2];
        let mut expected_slope = vec![0.0; 2];
        let mut expected_bias = vec![0.0; 2];
        let gain = 1.1;
        let slope = 0.7;
        let bias = 0.05;

        for row in 0..3 {
            for col in 0..2 {
                let idx = row * 2 + col;
                let x = input.data()[idx];
                let go = grad_output.data()[idx];
                let pre = slope * x + bias;
                let act = 1.0 / (1.0 + (-pre).exp());
                let deriv = act * (1.0 - act);
                let chain = go * gain;
                expected_gain[col] += go * act;
                let delta = chain * deriv;
                expected_bias[col] += delta;
                expected_slope[col] += delta * x;
                expected_grad_input[idx] = delta * slope;
            }
        }

        let inv_batch = 1.0 / 3.0;
        for value in expected_grad_input.iter_mut() {
            *value *= inv_batch;
        }
        for grad in [&mut expected_gain, &mut expected_slope, &mut expected_bias] {
            for value in grad.iter_mut() {
                *value *= inv_batch;
            }
        }

        approx_eq(grad_input.data(), &expected_grad_input);

        let gain_grad = layer.gain().gradient().expect("gain gradient");
        let slope_grad = layer.slope().gradient().expect("slope gradient");
        let bias_grad = layer.bias().gradient().expect("bias gradient");

        approx_eq(gain_grad.data(), &expected_gain);
        approx_eq(slope_grad.data(), &expected_slope);
        approx_eq(bias_grad.data(), &expected_bias);
    }
}
