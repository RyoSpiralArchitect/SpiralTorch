// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};
use st_tensor::TensorError;
use std::cell::{Cell, RefCell};

/// Layer normalisation with curvature-aware epsilon stabilisation.
#[derive(Debug)]
pub struct LayerNorm {
    features: usize,
    epsilon: f32,
    curvature: f32,
    gamma: Parameter,
    beta: Parameter,
}

impl LayerNorm {
    /// Builds a new layer normalisation module.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        curvature: f32,
        epsilon: f32,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "layernorm_epsilon",
                value: epsilon,
            });
        }
        let base_name: String = name.into();
        let gamma_name = format!("{}_gamma", base_name.as_str());
        let beta_name = format!("{}_beta", base_name.as_str());
        let gamma = Tensor::from_fn(1, features, |_, _| 1.0)?;
        let beta = Tensor::zeros(1, features)?;
        Ok(Self {
            features,
            epsilon,
            curvature,
            gamma: Parameter::new(gamma_name, gamma),
            beta: Parameter::new(beta_name, beta),
        })
    }

    /// Returns the number of features normalised per row.
    pub fn features(&self) -> usize {
        self.features
    }

    /// Returns the epsilon used for stabilisation.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Returns the enforced curvature.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    fn guard_input(&self, input: &Tensor) -> PureResult<()> {
        let (rows, cols) = input.shape();
        if cols != self.features {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (rows, self.features),
            });
        }
        Ok(())
    }

    fn effective_epsilon(&self) -> f32 {
        let scale = (-self.curvature).sqrt();
        self.epsilon * (1.0 + scale * 0.1)
    }
}

/// Batch normalisation over the batch dimension.
#[derive(Debug)]
pub struct BatchNorm1d {
    features: usize,
    epsilon: f32,
    momentum: f32,
    gamma: Parameter,
    beta: Parameter,
    running_mean: RefCell<Tensor>,
    running_var: RefCell<Tensor>,
    training: Cell<bool>,
    last_mean: RefCell<Option<Vec<f32>>>,
    last_inv_std: RefCell<Option<Vec<f32>>>,
}

impl BatchNorm1d {
    /// Creates a new batch normalisation layer operating over the feature axis.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        momentum: f32,
        epsilon: f32,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if !(0.0..=1.0).contains(&momentum) || !momentum.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "batchnorm_momentum",
            });
        }
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "batchnorm_epsilon",
                value: epsilon,
            });
        }
        let name = name.into();
        let gamma = Tensor::from_vec(1, features, vec![1.0; features])?;
        let beta = Tensor::zeros(1, features)?;
        let running_mean = Tensor::zeros(1, features)?;
        let running_var = Tensor::from_vec(1, features, vec![1.0; features])?;
        Ok(Self {
            features,
            epsilon,
            momentum,
            gamma: Parameter::new(format!("{name}::gamma"), gamma),
            beta: Parameter::new(format!("{name}::beta"), beta),
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),
            training: Cell::new(true),
            last_mean: RefCell::new(None),
            last_inv_std: RefCell::new(None),
        })
    }

    /// Number of normalised features.
    pub fn features(&self) -> usize {
        self.features
    }

    /// Returns the momentum applied to the running statistics.
    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Returns the epsilon used to stabilise the variance estimate.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Enables or disables training mode.
    pub fn set_training(&self, training: bool) {
        self.training.set(training);
    }

    /// Switches the layer to training mode.
    pub fn train(&self) {
        self.set_training(true);
    }

    /// Switches the layer to evaluation mode.
    pub fn eval(&self) {
        self.set_training(false);
    }

    fn guard_input(&self, input: &Tensor) -> PureResult<()> {
        let (rows, cols) = input.shape();
        if cols != self.features {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (rows, self.features),
            });
        }
        if rows == 0 {
            return Err(TensorError::EmptyInput("batchnorm_input"));
        }
        Ok(())
    }

    fn compute_stats(&self, input: &Tensor) -> (Vec<f32>, Vec<f32>) {
        let (batch, features) = input.shape();
        let mut mean = vec![0.0f32; features];
        for row in 0..batch {
            let slice = &input.data()[row * features..(row + 1) * features];
            for (idx, value) in slice.iter().enumerate() {
                mean[idx] += *value;
            }
        }
        let scale = 1.0 / batch as f32;
        for value in mean.iter_mut() {
            *value *= scale;
        }
        let mut variance = vec![0.0f32; features];
        for row in 0..batch {
            let slice = &input.data()[row * features..(row + 1) * features];
            for (idx, value) in slice.iter().enumerate() {
                let centered = *value - mean[idx];
                variance[idx] += centered * centered;
            }
        }
        for value in variance.iter_mut() {
            *value *= scale;
        }
        (mean, variance)
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        let (batch, features) = input.shape();
        let mut output = Vec::with_capacity(batch * features);
        let gamma = self.gamma.value().data();
        let beta = self.beta.value().data();
        let (mean, variance) = if self.training.get() {
            let (mean, variance) = self.compute_stats(input);
            {
                let mut running_mean = self.running_mean.borrow_mut();
                let data = running_mean.data_mut();
                for idx in 0..features {
                    data[idx] = self.momentum * mean[idx] + (1.0 - self.momentum) * data[idx];
                }
            }
            {
                let mut running_var = self.running_var.borrow_mut();
                let data = running_var.data_mut();
                for idx in 0..features {
                    data[idx] = self.momentum * variance[idx] + (1.0 - self.momentum) * data[idx];
                }
            }
            *self.last_mean.borrow_mut() = Some(mean.clone());
            let inv_std: Vec<f32> = variance
                .iter()
                .map(|v| 1.0 / (v + self.epsilon).sqrt())
                .collect();
            *self.last_inv_std.borrow_mut() = Some(inv_std.clone());
            (mean, variance)
        } else {
            let running_mean = self.running_mean.borrow();
            let running_var = self.running_var.borrow();
            (running_mean.data().to_vec(), running_var.data().to_vec())
        };

        let inv_std: Vec<f32> = if let Some(inv) = self.last_inv_std.borrow().clone() {
            if self.training.get() {
                inv
            } else {
                variance
                    .iter()
                    .map(|v| 1.0 / (v + self.epsilon).sqrt())
                    .collect()
            }
        } else {
            variance
                .iter()
                .map(|v| 1.0 / (v + self.epsilon).sqrt())
                .collect()
        };

        for row in 0..batch {
            let slice = &input.data()[row * features..(row + 1) * features];
            for feature in 0..features {
                let normed = (slice[feature] - mean[feature]) * inv_std[feature];
                output.push(normed * gamma[feature] + beta[feature]);
            }
        }
        Tensor::from_vec(batch, features, output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        if !self.training.get() {
            return Err(TensorError::InvalidValue {
                label: "batchnorm_backward_eval",
            });
        }
        let (batch, features) = input.shape();
        let mean = self
            .last_mean
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "batchnorm_cached_mean",
            })?;
        let inv_std = self
            .last_inv_std
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "batchnorm_cached_invstd",
            })?;

        let mut grad_input = vec![0.0f32; batch * features];
        let mut grad_gamma = vec![0.0f32; features];
        let mut grad_beta = vec![0.0f32; features];

        for feature in 0..features {
            let mut sum_grad = 0.0f32;
            let mut sum_grad_norm = 0.0f32;
            for row in 0..batch {
                let idx = row * features + feature;
                let normed = (input.data()[idx] - mean[feature]) * inv_std[feature];
                let g = grad_output.data()[idx];
                sum_grad += g;
                sum_grad_norm += g * normed;
                grad_gamma[feature] += g * normed;
                grad_beta[feature] += g;
            }
            for row in 0..batch {
                let idx = row * features + feature;
                let normed = (input.data()[idx] - mean[feature]) * inv_std[feature];
                let g = grad_output.data()[idx];
                let term = (batch as f32 * g - sum_grad - normed * sum_grad_norm) / batch as f32;
                grad_input[idx] = term * inv_std[feature];
            }
        }

        let grad_gamma = Tensor::from_vec(1, features, grad_gamma)?;
        let grad_beta = Tensor::from_vec(1, features, grad_beta)?;
        self.gamma.accumulate_euclidean(&grad_gamma)?;
        self.beta.accumulate_euclidean(&grad_beta)?;
        Tensor::from_vec(batch, features, grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.gamma)?;
        visitor(&self.beta)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.gamma)?;
        visitor(&mut self.beta)
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        let (rows, cols) = input.shape();
        let mut output = Vec::with_capacity(rows * cols);
        let gamma = self.gamma.value().data();
        let beta = self.beta.value().data();
        let epsilon = self.effective_epsilon();

        for r in 0..rows {
            let offset = r * cols;
            let slice = &input.data()[offset..offset + cols];
            let mean: f32 = slice.iter().copied().sum::<f32>() / cols as f32;
            let variance: f32 = slice
                .iter()
                .map(|x| {
                    let centered = *x - mean;
                    centered * centered
                })
                .sum::<f32>()
                / cols as f32;
            let denom = (variance + epsilon).sqrt();
            for c in 0..cols {
                let normed = (slice[c] - mean) / denom;
                output.push(normed * gamma[c] + beta[c]);
            }
        }
        Tensor::from_vec(rows, cols, output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = input.shape();
        let epsilon = self.effective_epsilon();
        let gamma = self.gamma.value().data().to_vec();
        let mut grad_input = vec![0.0f32; rows * cols];
        let mut grad_gamma = vec![0.0f32; cols];
        let mut grad_beta = vec![0.0f32; cols];

        for r in 0..rows {
            let offset = r * cols;
            let slice = &input.data()[offset..offset + cols];
            let grad_slice = &grad_output.data()[offset..offset + cols];
            let mean: f32 = slice.iter().copied().sum::<f32>() / cols as f32;
            let variance: f32 = slice
                .iter()
                .map(|x| {
                    let centered = *x - mean;
                    centered * centered
                })
                .sum::<f32>()
                / cols as f32;
            let denom = (variance + epsilon).sqrt();
            let inv_denom = 1.0 / denom;
            let mut normed = vec![0.0f32; cols];
            for c in 0..cols {
                normed[c] = (slice[c] - mean) * inv_denom;
                grad_gamma[c] += grad_slice[c] * normed[c];
                grad_beta[c] += grad_slice[c];
            }
            let dot_norm_grad: f32 = grad_slice
                .iter()
                .zip(normed.iter())
                .map(|(g, n)| g * n)
                .sum();
            let sum_grad: f32 = grad_slice.iter().sum();
            for c in 0..cols {
                let g = grad_slice[c];
                let n = normed[c];
                let term = (cols as f32 * g - sum_grad - n * dot_norm_grad) / cols as f32;
                grad_input[offset + c] = term * gamma[c] * inv_denom;
            }
        }

        let grad_gamma_tensor = Tensor::from_vec(1, cols, grad_gamma)?;
        let grad_beta_tensor = Tensor::from_vec(1, cols, grad_beta)?;
        self.gamma.accumulate_euclidean(&grad_gamma_tensor)?;
        self.beta.accumulate_euclidean(&grad_beta_tensor)?;

        Tensor::from_vec(rows, cols, grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.gamma)?;
        visitor(&self.beta)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.gamma)?;
        visitor(&mut self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_input() -> Tensor {
        Tensor::from_vec(2, 3, vec![0.5, -1.0, 1.5, 2.0, -0.5, 0.0]).unwrap()
    }

    #[test]
    fn layer_norm_zero_mean_unit_variance() {
        let layer = LayerNorm::new("demo", 3, -1.0, 1e-5).unwrap();
        let input = demo_input();
        let output = layer.forward(&input).unwrap();
        let (_, cols) = output.shape();
        for row in 0..2 {
            let start = row * cols;
            let slice = &output.data()[start..start + cols];
            let mean: f32 = slice.iter().sum::<f32>() / cols as f32;
            let var: f32 = slice
                .iter()
                .map(|v| {
                    let diff = *v - mean;
                    diff * diff
                })
                .sum::<f32>()
                / cols as f32;
            assert!(mean.abs() < 1e-4);
            assert!((var - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn layer_norm_backward_accumulates_parameters() {
        let mut layer = LayerNorm::new("demo", 3, -1.0, 1e-5).unwrap();
        let input = demo_input();
        let grad_output = Tensor::from_vec(2, 3, vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (2, 3));
        let gamma_grad = layer.gamma.gradient().unwrap();
        let beta_grad = layer.beta.gradient().unwrap();
        assert_eq!(gamma_grad.shape(), (1, 3));
        assert_eq!(beta_grad.shape(), (1, 3));
    }

    #[test]
    fn batch_norm_forward_normalises_features() {
        let layer = BatchNorm1d::new("bn", 3, 0.1, 1e-5).unwrap();
        let input = Tensor::from_vec(
            4,
            3,
            vec![
                0.5, 1.0, -0.5, // sample 0
                1.5, -0.5, 0.25, // sample 1
                -1.0, 0.2, 0.75, // sample 2
                0.0, -1.2, 1.5, // sample 3
            ],
        )
        .unwrap();
        let output = layer.forward(&input).unwrap();
        for feature in 0..3 {
            let mut mean = 0.0f32;
            let mut var = 0.0f32;
            for row in 0..4 {
                let value = output.data()[row * 3 + feature];
                mean += value;
                var += value * value;
            }
            mean /= 4.0;
            var /= 4.0;
            assert!(mean.abs() < 1e-4);
            assert!((var - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn batch_norm_backward_populates_parameter_grads() {
        let mut layer = BatchNorm1d::new("bn", 2, 0.2, 1e-5).unwrap();
        let input = Tensor::from_vec(3, 2, vec![0.2, -0.3, 1.0, 0.5, -1.5, 2.0]).unwrap();
        let grad_output = Tensor::from_vec(3, 2, vec![0.1, -0.2, 0.05, 0.3, -0.4, 0.6]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        let gamma_grad = layer.gamma.gradient().unwrap();
        let beta_grad = layer.beta.gradient().unwrap();
        assert_eq!(gamma_grad.shape(), (1, 2));
        assert_eq!(beta_grad.shape(), (1, 2));
        for value in grad_input.data() {
            assert!(value.is_finite());
        }
    }
}
