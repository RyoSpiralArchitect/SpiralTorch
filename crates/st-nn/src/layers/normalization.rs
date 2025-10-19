// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};
use st_tensor::TensorError;

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
}
