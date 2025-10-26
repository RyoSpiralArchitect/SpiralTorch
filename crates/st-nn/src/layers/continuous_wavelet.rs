// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use std::cell::{Ref, RefCell};

/// Continuous wavelet transform layer that focuses Z-space activity locally.
///
/// The layer performs a Morlet-style continuous wavelet transform across the
/// feature dimension, weighting each scale with a learnable focus vector.  The
/// result is a “local consciousness spotlight” that emphasises neighbourhoods on
/// the Z-lattice without discarding global coherence.
#[derive(Debug)]
pub struct ContinuousWaveletTransform {
    features: usize,
    scales: Vec<f32>,
    log_step: f32,
    omega0: f32,
    focus: Parameter,
    bias: Parameter,
    kernels: RefCell<Option<Vec<Vec<f32>>>>,
}

impl ContinuousWaveletTransform {
    /// Constructs a continuous wavelet transform layer.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        scales: Vec<f32>,
        log_step: f32,
        omega0: f32,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if scales.is_empty() {
            return Err(TensorError::InvalidDimensions { rows: 1, cols: 0 });
        }
        if !(log_step.is_finite() && log_step > 0.0) {
            return Err(TensorError::InvalidValue {
                label: "continuous_wavelet_log_step",
            });
        }
        if !(omega0.is_finite() && omega0 > 0.0) {
            return Err(TensorError::InvalidValue {
                label: "continuous_wavelet_omega0",
            });
        }
        for &scale in &scales {
            if !(scale.is_finite() && scale > 0.0) {
                return Err(TensorError::InvalidValue {
                    label: "continuous_wavelet_scale",
                });
            }
        }
        let focus = Tensor::from_vec(
            1,
            scales.len(),
            vec![1.0 / scales.len() as f32; scales.len()],
        )?;
        let bias = Tensor::zeros(1, features)?;
        let name = name.into();
        Ok(Self {
            features,
            scales,
            log_step,
            omega0,
            focus: Parameter::new(format!("{name}::focus"), focus),
            bias: Parameter::new(format!("{name}::bias"), bias),
            kernels: RefCell::new(None),
        })
    }

    /// Returns the number of features handled by the layer.
    pub fn features(&self) -> usize {
        self.features
    }

    /// Returns the learnable focus vector parameter.
    pub fn focus(&self) -> &Parameter {
        &self.focus
    }

    /// Returns the bias parameter.
    pub fn bias(&self) -> &Parameter {
        &self.bias
    }

    /// Exposes the Morlet central frequency.
    pub fn omega0(&self) -> f32 {
        self.omega0
    }

    /// Access the configured scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Computes a focus profile highlighting energy per Z-lattice coordinate.
    pub fn focus_profile(&self, input: &Tensor) -> PureResult<Vec<f32>> {
        let responses = self.compute_responses(input)?;
        let focus = self.focus.value().data();
        let (rows, cols) = input.shape();
        let mut profile = vec![0.0f32; cols];
        for c in 0..cols {
            let mut energy = 0.0f32;
            for (scale_idx, response) in responses.iter().enumerate() {
                let mut column_energy = 0.0f32;
                for r in 0..rows {
                    let value = response[r * cols + c];
                    column_energy += value * value;
                }
                energy += focus[scale_idx].abs() * column_energy.sqrt();
            }
            profile[c] = energy;
        }
        Ok(profile)
    }

    fn morlet(&self, delta: f32, scale: f32) -> f32 {
        let t = delta / scale;
        let envelope = (-0.5 * t * t).exp();
        let oscillation = (self.omega0 * t).cos();
        (1.0 / scale.sqrt()) * envelope * oscillation
    }

    fn build_kernels(&self, cols: usize) -> Vec<Vec<f32>> {
        let mut kernels = Vec::with_capacity(self.scales.len());
        for &scale in &self.scales {
            let mut kernel = vec![0.0f32; cols * cols];
            for center in 0..cols {
                let offset = center * cols;
                for sample in 0..cols {
                    let delta = (sample as f32 - center as f32) * self.log_step;
                    kernel[offset + sample] = self.morlet(delta, scale);
                }
            }
            kernels.push(kernel);
        }
        kernels
    }

    fn ensure_kernels(&self, cols: usize) -> Ref<'_, Vec<Vec<f32>>> {
        {
            let cached = self.kernels.borrow();
            if let Some(ref kernels) = *cached {
                if kernels.len() == self.scales.len()
                    && kernels
                        .first()
                        .map_or(true, |kernel| kernel.len() == cols * cols)
                {
                    return Ref::map(cached, |opt| opt.as_ref().unwrap());
                }
            }
        }

        let mut cache = self.kernels.borrow_mut();
        *cache = Some(self.build_kernels(cols));
        drop(cache);
        Ref::map(self.kernels.borrow(), |opt| opt.as_ref().unwrap())
    }

    fn compute_responses(&self, input: &Tensor) -> PureResult<Vec<Vec<f32>>> {
        let (rows, cols) = input.shape();
        if cols != self.features {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (rows, self.features),
            });
        }
        let input_data = input.data();
        let kernels = self.ensure_kernels(cols);
        let mut responses = Vec::with_capacity(kernels.len());
        for kernel in kernels.iter() {
            let mut response = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let row_offset = r * cols;
                for c in 0..cols {
                    let kernel_row = &kernel[c * cols..(c + 1) * cols];
                    let mut acc = 0.0f32;
                    for sample in 0..cols {
                        acc += input_data[row_offset + sample] * kernel_row[sample];
                    }
                    response[row_offset + c] = acc;
                }
            }
            responses.push(response);
        }
        Ok(responses)
    }
}

impl Module for ContinuousWaveletTransform {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let responses = self.compute_responses(input)?;
        let focus = self.focus.value().data();
        let bias = self.bias.value().data();
        let (rows, cols) = input.shape();
        let mut output = Tensor::zeros(rows, cols)?;
        {
            let out_data = output.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let mut value = bias[c];
                    for (scale_idx, response) in responses.iter().enumerate() {
                        value += focus[scale_idx] * response[offset + c];
                    }
                    out_data[offset + c] = value;
                }
            }
        }
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if grad_output.shape() != input.shape() {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: input.shape(),
            });
        }
        let responses = self.compute_responses(input)?;
        let focus = self.focus.value().data().to_vec();
        let (rows, cols) = input.shape();
        let grad_data = grad_output.data();
        let mut grad_input = Tensor::zeros(rows, cols)?;
        {
            let kernels = self.ensure_kernels(cols);
            let grad_input_data = grad_input.data_mut();
            for r in 0..rows {
                let row_offset = r * cols;
                for sample in 0..cols {
                    let mut acc = 0.0f32;
                    for c in 0..cols {
                        let go = grad_data[row_offset + c];
                        if go == 0.0 {
                            continue;
                        }
                        let mut wave_sum = 0.0f32;
                        for (scale_idx, kernel) in kernels.iter().enumerate() {
                            let kernel_row = &kernel[c * cols..(c + 1) * cols];
                            wave_sum += focus[scale_idx] * kernel_row[sample];
                        }
                        acc += go * wave_sum;
                    }
                    grad_input_data[row_offset + sample] = acc;
                }
            }
        }

        let mut grad_focus = vec![0.0f32; self.scales.len()];
        for (scale_idx, response) in responses.iter().enumerate() {
            let mut acc = 0.0f32;
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    acc += grad_data[offset + c] * response[offset + c];
                }
            }
            grad_focus[scale_idx] = acc;
        }

        let mut grad_bias = vec![0.0f32; cols];
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                grad_bias[c] += grad_data[offset + c];
            }
        }

        let grad_focus_tensor = Tensor::from_vec(1, grad_focus.len(), grad_focus)?;
        let grad_bias_tensor = Tensor::from_vec(1, cols, grad_bias)?;
        self.focus.accumulate_euclidean(&grad_focus_tensor)?;
        self.bias.accumulate_euclidean(&grad_bias_tensor)?;

        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.focus)?;
        visitor(&self.bias)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.focus)?;
        visitor(&mut self.bias)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_and_backward_shapes_match() {
        let mut layer =
            ContinuousWaveletTransform::new("cwt", 4, vec![0.75, 1.5], 0.25, 5.0).unwrap();
        let input =
            Tensor::from_vec(2, 4, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2, -0.3]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let grad_output = Tensor::from_vec(
            2,
            4,
            vec![0.01, -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, -0.03],
        )
        .unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        assert!(layer.focus().gradient().is_some());
        assert!(layer.bias().gradient().is_some());
    }

    #[test]
    fn focus_profile_tracks_energy() {
        let layer = ContinuousWaveletTransform::new("cwt", 3, vec![1.0, 2.0], 0.4, 3.5).unwrap();
        let input = Tensor::from_vec(1, 3, vec![0.5, -0.1, 0.2]).unwrap();
        let profile = layer.focus_profile(&input).unwrap();
        assert_eq!(profile.len(), 3);
        assert!(profile.iter().any(|v| *v > 0.0));
    }
}
