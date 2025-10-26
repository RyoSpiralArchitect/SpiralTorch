// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Continuous wavelet transform layer tailored for SpiralTorch's Z-space intuition.
//!
//! The layer sweeps a Morlet wavelet across the feature axis for each requested
//! scale, returning the real and imaginary coefficients per sample. Scales are
//! assumed to form a log-uniform lattice, which makes it straightforward to
//! stitch the output with Mellin-driven modules such as the renormalisation flow
//! DSL. When a fractal field is attached the coefficients are modulated by a
//! self-similar envelope, imitating "focus of consciousness" dynamics.

use crate::module::Module;
use st_frac::fractal_field::FractalField;
use st_tensor::{PureResult, Tensor, TensorError};

/// Deterministic continuous wavelet transform with optional fractal modulation.
#[derive(Clone, Debug)]
pub struct ContinuousWaveletTransformLayer {
    scales: Vec<f32>,
    omega0: f32,
    focus_width: f32,
    fractal: Option<FractalField>,
    fractal_mix: f32,
}

impl ContinuousWaveletTransformLayer {
    /// Builds a layer from explicit scales.
    pub fn new(scales: Vec<f32>, omega0: f32) -> PureResult<Self> {
        if scales.is_empty() {
            return Err(TensorError::InvalidDimensions { rows: 1, cols: 0 });
        }
        for &scale in &scales {
            if !scale.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "scale",
                    value: scale,
                });
            }
            if scale <= 0.0 {
                return Err(TensorError::InvalidValue { label: "scale" });
            }
        }
        Ok(Self {
            scales,
            omega0,
            focus_width: 0.0,
            fractal: None,
            fractal_mix: 0.5,
        })
    }

    /// Convenience constructor building a logarithmic lattice of scales.
    pub fn from_log_lattice(
        log_start: f32,
        log_step: f32,
        depth: usize,
        omega0: f32,
    ) -> PureResult<Self> {
        if depth == 0 {
            return Err(TensorError::InvalidDimensions { rows: 1, cols: 0 });
        }
        if !log_start.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "log_start",
                value: log_start,
            });
        }
        if !log_step.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "log_step",
                value: log_step,
            });
        }
        if log_step <= 0.0 {
            return Err(TensorError::InvalidValue { label: "log_step" });
        }
        let mut scales = Vec::with_capacity(depth);
        for i in 0..depth {
            let value = (log_start + log_step * i as f32).exp();
            scales.push(value.max(1e-6));
        }
        Self::new(scales, omega0)
    }

    /// Width of the Gaussian focus applied along the feature axis.
    pub fn with_focus_width(mut self, focus_width: f32) -> Self {
        self.focus_width = focus_width.max(0.0);
        self
    }

    /// Configures the fractal mix coefficient used to modulate the output.
    pub fn with_fractal_mix(mut self, mix: f32) -> Self {
        self.fractal_mix = mix;
        self
    }

    /// Attaches a pre-generated fractal field to modulate the coefficients.
    pub fn attach_fractal_field(&mut self, field: FractalField) -> PureResult<()> {
        if field.depth() != self.scales.len() {
            return Err(TensorError::ShapeMismatch {
                left: (field.depth(), field.dimension()),
                right: (self.scales.len(), field.dimension()),
            });
        }
        self.fractal = Some(field);
        Ok(())
    }

    fn morlet(&self, tau: f32) -> (f32, f32) {
        let gaussian = (-0.5 * tau * tau).exp();
        let phase = self.omega0 * tau;
        (gaussian * phase.cos(), gaussian * phase.sin())
    }

    fn focus_weight(&self, position: f32, centre: f32) -> f32 {
        if self.focus_width <= 0.0 {
            1.0
        } else {
            let normalised = (position - centre) / self.focus_width;
            (-0.5 * normalised * normalised).exp()
        }
    }

    fn fractal_factor(&self, scale_index: usize) -> f32 {
        if let Some(field) = &self.fractal {
            let mean = field.mean_amplitude(scale_index).unwrap_or(0.0);
            1.0 + self.fractal_mix * mean
        } else {
            1.0
        }
    }

    /// Returns the scales registered in the layer.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    fn output_width(&self) -> usize {
        self.scales.len() * 2
    }
}

impl Module for ContinuousWaveletTransformLayer {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        let out_cols = self.output_width();
        let mut output = Tensor::zeros(rows, out_cols)?;
        let centre = (cols as f32 - 1.0) * 0.5;
        let input_data = input.data();
        {
            let out_data = output.data_mut();
            for r in 0..rows {
                let in_offset = r * cols;
                let out_offset = r * out_cols;
                for (s_idx, &scale) in self.scales.iter().enumerate() {
                    let mut real = 0.0f32;
                    let mut imag = 0.0f32;
                    for c in 0..cols {
                        let tau = (c as f32 - centre) / scale;
                        let (psi_r, psi_i) = self.morlet(tau);
                        let focus = self.focus_weight(c as f32, centre);
                        let sample = input_data[in_offset + c] * focus;
                        real += sample * psi_r;
                        imag += sample * psi_i;
                    }
                    let gating = self.fractal_factor(s_idx);
                    let idx = out_offset + 2 * s_idx;
                    out_data[idx] = gating * real;
                    out_data[idx + 1] = gating * imag;
                }
            }
        }
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let expected = self.output_width();
        if grad_output.shape() != (rows, expected) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, expected),
            });
        }
        let mut grad_input = Tensor::zeros(rows, cols)?;
        let centre = (cols as f32 - 1.0) * 0.5;
        let out_cols = expected;
        let grad_out_data = grad_output.data();
        {
            let grad_in_data = grad_input.data_mut();
            for r in 0..rows {
                let in_offset = r * cols;
                let out_offset = r * out_cols;
                for (s_idx, &scale) in self.scales.iter().enumerate() {
                    let gating = self.fractal_factor(s_idx);
                    let grad_r = grad_out_data[out_offset + 2 * s_idx];
                    let grad_i = grad_out_data[out_offset + 2 * s_idx + 1];
                    for c in 0..cols {
                        let tau = (c as f32 - centre) / scale;
                        let (psi_r, psi_i) = self.morlet(tau);
                        let focus = self.focus_weight(c as f32, centre);
                        let contribution = gating * focus;
                        grad_in_data[in_offset + c] +=
                            contribution * (grad_r * psi_r + grad_i * psi_i);
                    }
                }
            }
        }
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_frac::fractal_field::FractalFieldGenerator;

    #[test]
    fn forward_produces_expected_shape() {
        let layer = ContinuousWaveletTransformLayer::from_log_lattice(-1.0, 0.5, 4, 5.0).unwrap();
        let input = Tensor::from_vec(2, 8, vec![0.5f32; 16]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 8));
    }

    #[test]
    fn fractal_modulation_changes_output() {
        let mut layer = ContinuousWaveletTransformLayer::from_log_lattice(-1.0, 0.5, 3, 6.0)
            .unwrap()
            .with_focus_width(2.0)
            .with_fractal_mix(1.0);
        let generator = FractalFieldGenerator::new(2, 0.75, 2024).unwrap();
        let log_scales: Vec<f32> = layer.scales().iter().map(|&s| s.ln()).collect();
        let field = generator.generate_for_scales(&log_scales).unwrap();
        layer.attach_fractal_field(field).unwrap();
        let input = Tensor::from_fn(1, 4, |_, c| (c as f32).sin()).unwrap();
        let output = layer.forward(&input).unwrap();
        assert!(output.data().iter().any(|&v| v.abs() > 1e-4));
    }

    #[test]
    fn backward_matches_input_shape() {
        let mut layer =
            ContinuousWaveletTransformLayer::from_log_lattice(-1.0, 0.5, 2, 4.5).unwrap();
        let input = Tensor::from_vec(1, 6, vec![0.1f32; 6]).unwrap();
        let output = layer.forward(&input).unwrap();
        let grad_output =
            Tensor::from_vec(1, output.shape().1, vec![0.2f32; output.shape().1]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
    }
}
