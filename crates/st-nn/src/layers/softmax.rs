// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;
use std::cell::RefCell;

const LOG_FLOOR: f32 = 1.0e-12;
const ADJUST_MIN: f32 = 0.25;
const ADJUST_MAX: f32 = 4.0;

/// Hyperbolic softmax that rescales logits through the Z-space curvature and
/// adapts its effective temperature to match a desired entropy range.
#[derive(Debug)]
pub struct ZSpaceSoftmax {
    curvature: f32,
    temperature: f32,
    min_temperature: f32,
    max_temperature: f32,
    entropy_target: Option<f32>,
    entropy_tolerance: f32,
    entropy_gain: f32,
    last_entropies: RefCell<Vec<f32>>,
    last_temperatures: RefCell<Vec<f32>>,
}

impl ZSpaceSoftmax {
    /// Builds the layer with the provided negative curvature and base
    /// temperature.
    pub fn new(curvature: f32, temperature: f32) -> PureResult<Self> {
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_softmax_temperature",
                value: temperature,
            });
        }
        let min_temperature = (temperature * 0.1).max(1.0e-3);
        let max_temperature = temperature * 10.0;
        Ok(Self {
            curvature,
            temperature,
            min_temperature,
            max_temperature,
            entropy_target: None,
            entropy_tolerance: 1.0e-4,
            entropy_gain: 0.5,
            last_entropies: RefCell::new(Vec::new()),
            last_temperatures: RefCell::new(Vec::new()),
        })
    }

    /// Overrides the entropy target used for the adaptive temperature.
    pub fn with_entropy_target(
        mut self,
        target: f32,
        tolerance: f32,
        gain: f32,
    ) -> PureResult<Self> {
        if !target.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_softmax_entropy_target",
                value: target,
            });
        }
        if tolerance < 0.0 || !tolerance.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_softmax_entropy_tolerance",
                value: tolerance,
            });
        }
        if gain < 0.0 || !gain.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_softmax_entropy_gain",
                value: gain,
            });
        }
        self.entropy_target = Some(target);
        self.entropy_tolerance = tolerance;
        self.entropy_gain = gain;
        Ok(self)
    }

    /// Tightens the admissible temperature interval enforced after adaptation.
    pub fn with_temperature_bounds(
        mut self,
        min_temperature: f32,
        max_temperature: f32,
    ) -> PureResult<Self> {
        if min_temperature <= 0.0 || !min_temperature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_softmax_min_temperature",
                value: min_temperature,
            });
        }
        if max_temperature <= 0.0 || !max_temperature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_softmax_max_temperature",
                value: max_temperature,
            });
        }
        if min_temperature > max_temperature {
            return Err(TensorError::InvalidDimensions {
                rows: min_temperature as usize,
                cols: max_temperature as usize,
            });
        }
        self.min_temperature = min_temperature;
        self.max_temperature = max_temperature;
        Ok(self)
    }

    /// Returns the enforced curvature.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the base temperature prior to entropy adjustment.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Returns the entropies captured during the most recent forward pass.
    pub fn last_entropies(&self) -> Vec<f32> {
        self.last_entropies.borrow().clone()
    }

    /// Returns the per-row effective temperatures emitted during the most
    /// recent forward pass.
    pub fn last_temperatures(&self) -> Vec<f32> {
        self.last_temperatures.borrow().clone()
    }

    /// Clears the cached entropy/temperature diagnostics.
    pub fn reset_metrics(&self) {
        self.last_entropies.replace(Vec::new());
        self.last_temperatures.replace(Vec::new());
    }

    fn curvature_scale(&self) -> f32 {
        (-self.curvature).sqrt()
    }

    fn compute_row(&self, row: &[f32]) -> (Vec<f32>, f32, f32) {
        if row.is_empty() {
            return (Vec::new(), 0.0, self.temperature);
        }
        let scale = self.curvature_scale();
        let mut effective_temperature = self
            .temperature
            .clamp(self.min_temperature, self.max_temperature);
        let mut probs;
        let mut entropy;
        let mut state = self.softmax_with_scale(row, scale / effective_temperature);
        probs = state.0;
        entropy = state.1;

        if let Some(target) = self.entropy_target {
            let delta = target - entropy;
            if delta.abs() > self.entropy_tolerance {
                let adjust = (1.0 + self.entropy_gain * delta).clamp(ADJUST_MIN, ADJUST_MAX);
                effective_temperature = (effective_temperature * adjust)
                    .clamp(self.min_temperature, self.max_temperature);
                state = self.softmax_with_scale(row, scale / effective_temperature);
                probs = state.0;
                entropy = state.1;
            }
        }

        (probs, entropy, effective_temperature)
    }

    fn softmax_with_scale(&self, row: &[f32], scale: f32) -> (Vec<f32>, f32) {
        if row.is_empty() {
            return (Vec::new(), 0.0);
        }
        let mut scaled: Vec<f32> = row.iter().map(|v| v * scale).collect();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in scaled.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        if !sum.is_finite() || sum <= 0.0 {
            let len = row.len() as f32;
            let uniform = 1.0 / len.max(1.0);
            let entropy = -len * uniform * uniform.max(LOG_FLOOR).ln();
            return (vec![uniform; row.len()], entropy);
        }
        let inv_sum = 1.0 / sum;
        let mut probs = Vec::with_capacity(row.len());
        let mut entropy = 0.0f32;
        for value in scaled {
            let prob = value * inv_sum;
            let guarded = prob.max(LOG_FLOOR);
            entropy -= prob * guarded.ln();
            probs.push(prob);
        }
        (probs, entropy)
    }
}

impl Module for ZSpaceSoftmax {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols == 0 {
            self.last_entropies.replace(vec![0.0; rows]);
            self.last_temperatures.replace(vec![self.temperature; rows]);
            return Tensor::zeros(rows, cols);
        }

        let mut output = Vec::with_capacity(rows * cols);
        let mut entropies = Vec::with_capacity(rows);
        let mut temperatures = Vec::with_capacity(rows);
        let data = input.data();
        for r in 0..rows {
            let offset = r * cols;
            let row_slice = &data[offset..offset + cols];
            let (prob, entropy, temp) = self.compute_row(row_slice);
            output.extend(prob);
            entropies.push(entropy);
            temperatures.push(temp);
        }
        self.last_entropies.replace(entropies);
        self.last_temperatures.replace(temperatures);
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
        let scale = self.curvature_scale();
        let mut grad = Vec::with_capacity(rows * cols);
        let input_data = input.data();
        let grad_output_data = grad_output.data();
        for r in 0..rows {
            let offset = r * cols;
            let row_slice = &input_data[offset..offset + cols];
            let grad_slice = &grad_output_data[offset..offset + cols];
            let (prob, _, temp) = self.compute_row(row_slice);
            let dot: f32 = grad_slice.iter().zip(prob.iter()).map(|(g, p)| g * p).sum();
            let factor = scale / temp.clamp(self.min_temperature, self.max_temperature);
            for (g_out, p) in grad_slice.iter().zip(prob.iter()) {
                grad.push(factor * p * (g_out - dot));
            }
        }
        Tensor::from_vec(rows, cols, grad)
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

    #[test]
    fn zspace_softmax_rows_sum_to_one() {
        let layer = ZSpaceSoftmax::new(-1.0, 1.0).unwrap();
        let input = Tensor::from_vec(2, 3, vec![1.0, 0.0, -1.0, 0.5, -0.25, 0.75]).unwrap();
        let output = layer.forward(&input).unwrap();
        for row in 0..2 {
            let start = row * 3;
            let sum: f32 = output.data()[start..start + 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
        let entropies = layer.last_entropies();
        assert_eq!(entropies.len(), 2);
        assert!(entropies[0] > 0.0);
    }

    #[test]
    fn zspace_softmax_backward_matches_formula() {
        let mut layer = ZSpaceSoftmax::new(-1.0, 1.5).unwrap();
        let input = Tensor::from_vec(1, 3, vec![0.2, -0.1, 0.3]).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![0.05, -0.02, 0.1]).unwrap();
        let forward = layer.forward(&input).unwrap();
        let grad_in = layer.backward(&input, &grad_out).unwrap();

        let probs = forward.data();
        let dot: f32 = grad_out
            .data()
            .iter()
            .zip(probs.iter())
            .map(|(g, p)| g * p)
            .sum();
        let scale = (-layer.curvature()).sqrt() / layer.last_temperatures()[0];
        for ((&prob, &grad), &grad_in) in probs
            .iter()
            .zip(grad_out.data().iter())
            .zip(grad_in.data().iter())
        {
            let expected = scale * prob * (grad - dot);
            assert!((expected - grad_in).abs() < 1e-5);
        }
    }

    #[test]
    fn zspace_softmax_entropy_controls_temperature() {
        let layer = ZSpaceSoftmax::new(-1.0, 1.0)
            .unwrap()
            .with_entropy_target(0.1, 1e-3, 1.0)
            .unwrap()
            .with_temperature_bounds(0.05, 2.0)
            .unwrap();
        let input = Tensor::from_vec(1, 3, vec![0.0, 0.0, 0.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 3));
        let temps = layer.last_temperatures();
        assert_eq!(temps.len(), 1);
        assert!(temps[0] < 1.0);
    }
}
