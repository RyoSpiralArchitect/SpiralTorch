// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_core::theory::zpulse::ZScale;
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

/// Hyperbolic geometry configuration used by [`NonLinerGeometry::Hyperbolic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonLinerHyperbolicConfig {
    curvature: f32,
    z_scale: ZScale,
    retention: f32,
}

impl NonLinerHyperbolicConfig {
    /// Creates a new hyperbolic configuration. `curvature` must be negative and finite while
    /// `retention` must lie in the unit interval.
    pub fn new(curvature: f32, z_scale: ZScale, retention: f32) -> PureResult<Self> {
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if !retention.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "non_liner_retention",
                value: retention,
            });
        }
        if !(0.0..=1.0).contains(&retention) {
            return Err(TensorError::InvalidValue {
                label: "non_liner_retention",
            });
        }
        Ok(Self {
            curvature,
            z_scale,
            retention,
        })
    }

    /// Returns the enforced curvature.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the Z-scale used to interpret the Euclidean activations.
    pub fn z_scale(&self) -> ZScale {
        self.z_scale
    }

    /// Returns the retention ratio blending the raw Euclidean signal with its hyperbolic projection.
    pub fn retention(&self) -> f32 {
        self.retention
    }

    fn curvature_scale(&self) -> f32 {
        (-self.curvature).sqrt()
    }

    fn mix(&self) -> f32 {
        1.0 - self.retention
    }
}

/// Geometry applied to the affine activation outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonLinerGeometry {
    /// Standard Euclidean geometry (no additional projection is applied).
    Euclidean,
    /// Hyperbolic projection into the Z-space manifold using the provided configuration.
    Hyperbolic(NonLinerHyperbolicConfig),
}

impl Default for NonLinerGeometry {
    fn default() -> Self {
        Self::Euclidean
    }
}

impl NonLinerGeometry {
    /// Convenience constructor for the hyperbolic variant.
    pub fn hyperbolic(config: NonLinerHyperbolicConfig) -> Self {
        Self::Hyperbolic(config)
    }

    /// Returns the underlying hyperbolic configuration when present.
    pub fn as_hyperbolic(&self) -> Option<NonLinerHyperbolicConfig> {
        match self {
            Self::Hyperbolic(config) => Some(*config),
            Self::Euclidean => None,
        }
    }

    /// Returns the curvature enforced by the geometry when hyperbolic.
    pub fn curvature(&self) -> Option<f32> {
        self.as_hyperbolic().map(|config| config.curvature())
    }

    /// Returns the Z-scale when hyperbolic.
    pub fn z_scale(&self) -> Option<ZScale> {
        self.as_hyperbolic().map(|config| config.z_scale())
    }

    /// Returns the retention ratio when hyperbolic.
    pub fn retention(&self) -> Option<f32> {
        self.as_hyperbolic().map(|config| config.retention())
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
/// Optionally the affine outputs are projected into a Z-space hyperbolic manifold before being
/// returned, ensuring they respect a curvature-aware radius.
#[derive(Debug)]
pub struct NonLiner {
    gain: Parameter,
    slope: Parameter,
    bias: Parameter,
    activation: NonLinerActivation,
    geometry: NonLinerGeometry,
    last_drift: Cell<Option<f32>>,
    last_radius: Cell<Option<f32>>,
}

impl NonLiner {
    /// Creates a new non-linear layer using the default tanh activation, unit initialisation, and
    /// Euclidean geometry.
    pub fn new(name: impl Into<String>, features: usize) -> PureResult<Self> {
        Self::with_activation(name, features, NonLinerActivation::Tanh)
    }

    /// Creates a new non-linear layer with the provided activation and Euclidean geometry.
    pub fn with_activation(
        name: impl Into<String>,
        features: usize,
        activation: NonLinerActivation,
    ) -> PureResult<Self> {
        Self::with_geometry(
            name,
            features,
            activation,
            1.0,
            1.0,
            0.0,
            NonLinerGeometry::default(),
        )
    }

    /// Creates a new non-linear layer with caller supplied initial values and Euclidean geometry.
    pub fn with_init(
        name: impl Into<String>,
        features: usize,
        activation: NonLinerActivation,
        slope: f32,
        gain: f32,
        bias: f32,
    ) -> PureResult<Self> {
        Self::with_geometry(
            name,
            features,
            activation,
            slope,
            gain,
            bias,
            NonLinerGeometry::default(),
        )
    }

    /// Creates a new non-linear layer with the provided activation, affine initialisation, and
    /// explicit geometry.
    pub fn with_geometry(
        name: impl Into<String>,
        features: usize,
        activation: NonLinerActivation,
        slope: f32,
        gain: f32,
        bias: f32,
        geometry: NonLinerGeometry,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions { rows: 0, cols: 0 });
        }
        let name = name.into();
        let slope_tensor = Tensor::from_vec(1, features, vec![slope; features])?;
        let gain_tensor = Tensor::from_vec(1, features, vec![gain; features])?;
        let bias_tensor = Tensor::from_vec(1, features, vec![bias; features])?;
        Ok(Self {
            gain: Parameter::new(format!("{name}::gain"), gain_tensor),
            slope: Parameter::new(format!("{name}::slope"), slope_tensor),
            bias: Parameter::new(format!("{name}::bias"), bias_tensor),
            activation,
            geometry,
            last_drift: Cell::new(None),
            last_radius: Cell::new(None),
        })
    }

    /// Creates a new hyperbolic non-linear layer with unit affine parameters.
    pub fn with_hyperbolic(
        name: impl Into<String>,
        features: usize,
        activation: NonLinerActivation,
        config: NonLinerHyperbolicConfig,
    ) -> PureResult<Self> {
        Self::with_geometry(
            name,
            features,
            activation,
            1.0,
            1.0,
            0.0,
            NonLinerGeometry::hyperbolic(config),
        )
    }

    /// Creates a new hyperbolic non-linear layer with caller supplied affine parameters.
    pub fn with_hyperbolic_init(
        name: impl Into<String>,
        features: usize,
        activation: NonLinerActivation,
        slope: f32,
        gain: f32,
        bias: f32,
        config: NonLinerHyperbolicConfig,
    ) -> PureResult<Self> {
        Self::with_geometry(
            name,
            features,
            activation,
            slope,
            gain,
            bias,
            NonLinerGeometry::hyperbolic(config),
        )
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

    /// Returns the geometry applied to the affine outputs.
    pub fn geometry(&self) -> NonLinerGeometry {
        self.geometry
    }

    /// Updates the geometry used during subsequent forward/backward passes.
    pub fn set_geometry(&mut self, geometry: NonLinerGeometry) {
        self.geometry = geometry;
        self.reset_metrics();
    }

    /// Clears cached telemetry (ψ drift and hyperbolic radii).
    pub fn reset_metrics(&self) {
        self.last_drift.set(None);
        self.last_radius.set(None);
    }

    /// Returns the average hyperbolic radius observed during the most recent forward pass.
    pub fn last_hyperbolic_radius(&self) -> Option<f32> {
        self.last_radius.get()
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

    fn hyperbolic_coefficients(config: &NonLinerHyperbolicConfig, norm: f32) -> (f32, f32, f32) {
        if norm <= 1.0e-6 {
            return (1.0, 0.0, 0.0);
        }
        let scale = config.curvature_scale();
        let scale_z = config.z_scale().value();
        if scale_z <= 0.0 || !scale.is_finite() {
            return (1.0, 0.0, 0.0);
        }
        let a = scale / scale_z;
        if !a.is_finite() || a <= 0.0 {
            return (1.0, 0.0, 0.0);
        }
        let u = norm / a;
        if !u.is_finite() {
            return (1.0, 0.0, scale);
        }
        let tanh_u = u.tanh();
        let factor = (a / norm) * tanh_u;
        let cosh_u = u.cosh();
        let sech_sq = if cosh_u.is_finite() {
            let denom = cosh_u * cosh_u;
            if denom == 0.0 {
                0.0
            } else {
                1.0 / denom
            }
        } else {
            0.0
        };
        let derivative = (sech_sq - factor) / norm;
        let target_radius = tanh_u * scale;
        (factor, derivative, target_radius)
    }

    fn apply_geometry_forward(&self, rows: usize, cols: usize, data: &mut [f32]) -> Option<f32> {
        match self.geometry {
            NonLinerGeometry::Euclidean => None,
            NonLinerGeometry::Hyperbolic(config) => {
                if rows == 0 || cols == 0 {
                    return Some(0.0);
                }
                let retention = config.retention();
                let mix = config.mix();
                let mut radius_sum = 0.0f32;
                let mut counted = 0usize;
                for row in 0..rows {
                    let start = row * cols;
                    let end = start + cols;
                    let chunk = &mut data[start..end];
                    if chunk.is_empty() {
                        continue;
                    }
                    let norm_sq: f32 = chunk.iter().map(|v| v * v).sum();
                    let norm = norm_sq.sqrt();
                    let (factor, _, radius) = Self::hyperbolic_coefficients(&config, norm);
                    if mix > 0.0 && norm > 1.0e-6 {
                        for value in chunk.iter_mut() {
                            let projected = *value * factor;
                            *value = retention * *value + mix * projected;
                        }
                    }
                    radius_sum += radius;
                    counted += 1;
                }
                Some(if counted > 0 {
                    radius_sum / counted as f32
                } else {
                    0.0
                })
            }
        }
    }

    fn apply_geometry_backward(&self, rows: usize, cols: usize, raw: &[f32], grad: &mut [f32]) {
        match self.geometry {
            NonLinerGeometry::Euclidean => {}
            NonLinerGeometry::Hyperbolic(config) => {
                if rows == 0 || cols == 0 {
                    return;
                }
                let retention = config.retention();
                let mix = config.mix();
                for row in 0..rows {
                    let start = row * cols;
                    let end = start + cols;
                    let raw_chunk = &raw[start..end];
                    let grad_chunk = &mut grad[start..end];
                    if raw_chunk.is_empty() {
                        continue;
                    }
                    let norm_sq: f32 = raw_chunk.iter().map(|v| v * v).sum();
                    let norm = norm_sq.sqrt();
                    let (factor, derivative, _) = Self::hyperbolic_coefficients(&config, norm);
                    if norm <= 1.0e-6 {
                        let scaling = retention + mix;
                        if (scaling - 1.0).abs() > f32::EPSILON {
                            for grad_value in grad_chunk.iter_mut() {
                                *grad_value *= scaling;
                            }
                        }
                        continue;
                    }
                    let dot = raw_chunk
                        .iter()
                        .zip(grad_chunk.iter())
                        .map(|(x, g)| x * g)
                        .sum::<f32>();
                    let scaling = retention + mix * factor;
                    let correction = mix * derivative * dot / norm;
                    for (raw_value, grad_value) in raw_chunk.iter().zip(grad_chunk.iter_mut()) {
                        *grad_value = scaling * *grad_value + correction * *raw_value;
                    }
                }
            }
        }
    }
}

impl Module for NonLiner {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols == 0 {
            self.last_drift.set(Some(0.0));
            self.last_radius
                .set(self.geometry.as_hyperbolic().map(|_| 0.0));
            return Tensor::zeros(rows, cols);
        }

        self.ensure_parameter_shapes(cols)?;

        let gain = self.gain.value().data().to_vec();
        let slope = self.slope.value().data().to_vec();
        let bias = self.bias.value().data().to_vec();

        let mut output = Vec::with_capacity(rows * cols);
        for chunk in input.data().chunks(cols) {
            for (col, value) in chunk.iter().enumerate() {
                let pre = slope[col] * *value + bias[col];
                let activated = self.activation.activate(pre);
                output.push(gain[col] * activated);
            }
        }

        let radius = self.apply_geometry_forward(rows, cols, &mut output);
        self.last_radius.set(radius);

        let total = rows * cols;
        let drift = if total == 0 {
            0.0
        } else {
            output.iter().map(|v| v.abs()).sum::<f32>() / total as f32
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

        let mut raw_outputs = vec![0.0f32; rows * cols];
        for row in 0..rows {
            let base = row * cols;
            for col in 0..cols {
                let idx = base + col;
                let input_value = input.data()[idx];
                let pre = slope[col] * input_value + bias[col];
                let activated = self.activation.activate(pre);
                raw_outputs[idx] = gain[col] * activated;
            }
        }

        let mut grad_raw = grad_output.data().to_vec();
        self.apply_geometry_backward(rows, cols, &raw_outputs, &mut grad_raw);

        let mut grad_input = vec![0.0; rows * cols];
        let mut grad_gain = vec![0.0; cols];
        let mut grad_slope = vec![0.0; cols];
        let mut grad_bias = vec![0.0; cols];

        for row in 0..rows {
            let base = row * cols;
            for col in 0..cols {
                let idx = base + col;
                let input_value = input.data()[idx];
                let pre = slope[col] * input_value + bias[col];
                let activated = self.activation.activate(pre);
                let derivative = self.activation.derivative(activated, pre);
                let grad = grad_raw[idx];

                grad_gain[col] += grad * activated;
                let chain = grad * gain[col];
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
        assert!(layer.last_hyperbolic_radius().is_none());
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

    #[test]
    fn hyperbolic_forward_projects_into_ball() {
        let config = NonLinerHyperbolicConfig::new(-0.9, ZScale::new(1.5).unwrap(), 0.2).unwrap();
        let layer =
            NonLiner::with_hyperbolic_init("z", 3, NonLinerActivation::Tanh, 0.9, 1.3, 0.1, config)
                .unwrap();
        let input = Tensor::from_vec(2, 3, vec![1.2, -0.8, 0.4, -1.5, 0.75, 0.0]).unwrap();
        let output = layer.forward(&input).unwrap();

        let curvature_scale = config.curvature_scale();
        let retention = config.retention();
        let mix = config.mix();
        let z_scale = config.z_scale().value();
        let slope = 0.9f32;
        let gain = 1.3f32;
        let bias = 0.1f32;
        for row in 0..2 {
            let start = row * 3;
            let slice = &output.data()[start..start + 3];
            let norm = slice.iter().map(|v| v * v).sum::<f32>().sqrt() * z_scale;

            let raw: Vec<f32> = input.data()[start..start + 3]
                .iter()
                .map(|value| gain * (slope * *value + bias).tanh())
                .collect();
            let raw_norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt() * z_scale;
            let allowable = retention * raw_norm + mix * curvature_scale;
            assert!(norm <= allowable + 1e-5);
        }
        let radius = layer.last_hyperbolic_radius().unwrap();
        assert!(radius <= curvature_scale + 1e-5);
    }

    #[test]
    fn hyperbolic_backward_matches_numeric_gradients() {
        let config = NonLinerHyperbolicConfig::new(-1.0, ZScale::new(1.2).unwrap(), 0.35).unwrap();
        let mut layer = NonLiner::with_hyperbolic_init(
            "geom",
            2,
            NonLinerActivation::Softsign,
            0.7,
            0.9,
            -0.05,
            config,
        )
        .unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.25, -0.4, 0.8, -1.1]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.3, -0.2, -0.45, 0.6]).unwrap();

        let grad_input = layer.backward(&input, &grad_output).unwrap();
        let grad_output_vec = grad_output.data().to_vec();
        let (rows, cols) = input.shape();
        let inv_rows = 1.0 / rows as f32;
        let epsilon = 1e-3;

        for idx in 0..(rows * cols) {
            let mut plus = input.clone();
            plus.data_mut()[idx] += epsilon;
            let out_plus = layer.forward(&plus).unwrap();
            let loss_plus = out_plus
                .data()
                .iter()
                .zip(grad_output_vec.iter())
                .map(|(o, g)| o * g)
                .sum::<f32>()
                * inv_rows;

            let mut minus = input.clone();
            minus.data_mut()[idx] -= epsilon;
            let out_minus = layer.forward(&minus).unwrap();
            let loss_minus = out_minus
                .data()
                .iter()
                .zip(grad_output_vec.iter())
                .map(|(o, g)| o * g)
                .sum::<f32>()
                * inv_rows;

            let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert!((numeric - grad_input.data()[idx]).abs() < 2e-3);
        }
    }
}
