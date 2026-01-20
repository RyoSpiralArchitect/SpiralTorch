// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};
use st_tensor::TensorError;
use std::cell::{Cell, RefCell};

const TANH_CLAMP: f32 = 25.0;

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

/// Batch normalisation that projects the whitened activations into the
/// hyperbolic Z-space ball before applying the affine parameters.
#[derive(Debug)]
pub struct ZSpaceBatchNorm1d {
    features: usize,
    curvature: f32,
    epsilon: f32,
    momentum: f32,
    projector_gain: Cell<f32>,
    gamma: Parameter,
    beta: Parameter,
    running_mean: RefCell<Tensor>,
    running_var: RefCell<Tensor>,
    training: Cell<bool>,
    last_batch: Cell<Option<usize>>,
    last_mean: RefCell<Option<Vec<f32>>>,
    last_inv_std: RefCell<Option<Vec<f32>>>,
    last_normed: RefCell<Option<Vec<f32>>>,
    last_projected: RefCell<Option<Vec<f32>>>,
    last_jacobian: RefCell<Option<Vec<f32>>>,
    last_radius: RefCell<Option<Vec<f32>>>,
}

/// Cached statistics captured during the last `ZSpaceBatchNorm1d` forward pass.
#[derive(Debug, Clone)]
pub struct ZSpaceBatchNormTelemetry {
    batch: usize,
    features: usize,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    normed: Vec<f32>,
    projected: Vec<f32>,
    jacobian: Vec<f32>,
    radius: Vec<f32>,
}

impl ZSpaceBatchNormTelemetry {
    /// Returns the batch size that produced the cached activations.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Returns the number of features per sample.
    pub fn features(&self) -> usize {
        self.features
    }

    /// Returns the per-feature means computed during the forward pass.
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Returns the inverse standard deviation used to whiten the inputs.
    pub fn inv_std(&self) -> &[f32] {
        &self.inv_std
    }

    /// Returns the normalised activations before projection, flattened in
    /// row-major order.
    pub fn normed(&self) -> &[f32] {
        &self.normed
    }

    /// Returns the projected activations that were fed into the affine branch.
    pub fn projected(&self) -> &[f32] {
        &self.projected
    }

    /// Returns the diagonal of the projection Jacobian for each activation.
    pub fn jacobian(&self) -> &[f32] {
        &self.jacobian
    }

    /// Returns the per-feature average Z-ball radius measured over the batch.
    pub fn radius(&self) -> &[f32] {
        &self.radius
    }
}

/// Layer normalisation that blends Euclidean statistics with Z-space projection.
#[derive(Debug)]
pub struct ZSpaceLayerNorm {
    features: usize,
    curvature: f32,
    epsilon: f32,
    projector_gain: Cell<f32>,
    gamma: Parameter,
    beta: Parameter,
    last_batch: Cell<Option<usize>>,
    last_mean: RefCell<Option<Vec<f32>>>,
    last_inv_std: RefCell<Option<Vec<f32>>>,
    last_normed: RefCell<Option<Vec<f32>>>,
    last_projected: RefCell<Option<Vec<f32>>>,
    last_jacobian: RefCell<Option<Vec<f32>>>,
    last_radius: RefCell<Option<Vec<f32>>>,
}

/// Cached statistics captured during the most recent `ZSpaceLayerNorm` forward pass.
#[derive(Debug, Clone)]
pub struct ZSpaceLayerNormTelemetry {
    batch: usize,
    features: usize,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    normed: Vec<f32>,
    projected: Vec<f32>,
    jacobian: Vec<f32>,
    radius: Vec<f32>,
}

impl ZSpaceLayerNormTelemetry {
    /// Returns the batch size of the cached activations.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Returns the number of features normalised per sample.
    pub fn features(&self) -> usize {
        self.features
    }

    /// Returns the per-sample means recorded during the forward pass.
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Returns the inverse standard deviations applied to each sample.
    pub fn inv_std(&self) -> &[f32] {
        &self.inv_std
    }

    /// Returns the whitened activations prior to projection.
    pub fn normed(&self) -> &[f32] {
        &self.normed
    }

    /// Returns the blended Euclidean/Z-space activations.
    pub fn projected(&self) -> &[f32] {
        &self.projected
    }

    /// Returns the projection Jacobian per activation.
    pub fn jacobian(&self) -> &[f32] {
        &self.jacobian
    }

    /// Returns the average ball radius measured per sample.
    pub fn radius(&self) -> &[f32] {
        &self.radius
    }
}

impl ZSpaceBatchNorm1d {
    /// Creates a new Z-space aware batch normalisation layer.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        curvature: f32,
        momentum: f32,
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
        if !(0.0..=1.0).contains(&momentum) || !momentum.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "zspace_batchnorm_momentum",
            });
        }
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_batchnorm_epsilon",
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
            curvature,
            epsilon,
            momentum,
            projector_gain: Cell::new(1.0),
            gamma: Parameter::new(format!("{name}::gamma"), gamma),
            beta: Parameter::new(format!("{name}::beta"), beta),
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),
            training: Cell::new(true),
            last_batch: Cell::new(None),
            last_mean: RefCell::new(None),
            last_inv_std: RefCell::new(None),
            last_normed: RefCell::new(None),
            last_projected: RefCell::new(None),
            last_jacobian: RefCell::new(None),
            last_radius: RefCell::new(None),
        })
    }

    /// Returns the number of features normalised per row.
    pub fn features(&self) -> usize {
        self.features
    }

    /// Returns the curvature enforced by the hyperbolic projection.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the mixing ratio between Euclidean and hyperbolic projections.
    pub fn projector_gain(&self) -> f32 {
        self.projector_gain.get()
    }

    /// Blends the Euclidean normalised activations with their Z-space projection.
    pub fn with_projector_gain(self, gain: f32) -> PureResult<Self> {
        self.set_projector_gain(gain)?;
        Ok(self)
    }

    /// Updates the projector gain in-place without rebuilding the layer.
    pub fn set_projector_gain(&self, gain: f32) -> PureResult<()> {
        if !(0.0..=1.0).contains(&gain) || !gain.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_batchnorm_projector_gain",
                value: gain,
            });
        }
        self.projector_gain.set(gain);
        Ok(())
    }

    /// Returns the epsilon used to stabilise the variance estimate.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Returns the momentum applied to the running statistics.
    pub fn momentum(&self) -> f32 {
        self.momentum
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

    /// Returns the cached telemetry captured during the most recent forward pass.
    pub fn telemetry(&self) -> Option<ZSpaceBatchNormTelemetry> {
        let batch = self.last_batch.get()?;
        let mean = self.last_mean.borrow().clone()?;
        let inv_std = self.last_inv_std.borrow().clone()?;
        let normed = self.last_normed.borrow().clone()?;
        let projected = self.last_projected.borrow().clone()?;
        let jacobian = self.last_jacobian.borrow().clone()?;
        let radius = self.last_radius.borrow().clone()?;
        Some(ZSpaceBatchNormTelemetry {
            batch,
            features: self.features,
            mean,
            inv_std,
            normed,
            projected,
            jacobian,
            radius,
        })
    }

    /// Returns the per-feature average ball radius captured during the most
    /// recent forward pass.
    pub fn last_ball_radius(&self) -> Option<Vec<f32>> {
        self.last_radius.borrow().clone()
    }

    /// Adjusts the projector gain to steer the average radius towards the
    /// provided target. Returns the updated gain.
    pub fn adapt_projector_gain(&self, target_radius: f32, smoothing: f32) -> PureResult<f32> {
        if !target_radius.is_finite() || target_radius < 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_batchnorm_target_radius",
                value: target_radius,
            });
        }
        if !smoothing.is_finite() || smoothing <= 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_batchnorm_smoothing",
                value: smoothing,
            });
        }
        let telemetry = self.telemetry().ok_or(TensorError::InvalidValue {
            label: "zspace_batchnorm_telemetry",
        })?;
        let avg_radius =
            telemetry.radius().iter().copied().sum::<f32>() / telemetry.radius().len() as f32;
        let gain =
            (self.projector_gain.get() + smoothing * (target_radius - avg_radius)).clamp(0.0, 1.0);
        self.projector_gain.set(gain);
        Ok(gain)
    }

    fn curvature_scale(&self) -> f32 {
        (-self.curvature).sqrt()
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
            return Err(TensorError::EmptyInput("zspace_batchnorm_input"));
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

    fn project_normed(&self, value: f32) -> (f32, f32, f32) {
        let gain = self.projector_gain.get();
        if gain <= f32::EPSILON {
            return (value, 1.0, 0.0);
        }
        let scale = self.curvature_scale();
        let scaled = (value * scale).clamp(-TANH_CLAMP, TANH_CLAMP);
        let tanh = scaled.tanh();
        let radial = tanh / scale;
        let sech_sq = 1.0 - tanh * tanh;
        let jacobian = (1.0 - gain) + gain * sech_sq;
        let blended = (1.0 - gain) * value + gain * radial;
        let radius = tanh.abs() / scale;
        (blended, jacobian, radius)
    }
}

impl ZSpaceLayerNorm {
    /// Creates a new Z-space aware layer normalisation module.
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
                label: "zspace_layernorm_epsilon",
                value: epsilon,
            });
        }
        let name = name.into();
        let gamma = Tensor::from_vec(1, features, vec![1.0; features])?;
        let beta = Tensor::zeros(1, features)?;
        Ok(Self {
            features,
            curvature,
            epsilon,
            projector_gain: Cell::new(1.0),
            gamma: Parameter::new(format!("{name}::gamma"), gamma),
            beta: Parameter::new(format!("{name}::beta"), beta),
            last_batch: Cell::new(None),
            last_mean: RefCell::new(None),
            last_inv_std: RefCell::new(None),
            last_normed: RefCell::new(None),
            last_projected: RefCell::new(None),
            last_jacobian: RefCell::new(None),
            last_radius: RefCell::new(None),
        })
    }

    /// Returns the number of features normalised per sample.
    pub fn features(&self) -> usize {
        self.features
    }

    /// Returns the enforced curvature of the hyperbolic projection.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the epsilon used to stabilise the variance estimate.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Returns the mixing ratio between Euclidean and Z-space activations.
    pub fn projector_gain(&self) -> f32 {
        self.projector_gain.get()
    }

    /// Configures the projector gain when constructing the layer.
    pub fn with_projector_gain(self, gain: f32) -> PureResult<Self> {
        self.set_projector_gain(gain)?;
        Ok(self)
    }

    /// Updates the projector gain in-place without rebuilding the module.
    pub fn set_projector_gain(&self, gain: f32) -> PureResult<()> {
        if !(0.0..=1.0).contains(&gain) || !gain.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_layernorm_projector_gain",
                value: gain,
            });
        }
        self.projector_gain.set(gain);
        Ok(())
    }

    /// Returns the cached telemetry captured during the most recent forward pass.
    pub fn telemetry(&self) -> Option<ZSpaceLayerNormTelemetry> {
        let batch = self.last_batch.get()?;
        let mean = self.last_mean.borrow().clone()?;
        let inv_std = self.last_inv_std.borrow().clone()?;
        let normed = self.last_normed.borrow().clone()?;
        let projected = self.last_projected.borrow().clone()?;
        let jacobian = self.last_jacobian.borrow().clone()?;
        let radius = self.last_radius.borrow().clone()?;
        Some(ZSpaceLayerNormTelemetry {
            batch,
            features: self.features,
            mean,
            inv_std,
            normed,
            projected,
            jacobian,
            radius,
        })
    }

    /// Returns the per-sample ball radius captured during the last forward pass.
    pub fn last_ball_radius(&self) -> Option<Vec<f32>> {
        self.last_radius.borrow().clone()
    }

    /// Adapts the projector gain towards the provided target radius.
    pub fn adapt_projector_gain(&self, target_radius: f32, smoothing: f32) -> PureResult<f32> {
        if !target_radius.is_finite() || target_radius < 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_layernorm_target_radius",
                value: target_radius,
            });
        }
        if !smoothing.is_finite() || smoothing <= 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "zspace_layernorm_smoothing",
                value: smoothing,
            });
        }
        let telemetry = self.telemetry().ok_or(TensorError::InvalidValue {
            label: "zspace_layernorm_telemetry",
        })?;
        let avg_radius =
            telemetry.radius().iter().copied().sum::<f32>() / telemetry.radius().len() as f32;
        let gain =
            (self.projector_gain.get() + smoothing * (target_radius - avg_radius)).clamp(0.0, 1.0);
        self.projector_gain.set(gain);
        Ok(gain)
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
            return Err(TensorError::EmptyInput("zspace_layernorm_input"));
        }
        Ok(())
    }

    fn effective_epsilon(&self) -> f32 {
        let scale = (-self.curvature).sqrt();
        self.epsilon * (1.0 + scale * 0.1)
    }

    fn curvature_scale(&self) -> f32 {
        (-self.curvature).sqrt()
    }

    fn project_normed(&self, value: f32) -> (f32, f32, f32) {
        let gain = self.projector_gain.get();
        if gain <= f32::EPSILON {
            return (value, 1.0, 0.0);
        }
        let scale = self.curvature_scale();
        let scaled = (value * scale).clamp(-TANH_CLAMP, TANH_CLAMP);
        let tanh = scaled.tanh();
        let radial = tanh / scale;
        let sech_sq = 1.0 - tanh * tanh;
        let jacobian = (1.0 - gain) + gain * sech_sq;
        let blended = (1.0 - gain) * value + gain * radial;
        let radius = tanh.abs() / scale;
        (blended, jacobian, radius)
    }
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
        let gamma = self.gamma.value().data();

        for feature in 0..features {
            let mut sum_grad = 0.0f32;
            let mut sum_grad_norm = 0.0f32;
            for row in 0..batch {
                let idx = row * features + feature;
                let normed = (input.data()[idx] - mean[feature]) * inv_std[feature];
                let g = grad_output.data()[idx];
                let g_gamma = g * gamma[feature];
                sum_grad += g_gamma;
                sum_grad_norm += g_gamma * normed;
                grad_gamma[feature] += g * normed;
                grad_beta[feature] += g;
            }
            for row in 0..batch {
                let idx = row * features + feature;
                let normed = (input.data()[idx] - mean[feature]) * inv_std[feature];
                let g = grad_output.data()[idx];
                let g_gamma = g * gamma[feature];
                let term =
                    (batch as f32 * g_gamma - sum_grad - normed * sum_grad_norm) / batch as f32;
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

impl Module for ZSpaceBatchNorm1d {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        let (batch, features) = input.shape();
        let mut output = Vec::with_capacity(batch * features);
        let mut normed_cache = vec![0.0f32; batch * features];
        let mut projected_cache = vec![0.0f32; batch * features];
        let mut jacobian_cache = vec![0.0f32; batch * features];
        let mut radius = vec![0.0f32; features];
        let gamma = self.gamma.value().data();
        let beta = self.beta.value().data();
        self.last_batch.set(Some(batch));
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
            (mean, variance)
        } else {
            let running_mean = self.running_mean.borrow();
            let running_var = self.running_var.borrow();
            (running_mean.data().to_vec(), running_var.data().to_vec())
        };

        let inv_std: Vec<f32> = variance
            .iter()
            .map(|v| 1.0 / (v + self.epsilon).sqrt())
            .collect();
        *self.last_mean.borrow_mut() = Some(mean.clone());
        *self.last_inv_std.borrow_mut() = Some(inv_std.clone());

        for row in 0..batch {
            let slice = &input.data()[row * features..(row + 1) * features];
            for feature in 0..features {
                let idx = row * features + feature;
                let normed = (slice[feature] - mean[feature]) * inv_std[feature];
                let (projected, jacobian, feature_radius) = self.project_normed(normed);
                normed_cache[idx] = normed;
                projected_cache[idx] = projected;
                jacobian_cache[idx] = jacobian;
                radius[feature] += feature_radius;
                output.push(projected * gamma[feature] + beta[feature]);
            }
        }

        for value in radius.iter_mut() {
            *value /= batch as f32;
        }
        *self.last_radius.borrow_mut() = Some(radius);
        *self.last_normed.borrow_mut() = Some(normed_cache);
        *self.last_projected.borrow_mut() = Some(projected_cache.clone());
        *self.last_jacobian.borrow_mut() = Some(jacobian_cache);

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
                label: "zspace_batchnorm_backward_eval",
            });
        }

        if self.last_mean.borrow().is_none() {
            return Err(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_mean",
            });
        }
        let inv_std = self
            .last_inv_std
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_invstd",
            })?;
        let normed = self
            .last_normed
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_normed",
            })?;
        let projected = self
            .last_projected
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_projected",
            })?;
        let jacobian = self
            .last_jacobian
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_jacobian",
            })?;

        let (batch, features) = input.shape();
        let gamma = self.gamma.value().data();
        let mut grad_input = vec![0.0f32; batch * features];
        let mut grad_gamma = vec![0.0f32; features];
        let mut grad_beta = vec![0.0f32; features];
        let mut grad_normed_cache = vec![0.0f32; batch * features];

        for feature in 0..features {
            let mut sum_grad = 0.0f32;
            let mut sum_grad_norm = 0.0f32;
            for row in 0..batch {
                let idx = row * features + feature;
                let go = grad_output.data()[idx];
                let proj = projected[idx];
                grad_gamma[feature] += go * proj;
                grad_beta[feature] += go;
                let go_gamma = go * gamma[feature];
                let grad_norm = go_gamma * jacobian[idx];
                grad_normed_cache[idx] = grad_norm;
                sum_grad += grad_norm;
                sum_grad_norm += grad_norm * normed[idx];
            }
            let inv = inv_std[feature];
            for row in 0..batch {
                let idx = row * features + feature;
                let grad_norm = grad_normed_cache[idx];
                let term = (batch as f32 * grad_norm - sum_grad - normed[idx] * sum_grad_norm)
                    / batch as f32;
                grad_input[idx] = term * inv;
            }
        }

        let grad_gamma_tensor = Tensor::from_vec(1, features, grad_gamma)?;
        let grad_beta_tensor = Tensor::from_vec(1, features, grad_beta)?;
        self.gamma.accumulate_euclidean(&grad_gamma_tensor)?;
        self.beta.accumulate_euclidean(&grad_beta_tensor)?;

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

impl Module for ZSpaceLayerNorm {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        let (rows, features) = input.shape();
        let gamma = self.gamma.value().data();
        let beta = self.beta.value().data();
        let epsilon = self.effective_epsilon();
        let mut output = Vec::with_capacity(rows * features);
        let mut normed_cache = vec![0.0f32; rows * features];
        let mut projected_cache = vec![0.0f32; rows * features];
        let mut jacobian_cache = vec![0.0f32; rows * features];
        let mut radius = vec![0.0f32; rows];
        let mut means = vec![0.0f32; rows];
        let mut inv_std = vec![0.0f32; rows];

        for row in 0..rows {
            let offset = row * features;
            let slice = &input.data()[offset..offset + features];
            let mean: f32 = slice.iter().copied().sum::<f32>() / features as f32;
            let variance: f32 = slice
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / features as f32;
            let denom = (variance + epsilon).sqrt();
            let inv = 1.0 / denom.max(f32::MIN_POSITIVE);
            means[row] = mean;
            inv_std[row] = inv;
            let mut row_radius = 0.0f32;
            for feature in 0..features {
                let idx = offset + feature;
                let normed = (slice[feature] - mean) * inv;
                let (projected, jacobian, feature_radius) = self.project_normed(normed);
                normed_cache[idx] = normed;
                projected_cache[idx] = projected;
                jacobian_cache[idx] = jacobian;
                row_radius += feature_radius;
                output.push(projected * gamma[feature] + beta[feature]);
            }
            radius[row] = row_radius / features as f32;
        }

        self.last_batch.set(Some(rows));
        *self.last_mean.borrow_mut() = Some(means);
        *self.last_inv_std.borrow_mut() = Some(inv_std);
        *self.last_normed.borrow_mut() = Some(normed_cache);
        *self.last_projected.borrow_mut() = Some(projected_cache.clone());
        *self.last_jacobian.borrow_mut() = Some(jacobian_cache);
        *self.last_radius.borrow_mut() = Some(radius);

        Tensor::from_vec(rows, features, output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, features) = input.shape();
        if self.last_batch.get().unwrap_or_default() != rows {
            return Err(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_batch",
            });
        }
        let inv_std = self
            .last_inv_std
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_invstd",
            })?;
        let normed = self
            .last_normed
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_normed",
            })?;
        let projected = self
            .last_projected
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_projected",
            })?;
        let jacobian = self
            .last_jacobian
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_jacobian",
            })?;

        let gamma = self.gamma.value().data();
        let mut grad_input = vec![0.0f32; rows * features];
        let mut grad_gamma = vec![0.0f32; features];
        let mut grad_beta = vec![0.0f32; features];
        let mut grad_normed_cache = vec![0.0f32; rows * features];

        for (row, &inv) in inv_std.iter().enumerate() {
            let offset = row * features;
            let mut sum_grad = 0.0f32;
            let mut sum_grad_norm = 0.0f32;
            for feature in 0..features {
                let idx = offset + feature;
                let go = grad_output.data()[idx];
                let proj = projected[idx];
                grad_gamma[feature] += go * proj;
                grad_beta[feature] += go;
                let go_gamma = go * gamma[feature];
                let grad_norm = go_gamma * jacobian[idx];
                grad_normed_cache[idx] = grad_norm;
                sum_grad += grad_norm;
                sum_grad_norm += grad_norm * normed[idx];
            }
            for feature in 0..features {
                let idx = offset + feature;
                let grad_norm = grad_normed_cache[idx];
                let norm_value = normed[idx];
                let term = (features as f32 * grad_norm - sum_grad - norm_value * sum_grad_norm)
                    / features as f32;
                grad_input[idx] = term * inv;
            }
        }

        let grad_gamma_tensor = Tensor::from_vec(1, features, grad_gamma)?;
        let grad_beta_tensor = Tensor::from_vec(1, features, grad_beta)?;
        self.gamma.accumulate_euclidean(&grad_gamma_tensor)?;
        self.beta.accumulate_euclidean(&grad_beta_tensor)?;

        Tensor::from_vec(rows, features, grad_input)
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
        input.layer_norm_affine(
            self.gamma.value(),
            self.beta.value(),
            self.effective_epsilon(),
        )
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

    #[test]
    fn batch_norm_backward_respects_gamma_scaling() {
        let mut layer = BatchNorm1d::new("bn", 2, 0.1, 1e-5).unwrap();
        {
            let gamma = layer.gamma.value_mut();
            for value in gamma.data_mut() {
                *value = 1.5;
            }
        }
        let input = Tensor::from_vec(2, 2, vec![0.5, -1.0, 0.25, 1.5]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.2, -0.3, -0.4, 0.6]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();

        let batch = input.shape().0;
        let features = input.shape().1;
        let mut mean = vec![0.0f32; features];
        for row in 0..batch {
            let slice = &input.data()[row * features..(row + 1) * features];
            for (feature, value) in slice.iter().enumerate() {
                mean[feature] += *value;
            }
        }
        for value in mean.iter_mut() {
            *value /= batch as f32;
        }
        let mut variance = vec![0.0f32; features];
        for row in 0..batch {
            let slice = &input.data()[row * features..(row + 1) * features];
            for (feature, value) in slice.iter().enumerate() {
                let centered = *value - mean[feature];
                variance[feature] += centered * centered;
            }
        }
        for value in variance.iter_mut() {
            *value /= batch as f32;
        }

        let epsilon = layer.epsilon();
        let gamma = layer.gamma.value().data();
        let mut expected = vec![0.0f32; batch * features];
        for feature in 0..features {
            let inv_std = 1.0 / (variance[feature] + epsilon).sqrt();
            let mut sum_grad = 0.0f32;
            let mut sum_grad_norm = 0.0f32;
            for row in 0..batch {
                let idx = row * features + feature;
                let normed = (input.data()[idx] - mean[feature]) * inv_std;
                let go = grad_output.data()[idx];
                let go_gamma = go * gamma[feature];
                sum_grad += go_gamma;
                sum_grad_norm += go_gamma * normed;
            }
            for row in 0..batch {
                let idx = row * features + feature;
                let normed = (input.data()[idx] - mean[feature]) * inv_std;
                let go = grad_output.data()[idx];
                let go_gamma = go * gamma[feature];
                let term =
                    (batch as f32 * go_gamma - sum_grad - normed * sum_grad_norm) / batch as f32;
                expected[idx] = term * inv_std;
            }
        }

        for (observed, anticipated) in grad_input.data().iter().zip(expected.iter()) {
            assert!((observed - anticipated).abs() < 1e-5);
        }
    }

    #[test]
    fn zspace_batch_norm_projects_into_ball() {
        let layer = ZSpaceBatchNorm1d::new("bnz", 3, -1.0, 0.2, 1e-5)
            .unwrap()
            .with_projector_gain(1.0)
            .unwrap();
        let input = Tensor::from_vec(
            4,
            3,
            vec![
                1.5, -2.0, 0.25, // sample 0
                0.5, -0.75, 1.2, // sample 1
                -1.0, 0.3, -0.9, // sample 2
                2.5, -1.5, 0.0, // sample 3
            ],
        )
        .unwrap();
        let output = layer.forward(&input).unwrap();
        let scale = (-layer.curvature()).sqrt();
        let limit = 1.0 / scale + 1e-5;
        for value in output.data() {
            assert!(value.is_finite());
        }
        for value in output.data() {
            assert!(value.abs() <= limit);
        }
        let radius = layer.last_ball_radius().unwrap();
        assert_eq!(radius.len(), 3);
        for value in radius {
            assert!(value >= 0.0);
            assert!(value <= limit);
        }
    }

    #[test]
    fn zspace_batch_norm_backward_matches_numeric_gradients() {
        let mut layer = ZSpaceBatchNorm1d::new("bnz", 2, -0.75, 0.5, 1e-4)
            .unwrap()
            .with_projector_gain(0.85)
            .unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.3, -0.6, 1.2, 0.4]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.2, -0.1, 0.05, 0.3]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();

        let eps = 1e-3;
        let mut numeric = [0.0f32; 4];
        let base = input.data().to_vec();
        for (idx, numeric_grad) in numeric.iter_mut().enumerate() {
            let mut plus = base.clone();
            plus[idx] += eps;
            let mut minus = base.clone();
            minus[idx] -= eps;
            let tensor_plus = Tensor::from_vec(2, 2, plus).unwrap();
            let tensor_minus = Tensor::from_vec(2, 2, minus).unwrap();
            let loss_plus = {
                let layer = ZSpaceBatchNorm1d::new("bnz", 2, -0.75, 0.5, 1e-4)
                    .unwrap()
                    .with_projector_gain(0.85)
                    .unwrap();
                let output = layer.forward(&tensor_plus).unwrap();
                output
                    .data()
                    .iter()
                    .zip(grad_output.data())
                    .map(|(o, g)| o * g)
                    .sum::<f32>()
            };
            let loss_minus = {
                let layer = ZSpaceBatchNorm1d::new("bnz", 2, -0.75, 0.5, 1e-4)
                    .unwrap()
                    .with_projector_gain(0.85)
                    .unwrap();
                let output = layer.forward(&tensor_minus).unwrap();
                output
                    .data()
                    .iter()
                    .zip(grad_output.data())
                    .map(|(o, g)| o * g)
                    .sum::<f32>()
            };
            *numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);
        }

        for (observed, expected) in grad_input.data().iter().zip(numeric.iter()) {
            assert!((observed - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn zspace_batch_norm_exposes_telemetry() {
        let layer = ZSpaceBatchNorm1d::new("bnz", 3, -1.0, 0.2, 1e-5)
            .unwrap()
            .with_projector_gain(0.6)
            .unwrap();
        let input = Tensor::from_vec(
            2,
            3,
            vec![
                0.25, -0.5, 0.75, // sample 0
                1.25, -1.0, 0.5, // sample 1
            ],
        )
        .unwrap();
        let _ = layer.forward(&input).unwrap();
        let telemetry = layer.telemetry().expect("telemetry available");
        assert_eq!(telemetry.batch(), 2);
        assert_eq!(telemetry.features(), 3);
        assert_eq!(telemetry.mean().len(), 3);
        assert_eq!(telemetry.inv_std().len(), 3);
        assert_eq!(telemetry.radius().len(), 3);
        assert_eq!(telemetry.normed().len(), 6);
        assert_eq!(telemetry.projected().len(), 6);
        assert_eq!(telemetry.jacobian().len(), 6);
        for &value in telemetry.jacobian() {
            assert!(value.is_finite());
            assert!(value > 0.0);
        }
    }

    #[test]
    fn zspace_batch_norm_adapts_projector_gain() {
        let layer = ZSpaceBatchNorm1d::new("bnz", 2, -1.0, 0.2, 1e-5)
            .unwrap()
            .with_projector_gain(0.15)
            .unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.5, -0.25, -0.75, 1.0]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let before = layer.projector_gain();
        let telemetry = layer.telemetry().expect("telemetry available");
        let avg_radius: f32 =
            telemetry.radius().iter().copied().sum::<f32>() / telemetry.radius().len() as f32;
        let target = (avg_radius + 0.25).min(0.9);
        let updated = layer
            .adapt_projector_gain(target, 0.8)
            .expect("gain adaptation succeeds");
        if target > avg_radius {
            assert!(updated >= before);
        } else if target < avg_radius {
            assert!(updated <= before);
        }
        assert!(updated <= 1.0);
        assert!((layer.projector_gain() - updated).abs() < f32::EPSILON);
    }

    #[test]
    fn zspace_layer_norm_projects_into_ball() {
        let layer = ZSpaceLayerNorm::new("lnz", 3, -1.0, 1e-5)
            .unwrap()
            .with_projector_gain(1.0)
            .unwrap();
        let input = Tensor::from_vec(
            2,
            3,
            vec![
                1.5, -2.0, 0.5, // sample 0
                -0.75, 0.25, 1.0, // sample 1
            ],
        )
        .unwrap();
        let output = layer.forward(&input).unwrap();
        let scale = (-layer.curvature()).sqrt();
        let limit = 1.0 / scale + 1e-5;
        for value in output.data() {
            assert!(value.is_finite());
            assert!(value.abs() <= limit);
        }
        let radius = layer.last_ball_radius().unwrap();
        assert_eq!(radius.len(), 2);
        for value in radius {
            assert!(value >= 0.0);
            assert!(value <= limit);
        }
    }

    #[test]
    fn zspace_layer_norm_backward_matches_numeric_gradients() {
        let mut layer = ZSpaceLayerNorm::new("lnz", 3, -0.9, 1e-5)
            .unwrap()
            .with_projector_gain(0.85)
            .unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.3, -0.6, 1.2, 0.4, -1.1, 0.8]).unwrap();
        let grad_output = Tensor::from_vec(2, 3, vec![0.2, -0.1, 0.05, 0.3, -0.2, 0.4]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();

        let eps = 1e-3;
        let mut numeric = [0.0f32; 6];
        let base = input.data().to_vec();
        for (idx, numeric_grad) in numeric.iter_mut().enumerate() {
            let mut plus = base.clone();
            plus[idx] += eps;
            let mut minus = base.clone();
            minus[idx] -= eps;
            let tensor_plus = Tensor::from_vec(2, 3, plus).unwrap();
            let tensor_minus = Tensor::from_vec(2, 3, minus).unwrap();
            let loss_plus = {
                let layer = ZSpaceLayerNorm::new("lnz", 3, -0.9, 1e-5)
                    .unwrap()
                    .with_projector_gain(0.85)
                    .unwrap();
                let output = layer.forward(&tensor_plus).unwrap();
                output
                    .data()
                    .iter()
                    .zip(grad_output.data())
                    .map(|(o, g)| o * g)
                    .sum::<f32>()
            };
            let loss_minus = {
                let layer = ZSpaceLayerNorm::new("lnz", 3, -0.9, 1e-5)
                    .unwrap()
                    .with_projector_gain(0.85)
                    .unwrap();
                let output = layer.forward(&tensor_minus).unwrap();
                output
                    .data()
                    .iter()
                    .zip(grad_output.data())
                    .map(|(o, g)| o * g)
                    .sum::<f32>()
            };
            *numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);
        }

        for (observed, expected) in grad_input.data().iter().zip(numeric.iter()) {
            assert!((observed - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn zspace_layer_norm_exposes_telemetry() {
        let layer = ZSpaceLayerNorm::new("lnz", 4, -1.1, 1e-4)
            .unwrap()
            .with_projector_gain(0.5)
            .unwrap();
        let input = Tensor::from_vec(
            3,
            4,
            vec![
                0.1, -0.2, 0.3, -0.4, // sample 0
                0.5, -0.6, 0.7, -0.8, // sample 1
                0.2, -0.1, 0.4, -0.3, // sample 2
            ],
        )
        .unwrap();
        let _ = layer.forward(&input).unwrap();
        let telemetry = layer.telemetry().expect("telemetry available");
        assert_eq!(telemetry.batch(), 3);
        assert_eq!(telemetry.features(), 4);
        assert_eq!(telemetry.mean().len(), 3);
        assert_eq!(telemetry.inv_std().len(), 3);
        assert_eq!(telemetry.radius().len(), 3);
        assert_eq!(telemetry.normed().len(), 12);
        assert_eq!(telemetry.projected().len(), 12);
        assert_eq!(telemetry.jacobian().len(), 12);
        for &value in telemetry.radius() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn zspace_layer_norm_adapts_projector_gain() {
        let layer = ZSpaceLayerNorm::new("lnz", 2, -0.8, 1e-5)
            .unwrap()
            .with_projector_gain(0.2)
            .unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.5, -0.25, -0.75, 1.0]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let before = layer.projector_gain();
        let telemetry = layer.telemetry().expect("telemetry available");
        let avg_radius: f32 =
            telemetry.radius().iter().copied().sum::<f32>() / telemetry.radius().len() as f32;
        let target = (avg_radius + 0.3).min(0.95);
        let updated = layer
            .adapt_projector_gain(target, 0.75)
            .expect("gain adaptation succeeds");
        if target > avg_radius {
            assert!(updated >= before);
        } else if target < avg_radius {
            assert!(updated <= before);
        }
        assert!(updated <= 1.0);
        assert!((layer.projector_gain() - updated).abs() < f32::EPSILON);
    }
}
