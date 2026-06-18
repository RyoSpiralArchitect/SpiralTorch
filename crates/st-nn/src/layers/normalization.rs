// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::{
    current_backend_policy, current_layer_norm_backend, current_tensor_util_backend_for_values,
};
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorError, TensorUtilBackend};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

const TANH_CLAMP: f32 = 25.0;

fn current_layer_norm_requested_label() -> &'static str {
    current_backend_policy()
        .map(|policy| policy.layer_norm_backend_label())
        .unwrap_or("auto")
}

fn emit_normalization_backward_meta(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    requested_backend: &'static str,
    epsilon: f32,
    zspace: bool,
    gradient_scale: Option<f32>,
    input_gradient_backend: Option<String>,
    input_gradient_reduction_backend: Option<&'static str>,
    normalization_backend: Option<&'static str>,
    affine_gradient_backend: Option<String>,
) {
    let input_gradient_axis = match op_name {
        "batch_norm_backward" | "zspace_batch_norm_backward" => "batch",
        "layer_norm_backward" | "zspace_layer_norm_backward" => "feature",
        _ => "unknown",
    };
    let input_gradient_formula = if zspace {
        "projected_affine_norm_vjp"
    } else {
        "affine_norm_vjp"
    };
    emit_tensor_op(op_name, &[rows, cols], &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        serde_json::json!({
            "backend": "hybrid",
            "requested_backend": requested_backend,
            "rows": rows,
            "cols": cols,
            "epsilon": epsilon,
            "input_gradient_backend": input_gradient_backend,
            "input_gradient_axis": input_gradient_axis,
            "input_gradient_formula": input_gradient_formula,
            "affine_gradient_backend": affine_gradient_backend,
            "gradient_scale": gradient_scale,
            "parameter_gradient_scale": gradient_scale,
            "input_gradient_scale": if gradient_scale.is_some() {
                Some(1.0f32)
            } else {
                None
            },
            "input_gradient_reduction_backend": input_gradient_reduction_backend,
            "normalization_backend": normalization_backend,
            "flags": {
                "zspace": zspace,
                "affine": true,
            }
        })
    });
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn backend_affine_gradients(
    grad_output: &Tensor,
    affine_input: &Tensor,
    gradient_scale: f32,
) -> PureResult<(Tensor, Tensor, String)> {
    let (_, cols) = grad_output.shape();
    let reduction_backend = current_tensor_util_backend_for_values(grad_output.data().len());
    let grad_gamma_product = grad_output.hadamard_with_backend(affine_input, reduction_backend)?;
    let grad_gamma = Tensor::from_vec(
        1,
        cols,
        grad_gamma_product.try_sum_axis0_scaled_with_backend(gradient_scale, reduction_backend)?,
    )?;
    let grad_beta = Tensor::from_vec(
        1,
        cols,
        grad_output.try_sum_axis0_scaled_with_backend(gradient_scale, reduction_backend)?,
    )?;
    Ok((
        grad_gamma,
        grad_beta,
        tensor_util_backend_label(reduction_backend).to_string(),
    ))
}

fn backend_batch_axis_stats(
    input: &Tensor,
    mean_label: &'static str,
    centered_label: &'static str,
    variance_sum_label: &'static str,
    variance_label: &'static str,
) -> PureResult<(Vec<f32>, Vec<f32>)> {
    let (batch, features) = input.shape();
    let scale = 1.0 / batch as f32;
    let reduction_backend = current_tensor_util_backend_for_values(input.data().len());
    let mean = input.try_sum_axis0_scaled_with_backend(scale, reduction_backend)?;
    validate_finite_slice(mean_label, &mean)?;

    let mut centered = Vec::with_capacity(input.data().len());
    for row in 0..batch {
        let slice = &input.data()[row * features..(row + 1) * features];
        for (feature, &value) in slice.iter().enumerate() {
            let centered_value = value - mean[feature];
            validate_finite_value(centered_label, centered_value)?;
            let squared = centered_value * centered_value;
            validate_finite_value(variance_sum_label, squared)?;
            centered.push(centered_value);
        }
    }
    let centered = Tensor::from_vec(batch, features, centered)?;
    let squared = centered.hadamard_with_backend(&centered, reduction_backend)?;
    let variance = squared.try_sum_axis0_scaled_with_backend(scale, reduction_backend)?;
    validate_finite_slice(variance_label, &variance)?;
    Ok((mean, variance))
}

fn validate_finite_value(label: &'static str, value: f32) -> PureResult<()> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
}

fn checked_finite_value(label: &'static str, value: f32) -> PureResult<f32> {
    validate_finite_value(label, value)?;
    Ok(value)
}

fn checked_sum(label: &'static str, values: impl IntoIterator<Item = f32>) -> PureResult<f32> {
    let mut sum = 0.0f64;
    for value in values {
        validate_finite_value(label, value)?;
        sum += f64::from(value);
    }
    if !sum.is_finite() || sum.abs() > f64::from(f32::MAX) {
        let value = if sum.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(sum as f32)
}

fn checked_mean(label: &'static str, values: &[f32]) -> PureResult<f32> {
    if values.is_empty() {
        return Err(TensorError::EmptyInput(label));
    }
    let sum = checked_sum(label, values.iter().copied())?;
    checked_finite_value(label, sum / values.len() as f32)
}

fn checked_projector_gain_update(
    current_gain: f32,
    target_radius: f32,
    smoothing: f32,
    radii: &[f32],
    target_label: &'static str,
    smoothing_label: &'static str,
    radius_label: &'static str,
    gain_label: &'static str,
) -> PureResult<Option<f32>> {
    if !target_radius.is_finite() || target_radius < 0.0 {
        return Err(TensorError::NonFiniteValue {
            label: target_label,
            value: target_radius,
        });
    }
    if !smoothing.is_finite() || smoothing <= 0.0 {
        return Err(TensorError::NonFiniteValue {
            label: smoothing_label,
            value: smoothing,
        });
    }
    validate_finite_value(gain_label, current_gain)?;
    if radii.is_empty() {
        return Ok(None);
    }
    let avg_radius = checked_mean(radius_label, radii)?;
    let delta = smoothing * (target_radius - avg_radius);
    validate_finite_value(gain_label, delta)?;
    let gain = (current_gain + delta).clamp(0.0, 1.0);
    validate_finite_value(gain_label, gain)?;
    Ok(Some(gain))
}

fn validate_finite_slice(label: &'static str, values: &[f32]) -> PureResult<()> {
    for &value in values {
        validate_finite_value(label, value)?;
    }
    Ok(())
}

fn validate_finite_tensor(label: &'static str, tensor: &Tensor) -> PureResult<()> {
    validate_finite_slice(label, tensor.data())
}

fn relabel_non_finite<T>(result: PureResult<T>, label: &'static str) -> PureResult<T> {
    match result {
        Err(TensorError::NonFiniteValue { value, .. }) => {
            Err(TensorError::NonFiniteValue { label, value })
        }
        other => other,
    }
}

fn backend_grad_normed(
    grad_output: &Tensor,
    gamma: &[f32],
    jacobian: Option<&[f32]>,
    backend: TensorUtilBackend,
    label: &'static str,
) -> PureResult<Tensor> {
    let (rows, cols) = grad_output.shape();
    let mut grad_normed =
        relabel_non_finite(grad_output.mul_row_with_backend(gamma, backend), label)?;
    if let Some(jacobian) = jacobian {
        let jacobian = Tensor::from_vec(rows, cols, jacobian.to_vec())?;
        grad_normed =
            relabel_non_finite(grad_normed.hadamard_with_backend(&jacobian, backend), label)?;
    }
    validate_finite_tensor(label, &grad_normed)?;
    Ok(grad_normed)
}

fn backend_batch_axis_input_gradient(
    grad_normed: &Tensor,
    normed: &Tensor,
    inv_std: &[f32],
    divisor: usize,
    backend: TensorUtilBackend,
    label: &'static str,
) -> PureResult<Tensor> {
    if divisor == 0 {
        return Err(TensorError::EmptyInput(label));
    }
    if grad_normed.shape() != normed.shape() {
        return Err(TensorError::ShapeMismatch {
            left: grad_normed.shape(),
            right: normed.shape(),
        });
    }
    let (_, cols) = grad_normed.shape();
    if inv_std.len() != cols {
        return Err(TensorError::DataLength {
            expected: cols,
            got: inv_std.len(),
        });
    }
    validate_finite_tensor(label, grad_normed)?;
    validate_finite_tensor(label, normed)?;
    validate_finite_slice(label, inv_std)?;
    let sum_grad = relabel_non_finite(grad_normed.try_sum_axis0_with_backend(backend), label)?;
    validate_finite_slice(label, &sum_grad)?;
    let grad_normed_times_normed =
        relabel_non_finite(grad_normed.hadamard_with_backend(normed, backend), label)?;
    let sum_grad_norm = relabel_non_finite(
        grad_normed_times_normed.try_sum_axis0_with_backend(backend),
        label,
    )?;
    validate_finite_slice(label, &sum_grad_norm)?;
    let mut correction =
        relabel_non_finite(normed.mul_row_with_backend(&sum_grad_norm, backend), label)?;
    relabel_non_finite(
        correction.add_row_inplace_with_backend(&sum_grad, backend),
        label,
    )?;
    let correction = relabel_non_finite(
        correction.scale_with_backend(1.0 / divisor as f32, backend),
        label,
    )?;
    let mut output = grad_normed.clone();
    relabel_non_finite(
        output.add_scaled_with_backend(&correction, -1.0, backend),
        label,
    )?;
    let output = relabel_non_finite(output.mul_row_with_backend(inv_std, backend), label)?;
    validate_finite_tensor(label, &output)?;
    Ok(output)
}

fn backend_layer_axis_input_gradient(
    grad_normed: &Tensor,
    normed: &Tensor,
    inv_std: &[f32],
    backend: TensorUtilBackend,
    label: &'static str,
) -> PureResult<Tensor> {
    let (rows, cols) = grad_normed.shape();
    if inv_std.len() != rows {
        return Err(TensorError::DataLength {
            expected: rows,
            got: inv_std.len(),
        });
    }
    let grad_normed_t = relabel_non_finite(grad_normed.transpose_with_backend(backend), label)?;
    let normed_t = relabel_non_finite(normed.transpose_with_backend(backend), label)?;
    let output_t = backend_batch_axis_input_gradient(
        &grad_normed_t,
        &normed_t,
        inv_std,
        cols,
        backend,
        label,
    )?;
    let output = relabel_non_finite(output_t.transpose_with_backend(backend), label)?;
    validate_finite_tensor(label, &output)?;
    Ok(output)
}

fn inverse_stddev(label: &'static str, variance: &[f32], epsilon: f32) -> PureResult<Vec<f32>> {
    let mut inv_std = Vec::with_capacity(variance.len());
    for &value in variance {
        validate_finite_value(label, value)?;
        let denom = value + epsilon;
        validate_finite_value(label, denom)?;
        if denom <= 0.0 {
            return Err(TensorError::InvalidValue { label });
        }
        let inv = denom.sqrt().recip();
        validate_finite_value(label, inv)?;
        inv_std.push(inv);
    }
    Ok(inv_std)
}

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

    /// Returns `true` when the layer is operating in training mode.
    pub fn training(&self) -> bool {
        self.training.get()
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
        let telemetry = self.telemetry().ok_or(TensorError::InvalidValue {
            label: "zspace_batchnorm_telemetry",
        })?;
        let current = self.projector_gain.get();
        if let Some(gain) = checked_projector_gain_update(
            current,
            target_radius,
            smoothing,
            telemetry.radius(),
            "zspace_batchnorm_target_radius",
            "zspace_batchnorm_smoothing",
            "zspace_batchnorm_radius_mean",
            "zspace_batchnorm_projector_gain",
        )? {
            self.projector_gain.set(gain);
            Ok(gain)
        } else {
            Ok(current)
        }
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
        Ok(())
    }

    fn compute_stats(&self, input: &Tensor) -> PureResult<(Vec<f32>, Vec<f32>)> {
        backend_batch_axis_stats(
            input,
            "zspace_batchnorm_mean",
            "zspace_batchnorm_centered",
            "zspace_batchnorm_variance_sum",
            "zspace_batchnorm_variance",
        )
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
        let telemetry = self.telemetry().ok_or(TensorError::InvalidValue {
            label: "zspace_layernorm_telemetry",
        })?;
        let current = self.projector_gain.get();
        if let Some(gain) = checked_projector_gain_update(
            current,
            target_radius,
            smoothing,
            telemetry.radius(),
            "zspace_layernorm_target_radius",
            "zspace_layernorm_smoothing",
            "zspace_layernorm_radius_mean",
            "zspace_layernorm_projector_gain",
        )? {
            self.projector_gain.set(gain);
            Ok(gain)
        } else {
            Ok(current)
        }
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

    /// Returns `true` when the layer is operating in training mode.
    pub fn training(&self) -> bool {
        self.training.get()
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

    fn compute_stats(&self, input: &Tensor) -> PureResult<(Vec<f32>, Vec<f32>)> {
        backend_batch_axis_stats(
            input,
            "batchnorm_mean",
            "batchnorm_centered",
            "batchnorm_variance_sum",
            "batchnorm_variance",
        )
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        let (batch, features) = input.shape();
        validate_finite_tensor("batchnorm_input", input)?;
        let gamma = self.gamma.value().data();
        let beta = self.beta.value().data();
        validate_finite_slice("batchnorm_gamma", gamma)?;
        validate_finite_slice("batchnorm_beta", beta)?;
        if batch == 0 {
            *self.last_mean.borrow_mut() = Some(vec![0.0; features]);
            *self.last_inv_std.borrow_mut() = Some(vec![0.0; features]);
            return Tensor::zeros(batch, features);
        }
        let mut output = Vec::with_capacity(batch * features);
        let training = self.training.get();
        let mut next_running_mean = None;
        let mut next_running_var = None;
        let (mean, variance) = if training {
            let (mean, variance) = self.compute_stats(input)?;
            let current_running_mean = self.running_mean.borrow().data().to_vec();
            let current_running_var = self.running_var.borrow().data().to_vec();
            validate_finite_slice("batchnorm_running_mean", &current_running_mean)?;
            validate_finite_slice("batchnorm_running_var", &current_running_var)?;
            let mut updated_mean = current_running_mean;
            let mut updated_var = current_running_var;
            for idx in 0..features {
                updated_mean[idx] =
                    self.momentum * mean[idx] + (1.0 - self.momentum) * updated_mean[idx];
                validate_finite_value("batchnorm_running_mean", updated_mean[idx])?;
                updated_var[idx] =
                    self.momentum * variance[idx] + (1.0 - self.momentum) * updated_var[idx];
                validate_finite_value("batchnorm_running_var", updated_var[idx])?;
            }
            next_running_mean = Some(updated_mean);
            next_running_var = Some(updated_var);
            (mean, variance)
        } else {
            let running_mean = self.running_mean.borrow();
            let running_var = self.running_var.borrow();
            validate_finite_slice("batchnorm_running_mean", running_mean.data())?;
            validate_finite_slice("batchnorm_running_var", running_var.data())?;
            (running_mean.data().to_vec(), running_var.data().to_vec())
        };

        let inv_std = inverse_stddev("batchnorm_invstd", &variance, self.epsilon)?;

        for row in 0..batch {
            let slice = &input.data()[row * features..(row + 1) * features];
            for feature in 0..features {
                let normed = (slice[feature] - mean[feature]) * inv_std[feature];
                validate_finite_value("batchnorm_normed", normed)?;
                let value = normed * gamma[feature] + beta[feature];
                validate_finite_value("batchnorm_output", value)?;
                output.push(value);
            }
        }
        let tensor = Tensor::from_vec(batch, features, output)?;
        if training {
            if let Some(updated) = next_running_mean {
                self.running_mean
                    .borrow_mut()
                    .data_mut()
                    .copy_from_slice(&updated);
            }
            if let Some(updated) = next_running_var {
                self.running_var
                    .borrow_mut()
                    .data_mut()
                    .copy_from_slice(&updated);
            }
            *self.last_mean.borrow_mut() = Some(mean);
            *self.last_inv_std.borrow_mut() = Some(inv_std);
        }
        Ok(tensor)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        validate_finite_tensor("batchnorm_backward_input", input)?;
        validate_finite_tensor("batchnorm_backward_grad_output", grad_output)?;
        if !self.training.get() {
            return Err(TensorError::InvalidValue {
                label: "batchnorm_backward_eval",
            });
        }
        let (batch, features) = input.shape();
        if batch == 0 {
            let output = Tensor::zeros(batch, features)?;
            emit_normalization_backward_meta(
                "batch_norm_backward",
                batch,
                features,
                "auto",
                self.epsilon,
                false,
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(output);
        }
        let mean = self
            .last_mean
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "batchnorm_cached_mean",
            })?;
        validate_finite_slice("batchnorm_cached_mean", &mean)?;
        let inv_std = self
            .last_inv_std
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "batchnorm_cached_invstd",
            })?;
        validate_finite_slice("batchnorm_cached_invstd", &inv_std)?;

        let gamma = self.gamma.value().data();
        validate_finite_slice("batchnorm_gamma", gamma)?;
        let input_gradient_backend =
            current_tensor_util_backend_for_values(batch.saturating_mul(features));
        let normed_bias = mean
            .iter()
            .zip(inv_std.iter())
            .map(|(&mean, &inv)| {
                let bias = -mean * inv;
                validate_finite_value("batchnorm_backward_normed", bias)?;
                Ok(bias)
            })
            .collect::<PureResult<Vec<_>>>()?;
        let normed = relabel_non_finite(
            input.row_affine_with_backend(&inv_std, &normed_bias, input_gradient_backend),
            "batchnorm_backward_normed",
        )?;
        validate_finite_tensor("batchnorm_backward_normed", &normed)?;
        let grad_normed = backend_grad_normed(
            grad_output,
            gamma,
            None,
            input_gradient_backend,
            "batchnorm_backward_grad_normed",
        )?;
        let output = backend_batch_axis_input_gradient(
            &grad_normed,
            &normed,
            &inv_std,
            batch,
            input_gradient_backend,
            "batchnorm_backward_input_grad",
        )?;

        let gradient_scale = 1.0 / batch as f32;
        let (grad_gamma, grad_beta, affine_gradient_backend) =
            backend_affine_gradients(grad_output, &normed, gradient_scale)?;
        self.gamma.accumulate_euclidean(&grad_gamma)?;
        self.beta.accumulate_euclidean(&grad_beta)?;
        emit_normalization_backward_meta(
            "batch_norm_backward",
            batch,
            features,
            "auto",
            self.epsilon,
            false,
            Some(gradient_scale),
            Some(tensor_util_backend_label(input_gradient_backend).to_string()),
            Some(tensor_util_backend_label(input_gradient_backend)),
            Some(tensor_util_backend_label(input_gradient_backend)),
            Some(affine_gradient_backend),
        );
        Ok(output)
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

    fn state_dict(&self) -> PureResult<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        self.visit_parameters(&mut |param| {
            state.insert(param.name().to_string(), param.value().clone());
            Ok(())
        })?;

        let base = self
            .gamma
            .name()
            .strip_suffix("::gamma")
            .unwrap_or(self.gamma.name());
        state.insert(
            format!("{base}::running_mean"),
            (*self.running_mean.borrow()).clone(),
        );
        state.insert(
            format!("{base}::running_var"),
            (*self.running_var.borrow()).clone(),
        );
        Ok(state)
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

        let base = self
            .gamma
            .name()
            .strip_suffix("::gamma")
            .unwrap_or(self.gamma.name());
        if let Some(value) = state.get(&format!("{base}::running_mean")) {
            let expected = { self.running_mean.borrow().shape() };
            if value.shape() != expected {
                return Err(TensorError::ShapeMismatch {
                    left: value.shape(),
                    right: expected,
                });
            }
            *self.running_mean.borrow_mut() = value.clone();
        }
        if let Some(value) = state.get(&format!("{base}::running_var")) {
            let expected = { self.running_var.borrow().shape() };
            if value.shape() != expected {
                return Err(TensorError::ShapeMismatch {
                    left: value.shape(),
                    right: expected,
                });
            }
            *self.running_var.borrow_mut() = value.clone();
        }
        Ok(())
    }

    fn set_training(&mut self, training: bool) -> PureResult<()> {
        BatchNorm1d::set_training(self, training);
        Ok(())
    }
}

impl Module for ZSpaceBatchNorm1d {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        let (batch, features) = input.shape();
        validate_finite_tensor("zspace_batchnorm_input", input)?;
        let gamma = self.gamma.value().data();
        let beta = self.beta.value().data();
        validate_finite_slice("zspace_batchnorm_gamma", gamma)?;
        validate_finite_slice("zspace_batchnorm_beta", beta)?;
        if batch == 0 {
            let tensor = Tensor::zeros(batch, features)?;
            self.last_batch.set(Some(batch));
            *self.last_mean.borrow_mut() = Some(vec![0.0; features]);
            *self.last_inv_std.borrow_mut() = Some(vec![0.0; features]);
            *self.last_normed.borrow_mut() = Some(Vec::new());
            *self.last_projected.borrow_mut() = Some(Vec::new());
            *self.last_jacobian.borrow_mut() = Some(Vec::new());
            *self.last_radius.borrow_mut() = Some(Vec::new());
            return Ok(tensor);
        }
        let mut output = Vec::with_capacity(batch * features);
        let mut normed_cache = vec![0.0f32; batch * features];
        let mut projected_cache = vec![0.0f32; batch * features];
        let mut jacobian_cache = vec![0.0f32; batch * features];
        let mut radius = vec![0.0f32; features];
        let training = self.training.get();
        let mut next_running_mean = None;
        let mut next_running_var = None;
        let (mean, variance) = if training {
            let (mean, variance) = self.compute_stats(input)?;
            let current_running_mean = self.running_mean.borrow().data().to_vec();
            let current_running_var = self.running_var.borrow().data().to_vec();
            validate_finite_slice("zspace_batchnorm_running_mean", &current_running_mean)?;
            validate_finite_slice("zspace_batchnorm_running_var", &current_running_var)?;
            let mut updated_mean = current_running_mean;
            let mut updated_var = current_running_var;
            for idx in 0..features {
                updated_mean[idx] =
                    self.momentum * mean[idx] + (1.0 - self.momentum) * updated_mean[idx];
                validate_finite_value("zspace_batchnorm_running_mean", updated_mean[idx])?;
                updated_var[idx] =
                    self.momentum * variance[idx] + (1.0 - self.momentum) * updated_var[idx];
                validate_finite_value("zspace_batchnorm_running_var", updated_var[idx])?;
            }
            next_running_mean = Some(updated_mean);
            next_running_var = Some(updated_var);
            (mean, variance)
        } else {
            let running_mean = self.running_mean.borrow();
            let running_var = self.running_var.borrow();
            validate_finite_slice("zspace_batchnorm_running_mean", running_mean.data())?;
            validate_finite_slice("zspace_batchnorm_running_var", running_var.data())?;
            (running_mean.data().to_vec(), running_var.data().to_vec())
        };

        let inv_std = inverse_stddev("zspace_batchnorm_invstd", &variance, self.epsilon)?;

        for row in 0..batch {
            let slice = &input.data()[row * features..(row + 1) * features];
            for feature in 0..features {
                let idx = row * features + feature;
                let normed = (slice[feature] - mean[feature]) * inv_std[feature];
                validate_finite_value("zspace_batchnorm_normed", normed)?;
                let (projected, jacobian, feature_radius) = self.project_normed(normed);
                validate_finite_value("zspace_batchnorm_projected", projected)?;
                validate_finite_value("zspace_batchnorm_jacobian", jacobian)?;
                validate_finite_value("zspace_batchnorm_radius", feature_radius)?;
                normed_cache[idx] = normed;
                projected_cache[idx] = projected;
                jacobian_cache[idx] = jacobian;
                radius[feature] += feature_radius;
                validate_finite_value("zspace_batchnorm_radius", radius[feature])?;
                let value = projected * gamma[feature] + beta[feature];
                validate_finite_value("zspace_batchnorm_output", value)?;
                output.push(value);
            }
        }

        for value in radius.iter_mut() {
            *value /= batch as f32;
            validate_finite_value("zspace_batchnorm_radius", *value)?;
        }

        let tensor = Tensor::from_vec(batch, features, output)?;
        if training {
            if let Some(updated) = next_running_mean {
                self.running_mean
                    .borrow_mut()
                    .data_mut()
                    .copy_from_slice(&updated);
            }
            if let Some(updated) = next_running_var {
                self.running_var
                    .borrow_mut()
                    .data_mut()
                    .copy_from_slice(&updated);
            }
        }
        self.last_batch.set(Some(batch));
        *self.last_mean.borrow_mut() = Some(mean);
        *self.last_inv_std.borrow_mut() = Some(inv_std);
        *self.last_radius.borrow_mut() = Some(radius);
        *self.last_normed.borrow_mut() = Some(normed_cache);
        *self.last_projected.borrow_mut() = Some(projected_cache);
        *self.last_jacobian.borrow_mut() = Some(jacobian_cache);

        Ok(tensor)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        validate_finite_tensor("zspace_batchnorm_backward_input", input)?;
        validate_finite_tensor("zspace_batchnorm_backward_grad_output", grad_output)?;
        if !self.training.get() {
            return Err(TensorError::InvalidValue {
                label: "zspace_batchnorm_backward_eval",
            });
        }

        let (batch, features) = input.shape();
        if batch == 0 {
            let output = Tensor::zeros(batch, features)?;
            emit_normalization_backward_meta(
                "zspace_batch_norm_backward",
                batch,
                features,
                "auto",
                self.epsilon,
                true,
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(output);
        }

        let mean = self
            .last_mean
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_mean",
            })?;
        if mean.len() != features {
            return Err(TensorError::DataLength {
                expected: features,
                got: mean.len(),
            });
        }
        validate_finite_slice("zspace_batchnorm_cached_mean", &mean)?;
        let inv_std = self
            .last_inv_std
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_invstd",
            })?;
        if inv_std.len() != features {
            return Err(TensorError::DataLength {
                expected: features,
                got: inv_std.len(),
            });
        }
        validate_finite_slice("zspace_batchnorm_cached_invstd", &inv_std)?;
        let normed = self
            .last_normed
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_normed",
            })?;
        if normed.len() != batch * features {
            return Err(TensorError::DataLength {
                expected: batch * features,
                got: normed.len(),
            });
        }
        validate_finite_slice("zspace_batchnorm_cached_normed", &normed)?;
        let projected = self
            .last_projected
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_projected",
            })?;
        if projected.len() != batch * features {
            return Err(TensorError::DataLength {
                expected: batch * features,
                got: projected.len(),
            });
        }
        validate_finite_slice("zspace_batchnorm_cached_projected", &projected)?;
        let jacobian = self
            .last_jacobian
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_batchnorm_cached_jacobian",
            })?;
        if jacobian.len() != batch * features {
            return Err(TensorError::DataLength {
                expected: batch * features,
                got: jacobian.len(),
            });
        }
        validate_finite_slice("zspace_batchnorm_cached_jacobian", &jacobian)?;

        let gamma = self.gamma.value().data();
        validate_finite_slice("zspace_batchnorm_gamma", gamma)?;
        let input_gradient_backend =
            current_tensor_util_backend_for_values(batch.saturating_mul(features));
        let normed = Tensor::from_vec(batch, features, normed)?;
        let grad_normed = backend_grad_normed(
            grad_output,
            gamma,
            Some(&jacobian),
            input_gradient_backend,
            "zspace_batchnorm_backward_grad_norm",
        )?;
        let output = backend_batch_axis_input_gradient(
            &grad_normed,
            &normed,
            &inv_std,
            batch,
            input_gradient_backend,
            "zspace_batchnorm_backward_input_grad",
        )?;

        let gradient_scale = 1.0 / batch as f32;
        let projected = Tensor::from_vec(batch, features, projected)?;
        let (grad_gamma_tensor, grad_beta_tensor, affine_gradient_backend) =
            backend_affine_gradients(grad_output, &projected, gradient_scale)?;
        self.gamma.accumulate_euclidean(&grad_gamma_tensor)?;
        self.beta.accumulate_euclidean(&grad_beta_tensor)?;

        emit_normalization_backward_meta(
            "zspace_batch_norm_backward",
            batch,
            features,
            "auto",
            self.epsilon,
            true,
            Some(gradient_scale),
            Some(tensor_util_backend_label(input_gradient_backend).to_string()),
            Some(tensor_util_backend_label(input_gradient_backend)),
            Some(tensor_util_backend_label(input_gradient_backend)),
            Some(affine_gradient_backend),
        );
        Ok(output)
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

    fn state_dict(&self) -> PureResult<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        self.visit_parameters(&mut |param| {
            state.insert(param.name().to_string(), param.value().clone());
            Ok(())
        })?;

        let base = self
            .gamma
            .name()
            .strip_suffix("::gamma")
            .unwrap_or(self.gamma.name());
        state.insert(
            format!("{base}::running_mean"),
            (*self.running_mean.borrow()).clone(),
        );
        state.insert(
            format!("{base}::running_var"),
            (*self.running_var.borrow()).clone(),
        );
        state.insert(
            format!("{base}::projector_gain"),
            Tensor::from_vec(1, 1, vec![self.projector_gain.get()])?,
        );
        Ok(state)
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

        let base = self
            .gamma
            .name()
            .strip_suffix("::gamma")
            .unwrap_or(self.gamma.name());
        if let Some(value) = state.get(&format!("{base}::running_mean")) {
            let expected = { self.running_mean.borrow().shape() };
            if value.shape() != expected {
                return Err(TensorError::ShapeMismatch {
                    left: value.shape(),
                    right: expected,
                });
            }
            *self.running_mean.borrow_mut() = value.clone();
        }
        if let Some(value) = state.get(&format!("{base}::running_var")) {
            let expected = { self.running_var.borrow().shape() };
            if value.shape() != expected {
                return Err(TensorError::ShapeMismatch {
                    left: value.shape(),
                    right: expected,
                });
            }
            *self.running_var.borrow_mut() = value.clone();
        }
        if let Some(value) = state.get(&format!("{base}::projector_gain")) {
            if value.shape() != (1, 1) {
                return Err(TensorError::ShapeMismatch {
                    left: value.shape(),
                    right: (1, 1),
                });
            }
            let gain = value.data().first().copied().unwrap_or(1.0);
            self.set_projector_gain(gain)?;
        }
        Ok(())
    }

    fn set_training(&mut self, training: bool) -> PureResult<()> {
        ZSpaceBatchNorm1d::set_training(self, training);
        Ok(())
    }
}

impl Module for ZSpaceLayerNorm {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        let (rows, features) = input.shape();
        validate_finite_tensor("zspace_layernorm_input", input)?;
        let gamma = self.gamma.value().data();
        let beta = self.beta.value().data();
        validate_finite_slice("zspace_layernorm_gamma", gamma)?;
        validate_finite_slice("zspace_layernorm_beta", beta)?;
        let epsilon = self.effective_epsilon();
        validate_finite_value("zspace_layernorm_epsilon", epsilon)?;
        if rows == 0 {
            let tensor = Tensor::zeros(rows, features)?;
            self.last_batch.set(Some(rows));
            *self.last_mean.borrow_mut() = Some(Vec::new());
            *self.last_inv_std.borrow_mut() = Some(Vec::new());
            *self.last_normed.borrow_mut() = Some(Vec::new());
            *self.last_projected.borrow_mut() = Some(Vec::new());
            *self.last_jacobian.borrow_mut() = Some(Vec::new());
            *self.last_radius.borrow_mut() = Some(Vec::new());
            return Ok(tensor);
        }
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
            let mut mean_sum = 0.0f32;
            for &value in slice {
                mean_sum += value;
                validate_finite_value("zspace_layernorm_mean_sum", mean_sum)?;
            }
            let mean = mean_sum / features as f32;
            validate_finite_value("zspace_layernorm_mean", mean)?;
            let mut variance_sum = 0.0f32;
            for &value in slice {
                let centered = value - mean;
                validate_finite_value("zspace_layernorm_centered", centered)?;
                let squared = centered * centered;
                validate_finite_value("zspace_layernorm_variance_sum", squared)?;
                variance_sum += squared;
                validate_finite_value("zspace_layernorm_variance_sum", variance_sum)?;
            }
            let variance = variance_sum / features as f32;
            validate_finite_value("zspace_layernorm_variance", variance)?;
            let denom = variance + epsilon;
            validate_finite_value("zspace_layernorm_invstd", denom)?;
            if denom <= 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "zspace_layernorm_invstd",
                });
            }
            let inv = denom.sqrt().recip();
            validate_finite_value("zspace_layernorm_invstd", inv)?;
            means[row] = mean;
            inv_std[row] = inv;
            let mut row_radius = 0.0f32;
            for feature in 0..features {
                let idx = offset + feature;
                let normed = (slice[feature] - mean) * inv;
                validate_finite_value("zspace_layernorm_normed", normed)?;
                let (projected, jacobian, feature_radius) = self.project_normed(normed);
                validate_finite_value("zspace_layernorm_projected", projected)?;
                validate_finite_value("zspace_layernorm_jacobian", jacobian)?;
                validate_finite_value("zspace_layernorm_radius", feature_radius)?;
                normed_cache[idx] = normed;
                projected_cache[idx] = projected;
                jacobian_cache[idx] = jacobian;
                row_radius += feature_radius;
                validate_finite_value("zspace_layernorm_radius", row_radius)?;
                let value = projected * gamma[feature] + beta[feature];
                validate_finite_value("zspace_layernorm_output", value)?;
                output.push(value);
            }
            radius[row] = row_radius / features as f32;
            validate_finite_value("zspace_layernorm_radius", radius[row])?;
        }

        let tensor = Tensor::from_vec(rows, features, output)?;
        self.last_batch.set(Some(rows));
        *self.last_mean.borrow_mut() = Some(means);
        *self.last_inv_std.borrow_mut() = Some(inv_std);
        *self.last_normed.borrow_mut() = Some(normed_cache);
        *self.last_projected.borrow_mut() = Some(projected_cache);
        *self.last_jacobian.borrow_mut() = Some(jacobian_cache);
        *self.last_radius.borrow_mut() = Some(radius);

        Ok(tensor)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        validate_finite_tensor("zspace_layernorm_backward_input", input)?;
        validate_finite_tensor("zspace_layernorm_backward_grad_output", grad_output)?;
        let (rows, features) = input.shape();
        if rows == 0 {
            let output = Tensor::zeros(rows, features)?;
            emit_normalization_backward_meta(
                "zspace_layer_norm_backward",
                rows,
                features,
                current_layer_norm_requested_label(),
                self.effective_epsilon(),
                true,
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(output);
        }
        if self.last_batch.get().unwrap_or_default() != rows {
            return Err(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_batch",
            });
        }
        let mean = self
            .last_mean
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_mean",
            })?;
        if mean.len() != rows {
            return Err(TensorError::DataLength {
                expected: rows,
                got: mean.len(),
            });
        }
        validate_finite_slice("zspace_layernorm_cached_mean", &mean)?;
        let inv_std = self
            .last_inv_std
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_invstd",
            })?;
        if inv_std.len() != rows {
            return Err(TensorError::DataLength {
                expected: rows,
                got: inv_std.len(),
            });
        }
        validate_finite_slice("zspace_layernorm_cached_invstd", &inv_std)?;
        let normed = self
            .last_normed
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_normed",
            })?;
        if normed.len() != rows * features {
            return Err(TensorError::DataLength {
                expected: rows * features,
                got: normed.len(),
            });
        }
        validate_finite_slice("zspace_layernorm_cached_normed", &normed)?;
        let projected = self
            .last_projected
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_projected",
            })?;
        if projected.len() != rows * features {
            return Err(TensorError::DataLength {
                expected: rows * features,
                got: projected.len(),
            });
        }
        validate_finite_slice("zspace_layernorm_cached_projected", &projected)?;
        let jacobian = self
            .last_jacobian
            .borrow()
            .clone()
            .ok_or(TensorError::InvalidValue {
                label: "zspace_layernorm_cached_jacobian",
            })?;
        if jacobian.len() != rows * features {
            return Err(TensorError::DataLength {
                expected: rows * features,
                got: jacobian.len(),
            });
        }
        validate_finite_slice("zspace_layernorm_cached_jacobian", &jacobian)?;

        let gamma = self.gamma.value().data();
        validate_finite_slice("zspace_layernorm_gamma", gamma)?;
        let input_gradient_backend =
            current_tensor_util_backend_for_values(rows.saturating_mul(features));
        let normed = Tensor::from_vec(rows, features, normed)?;
        let grad_normed = backend_grad_normed(
            grad_output,
            gamma,
            Some(&jacobian),
            input_gradient_backend,
            "zspace_layernorm_backward_grad_norm",
        )?;
        let output = backend_layer_axis_input_gradient(
            &grad_normed,
            &normed,
            &inv_std,
            input_gradient_backend,
            "zspace_layernorm_backward_input_grad",
        )?;

        let gradient_scale = 1.0 / rows as f32;
        let projected = Tensor::from_vec(rows, features, projected)?;
        let (grad_gamma_tensor, grad_beta_tensor, affine_gradient_backend) =
            backend_affine_gradients(grad_output, &projected, gradient_scale)?;
        self.gamma.accumulate_euclidean(&grad_gamma_tensor)?;
        self.beta.accumulate_euclidean(&grad_beta_tensor)?;

        emit_normalization_backward_meta(
            "zspace_layer_norm_backward",
            rows,
            features,
            current_layer_norm_requested_label(),
            self.effective_epsilon(),
            true,
            Some(gradient_scale),
            Some(tensor_util_backend_label(input_gradient_backend).to_string()),
            Some(tensor_util_backend_label(input_gradient_backend)),
            Some(tensor_util_backend_label(input_gradient_backend)),
            Some(affine_gradient_backend),
        );
        Ok(output)
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
        if rows == 0 {
            return Tensor::zeros(rows, cols);
        }
        input.layer_norm_affine_with_backend(
            self.gamma.value(),
            self.beta.value(),
            self.effective_epsilon(),
            current_layer_norm_backend(),
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
        validate_finite_tensor("layernorm_backward_input", input)?;
        validate_finite_tensor("layernorm_backward_grad_output", grad_output)?;
        if rows == 0 {
            let output = Tensor::zeros(rows, cols)?;
            emit_normalization_backward_meta(
                "layer_norm_backward",
                rows,
                cols,
                current_layer_norm_requested_label(),
                epsilon,
                false,
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(output);
        }
        let gamma = self.gamma.value().data().to_vec();
        validate_finite_slice("layernorm_gamma", &gamma)?;
        let mut normed_values = vec![0.0f32; rows * cols];
        let mut inv_std = vec![0.0f32; rows];

        for r in 0..rows {
            let offset = r * cols;
            let slice = &input.data()[offset..offset + cols];
            let mean =
                checked_sum("layernorm_backward_mean_sum", slice.iter().copied())? / cols as f32;
            validate_finite_value("layernorm_backward_mean", mean)?;
            let variance_sum = checked_sum(
                "layernorm_backward_variance_sum",
                slice.iter().map(|&value| {
                    let centered = value - mean;
                    centered * centered
                }),
            )?;
            let variance =
                checked_finite_value("layernorm_backward_variance", variance_sum / cols as f32)?;
            let denom =
                checked_finite_value("layernorm_backward_denom", (variance + epsilon).sqrt())?;
            if denom <= 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "layernorm_backward_denom",
                });
            }
            let inv_denom = checked_finite_value("layernorm_backward_inv_denom", 1.0 / denom)?;
            inv_std[r] = inv_denom;
            for c in 0..cols {
                let normed = checked_finite_value(
                    "layernorm_backward_normed",
                    (slice[c] - mean) * inv_denom,
                )?;
                normed_values[offset + c] = normed;
            }
        }

        let gradient_scale = 1.0 / rows as f32;
        validate_finite_slice("layernorm_backward_normed", &normed_values)?;
        validate_finite_slice("layernorm_backward_inv_denom", &inv_std)?;
        let normed = Tensor::from_vec(rows, cols, normed_values)?;
        let input_gradient_backend =
            current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let grad_normed = backend_grad_normed(
            grad_output,
            &gamma,
            None,
            input_gradient_backend,
            "layernorm_backward_grad_normed",
        )?;
        let output = backend_layer_axis_input_gradient(
            &grad_normed,
            &normed,
            &inv_std,
            input_gradient_backend,
            "layernorm_backward_input_grad",
        )?;
        let (grad_gamma_tensor, grad_beta_tensor, affine_gradient_backend) =
            backend_affine_gradients(grad_output, &normed, gradient_scale)?;
        self.gamma.accumulate_euclidean(&grad_gamma_tensor)?;
        self.beta.accumulate_euclidean(&grad_beta_tensor)?;

        emit_normalization_backward_meta(
            "layer_norm_backward",
            rows,
            cols,
            current_layer_norm_requested_label(),
            epsilon,
            false,
            Some(gradient_scale),
            Some("hybrid".to_string()),
            Some(tensor_util_backend_label(input_gradient_backend)),
            Some("cpu"),
            Some(affine_gradient_backend),
        );
        Ok(output)
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
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    #[cfg(feature = "wgpu")]
    use st_tensor::backend::wgpu_dense;
    use std::sync::{Arc, Mutex, OnceLock};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn demo_input() -> Tensor {
        Tensor::from_vec(2, 3, vec![0.5, -1.0, 1.5, 2.0, -0.5, 0.0]).unwrap()
    }

    fn assert_same_tensor(left: &Tensor, right: &Tensor) {
        assert_eq!(left.shape(), right.shape());
        assert_eq!(left.data(), right.data());
    }

    fn assert_close_slice(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (idx, (&left, &right)) in left.iter().zip(right.iter()).enumerate() {
            let delta = (left - right).abs();
            assert!(
                delta <= 1.0e-6,
                "mismatch at {idx}: left={left} right={right} delta={delta}"
            );
        }
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
    fn layer_norm_backward_matches_finite_difference_with_nonuniform_gamma() {
        let mut layer = LayerNorm::new("demo", 3, -1.0, 1e-5).unwrap();
        layer
            .gamma
            .value_mut()
            .data_mut()
            .copy_from_slice(&[1.5, -0.75, 0.35]);
        layer
            .beta
            .value_mut()
            .data_mut()
            .copy_from_slice(&[0.1, -0.2, 0.05]);
        let input_values = vec![0.45, -0.8, 1.25, -0.35, 0.9, -1.1];
        let input = Tensor::from_vec(2, 3, input_values.clone()).unwrap();
        let grad_output = Tensor::from_vec(2, 3, vec![0.2, -0.15, 0.35, -0.25, 0.1, 0.3]).unwrap();

        let grad_input = layer.backward(&input, &grad_output).unwrap();
        let analytic = grad_input.data()[0];

        let epsilon = 1.0e-3f32;
        let loss_at = |values: Vec<f32>| {
            let tensor = Tensor::from_vec(2, 3, values).unwrap();
            let output = layer.forward(&tensor).unwrap();
            output
                .data()
                .iter()
                .zip(grad_output.data().iter())
                .map(|(&value, &grad)| value * grad)
                .sum::<f32>()
        };
        let mut plus = input_values.clone();
        plus[0] += epsilon;
        let mut minus = input_values;
        minus[0] -= epsilon;
        let finite_difference = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon);

        assert!(
            (analytic - finite_difference).abs() < 2.5e-3,
            "analytic={analytic} finite_difference={finite_difference}"
        );
    }

    #[test]
    fn normalization_affine_gradients_are_batch_normalized() {
        let mut ln_single = LayerNorm::new("ln", 3, -1.0, 1e-5).unwrap();
        let input_single = Tensor::from_vec(1, 3, vec![0.45, -0.8, 1.25]).unwrap();
        let grad_single = Tensor::from_vec(1, 3, vec![0.2, -0.15, 0.35]).unwrap();
        let _ = ln_single.backward(&input_single, &grad_single).unwrap();

        let mut ln_repeated = LayerNorm::new("ln", 3, -1.0, 1e-5).unwrap();
        let input_repeated =
            Tensor::from_vec(2, 3, vec![0.45, -0.8, 1.25, 0.45, -0.8, 1.25]).unwrap();
        let grad_repeated =
            Tensor::from_vec(2, 3, vec![0.2, -0.15, 0.35, 0.2, -0.15, 0.35]).unwrap();
        let _ = ln_repeated
            .backward(&input_repeated, &grad_repeated)
            .unwrap();
        assert_close_slice(
            ln_single.gamma.gradient().unwrap().data(),
            ln_repeated.gamma.gradient().unwrap().data(),
        );
        assert_close_slice(
            ln_single.beta.gradient().unwrap().data(),
            ln_repeated.beta.gradient().unwrap().data(),
        );

        let bn_input = Tensor::from_vec(2, 2, vec![0.2, -0.3, 1.0, 0.5]).unwrap();
        let bn_grad = Tensor::from_vec(2, 2, vec![0.1, -0.2, 0.05, 0.3]).unwrap();
        let mut bn_base = BatchNorm1d::new("bn", 2, 0.2, 1e-5).unwrap();
        let _ = bn_base.forward(&bn_input).unwrap();
        let _ = bn_base.backward(&bn_input, &bn_grad).unwrap();

        let bn_input_repeated =
            Tensor::from_vec(4, 2, vec![0.2, -0.3, 1.0, 0.5, 0.2, -0.3, 1.0, 0.5]).unwrap();
        let bn_grad_repeated =
            Tensor::from_vec(4, 2, vec![0.1, -0.2, 0.05, 0.3, 0.1, -0.2, 0.05, 0.3]).unwrap();
        let mut bn_repeated = BatchNorm1d::new("bn", 2, 0.2, 1e-5).unwrap();
        let _ = bn_repeated.forward(&bn_input_repeated).unwrap();
        let _ = bn_repeated
            .backward(&bn_input_repeated, &bn_grad_repeated)
            .unwrap();
        assert_close_slice(
            bn_base.gamma.gradient().unwrap().data(),
            bn_repeated.gamma.gradient().unwrap().data(),
        );
        assert_close_slice(
            bn_base.beta.gradient().unwrap().data(),
            bn_repeated.beta.gradient().unwrap().data(),
        );

        let mut zln_single = ZSpaceLayerNorm::new("zln", 3, -0.9, 1e-5)
            .unwrap()
            .with_projector_gain(0.7)
            .unwrap();
        let _ = zln_single.forward(&input_single).unwrap();
        let _ = zln_single.backward(&input_single, &grad_single).unwrap();
        let mut zln_repeated = ZSpaceLayerNorm::new("zln", 3, -0.9, 1e-5)
            .unwrap()
            .with_projector_gain(0.7)
            .unwrap();
        let _ = zln_repeated.forward(&input_repeated).unwrap();
        let _ = zln_repeated
            .backward(&input_repeated, &grad_repeated)
            .unwrap();
        assert_close_slice(
            zln_single.gamma.gradient().unwrap().data(),
            zln_repeated.gamma.gradient().unwrap().data(),
        );
        assert_close_slice(
            zln_single.beta.gradient().unwrap().data(),
            zln_repeated.beta.gradient().unwrap().data(),
        );

        let mut zbn_base = ZSpaceBatchNorm1d::new("zbn", 2, -0.75, 0.5, 1e-4)
            .unwrap()
            .with_projector_gain(0.6)
            .unwrap();
        let _ = zbn_base.forward(&bn_input).unwrap();
        let _ = zbn_base.backward(&bn_input, &bn_grad).unwrap();
        let mut zbn_repeated = ZSpaceBatchNorm1d::new("zbn", 2, -0.75, 0.5, 1e-4)
            .unwrap()
            .with_projector_gain(0.6)
            .unwrap();
        let _ = zbn_repeated.forward(&bn_input_repeated).unwrap();
        let _ = zbn_repeated
            .backward(&bn_input_repeated, &bn_grad_repeated)
            .unwrap();
        assert_close_slice(
            zbn_base.gamma.gradient().unwrap().data(),
            zbn_repeated.gamma.gradient().unwrap().data(),
        );
        assert_close_slice(
            zbn_base.beta.gradient().unwrap().data(),
            zbn_repeated.beta.gradient().unwrap().data(),
        );
    }

    #[test]
    fn layer_norm_backward_emits_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut layer = LayerNorm::new("demo", 3, -1.0, 1e-5).unwrap();
        let input = demo_input();
        let grad_output = Tensor::from_vec(2, 3, vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let _ = layer.backward(&input, &grad_output).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "layer_norm_backward" && data["rows"] == 2 && data["cols"] == 3
            })
            .expect("layer norm backward metadata event");
        assert_eq!(meta.1["backend"], "hybrid");
        assert_eq!(meta.1["requested_backend"], "auto");
        assert_eq!(meta.1["input_gradient_backend"], "hybrid");
        assert_eq!(meta.1["input_gradient_reduction_backend"], "auto");
        assert_eq!(meta.1["normalization_backend"], "cpu");
        assert_eq!(meta.1["input_gradient_axis"], "feature");
        assert_eq!(meta.1["input_gradient_formula"], "affine_norm_vjp");
        assert_eq!(meta.1["affine_gradient_backend"], "auto");
        assert_eq!(meta.1["gradient_scale"], 0.5);
        assert_eq!(meta.1["parameter_gradient_scale"], 0.5);
        assert_eq!(meta.1["input_gradient_scale"], 1.0);
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "hadamard" && data["rows"] == 2 && data["cols"] == 3
        }));
        let scaled_reductions = events
            .iter()
            .filter(|(op_name, data)| {
                *op_name == "sum_axis0_scaled"
                    && data["rows"] == 2
                    && data["cols"] == 3
                    && data["scale"] == 0.5
            })
            .count();
        assert!(
            scaled_reductions >= 2,
            "expected gamma and beta reductions, saw {scaled_reductions}"
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn layer_norm_forced_wgpu_routes_input_and_affine_reducers() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut layer = LayerNorm::new("demo", 3, -1.0, 1e-5).unwrap();
        let input = demo_input();
        let grad_output = Tensor::from_vec(2, 3, vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3]).unwrap();
        {
            let _guard = push_backend_policy(policy);
            let _ = layer.forward(&input).unwrap();
            let _ = layer.backward(&input, &grad_output).unwrap();
        }

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "layer_norm_backward" && data["rows"] == 2 && data["cols"] == 3
            })
            .expect("layer norm backward metadata event");
        assert_eq!(meta.1["backend"], "hybrid");
        assert_eq!(meta.1["requested_backend"], "wgpu");
        assert_eq!(meta.1["input_gradient_backend"], "hybrid");
        assert_eq!(meta.1["input_gradient_reduction_backend"], "wgpu");
        assert_eq!(meta.1["normalization_backend"], "cpu");
        assert_eq!(meta.1["input_gradient_axis"], "feature");
        assert_eq!(meta.1["input_gradient_formula"], "affine_norm_vjp");
        assert_eq!(meta.1["affine_gradient_backend"], "wgpu");
        assert!(events
            .iter()
            .any(|(op_name, data)| *op_name == "hadamard" && data["backend"] == "wgpu_dense"));
        let scaled_wgpu_reductions = events
            .iter()
            .filter(|(op_name, data)| {
                *op_name == "sum_axis0_scaled"
                    && data["rows"] == 2
                    && data["cols"] == 3
                    && data["backend"] == "wgpu_dense"
            })
            .count();
        assert!(
            scaled_wgpu_reductions >= 2,
            "expected WGPU gamma and beta reductions, saw {scaled_wgpu_reductions}"
        );
        assert!(events
            .iter()
            .any(|(op_name, data)| { *op_name == "sum_axis0" && data["backend"] == "wgpu_dense" }));
        assert!(events
            .iter()
            .any(|(op_name, data)| { *op_name == "mul_row" && data["backend"] == "wgpu_dense" }));
    }

    #[test]
    fn normalization_backward_affine_reducers_emit_tensor_utility_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut batch_norm = BatchNorm1d::new("bn", 2, 0.2, 1e-5).unwrap();
        let bn_input = Tensor::from_vec(3, 2, vec![0.2, -0.3, 1.0, 0.5, -0.4, 0.9]).unwrap();
        let bn_grad = Tensor::from_vec(3, 2, vec![0.1, -0.2, 0.05, 0.3, -0.15, 0.25]).unwrap();
        let _ = batch_norm.forward(&bn_input).unwrap();
        let _ = batch_norm.backward(&bn_input, &bn_grad).unwrap();

        let mut zspace_batch_norm = ZSpaceBatchNorm1d::new("zbn", 2, -0.75, 0.5, 1e-4)
            .unwrap()
            .with_projector_gain(0.6)
            .unwrap();
        let zbn_input = Tensor::from_vec(2, 2, vec![0.2, -0.3, 1.0, 0.5]).unwrap();
        let zbn_grad = Tensor::from_vec(2, 2, vec![0.1, -0.2, 0.05, 0.3]).unwrap();
        let _ = zspace_batch_norm.forward(&zbn_input).unwrap();
        let _ = zspace_batch_norm.backward(&zbn_input, &zbn_grad).unwrap();

        let mut zspace_layer_norm = ZSpaceLayerNorm::new("zln", 3, -0.9, 1e-5)
            .unwrap()
            .with_projector_gain(0.7)
            .unwrap();
        let zln_input = Tensor::from_vec(2, 3, vec![0.45, -0.8, 1.25, -0.2, 0.4, 0.9]).unwrap();
        let zln_grad = Tensor::from_vec(2, 3, vec![0.2, -0.15, 0.35, -0.25, 0.1, 0.3]).unwrap();
        let _ = zspace_layer_norm.forward(&zln_input).unwrap();
        let _ = zspace_layer_norm.backward(&zln_input, &zln_grad).unwrap();

        st_tensor::set_tensor_op_meta_observer(previous);
        let events = events.lock().unwrap();
        for (rows, cols) in [(3, 2), (2, 2), (2, 3)] {
            assert!(
                events.iter().any(|(op_name, data)| {
                    *op_name == "hadamard" && data["rows"] == rows && data["cols"] == cols
                }),
                "missing hadamard reducer meta for {rows}x{cols}"
            );
            let scaled_reductions = events
                .iter()
                .filter(|(op_name, data)| {
                    *op_name == "sum_axis0_scaled" && data["rows"] == rows && data["cols"] == cols
                })
                .count();
            assert!(
                scaled_reductions >= 2,
                "expected gamma/beta reductions for {rows}x{cols}, saw {scaled_reductions}"
            );
        }
    }

    #[test]
    fn layer_norm_empty_batch_forward_backward_skips_parameter_updates() {
        let mut layer = LayerNorm::new("demo", 3, -1.0, 1e-5).unwrap();
        let input = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (0, 3));
        assert!(output.data().is_empty());

        let grad_output = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (0, 3));
        assert!(grad_input.data().is_empty());
        assert!(layer.gamma.gradient().is_none());
        assert!(layer.beta.gradient().is_none());
    }

    #[test]
    fn layer_norm_backward_rejects_overflow_without_accumulating() {
        let mut layer = LayerNorm::new("ln", 3, -1.0, 1e-5).unwrap();
        let input = Tensor::from_vec(1, 3, vec![f32::MAX, -f32::MAX, 0.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 3, vec![0.1, -0.2, 0.3]).unwrap();

        let err = layer.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "layernorm_backward_variance_sum",
                value,
            } if value.is_infinite()
        ));
        assert!(layer.gamma.gradient().is_none());
        assert!(layer.beta.gradient().is_none());
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
    fn batch_norm_forward_stats_emit_tensor_utility_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let batch_norm = BatchNorm1d::new("bn", 2, 0.2, 1e-5).unwrap();
        let bn_input = Tensor::from_vec(3, 2, vec![0.2, -0.3, 1.0, 0.5, -0.4, 0.9]).unwrap();
        let _ = batch_norm.forward(&bn_input).unwrap();

        let zspace_batch_norm = ZSpaceBatchNorm1d::new("zbn", 2, -0.75, 0.5, 1e-4)
            .unwrap()
            .with_projector_gain(0.6)
            .unwrap();
        let zbn_input = Tensor::from_vec(2, 2, vec![0.2, -0.3, 1.0, 0.5]).unwrap();
        let _ = zspace_batch_norm.forward(&zbn_input).unwrap();

        st_tensor::set_tensor_op_meta_observer(previous);
        let events = events.lock().unwrap();
        for (rows, cols) in [(3, 2), (2, 2)] {
            assert!(
                events.iter().any(|(op_name, data)| {
                    *op_name == "hadamard" && data["rows"] == rows && data["cols"] == cols
                }),
                "missing variance hadamard meta for {rows}x{cols}"
            );
            let scaled_reductions = events
                .iter()
                .filter(|(op_name, data)| {
                    *op_name == "sum_axis0_scaled" && data["rows"] == rows && data["cols"] == cols
                })
                .count();
            assert!(
                scaled_reductions >= 2,
                "expected mean and variance reductions for {rows}x{cols}, saw {scaled_reductions}"
            );
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
    fn batch_norm_empty_batch_forward_backward_skips_stats_and_updates() {
        let mut layer = BatchNorm1d::new("bn", 2, 0.2, 1e-5).unwrap();
        let running_mean_before = layer.running_mean.borrow().clone();
        let running_var_before = layer.running_var.borrow().clone();
        let input = Tensor::from_vec(0, 2, Vec::new()).unwrap();

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (0, 2));
        assert!(output.data().is_empty());
        assert_same_tensor(&layer.running_mean.borrow(), &running_mean_before);
        assert_same_tensor(&layer.running_var.borrow(), &running_var_before);

        let grad_output = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (0, 2));
        assert!(grad_input.data().is_empty());
        assert!(layer.gamma.gradient().is_none());
        assert!(layer.beta.gradient().is_none());
    }

    #[test]
    fn batch_norm_rejects_non_finite_input_without_mutating_stats() {
        let layer = BatchNorm1d::new("bn", 2, 0.2, 1e-5).unwrap();
        let running_mean_before = layer.running_mean.borrow().clone();
        let running_var_before = layer.running_var.borrow().clone();
        let input = Tensor::from_vec(2, 2, vec![0.2, f32::NAN, 1.0, 0.5]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "batchnorm_input",
                value,
            } if value.is_nan()
        ));
        assert_same_tensor(&layer.running_mean.borrow(), &running_mean_before);
        assert_same_tensor(&layer.running_var.borrow(), &running_var_before);
        assert!(layer.last_mean.borrow().is_none());
        assert!(layer.last_inv_std.borrow().is_none());
    }

    #[test]
    fn batch_norm_rejects_overflowing_variance_without_mutating_stats() {
        let layer = BatchNorm1d::new("bn", 1, 0.2, 1e-5).unwrap();
        let running_mean_before = layer.running_mean.borrow().clone();
        let running_var_before = layer.running_var.borrow().clone();
        let input = Tensor::from_vec(2, 1, vec![f32::MAX, -f32::MAX]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "batchnorm_variance_sum",
                value,
            } if value.is_infinite()
        ));
        assert_same_tensor(&layer.running_mean.borrow(), &running_mean_before);
        assert_same_tensor(&layer.running_var.borrow(), &running_var_before);
        assert!(layer.last_mean.borrow().is_none());
        assert!(layer.last_inv_std.borrow().is_none());
    }

    #[test]
    fn batch_norm_rejects_non_finite_backward_grad_without_accumulating() {
        let mut layer = BatchNorm1d::new("bn", 2, 0.2, 1e-5).unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.2, -0.3, 1.0, 0.5]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.1, f32::NAN, 0.05, 0.3]).unwrap();

        let err = layer.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "batchnorm_backward_grad_output",
                value,
            } if value.is_nan()
        ));
        assert!(layer.gamma.gradient().is_none());
        assert!(layer.beta.gradient().is_none());
    }

    #[test]
    fn zspace_batch_norm_rejects_non_finite_input_without_mutating_state() {
        let layer = ZSpaceBatchNorm1d::new("bnz", 2, -1.0, 0.2, 1e-5).unwrap();
        let running_mean_before = layer.running_mean.borrow().clone();
        let running_var_before = layer.running_var.borrow().clone();
        let input = Tensor::from_vec(2, 2, vec![0.2, f32::NAN, 1.0, 0.5]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_batchnorm_input",
                value,
            } if value.is_nan()
        ));
        assert_same_tensor(&layer.running_mean.borrow(), &running_mean_before);
        assert_same_tensor(&layer.running_var.borrow(), &running_var_before);
        assert!(layer.telemetry().is_none());
    }

    #[test]
    fn zspace_batch_norm_rejects_overflowing_variance_without_mutating_state() {
        let layer = ZSpaceBatchNorm1d::new("bnz", 1, -1.0, 0.2, 1e-5).unwrap();
        let running_mean_before = layer.running_mean.borrow().clone();
        let running_var_before = layer.running_var.borrow().clone();
        let input = Tensor::from_vec(2, 1, vec![f32::MAX, -f32::MAX]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_batchnorm_variance_sum",
                value,
            } if value.is_infinite()
        ));
        assert_same_tensor(&layer.running_mean.borrow(), &running_mean_before);
        assert_same_tensor(&layer.running_var.borrow(), &running_var_before);
        assert!(layer.telemetry().is_none());
    }

    #[test]
    fn zspace_batch_norm_rejects_non_finite_backward_grad_without_accumulating() {
        let mut layer = ZSpaceBatchNorm1d::new("bnz", 2, -1.0, 0.2, 1e-5).unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.2, -0.3, 1.0, 0.5]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.1, f32::NAN, 0.05, 0.3]).unwrap();

        let err = layer.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_batchnorm_backward_grad_output",
                value,
            } if value.is_nan()
        ));
        assert!(layer.gamma.gradient().is_none());
        assert!(layer.beta.gradient().is_none());
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
        let avg_radius = checked_mean("zspace_batchnorm_radius_mean", telemetry.radius()).unwrap();
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
    fn zspace_batch_norm_rejects_bad_projector_radius_without_commit() {
        let layer = ZSpaceBatchNorm1d::new("bnz", 2, -1.0, 0.2, 1e-5)
            .unwrap()
            .with_projector_gain(0.35)
            .unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.5, -0.25, -0.75, 1.0]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let before = layer.projector_gain();
        *layer.last_radius.borrow_mut() = Some(vec![f32::MAX, f32::MAX]);

        let err = layer.adapt_projector_gain(0.5, 0.8).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_batchnorm_radius_mean",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(layer.projector_gain(), before);
    }

    #[test]
    fn zspace_batch_norm_empty_batch_keeps_telemetry_without_updates() {
        let mut layer = ZSpaceBatchNorm1d::new("bnz", 2, -1.0, 0.2, 1e-5)
            .unwrap()
            .with_projector_gain(0.4)
            .unwrap();
        let running_mean_before = layer.running_mean.borrow().clone();
        let running_var_before = layer.running_var.borrow().clone();
        let input = Tensor::from_vec(0, 2, Vec::new()).unwrap();

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (0, 2));
        assert!(output.data().is_empty());
        assert_same_tensor(&layer.running_mean.borrow(), &running_mean_before);
        assert_same_tensor(&layer.running_var.borrow(), &running_var_before);

        let telemetry = layer.telemetry().expect("empty telemetry available");
        assert_eq!(telemetry.batch(), 0);
        assert_eq!(telemetry.features(), 2);
        assert_eq!(telemetry.mean().len(), 2);
        assert_eq!(telemetry.inv_std().len(), 2);
        assert!(telemetry.normed().is_empty());
        assert!(telemetry.projected().is_empty());
        assert!(telemetry.jacobian().is_empty());
        assert!(telemetry.radius().is_empty());
        let gain_before = layer.projector_gain();
        let updated = layer.adapt_projector_gain(0.7, 0.5).unwrap();
        assert_eq!(updated, gain_before);
        assert_eq!(layer.projector_gain(), gain_before);

        let grad_output = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (0, 2));
        assert!(grad_input.data().is_empty());
        assert!(layer.gamma.gradient().is_none());
        assert!(layer.beta.gradient().is_none());
    }

    #[test]
    fn zspace_layer_norm_rejects_non_finite_input_without_mutating_telemetry() {
        let layer = ZSpaceLayerNorm::new("lnz", 3, -1.0, 1e-5).unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.2, f32::NAN, 0.4, 1.0, -0.5, 0.25]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_layernorm_input",
                value,
            } if value.is_nan()
        ));
        assert!(layer.telemetry().is_none());
    }

    #[test]
    fn zspace_layer_norm_rejects_overflowing_variance_without_mutating_telemetry() {
        let layer = ZSpaceLayerNorm::new("lnz", 2, -1.0, 1e-5).unwrap();
        let input = Tensor::from_vec(1, 2, vec![f32::MAX, -f32::MAX]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_layernorm_variance_sum",
                value,
            } if value.is_infinite()
        ));
        assert!(layer.telemetry().is_none());
    }

    #[test]
    fn zspace_layer_norm_rejects_non_finite_backward_grad_without_accumulating() {
        let mut layer = ZSpaceLayerNorm::new("lnz", 3, -1.0, 1e-5).unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.2, -0.3, 0.4, 1.0, -0.5, 0.25]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_output =
            Tensor::from_vec(2, 3, vec![0.1, f32::NAN, 0.05, 0.3, -0.2, 0.4]).unwrap();

        let err = layer.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_layernorm_backward_grad_output",
                value,
            } if value.is_nan()
        ));
        assert!(layer.gamma.gradient().is_none());
        assert!(layer.beta.gradient().is_none());
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
        let avg_radius = checked_mean("zspace_layernorm_radius_mean", telemetry.radius()).unwrap();
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

    #[test]
    fn zspace_layer_norm_rejects_bad_projector_radius_without_commit() {
        let layer = ZSpaceLayerNorm::new("lnz", 2, -0.8, 1e-5)
            .unwrap()
            .with_projector_gain(0.45)
            .unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.5, -0.25, -0.75, 1.0]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let before = layer.projector_gain();
        *layer.last_radius.borrow_mut() = Some(vec![f32::MAX, f32::MAX]);

        let err = layer.adapt_projector_gain(0.5, 0.75).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_layernorm_radius_mean",
                value,
            } if value.is_infinite()
        ));
        assert_eq!(layer.projector_gain(), before);
    }

    #[test]
    fn zspace_layer_norm_empty_batch_keeps_telemetry_without_updates() {
        let mut layer = ZSpaceLayerNorm::new("lnz", 3, -1.0, 1e-5)
            .unwrap()
            .with_projector_gain(0.4)
            .unwrap();
        let input = Tensor::from_vec(0, 3, Vec::new()).unwrap();

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (0, 3));
        assert!(output.data().is_empty());

        let telemetry = layer.telemetry().expect("empty telemetry available");
        assert_eq!(telemetry.batch(), 0);
        assert_eq!(telemetry.features(), 3);
        assert!(telemetry.mean().is_empty());
        assert!(telemetry.inv_std().is_empty());
        assert!(telemetry.normed().is_empty());
        assert!(telemetry.projected().is_empty());
        assert!(telemetry.jacobian().is_empty());
        assert!(telemetry.radius().is_empty());
        let gain_before = layer.projector_gain();
        let updated = layer.adapt_projector_gain(0.7, 0.5).unwrap();
        assert_eq!(updated, gain_before);
        assert_eq!(layer.projector_gain(), gain_before);

        let grad_output = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (0, 3));
        assert!(grad_input.data().is_empty());
        assert!(layer.gamma.gradient().is_none());
        assert!(layer.beta.gradient().is_none());
    }
}
