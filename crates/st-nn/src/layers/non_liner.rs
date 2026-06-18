// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_core::theory::zpulse::ZScale;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};
use std::cell::Cell;

fn validate_finite_value(label: &'static str, value: f32) -> PureResult<()> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
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
    fn label(self) -> &'static str {
        match self {
            Self::Tanh => "tanh",
            Self::Sigmoid => "sigmoid",
            Self::Softsign => "softsign",
        }
    }

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

/// Elliptic geometry configuration used by [`NonLinerGeometry::Elliptic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonLinerEllipticConfig {
    curvature: f32,
    z_scale: ZScale,
    retention: f32,
}

impl NonLinerEllipticConfig {
    /// Creates a new elliptic configuration. `curvature` must be positive and finite while
    /// `retention` must lie in the unit interval.
    pub fn new(curvature: f32, z_scale: ZScale, retention: f32) -> PureResult<Self> {
        if curvature <= 0.0 || !curvature.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "non_liner_curvature",
            });
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

    /// Returns the retention ratio blending the raw Euclidean signal with its elliptic projection.
    pub fn retention(&self) -> f32 {
        self.retention
    }

    fn curvature_scale(&self) -> f32 {
        self.curvature.sqrt()
    }

    fn mix(&self) -> f32 {
        1.0 - self.retention
    }
}

/// Geometry applied to the affine activation outputs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NonLinerGeometry {
    /// Standard Euclidean geometry (no additional projection is applied).
    #[default]
    Euclidean,
    /// Hyperbolic projection into the Z-space manifold using the provided configuration.
    Hyperbolic(NonLinerHyperbolicConfig),
    /// Elliptic projection into the Z-space manifold using the provided configuration.
    Elliptic(NonLinerEllipticConfig),
}

impl NonLinerGeometry {
    fn label(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::Hyperbolic(_) => "hyperbolic",
            Self::Elliptic(_) => "elliptic",
        }
    }

    /// Convenience constructor for the hyperbolic variant.
    pub fn hyperbolic(config: NonLinerHyperbolicConfig) -> Self {
        Self::Hyperbolic(config)
    }

    /// Convenience constructor for the elliptic variant.
    pub fn elliptic(config: NonLinerEllipticConfig) -> Self {
        Self::Elliptic(config)
    }

    /// Returns the underlying hyperbolic configuration when present.
    pub fn as_hyperbolic(&self) -> Option<NonLinerHyperbolicConfig> {
        match self {
            Self::Hyperbolic(config) => Some(*config),
            _ => None,
        }
    }

    /// Returns the underlying elliptic configuration when present.
    pub fn as_elliptic(&self) -> Option<NonLinerEllipticConfig> {
        match self {
            Self::Elliptic(config) => Some(*config),
            _ => None,
        }
    }

    /// Returns the curvature enforced by the geometry when non-Euclidean.
    pub fn curvature(&self) -> Option<f32> {
        match self {
            Self::Hyperbolic(config) => Some(config.curvature()),
            Self::Elliptic(config) => Some(config.curvature()),
            Self::Euclidean => None,
        }
    }

    /// Returns the Z-scale when non-Euclidean.
    pub fn z_scale(&self) -> Option<ZScale> {
        match self {
            Self::Hyperbolic(config) => Some(config.z_scale()),
            Self::Elliptic(config) => Some(config.z_scale()),
            Self::Euclidean => None,
        }
    }

    /// Returns the retention ratio when non-Euclidean.
    pub fn retention(&self) -> Option<f32> {
        match self {
            Self::Hyperbolic(config) => Some(config.retention()),
            Self::Elliptic(config) => Some(config.retention()),
            Self::Euclidean => None,
        }
    }
}

fn emit_non_liner_meta(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    backward: bool,
    activation: NonLinerActivation,
    geometry: NonLinerGeometry,
    drift: Option<f32>,
    radius: Option<f32>,
    preactivation_backend: Option<String>,
    broadcast_backend: Option<String>,
    gradient_backend: Option<String>,
    input_gradient_backend: Option<String>,
    gradient_scale: Option<f32>,
) {
    let input_shape = if backward {
        vec![rows, cols, rows, cols, 1, cols, 1, cols, 1, cols]
    } else {
        vec![rows, cols, 1, cols, 1, cols, 1, cols]
    };
    emit_tensor_op(op_name, &input_shape, &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        let values = rows.saturating_mul(cols);
        let geometry_backend = if matches!(geometry, NonLinerGeometry::Euclidean) {
            None
        } else {
            Some("cpu")
        };
        serde_json::json!({
            "backend": "composite",
            "requested_backend": "auto",
            "kernel": "non_liner.scalar",
            "kind": if backward { "activation_backward" } else { "activation_forward" },
            "activation": activation.label(),
            "activation_backend": "cpu",
            "geometry": geometry.label(),
            "geometry_backend": geometry_backend,
            "curvature": geometry.curvature(),
            "z_scale": geometry.z_scale().map(|scale| scale.value()),
            "retention": geometry.retention(),
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "trainable_parameters": cols.saturating_mul(3),
            "psi_drift": drift,
            "geometry_radius": radius,
            "preactivation_backend": preactivation_backend,
            "broadcast_backend": broadcast_backend,
            "input_gradient_backend": input_gradient_backend,
            "gradient_reduction_backend": gradient_backend,
            "gradient_scale": gradient_scale,
            "parameter_gradient_scale": gradient_scale,
            "input_gradient_scale": if backward && gradient_scale.is_some() {
                Some(1.0f32)
            } else {
                None
            },
            "estimated_activation_ops": values,
            "estimated_geometry_ops": if matches!(geometry, NonLinerGeometry::Euclidean) {
                0
            } else {
                values.saturating_mul(4)
            },
            "estimated_parameter_gradient_ops": if backward {
                values.saturating_mul(6)
            } else {
                0
            },
            "estimated_total_ops": if backward {
                values.saturating_mul(8)
            } else if matches!(geometry, NonLinerGeometry::Euclidean) {
                values
            } else {
                values.saturating_mul(5)
            },
            "empty": rows == 0 || cols == 0,
        })
    });
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
        validate_finite_value("non_liner_slope", slope)?;
        validate_finite_value("non_liner_gain", gain)?;
        validate_finite_value("non_liner_bias", bias)?;
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

    /// Creates a new elliptic non-linear layer with unit affine parameters.
    pub fn with_elliptic(
        name: impl Into<String>,
        features: usize,
        activation: NonLinerActivation,
        config: NonLinerEllipticConfig,
    ) -> PureResult<Self> {
        Self::with_geometry(
            name,
            features,
            activation,
            1.0,
            1.0,
            0.0,
            NonLinerGeometry::elliptic(config),
        )
    }

    /// Creates a new elliptic non-linear layer with caller supplied affine parameters.
    pub fn with_elliptic_init(
        name: impl Into<String>,
        features: usize,
        activation: NonLinerActivation,
        slope: f32,
        gain: f32,
        bias: f32,
        config: NonLinerEllipticConfig,
    ) -> PureResult<Self> {
        Self::with_geometry(
            name,
            features,
            activation,
            slope,
            gain,
            bias,
            NonLinerGeometry::elliptic(config),
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

    fn validate_parameters(&self) -> PureResult<()> {
        validate_finite_tensor("non_liner_gain", self.gain.value())?;
        validate_finite_tensor("non_liner_slope", self.slope.value())?;
        validate_finite_tensor("non_liner_bias", self.bias.value())
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

    fn elliptic_coefficients(config: &NonLinerEllipticConfig, norm: f32) -> (f32, f32, f32) {
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
            let factor = a / norm;
            let derivative = -factor / norm;
            return (factor, derivative, scale);
        }
        let limit = core::f32::consts::FRAC_PI_2 - 1.0e-6;
        if u >= limit {
            let sin_limit = limit.sin();
            let factor = (a / norm) * sin_limit;
            let derivative = -factor / norm;
            let target_radius = sin_limit * scale;
            return (factor, derivative, target_radius);
        }
        let sin_u = u.sin();
        let cos_u = u.cos();
        let factor = (a / norm) * sin_u;
        let derivative = (cos_u - factor) / norm;
        let target_radius = sin_u * scale;
        (factor, derivative, target_radius)
    }

    fn apply_geometry_forward(
        &self,
        rows: usize,
        cols: usize,
        data: &mut [f32],
    ) -> PureResult<Option<f32>> {
        match self.geometry {
            NonLinerGeometry::Euclidean => Ok(None),
            NonLinerGeometry::Hyperbolic(config) => Self::apply_curved_geometry_forward(
                rows,
                cols,
                data,
                config.retention(),
                config.mix(),
                |norm| Self::hyperbolic_coefficients(&config, norm),
            ),
            NonLinerGeometry::Elliptic(config) => Self::apply_curved_geometry_forward(
                rows,
                cols,
                data,
                config.retention(),
                config.mix(),
                |norm| Self::elliptic_coefficients(&config, norm),
            ),
        }
    }

    fn apply_curved_geometry_forward<F>(
        rows: usize,
        cols: usize,
        data: &mut [f32],
        retention: f32,
        mix: f32,
        coefficients: F,
    ) -> PureResult<Option<f32>>
    where
        F: Fn(f32) -> (f32, f32, f32),
    {
        if rows == 0 || cols == 0 {
            return Ok(Some(0.0));
        }

        validate_finite_value("non_liner_geometry_retention", retention)?;
        validate_finite_value("non_liner_geometry_mix", mix)?;

        let mut radius_sum = 0.0f32;
        let mut counted = 0usize;
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let chunk = &mut data[start..end];
            if chunk.is_empty() {
                continue;
            }

            let mut norm_sq = 0.0f32;
            for &value in chunk.iter() {
                validate_finite_value("non_liner_geometry_raw", value)?;
                let square = value * value;
                validate_finite_value("non_liner_geometry_norm_sq", square)?;
                norm_sq += square;
                validate_finite_value("non_liner_geometry_norm_sq", norm_sq)?;
            }
            let norm = norm_sq.sqrt();
            validate_finite_value("non_liner_geometry_norm", norm)?;

            let (factor, _, radius) = coefficients(norm);
            validate_finite_value("non_liner_geometry_factor", factor)?;
            validate_finite_value("non_liner_geometry_radius", radius)?;
            if mix > 0.0 && norm > 1.0e-6 {
                for value in chunk.iter_mut() {
                    let projected = *value * factor;
                    validate_finite_value("non_liner_geometry_projected", projected)?;
                    let mixed = retention * *value + mix * projected;
                    validate_finite_value("non_liner_geometry_output", mixed)?;
                    *value = mixed;
                }
            }
            radius_sum += radius;
            validate_finite_value("non_liner_geometry_radius_sum", radius_sum)?;
            counted += 1;
        }

        let radius_mean = if counted > 0 {
            radius_sum / counted as f32
        } else {
            0.0
        };
        validate_finite_value("non_liner_geometry_radius_mean", radius_mean)?;
        Ok(Some(radius_mean))
    }

    fn apply_geometry_backward(
        &self,
        rows: usize,
        cols: usize,
        raw: &[f32],
        grad: &mut [f32],
    ) -> PureResult<()> {
        match self.geometry {
            NonLinerGeometry::Euclidean => Ok(()),
            NonLinerGeometry::Hyperbolic(config) => Self::apply_curved_geometry_backward(
                rows,
                cols,
                raw,
                grad,
                config.retention(),
                config.mix(),
                |norm| Self::hyperbolic_coefficients(&config, norm),
            ),
            NonLinerGeometry::Elliptic(config) => Self::apply_curved_geometry_backward(
                rows,
                cols,
                raw,
                grad,
                config.retention(),
                config.mix(),
                |norm| Self::elliptic_coefficients(&config, norm),
            ),
        }
    }

    fn apply_curved_geometry_backward<F>(
        rows: usize,
        cols: usize,
        raw: &[f32],
        grad: &mut [f32],
        retention: f32,
        mix: f32,
        coefficients: F,
    ) -> PureResult<()>
    where
        F: Fn(f32) -> (f32, f32, f32),
    {
        if rows == 0 || cols == 0 {
            return Ok(());
        }

        validate_finite_value("non_liner_geometry_backward_retention", retention)?;
        validate_finite_value("non_liner_geometry_backward_mix", mix)?;

        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let raw_chunk = &raw[start..end];
            let grad_chunk = &mut grad[start..end];
            if raw_chunk.is_empty() {
                continue;
            }

            let mut norm_sq = 0.0f32;
            for &value in raw_chunk.iter() {
                validate_finite_value("non_liner_geometry_backward_raw", value)?;
                let square = value * value;
                validate_finite_value("non_liner_geometry_backward_norm_sq", square)?;
                norm_sq += square;
                validate_finite_value("non_liner_geometry_backward_norm_sq", norm_sq)?;
            }
            validate_finite_slice("non_liner_geometry_backward_grad", grad_chunk)?;
            let norm = norm_sq.sqrt();
            validate_finite_value("non_liner_geometry_backward_norm", norm)?;

            let (factor, derivative, _) = coefficients(norm);
            validate_finite_value("non_liner_geometry_backward_factor", factor)?;
            validate_finite_value("non_liner_geometry_backward_derivative", derivative)?;
            if norm <= 1.0e-6 {
                let scaling = retention + mix;
                validate_finite_value("non_liner_geometry_backward_scaling", scaling)?;
                if (scaling - 1.0).abs() > f32::EPSILON {
                    for grad_value in grad_chunk.iter_mut() {
                        let updated = *grad_value * scaling;
                        validate_finite_value("non_liner_geometry_backward_output", updated)?;
                        *grad_value = updated;
                    }
                }
                continue;
            }

            let mut dot = 0.0f32;
            for (&raw_value, &grad_value) in raw_chunk.iter().zip(grad_chunk.iter()) {
                let product = raw_value * grad_value;
                validate_finite_value("non_liner_geometry_backward_dot", product)?;
                dot += product;
                validate_finite_value("non_liner_geometry_backward_dot", dot)?;
            }
            let scaling = retention + mix * factor;
            validate_finite_value("non_liner_geometry_backward_scaling", scaling)?;
            let correction = mix * derivative * dot / norm;
            validate_finite_value("non_liner_geometry_backward_correction", correction)?;
            for (raw_value, grad_value) in raw_chunk.iter().zip(grad_chunk.iter_mut()) {
                let updated = scaling * *grad_value + correction * *raw_value;
                validate_finite_value("non_liner_geometry_backward_output", updated)?;
                *grad_value = updated;
            }
        }
        Ok(())
    }
}

impl Module for NonLiner {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        validate_finite_tensor("non_liner_input", input)?;
        if cols == 0 {
            let radius = match self.geometry {
                NonLinerGeometry::Euclidean => None,
                NonLinerGeometry::Hyperbolic(_) | NonLinerGeometry::Elliptic(_) => Some(0.0),
            };
            let output = Tensor::zeros(rows, cols)?;
            self.last_drift.set(Some(0.0));
            self.last_radius.set(radius);
            emit_non_liner_meta(
                "non_liner_forward",
                rows,
                cols,
                false,
                self.activation,
                self.geometry,
                Some(0.0),
                radius,
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(output);
        }

        self.ensure_parameter_shapes(cols)?;
        self.validate_parameters()?;
        if rows == 0 {
            let radius = match self.geometry {
                NonLinerGeometry::Euclidean => None,
                NonLinerGeometry::Hyperbolic(_) | NonLinerGeometry::Elliptic(_) => Some(0.0),
            };
            let output = Tensor::zeros(rows, cols)?;
            self.last_drift.set(Some(0.0));
            self.last_radius.set(radius);
            emit_non_liner_meta(
                "non_liner_forward",
                rows,
                cols,
                false,
                self.activation,
                self.geometry,
                Some(0.0),
                radius,
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(output);
        }

        let gain = self.gain.value().data().to_vec();
        let slope = self.slope.value().data().to_vec();
        let bias = self.bias.value().data().to_vec();
        let util_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let preactivation = input.row_affine_with_backend(&slope, &bias, util_backend)?;

        let mut activated_values = Vec::with_capacity(rows * cols);
        for &pre in preactivation.data() {
            validate_finite_value("non_liner_pre_activation", pre)?;
            let activated = self.activation.activate(pre);
            validate_finite_value("non_liner_activation", activated)?;
            activated_values.push(activated);
        }
        let activated = Tensor::from_vec(rows, cols, activated_values)?;
        let output_tensor = activated.mul_row_with_backend(&gain, util_backend)?;
        let mut output = output_tensor.data().to_vec();

        let radius = self.apply_geometry_forward(rows, cols, &mut output)?;
        validate_finite_slice("non_liner_output", &output)?;

        let total = rows * cols;
        let mut drift_sum = 0.0f32;
        for value in output.iter() {
            drift_sum += value.abs();
            validate_finite_value("non_liner_drift_sum", drift_sum)?;
        }
        let drift = drift_sum / total as f32;
        validate_finite_value("non_liner_drift", drift)?;

        let output = Tensor::from_vec(rows, cols, output)?;
        self.last_radius.set(radius);
        self.last_drift.set(Some(drift));
        emit_non_liner_meta(
            "non_liner_forward",
            rows,
            cols,
            false,
            self.activation,
            self.geometry,
            Some(drift),
            radius,
            Some(util_backend.to_string()),
            Some(util_backend.to_string()),
            None,
            None,
            None,
        );
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
        validate_finite_tensor("non_liner_backward_input", input)?;
        validate_finite_tensor("non_liner_backward_grad_output", grad_output)?;
        if cols == 0 {
            let output = Tensor::zeros(rows, cols)?;
            emit_non_liner_meta(
                "non_liner_backward",
                rows,
                cols,
                true,
                self.activation,
                self.geometry,
                self.last_drift.get(),
                self.last_radius.get(),
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(output);
        }

        self.ensure_parameter_shapes(cols)?;
        self.validate_parameters()?;
        if rows == 0 {
            let output = Tensor::zeros(rows, cols)?;
            emit_non_liner_meta(
                "non_liner_backward",
                rows,
                cols,
                true,
                self.activation,
                self.geometry,
                self.last_drift.get(),
                self.last_radius.get(),
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(output);
        }

        let gain = self.gain.value().data().to_vec();
        let slope = self.slope.value().data().to_vec();
        let bias = self.bias.value().data().to_vec();
        let util_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let preactivation = input.row_affine_with_backend(&slope, &bias, util_backend)?;

        let mut raw_outputs = vec![0.0f32; rows * cols];
        let mut activated_values = vec![0.0; rows * cols];
        for row in 0..rows {
            let base = row * cols;
            for col in 0..cols {
                let idx = base + col;
                let pre = preactivation.data()[idx];
                validate_finite_value("non_liner_backward_pre_activation", pre)?;
                let activated = self.activation.activate(pre);
                validate_finite_value("non_liner_backward_activation", activated)?;
                let raw_output = gain[col] * activated;
                validate_finite_value("non_liner_backward_raw_output", raw_output)?;
                activated_values[idx] = activated;
                raw_outputs[idx] = raw_output;
            }
        }

        let mut grad_raw = grad_output.data().to_vec();
        validate_finite_slice("non_liner_backward_grad_raw", &grad_raw)?;
        self.apply_geometry_backward(rows, cols, &raw_outputs, &mut grad_raw)?;
        validate_finite_slice("non_liner_backward_grad_raw", &grad_raw)?;

        let mut delta_values = vec![0.0; rows * cols];

        for row in 0..rows {
            let base = row * cols;
            for col in 0..cols {
                let idx = base + col;
                let pre = preactivation.data()[idx];
                validate_finite_value("non_liner_backward_pre_activation", pre)?;
                let activated = activated_values[idx];
                let derivative = self.activation.derivative(activated, pre);
                validate_finite_value("non_liner_backward_derivative", derivative)?;
                let grad = grad_raw[idx];
                validate_finite_value("non_liner_backward_grad_raw", grad)?;

                activated_values[idx] = activated;
                let chain = grad * gain[col];
                validate_finite_value("non_liner_backward_chain", chain)?;
                let delta = chain * derivative;
                validate_finite_value("non_liner_backward_delta", delta)?;
                delta_values[idx] = delta;
            }
        }

        let inv_batch = 1.0 / rows as f32;
        let grad_raw_tensor = Tensor::from_vec(rows, cols, grad_raw)?;
        let activated_tensor = Tensor::from_vec(rows, cols, activated_values)?;
        let delta_tensor = Tensor::from_vec(rows, cols, delta_values)?;
        let grad_input = delta_tensor.mul_row_with_backend(&slope, util_backend)?;

        let gain_product =
            grad_raw_tensor.hadamard_with_backend(&activated_tensor, util_backend)?;
        let gain_grad = Tensor::from_vec(
            1,
            cols,
            gain_product.try_sum_axis0_scaled_with_backend(inv_batch, util_backend)?,
        )?;
        validate_finite_tensor("non_liner_gain_grad", &gain_grad)?;
        let slope_product = delta_tensor.hadamard_with_backend(input, util_backend)?;
        let slope_grad = Tensor::from_vec(
            1,
            cols,
            slope_product.try_sum_axis0_scaled_with_backend(inv_batch, util_backend)?,
        )?;
        validate_finite_tensor("non_liner_slope_grad", &slope_grad)?;
        let bias_grad = Tensor::from_vec(
            1,
            cols,
            delta_tensor.try_sum_axis0_scaled_with_backend(inv_batch, util_backend)?,
        )?;
        validate_finite_tensor("non_liner_bias_grad", &bias_grad)?;

        self.gain.accumulate_euclidean(&gain_grad)?;
        self.slope.accumulate_euclidean(&slope_grad)?;
        self.bias.accumulate_euclidean(&bias_grad)?;

        emit_non_liner_meta(
            "non_liner_backward",
            rows,
            cols,
            true,
            self.activation,
            self.geometry,
            self.last_drift.get(),
            self.last_radius.get(),
            Some(util_backend.to_string()),
            None,
            Some(util_backend.to_string()),
            Some(util_backend.to_string()),
            Some(inv_batch),
        );
        Ok(grad_input)
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
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    #[cfg(feature = "wgpu")]
    use st_tensor::backend::wgpu_dense;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn approx_eq(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (l, r) in left.iter().zip(right.iter()) {
            let diff = (l - r).abs();
            assert!(diff < 1e-5, "expected {l} ≈ {r} (diff={diff})");
        }
    }

    fn assert_non_finite_label<T>(result: PureResult<T>, expected: &'static str) {
        match result {
            Err(TensorError::NonFiniteValue { label, .. }) => assert_eq!(label, expected),
            Err(error) => panic!("expected non-finite label {expected}, got {error:?}"),
            Ok(_) => panic!("expected non-finite label {expected}"),
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
    fn forward_rejects_non_finite_input_without_mutating_metrics() {
        let config = NonLinerHyperbolicConfig::new(-0.9, ZScale::new(1.5).unwrap(), 0.2).unwrap();
        let layer =
            NonLiner::with_hyperbolic_init("z", 2, NonLinerActivation::Tanh, 0.8, 1.1, 0.1, config)
                .unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.2, -0.4]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let drift_before = layer.psi_probe();
        let radius_before = layer.last_hyperbolic_radius();

        let mut bad_input = input.clone();
        bad_input.data_mut()[1] = f32::NAN;

        assert_non_finite_label(layer.forward(&bad_input), "non_liner_input");
        assert_eq!(layer.psi_probe(), drift_before);
        assert_eq!(layer.last_hyperbolic_radius(), radius_before);
    }

    #[test]
    fn forward_rejects_non_finite_parameter_without_mutating_metrics() {
        let mut layer =
            NonLiner::with_init("nl", 2, NonLinerActivation::Sigmoid, 0.7, 1.1, 0.05).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.2, -0.3]).unwrap();
        layer.gain.value_mut().data_mut()[0] = f32::NAN;

        assert_non_finite_label(layer.forward(&input), "non_liner_gain");
        assert!(layer.psi_probe().is_none());
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
    fn backward_rejects_non_finite_grad_without_accumulating() {
        let mut layer =
            NonLiner::with_init("nl", 2, NonLinerActivation::Sigmoid, 0.7, 1.1, 0.05).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.2, -0.3]).unwrap();
        let mut grad_output = Tensor::from_vec(1, 2, vec![0.6, -0.2]).unwrap();
        grad_output.data_mut()[0] = f32::NAN;

        assert_non_finite_label(
            layer.backward(&input, &grad_output),
            "non_liner_backward_grad_output",
        );
        assert!(layer.gain.gradient().is_none());
        assert!(layer.slope.gradient().is_none());
        assert!(layer.bias.gradient().is_none());
    }

    #[test]
    fn backward_rejects_overflowing_chain_without_accumulating() {
        let mut layer =
            NonLiner::with_init("nl", 1, NonLinerActivation::Tanh, 1.0, f32::MAX, 0.0).unwrap();
        let input = Tensor::from_vec(1, 1, vec![0.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 1, vec![2.0]).unwrap();

        assert_non_finite_label(
            layer.backward(&input, &grad_output),
            "non_liner_backward_chain",
        );
        assert!(layer.gain.gradient().is_none());
        assert!(layer.slope.gradient().is_none());
        assert!(layer.bias.gradient().is_none());
    }

    #[test]
    fn forward_backward_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut layer =
            NonLiner::with_init("nl", 2, NonLinerActivation::Sigmoid, 0.7, 1.1, 0.05).unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.2, -0.3, 0.5, 0.8]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.6, -0.2, -0.4, 0.9]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let _ = layer.backward(&input, &grad_output).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "non_liner_forward"
                    && data["rows"] == 2
                    && data["cols"] == 2
                    && data["activation"] == "sigmoid"
                    && data["geometry"] == "euclidean"
            })
            .expect("non liner forward metadata event");
        assert_eq!(forward.1["backend"], "composite");
        assert_eq!(forward.1["kind"], "activation_forward");
        assert_eq!(forward.1["activation"], "sigmoid");
        assert_eq!(forward.1["activation_backend"], "cpu");
        assert_eq!(forward.1["geometry"], "euclidean");
        assert!(forward.1["geometry_backend"].is_null());
        assert_eq!(forward.1["preactivation_backend"], "auto");
        assert_eq!(forward.1["broadcast_backend"], "auto");

        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "non_liner_backward"
                    && data["rows"] == 2
                    && data["cols"] == 2
                    && data["activation"] == "sigmoid"
                    && data["geometry"] == "euclidean"
            })
            .expect("non liner backward metadata event");
        assert_eq!(backward.1["backend"], "composite");
        assert_eq!(backward.1["kind"], "activation_backward");
        assert_eq!(backward.1["activation_backend"], "cpu");
        assert_eq!(backward.1["preactivation_backend"], "auto");
        assert_eq!(backward.1["input_gradient_backend"], "auto");
        assert_eq!(backward.1["gradient_reduction_backend"], "auto");
        assert_eq!(backward.1["gradient_scale"], 0.5);
        assert_eq!(backward.1["parameter_gradient_scale"], 0.5);
        assert_eq!(backward.1["input_gradient_scale"], 1.0);
        assert_eq!(backward.1["trainable_parameters"], 6);

        let hadamard_count = events
            .iter()
            .filter(|(op_name, data)| *op_name == "hadamard" && data["rows"] == 2)
            .count();
        let mul_row_count = events
            .iter()
            .filter(|(op_name, data)| *op_name == "mul_row" && data["rows"] == 2)
            .count();
        let row_affine_count = events
            .iter()
            .filter(|(op_name, data)| *op_name == "row_affine" && data["rows"] == 2)
            .count();
        let reduction_count = events
            .iter()
            .filter(|(op_name, data)| *op_name == "sum_axis0_scaled" && data["cols"] == 2)
            .count();
        assert!(hadamard_count >= 2);
        assert_eq!(mul_row_count, 2);
        assert_eq!(row_affine_count, 2);
        assert!(reduction_count >= 3);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn backward_forced_wgpu_uses_mul_row_and_matches_cpu_reference() {
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

        let input = Tensor::from_fn(4, 3, |row, col| {
            ((row * 17 + col * 7) % 19) as f32 * 0.039 - 0.31
        })
        .unwrap();
        let grad_output = Tensor::from_fn(4, 3, |row, col| {
            ((row * 13 + col * 5) % 17) as f32 * 0.027 - 0.19
        })
        .unwrap();
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_layer =
            NonLiner::with_init("nl_cpu", 3, NonLinerActivation::Sigmoid, 0.8, 1.15, -0.04)
                .unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.backward(&input, &grad_output).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_layer =
            NonLiner::with_init("nl_wgpu", 3, NonLinerActivation::Sigmoid, 0.8, 1.15, -0.04)
                .unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.forward(&input).unwrap(),
                wgpu_layer.backward(&input, &grad_output).unwrap(),
            )
        };

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        approx_eq(cpu_forward.data(), wgpu_forward.data());
        approx_eq(cpu_grad_input.data(), wgpu_grad_input.data());

        for (cpu_param, wgpu_param) in [
            (
                cpu_layer.gain.gradient().unwrap(),
                wgpu_layer.gain.gradient().unwrap(),
            ),
            (
                cpu_layer.slope.gradient().unwrap(),
                wgpu_layer.slope.gradient().unwrap(),
            ),
            (
                cpu_layer.bias.gradient().unwrap(),
                wgpu_layer.bias.gradient().unwrap(),
            ),
        ] {
            approx_eq(cpu_param.data(), wgpu_param.data());
        }

        let events = events.lock().unwrap();
        assert!(events
            .iter()
            .any(|(op_name, data)| *op_name == "mul_row" && data["backend"] == "wgpu_dense"));
        assert!(events
            .iter()
            .any(|(op_name, data)| *op_name == "row_affine" && data["backend"] == "wgpu_dense"));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "non_liner_forward"
                && data["backend"] == "composite"
                && data["activation_backend"] == "cpu"
                && data["preactivation_backend"] == "wgpu"
                && data["broadcast_backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "non_liner_backward"
                && data["backend"] == "composite"
                && data["activation_backend"] == "cpu"
                && data["preactivation_backend"] == "wgpu"
                && data["input_gradient_backend"] == "wgpu"
        }));
    }

    #[test]
    fn backward_input_gradient_matches_numeric_gradients_without_batch_scaling() {
        let mut layer =
            NonLiner::with_init("nl", 2, NonLinerActivation::Sigmoid, 0.85, 1.2, -0.05).unwrap();
        let input = Tensor::from_vec(3, 2, vec![0.15, -0.25, 0.45, 0.7, -0.6, 0.05]).unwrap();
        let grad_output =
            Tensor::from_vec(3, 2, vec![0.35, -0.15, -0.45, 0.55, 0.25, -0.65]).unwrap();

        let grad_input = layer.backward(&input, &grad_output).unwrap();
        let grad_output_vec = grad_output.data().to_vec();
        let (rows, cols) = input.shape();
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
                .sum::<f32>();

            let mut minus = input.clone();
            minus.data_mut()[idx] -= epsilon;
            let out_minus = layer.forward(&minus).unwrap();
            let loss_minus = out_minus
                .data()
                .iter()
                .zip(grad_output_vec.iter())
                .map(|(o, g)| o * g)
                .sum::<f32>();

            let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert!((numeric - grad_input.data()[idx]).abs() < 2e-3);
        }
    }

    #[test]
    fn backward_empty_batch_returns_empty_grad_without_parameter_updates() {
        let mut layer =
            NonLiner::with_init("nl", 2, NonLinerActivation::Sigmoid, 0.7, 1.1, 0.05).unwrap();
        let input = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (0, 2));
        assert!(output.data().is_empty());

        let grad_output = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(layer.gain.gradient().is_none());
        assert!(layer.slope.gradient().is_none());
        assert!(layer.bias.gradient().is_none());
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
                .sum::<f32>();

            let mut minus = input.clone();
            minus.data_mut()[idx] -= epsilon;
            let out_minus = layer.forward(&minus).unwrap();
            let loss_minus = out_minus
                .data()
                .iter()
                .zip(grad_output_vec.iter())
                .map(|(o, g)| o * g)
                .sum::<f32>();

            let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert!((numeric - grad_input.data()[idx]).abs() < 2e-3);
        }
    }

    #[test]
    fn elliptic_config_requires_positive_curvature() {
        let scale = ZScale::new(1.0).unwrap();
        assert!(NonLinerEllipticConfig::new(0.0, scale, 0.5).is_err());
        assert!(NonLinerEllipticConfig::new(-0.1, scale, 0.5).is_err());
        assert!(NonLinerEllipticConfig::new(0.5, scale, 1.2).is_err());
    }

    #[test]
    fn elliptic_forward_projects_onto_sphere() {
        let config = NonLinerEllipticConfig::new(0.8, ZScale::new(1.4).unwrap(), 0.25).unwrap();
        let layer = NonLiner::with_elliptic_init(
            "sphere",
            3,
            NonLinerActivation::Tanh,
            1.1,
            0.85,
            -0.05,
            config,
        )
        .unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.8, -1.2, 0.4, -0.9, 0.6, 0.3]).unwrap();
        let output = layer.forward(&input).unwrap();

        let curvature_scale = config.curvature_scale();
        let retention = config.retention();
        let mix = 1.0 - retention;
        let z_scale = config.z_scale().value();
        let slope = 1.1f32;
        let gain = 0.85f32;
        let bias = -0.05f32;
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
    fn elliptic_backward_matches_numeric_gradients() {
        let config = NonLinerEllipticConfig::new(0.6, ZScale::new(1.1).unwrap(), 0.3).unwrap();
        let mut layer = NonLiner::with_elliptic_init(
            "ellipse",
            2,
            NonLinerActivation::Softsign,
            0.9,
            1.05,
            0.1,
            config,
        )
        .unwrap();
        let input = Tensor::from_vec(2, 2, vec![0.35, -0.45, 0.7, -0.8]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.2, -0.15, -0.35, 0.4]).unwrap();

        let grad_input = layer.backward(&input, &grad_output).unwrap();
        let grad_output_vec = grad_output.data().to_vec();
        let (rows, cols) = input.shape();
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
                .sum::<f32>();

            let mut minus = input.clone();
            minus.data_mut()[idx] -= epsilon;
            let out_minus = layer.forward(&minus).unwrap();
            let loss_minus = out_minus
                .data()
                .iter()
                .zip(grad_output_vec.iter())
                .map(|(o, g)| o * g)
                .sum::<f32>();

            let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert!((numeric - grad_input.data()[idx]).abs() < 2e-3);
        }
    }
}
