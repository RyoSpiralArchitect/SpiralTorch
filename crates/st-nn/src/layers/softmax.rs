// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::{
    current_backend_policy, current_softmax_backend, current_tensor_util_backend,
    current_tensor_util_backend_for_values,
};
use crate::module::Module;
use crate::{PureResult, Tensor};
use st_core::inference::generation_control::{
    project_zspace_softmax, zspace_entropy, ZSpaceGenerationControlError, ZSpaceSoftmaxConfig,
    ZSpaceSoftmaxProjection, ZSPACE_SOFTMAX_LOG_FLOOR,
};
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorError, TensorUtilBackend};
use std::cell::RefCell;

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

fn semantic_error_to_tensor(error: ZSpaceGenerationControlError) -> TensorError {
    let label = match error {
        ZSpaceGenerationControlError::NonFinite { field }
        | ZSpaceGenerationControlError::NonPositive { field }
        | ZSpaceGenerationControlError::Negative { field }
        | ZSpaceGenerationControlError::OutOfRange { field, .. }
        | ZSpaceGenerationControlError::NonFiniteDerived { field } => field,
        ZSpaceGenerationControlError::InvalidTemperatureBounds => {
            "zspace_softmax_temperature_bounds"
        }
        ZSpaceGenerationControlError::CandidateLengthMismatch { .. } => {
            "zspace_softmax_candidate_lengths"
        }
    };
    TensorError::InvalidValue { label }
}

fn f64_to_f32(label: &'static str, value: f64) -> PureResult<f32> {
    let value = value as f32;
    validate_finite_value(label, value)?;
    Ok(value)
}

fn f64_slice_to_f32(label: &'static str, values: &[f64]) -> PureResult<Vec<f32>> {
    values
        .iter()
        .copied()
        .map(|value| f64_to_f32(label, value))
        .collect()
}

fn current_softmax_requested_label() -> &'static str {
    current_backend_policy()
        .map(|policy| policy.softmax_backend_label())
        .unwrap_or("auto")
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

#[cfg(feature = "wgpu")]
fn strict_gpu_path() -> bool {
    crate::execution::current_accelerator_fallback().is_strict()
}

#[cfg(feature = "wgpu")]
fn zspace_softmax_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_zspace_softmax_forward_meta(
    rows: usize,
    cols: usize,
    requested_backend: &'static str,
    backend: &'static str,
    kernel: &'static str,
    adaptive_temperature: bool,
    entropy_rows: usize,
    temperature_rows: usize,
    curvature: f32,
    base_temperature: f32,
    min_temperature: f32,
    max_temperature: f32,
    entropy_target: Option<f32>,
) {
    emit_tensor_op("zspace_softmax_forward", &[rows, cols], &[rows, cols]);
    emit_tensor_op_meta("zspace_softmax_forward", || {
        let mut data = serde_json::Map::new();
        data.insert("backend".to_string(), serde_json::json!(backend));
        data.insert(
            "requested_backend".to_string(),
            serde_json::json!(requested_backend),
        );
        data.insert(
            "softmax_requested_backend".to_string(),
            serde_json::json!(current_softmax_requested_label()),
        );
        data.insert("kernel".to_string(), serde_json::json!(kernel));
        data.insert("kind".to_string(), serde_json::json!("activation_forward"));
        data.insert("rows".to_string(), serde_json::json!(rows));
        data.insert("cols".to_string(), serde_json::json!(cols));
        data.insert(
            "values".to_string(),
            serde_json::json!(rows.saturating_mul(cols)),
        );
        data.insert("output_rows".to_string(), serde_json::json!(rows));
        data.insert("output_cols".to_string(), serde_json::json!(cols));
        data.insert(
            "output_values".to_string(),
            serde_json::json!(rows.saturating_mul(cols)),
        );
        data.insert("curvature".to_string(), serde_json::json!(curvature));
        data.insert(
            "temperature".to_string(),
            serde_json::json!(base_temperature),
        );
        data.insert(
            "min_temperature".to_string(),
            serde_json::json!(min_temperature),
        );
        data.insert(
            "max_temperature".to_string(),
            serde_json::json!(max_temperature),
        );
        data.insert(
            "adaptive_temperature".to_string(),
            serde_json::json!(adaptive_temperature),
        );
        data.insert(
            "entropy_target".to_string(),
            serde_json::json!(entropy_target),
        );
        data.insert("entropy_rows".to_string(), serde_json::json!(entropy_rows));
        data.insert(
            "temperature_rows".to_string(),
            serde_json::json!(temperature_rows),
        );
        data.insert(
            "empty".to_string(),
            serde_json::json!(rows == 0 || cols == 0),
        );
        serde_json::Value::Object(data)
    });
}

#[allow(clippy::too_many_arguments)]
fn emit_zspace_softmax_backward_meta(
    rows: usize,
    cols: usize,
    requested_backend: &'static str,
    backend: &'static str,
    kernel: &'static str,
    adaptive_temperature: bool,
    temperature_gradient_rows: usize,
    curvature: f32,
    base_temperature: f32,
    fallback_message: Option<&str>,
) {
    emit_tensor_op(
        "zspace_softmax_backward",
        &[rows, cols, rows, cols],
        &[rows, cols],
    );
    emit_tensor_op_meta("zspace_softmax_backward", || {
        let mut data = serde_json::Map::new();
        data.insert("backend".to_string(), serde_json::json!(backend));
        data.insert(
            "requested_backend".to_string(),
            serde_json::json!(requested_backend),
        );
        data.insert(
            "softmax_requested_backend".to_string(),
            serde_json::json!(current_softmax_requested_label()),
        );
        data.insert("kernel".to_string(), serde_json::json!(kernel));
        data.insert("kind".to_string(), serde_json::json!("activation_backward"));
        data.insert("rows".to_string(), serde_json::json!(rows));
        data.insert("cols".to_string(), serde_json::json!(cols));
        data.insert(
            "values".to_string(),
            serde_json::json!(rows.saturating_mul(cols)),
        );
        data.insert("output_rows".to_string(), serde_json::json!(rows));
        data.insert("output_cols".to_string(), serde_json::json!(cols));
        data.insert(
            "output_values".to_string(),
            serde_json::json!(rows.saturating_mul(cols)),
        );
        data.insert("curvature".to_string(), serde_json::json!(curvature));
        data.insert(
            "temperature".to_string(),
            serde_json::json!(base_temperature),
        );
        data.insert(
            "adaptive_temperature".to_string(),
            serde_json::json!(adaptive_temperature),
        );
        data.insert(
            "temperature_gradient".to_string(),
            serde_json::json!(if temperature_gradient_rows > 0 {
                "one_step_entropy_exact"
            } else {
                "constant"
            }),
        );
        data.insert(
            "temperature_gradient_rows".to_string(),
            serde_json::json!(temperature_gradient_rows),
        );
        data.insert(
            "empty".to_string(),
            serde_json::json!(rows == 0 || cols == 0),
        );
        if let Some(message) = fallback_message {
            data.insert(
                "fallback".to_string(),
                serde_json::json!({"from": "wgpu", "message": message}),
            );
        }
        serde_json::Value::Object(data)
    });
}

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
        let semantic = ZSpaceSoftmaxConfig::new(f64::from(curvature), f64::from(temperature))
            .map_err(semantic_error_to_tensor)?;
        let min_temperature =
            f64_to_f32("zspace_softmax_min_temperature", semantic.min_temperature())?;
        let max_temperature =
            f64_to_f32("zspace_softmax_max_temperature", semantic.max_temperature())?;
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
        self.semantic_config()?
            .with_entropy_target(f64::from(target), f64::from(tolerance), f64::from(gain))
            .map_err(semantic_error_to_tensor)?;
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
            return Err(TensorError::InvalidValue {
                label: "zspace_softmax_temperature_bounds",
            });
        }
        self.semantic_config()?
            .with_temperature_bounds(f64::from(min_temperature), f64::from(max_temperature))
            .map_err(semantic_error_to_tensor)?;
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

    fn semantic_config(&self) -> PureResult<ZSpaceSoftmaxConfig> {
        let mut config =
            ZSpaceSoftmaxConfig::new(f64::from(self.curvature), f64::from(self.temperature))
                .map_err(semantic_error_to_tensor)?;
        if let Some(target) = self.entropy_target {
            config = config
                .with_entropy_target(
                    f64::from(target),
                    f64::from(self.entropy_tolerance),
                    f64::from(self.entropy_gain),
                )
                .map_err(semantic_error_to_tensor)?;
        }
        config
            .with_temperature_bounds(
                f64::from(self.min_temperature),
                f64::from(self.max_temperature),
            )
            .map_err(semantic_error_to_tensor)
    }

    fn curvature_scale(&self) -> PureResult<f32> {
        f64_to_f32(
            "zspace_softmax_curvature_scale",
            self.semantic_config()?.curvature_scale(),
        )
    }

    fn fixed_temperature(&self) -> PureResult<f32> {
        f64_to_f32(
            "zspace_softmax_temperature",
            self.semantic_config()?.fixed_temperature(),
        )
    }

    fn project_row(&self, row: &[f32]) -> PureResult<ZSpaceSoftmaxProjection> {
        validate_finite_slice("zspace_softmax_input", row)?;
        let logits = row.iter().copied().map(f64::from).collect::<Vec<_>>();
        project_zspace_softmax(&logits, self.semantic_config()?).map_err(semantic_error_to_tensor)
    }

    fn batch_softmax_fixed_temperature(
        &self,
        input: &Tensor,
        route_backend: TensorUtilBackend,
    ) -> PureResult<Tensor> {
        let (_rows, cols) = input.shape();
        let factor = self.curvature_scale()? / self.fixed_temperature()?;
        validate_finite_value("zspace_softmax_scale", factor)?;
        let logits = input.scale_with_backend(factor, route_backend)?;
        let mut probs = logits.row_softmax_with_backend(current_softmax_backend())?;
        let data = probs.data_mut();
        for row in data.chunks_exact_mut(cols) {
            Self::sanitize_prob_row(row)?;
        }
        Ok(probs)
    }

    fn sanitize_prob_row(probs: &mut [f32]) -> PureResult<()> {
        validate_finite_slice("zspace_softmax_probability", probs)?;
        let prob_sum = probs.iter().copied().try_fold(0.0f32, |acc, prob| {
            let next = acc + prob;
            validate_finite_value("zspace_softmax_probability_sum", next)?;
            Ok(next)
        })?;
        if !prob_sum.is_finite() || prob_sum <= f32::EPSILON {
            let uniform = 1.0 / (probs.len() as f32).max(1.0);
            validate_finite_value("zspace_softmax_probability", uniform)?;
            probs.fill(uniform);
        }
        validate_finite_slice("zspace_softmax_probability", probs)
    }

    fn compute_row(&self, row: &[f32]) -> PureResult<(Vec<f32>, f32, f32)> {
        if row.is_empty() {
            return Ok((Vec::new(), 0.0, self.fixed_temperature()?));
        }
        let projection = self.project_row(row)?;
        Ok((
            f64_slice_to_f32("zspace_softmax_probability", &projection.probabilities)?,
            f64_to_f32("zspace_softmax_entropy", projection.entropy)?,
            f64_to_f32(
                "zspace_softmax_temperature",
                projection.effective_temperature,
            )?,
        ))
    }

    fn entropy_from_probs(probs: &[f32]) -> PureResult<f32> {
        validate_finite_slice("zspace_softmax_probability", probs)?;
        let probabilities = probs.iter().copied().map(f64::from).collect::<Vec<_>>();
        let entropy = zspace_entropy(&probabilities).map_err(semantic_error_to_tensor)?;
        f64_to_f32("zspace_softmax_entropy", entropy)
    }
}

impl Module for ZSpaceSoftmax {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        validate_finite_tensor("zspace_softmax_input", input)?;
        if cols == 0 {
            let output = Tensor::zeros(rows, cols)?;
            let entropies = vec![0.0; rows];
            let temperatures = vec![self.fixed_temperature()?; rows];
            validate_finite_slice("zspace_softmax_entropy", &entropies)?;
            validate_finite_slice("zspace_softmax_temperature", &temperatures)?;
            self.last_entropies.replace(entropies);
            self.last_temperatures.replace(temperatures);
            emit_zspace_softmax_forward_meta(
                rows,
                cols,
                tensor_util_backend_label(current_tensor_util_backend()),
                "cpu",
                "zspace_softmax.forward_empty",
                self.entropy_target.is_some(),
                rows,
                rows,
                self.curvature,
                self.temperature,
                self.min_temperature,
                self.max_temperature,
                self.entropy_target,
            );
            return Ok(output);
        }
        if self.entropy_target.is_none() {
            let route_backend = current_tensor_util_backend_for_values(input.data().len());
            let output = self.batch_softmax_fixed_temperature(input, route_backend)?;
            let entropies = output
                .data()
                .chunks_exact(cols)
                .map(Self::entropy_from_probs)
                .collect::<PureResult<Vec<_>>>()?;
            validate_finite_slice("zspace_softmax_entropy", &entropies)?;
            let temperatures = vec![self.fixed_temperature()?; rows];
            validate_finite_slice("zspace_softmax_temperature", &temperatures)?;
            self.last_entropies.replace(entropies);
            self.last_temperatures.replace(temperatures);
            emit_zspace_softmax_forward_meta(
                rows,
                cols,
                tensor_util_backend_label(current_tensor_util_backend()),
                tensor_util_backend_label(route_backend),
                "zspace_softmax.forward_fixed",
                false,
                rows,
                rows,
                self.curvature,
                self.temperature,
                self.min_temperature,
                self.max_temperature,
                self.entropy_target,
            );
            return Ok(output);
        }

        let mut output = Vec::with_capacity(rows * cols);
        let mut entropies = Vec::with_capacity(rows);
        let mut temperatures = Vec::with_capacity(rows);
        let data = input.data();
        for r in 0..rows {
            let offset = r * cols;
            let row_slice = &data[offset..offset + cols];
            let (prob, entropy, temp) = self.compute_row(row_slice)?;
            output.extend(prob);
            entropies.push(entropy);
            temperatures.push(temp);
        }
        validate_finite_slice("zspace_softmax_output", &output)?;
        validate_finite_slice("zspace_softmax_entropy", &entropies)?;
        validate_finite_slice("zspace_softmax_temperature", &temperatures)?;
        let output = Tensor::from_vec(rows, cols, output)?;
        self.last_entropies.replace(entropies);
        self.last_temperatures.replace(temperatures);
        emit_zspace_softmax_forward_meta(
            rows,
            cols,
            tensor_util_backend_label(current_tensor_util_backend()),
            "cpu_adaptive",
            "zspace_softmax.forward_adaptive",
            true,
            rows,
            rows,
            self.curvature,
            self.temperature,
            self.min_temperature,
            self.max_temperature,
            self.entropy_target,
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
        validate_finite_tensor("zspace_softmax_backward_input", input)?;
        validate_finite_tensor("zspace_softmax_backward_grad_output", grad_output)?;
        if cols == 0 {
            let output = Tensor::zeros(rows, cols)?;
            emit_zspace_softmax_backward_meta(
                rows,
                cols,
                tensor_util_backend_label(current_tensor_util_backend_for_values(0)),
                "cpu",
                "zspace_softmax.backward",
                self.entropy_target.is_some(),
                0,
                self.curvature,
                self.temperature,
                None,
            );
            return Ok(output);
        }
        let scale = self.curvature_scale()?;
        validate_finite_value("zspace_softmax_scale", scale)?;
        if self.entropy_target.is_none() {
            let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
            let requested_backend = tensor_util_backend_label(route_backend);
            #[cfg(feature = "wgpu")]
            let mut wgpu_failure: Option<String> = None;

            #[cfg(feature = "wgpu")]
            {
                if matches!(route_backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available()
                {
                    let factor = scale / self.fixed_temperature()?;
                    validate_finite_value("zspace_softmax_backward_factor", factor)?;
                    match wgpu_dense::zspace_softmax_backward_fixed(
                        input.data(),
                        grad_output.data(),
                        rows,
                        cols,
                        factor,
                    ) {
                        Ok(grad) => {
                            validate_finite_slice("zspace_softmax_backward_grad", &grad)?;
                            let output = Tensor::from_vec(rows, cols, grad)?;
                            emit_zspace_softmax_backward_meta(
                                rows,
                                cols,
                                requested_backend,
                                "wgpu_dense",
                                "zspace_softmax.backward_fixed_wgpu",
                                false,
                                0,
                                self.curvature,
                                self.temperature,
                                None,
                            );
                            return Ok(output);
                        }
                        Err(message) if strict_gpu_path() => {
                            return Err(zspace_softmax_wgpu_error(
                                "zspace_softmax_backward",
                                message,
                            ));
                        }
                        Err(message) => {
                            wgpu_failure = Some(message);
                        }
                    }
                }
            }

            let probs = self.batch_softmax_fixed_temperature(input, route_backend)?;
            let factor = scale / self.fixed_temperature()?;
            validate_finite_value("zspace_softmax_backward_factor", factor)?;
            let mut grad = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                let offset = r * cols;
                let prob = &probs.data()[offset..offset + cols];
                let grad_slice = &grad_output.data()[offset..offset + cols];
                let mut dot = 0.0f32;
                for (g, p) in grad_slice.iter().zip(prob.iter()) {
                    let term = g * p;
                    validate_finite_value("zspace_softmax_backward_dot_term", term)?;
                    dot += term;
                    validate_finite_value("zspace_softmax_backward_dot", dot)?;
                }
                for (g_out, p) in grad_slice.iter().zip(prob.iter()) {
                    let centered = g_out - dot;
                    validate_finite_value("zspace_softmax_backward_centered_grad", centered)?;
                    let value = factor * p * centered;
                    validate_finite_value("zspace_softmax_backward_grad", value)?;
                    grad.push(value);
                }
            }
            let output = Tensor::from_vec(rows, cols, grad)?;
            emit_zspace_softmax_backward_meta(
                rows,
                cols,
                requested_backend,
                "cpu",
                "zspace_softmax.backward",
                false,
                0,
                self.curvature,
                self.temperature,
                {
                    #[cfg(feature = "wgpu")]
                    {
                        wgpu_failure.as_deref()
                    }
                    #[cfg(not(feature = "wgpu"))]
                    {
                        None
                    }
                },
            );
            return Ok(output);
        }
        let mut grad = Vec::with_capacity(rows * cols);
        let input_data = input.data();
        let grad_output_data = grad_output.data();
        let mut temperature_gradient_rows = 0usize;
        let semantic_config = self.semantic_config()?;
        let curvature_scale = semantic_config.curvature_scale();
        let entropy_gain = semantic_config.entropy_gain();
        for r in 0..rows {
            let offset = r * cols;
            let row_slice = &input_data[offset..offset + cols];
            let grad_slice = &grad_output_data[offset..offset + cols];
            let projection = self.project_row(row_slice)?;
            let mut dot = 0.0f64;
            for (&g, &p) in grad_slice.iter().zip(projection.probabilities.iter()) {
                let term = f64::from(g) * p;
                if !term.is_finite() {
                    return Err(TensorError::InvalidValue {
                        label: "zspace_softmax_backward_dot_term",
                    });
                }
                dot += term;
                if !dot.is_finite() {
                    return Err(TensorError::InvalidValue {
                        label: "zspace_softmax_backward_dot",
                    });
                }
            }
            let factor = projection.scale;
            let mut row_grad = Vec::with_capacity(cols);
            let mut dloss_dscale = 0.0f64;
            for ((&g_out, &p), &input_value) in grad_slice
                .iter()
                .zip(projection.probabilities.iter())
                .zip(row_slice.iter())
            {
                let grad_scaled_logit = p * (f64::from(g_out) - dot);
                if !grad_scaled_logit.is_finite() {
                    return Err(TensorError::InvalidValue {
                        label: "zspace_softmax_backward_scaled_logit_grad",
                    });
                }
                dloss_dscale += grad_scaled_logit * f64::from(input_value);
                if !dloss_dscale.is_finite() {
                    return Err(TensorError::InvalidValue {
                        label: "zspace_softmax_backward_dloss_dscale",
                    });
                }
                let value = factor * grad_scaled_logit;
                if !value.is_finite() {
                    return Err(TensorError::InvalidValue {
                        label: "zspace_softmax_backward_grad",
                    });
                }
                row_grad.push(value);
            }
            if projection.temperature_gradient_active {
                temperature_gradient_rows = temperature_gradient_rows.saturating_add(1);
                let dloss_dtemperature = dloss_dscale
                    * (-curvature_scale
                        / (projection.effective_temperature * projection.effective_temperature));
                if !dloss_dtemperature.is_finite() {
                    return Err(TensorError::InvalidValue {
                        label: "zspace_softmax_backward_dloss_dtemperature",
                    });
                }
                for (idx, grad_value) in row_grad.iter_mut().enumerate() {
                    let prob0 = projection.base_probabilities[idx];
                    let log_prob0 = prob0.max(ZSPACE_SOFTMAX_LOG_FLOOR).ln();
                    let dentropy_dx =
                        -projection.base_scale * prob0 * (log_prob0 + projection.base_entropy);
                    if !dentropy_dx.is_finite() {
                        return Err(TensorError::InvalidValue {
                            label: "zspace_softmax_backward_dentropy_dx",
                        });
                    }
                    let dtemperature_dx = -projection.base_temperature * entropy_gain * dentropy_dx;
                    if !dtemperature_dx.is_finite() {
                        return Err(TensorError::InvalidValue {
                            label: "zspace_softmax_backward_dtemperature_dx",
                        });
                    }
                    *grad_value += dloss_dtemperature * dtemperature_dx;
                    if !grad_value.is_finite() {
                        return Err(TensorError::InvalidValue {
                            label: "zspace_softmax_backward_grad",
                        });
                    }
                }
            }
            grad.extend(f64_slice_to_f32("zspace_softmax_backward_grad", &row_grad)?);
        }
        let output = Tensor::from_vec(rows, cols, grad)?;
        emit_zspace_softmax_backward_meta(
            rows,
            cols,
            tensor_util_backend_label(current_tensor_util_backend_for_values(
                rows.saturating_mul(cols),
            )),
            "cpu",
            "zspace_softmax.backward_adaptive",
            true,
            temperature_gradient_rows,
            self.curvature,
            self.temperature,
            None,
        );
        Ok(output)
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
    use std::sync::{Arc, Mutex};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_global_state_lock()
    }

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
    fn zspace_softmax_forward_emits_row_softmax_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let layer = ZSpaceSoftmax::new(-1.0, 1.0).unwrap();
        let input = Tensor::from_vec(2, 3, vec![1.0, 0.0, -1.0, 0.5, -0.25, 0.75]).unwrap();
        let _ = layer.forward(&input).unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let softmax = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "row_softmax" && data["rows"] == 2 && data["cols"] == 3
            })
            .expect("row softmax metadata event");
        assert!(softmax.1["backend"].as_str().is_some());

        let zspace = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_softmax_forward" && data["rows"] == 2 && data["cols"] == 3
            })
            .expect("zspace softmax forward metadata event");
        assert_eq!(zspace.1["kind"], "activation_forward");
        assert_eq!(zspace.1["kernel"], "zspace_softmax.forward_fixed");
        assert_eq!(zspace.1["requested_backend"], "auto");
        assert_eq!(zspace.1["backend"], "auto");
        assert_eq!(zspace.1["softmax_requested_backend"], "auto");
        assert_eq!(zspace.1["adaptive_temperature"], false);
        assert_eq!(zspace.1["entropy_rows"], 2);
        assert_eq!(zspace.1["temperature_rows"], 2);
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
    fn zspace_softmax_backward_emits_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut layer = ZSpaceSoftmax::new(-1.0, 1.5).unwrap();
        let input = Tensor::from_vec(1, 3, vec![0.2, -0.1, 0.3]).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![0.05, -0.02, 0.1]).unwrap();
        let _ = layer.backward(&input, &grad_out).unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_softmax_backward" && data["rows"] == 1 && data["cols"] == 3
            })
            .expect("zspace softmax backward metadata event");
        assert_eq!(backward.1["backend"], "cpu");
        assert_eq!(backward.1["requested_backend"], "auto");
        assert_eq!(backward.1["kind"], "activation_backward");
        assert_eq!(backward.1["adaptive_temperature"], false);
        assert_eq!(backward.1["temperature_gradient"], "constant");
        assert_eq!(backward.1["temperature_gradient_rows"], 0);
    }

    #[test]
    fn zspace_softmax_empty_rows_preserve_shape_and_metrics() {
        let mut layer = ZSpaceSoftmax::new(-1.0, 1.0)
            .unwrap()
            .with_entropy_target(0.5, 1e-3, 0.5)
            .unwrap();
        let input = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (0, 3));
        assert!(output.data().is_empty());
        assert!(layer.last_entropies().is_empty());
        assert!(layer.last_temperatures().is_empty());

        let grad_out = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let grad_in = layer.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), (0, 3));
        assert!(grad_in.data().is_empty());
    }

    #[test]
    fn zspace_softmax_rejects_non_finite_input_without_mutating_metrics() {
        let layer = ZSpaceSoftmax::new(-1.0, 1.0)
            .unwrap()
            .with_entropy_target(0.5, 1e-3, 0.5)
            .unwrap();
        let input = Tensor::from_vec(1, 3, vec![0.2, -0.1, 0.3]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let entropies_before = layer.last_entropies();
        let temperatures_before = layer.last_temperatures();
        let bad_input = Tensor::from_vec(1, 3, vec![0.2, f32::NAN, 0.3]).unwrap();

        let err = layer.forward(&bad_input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_softmax_input",
                value,
            } if value.is_nan()
        ));
        assert_eq!(layer.last_entropies(), entropies_before);
        assert_eq!(layer.last_temperatures(), temperatures_before);
    }

    #[test]
    fn zspace_softmax_handles_extreme_finite_logits_without_overflow() {
        let layer = ZSpaceSoftmax::new(-4.0, 1.0)
            .unwrap()
            .with_entropy_target(0.5, 1e-3, 0.5)
            .unwrap();
        let input = Tensor::from_vec(1, 3, vec![f32::MAX, 0.0, -f32::MAX]).unwrap();

        let output = layer.forward(&input).unwrap();

        assert!(output.data().iter().all(|value| value.is_finite()));
        assert!((output.data().iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(output.data()[0], 1.0);
        assert_eq!(output.data()[1], 0.0);
        assert_eq!(output.data()[2], 0.0);
    }

    #[test]
    fn zspace_softmax_backward_rejects_non_finite_grad_output() {
        let mut layer = ZSpaceSoftmax::new(-1.0, 1.0).unwrap();
        let input = Tensor::from_vec(1, 3, vec![0.2, -0.1, 0.3]).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![0.05, f32::NAN, 0.1]).unwrap();

        let err = layer.backward(&input, &grad_out).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_softmax_backward_grad_output",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn zspace_softmax_backward_rejects_overflowing_gradient() {
        let mut layer = ZSpaceSoftmax::new(-4.0, 1.0e-3).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.0, 0.0]).unwrap();
        let grad_out = Tensor::from_vec(1, 2, vec![f32::MAX, -f32::MAX]).unwrap();

        let err = layer.backward(&input, &grad_out).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zspace_softmax_backward_grad",
                value,
            } if value.is_infinite()
        ));
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

    #[test]
    fn zspace_softmax_fixed_forward_matches_the_st_core_projection() {
        let layer = ZSpaceSoftmax::new(-1.0, 1.5)
            .unwrap()
            .with_temperature_bounds(0.1, 4.0)
            .unwrap();
        let logits = [2.0_f32, 0.5, -1.0];
        let input = Tensor::from_vec(1, logits.len(), logits.to_vec()).unwrap();

        let output = layer.forward(&input).unwrap();
        let expected = project_zspace_softmax(
            &logits.iter().copied().map(f64::from).collect::<Vec<_>>(),
            ZSpaceSoftmaxConfig::new(-1.0, 1.5)
                .unwrap()
                .with_temperature_bounds(0.1, 4.0)
                .unwrap(),
        )
        .unwrap();

        for (&actual, &expected) in output.data().iter().zip(expected.probabilities.iter()) {
            assert!((f64::from(actual) - expected).abs() < 1.0e-6);
        }
        assert!((f64::from(layer.last_entropies()[0]) - expected.entropy).abs() < 1.0e-6);
        assert!(
            (f64::from(layer.last_temperatures()[0]) - expected.effective_temperature).abs()
                < 1.0e-6
        );
    }

    #[test]
    fn zspace_softmax_adaptive_forward_matches_the_st_core_projection() {
        let layer = ZSpaceSoftmax::new(-1.0, 1.5)
            .unwrap()
            .with_entropy_target(1.3, 1.0e-6, 0.8)
            .unwrap()
            .with_temperature_bounds(0.1, 4.0)
            .unwrap();
        let logits = [2.0_f32, 0.5, -1.0];
        let input = Tensor::from_vec(1, logits.len(), logits.to_vec()).unwrap();

        let output = layer.forward(&input).unwrap();
        let expected = project_zspace_softmax(
            &logits.iter().copied().map(f64::from).collect::<Vec<_>>(),
            ZSpaceSoftmaxConfig::new(-1.0, 1.5)
                .unwrap()
                .with_entropy_target(1.3, 1.0e-6, 0.8)
                .unwrap()
                .with_temperature_bounds(0.1, 4.0)
                .unwrap(),
        )
        .unwrap();

        for (&actual, &expected) in output.data().iter().zip(expected.probabilities.iter()) {
            assert!((f64::from(actual) - expected).abs() < 1.0e-6);
        }
        assert!((f64::from(layer.last_entropies()[0]) - expected.entropy).abs() < 1.0e-6);
        assert!(
            (f64::from(layer.last_temperatures()[0]) - expected.effective_temperature).abs()
                < 1.0e-6
        );
    }

    #[test]
    fn zspace_softmax_adaptive_forward_emits_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let layer = ZSpaceSoftmax::new(-1.0, 1.0)
            .unwrap()
            .with_entropy_target(0.1, 1e-3, 1.0)
            .unwrap()
            .with_temperature_bounds(0.05, 2.0)
            .unwrap();
        let input = Tensor::from_vec(1, 3, vec![0.0, 0.0, 0.0]).unwrap();
        let _ = layer.forward(&input).unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let zspace = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_softmax_forward" && data["rows"] == 1 && data["cols"] == 3
            })
            .expect("adaptive zspace softmax forward metadata event");
        assert_eq!(zspace.1["backend"], "cpu_adaptive");
        assert_eq!(zspace.1["kernel"], "zspace_softmax.forward_adaptive");
        assert_eq!(zspace.1["adaptive_temperature"], true);
        let entropy_target = zspace.1["entropy_target"].as_f64().unwrap();
        assert!((entropy_target - 0.1).abs() < 1.0e-6);
        assert_eq!(zspace.1["entropy_rows"], 1);
        assert_eq!(zspace.1["temperature_rows"], 1);
    }

    #[test]
    fn zspace_softmax_adaptive_backward_matches_finite_difference() {
        let mut layer = ZSpaceSoftmax::new(-1.0, 1.0)
            .unwrap()
            .with_entropy_target(0.45, 1e-5, 0.6)
            .unwrap()
            .with_temperature_bounds(0.1, 4.0)
            .unwrap();
        let input_values = vec![1.1, -0.45, 0.25];
        let input = Tensor::from_vec(1, 3, input_values.clone()).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![0.4, -0.15, 0.25]).unwrap();

        let _ = layer.forward(&input).unwrap();
        let temperatures = layer.last_temperatures();
        assert_eq!(temperatures.len(), 1);
        assert!((temperatures[0] - layer.fixed_temperature().unwrap()).abs() > 1.0e-4);
        let grad_input = layer.backward(&input, &grad_out).unwrap();
        let analytic = grad_input.data()[0];

        let epsilon = 1.0e-3f32;
        let loss_at = |values: Vec<f32>| {
            let tensor = Tensor::from_vec(1, 3, values).unwrap();
            let out = layer.forward(&tensor).unwrap();
            out.data()
                .iter()
                .zip(grad_out.data().iter())
                .map(|(&value, &grad)| value * grad)
                .sum::<f32>()
        };
        let mut plus = input_values.clone();
        plus[0] += epsilon;
        let mut minus = input_values;
        minus[0] -= epsilon;
        let finite_difference = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon);

        assert!(
            (analytic - finite_difference).abs() < 3.0e-3,
            "analytic={analytic} finite_difference={finite_difference}"
        );
    }

    #[test]
    fn zspace_softmax_adaptive_backward_meta_reports_temperature_gradient() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut layer = ZSpaceSoftmax::new(-1.0, 1.0)
            .unwrap()
            .with_entropy_target(0.45, 1e-5, 0.6)
            .unwrap()
            .with_temperature_bounds(0.1, 4.0)
            .unwrap();
        let input = Tensor::from_vec(1, 3, vec![1.1, -0.45, 0.25]).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![0.4, -0.15, 0.25]).unwrap();
        let _ = layer.backward(&input, &grad_out).unwrap();
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_softmax_backward" && data["rows"] == 1 && data["cols"] == 3
            })
            .expect("zspace softmax backward metadata event");
        assert_eq!(backward.1["adaptive_temperature"], true);
        assert_eq!(backward.1["temperature_gradient"], "one_step_entropy_exact");
        assert_eq!(backward.1["temperature_gradient_rows"], 1);
    }
}
