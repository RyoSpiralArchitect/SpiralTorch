// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::{current_tensor_util_backend, current_tensor_util_backend_for_values};
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;
use st_tensor::{
    emit_tensor_op, emit_tensor_op_meta,
    topos::{OpenCartesianTopos, RewriteMonad},
    LanguageWaveEncoder, TensorUtilBackend,
};

#[allow(clippy::too_many_arguments)]
fn emit_wave_gate_forward_meta(
    rows: usize,
    cols: usize,
    backend: &'static str,
    requested_backend: &'static str,
    selected_backend: TensorUtilBackend,
    kernel: &'static str,
    curvature: f32,
    saturation: f32,
    porosity: f32,
    fused_projection: bool,
) {
    emit_tensor_op(
        "wave_gate_forward",
        &[rows, cols, 1, cols, 1, cols],
        &[rows, cols],
    );
    emit_tensor_op_meta("wave_gate_forward", || {
        let values = rows.saturating_mul(cols);
        serde_json::json!({
            "backend": backend,
            "requested_backend": requested_backend,
            "selected_backend": tensor_util_backend_label(selected_backend),
            "kernel": kernel,
            "kind": "broadcast_gate_forward",
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "gate_cols": cols,
            "bias_cols": cols,
            "trainable_parameters": cols.saturating_mul(2),
            "curvature": curvature,
            "saturation": saturation,
            "porosity": porosity,
            "effective_gate_rewrite": true,
            "projection": "poincare",
            "fused_projection": fused_projection,
            "estimated_broadcast_ops": values,
            "estimated_projection_ops": values.saturating_mul(3),
            "estimated_total_ops": values.saturating_mul(if fused_projection { 4 } else { 5 }),
            "empty": rows == 0 || cols == 0,
        })
    });
}

fn emit_wave_gate_backward_meta(
    rows: usize,
    cols: usize,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    gradient_reduction_backend: TensorUtilBackend,
    curvature: f32,
    saturation: f32,
    porosity: f32,
    effective_gate_rewrite: bool,
    projection_gradient: bool,
    gradient_scale: Option<f32>,
    fallback_message: Option<&str>,
) {
    emit_tensor_op(
        "wave_gate_backward",
        &[rows, cols, rows, cols, 1, cols, 1, cols],
        &[rows, cols],
    );
    emit_tensor_op_meta("wave_gate_backward", || {
        let values = rows.saturating_mul(cols);
        let mut data = serde_json::json!({
            "backend": backend,
            "requested_backend": requested_backend,
            "selected_backend": tensor_util_backend_label(gradient_reduction_backend),
            "kernel": kernel,
            "kind": "broadcast_gate_backward",
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "gate_cols": cols,
            "bias_cols": cols,
            "trainable_parameters": cols.saturating_mul(2),
            "gradient_reduction_backend": gradient_reduction_backend.to_string(),
            "gradient_scale": gradient_scale,
            "parameter_gradient_scale": gradient_scale,
            "input_gradient_scale": if gradient_scale.is_some() {
                Some(1.0f32)
            } else {
                None
            },
            "curvature": curvature,
            "saturation": saturation,
            "porosity": porosity,
            "effective_gate_rewrite": effective_gate_rewrite,
            "projection_gradient": projection_gradient,
            "saturation_gradient": "porous_mix_exact",
            "estimated_broadcast_ops": values,
            "estimated_gate_gradient_ops": values.saturating_mul(2),
            "estimated_bias_gradient_ops": values,
            "estimated_projection_gradient_ops": values.saturating_mul(4),
            "estimated_saturation_gradient_ops": values.saturating_mul(3),
            "estimated_total_ops": values.saturating_mul(11),
            "empty": rows == 0 || cols == 0,
        });
        if let Some(message) = fallback_message {
            data["fallback"] = serde_json::json!({"from": "wgpu", "message": message});
        }
        data
    });
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
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(feature = "wgpu")]
fn wave_gate_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
    }
}

fn porous_saturation_backward_factor(value: f32, saturation: f32, porosity: f32) -> f32 {
    if !value.is_finite() || saturation <= 0.0 {
        return 0.0;
    }
    let limit = saturation.abs();
    let magnitude = value.abs();
    if magnitude <= limit {
        return 1.0;
    }
    if porosity <= f32::EPSILON {
        return 0.0;
    }
    let absorb = (porosity * 0.25).min(1.0);
    let denom = magnitude + limit;
    if denom <= f32::EPSILON {
        return 0.0;
    }
    -2.0 * limit * limit * absorb / (denom * denom)
}

fn porous_saturation_backward(
    pre_saturation: &Tensor,
    grad_saturated: &Tensor,
    saturation: f32,
    porosity: f32,
) -> PureResult<Tensor> {
    if pre_saturation.shape() != grad_saturated.shape() {
        return Err(TensorError::ShapeMismatch {
            left: pre_saturation.shape(),
            right: grad_saturated.shape(),
        });
    }
    let (rows, cols) = pre_saturation.shape();
    let data = pre_saturation
        .data()
        .iter()
        .zip(grad_saturated.data().iter())
        .map(|(&value, &grad)| {
            grad * porous_saturation_backward_factor(value, saturation, porosity)
        })
        .collect();
    Tensor::from_vec(rows, cols, data)
}

fn poincare_projection_backward(
    preprojected: &Tensor,
    grad_projected: &Tensor,
    curvature: f32,
) -> PureResult<Tensor> {
    if preprojected.shape() != grad_projected.shape() {
        return Err(TensorError::ShapeMismatch {
            left: preprojected.shape(),
            right: grad_projected.shape(),
        });
    }
    if curvature >= 0.0 {
        return Err(TensorError::NonHyperbolicCurvature { curvature });
    }
    let (rows, cols) = preprojected.shape();
    let scale = (-curvature).sqrt();
    let mut data = vec![0.0f32; rows.saturating_mul(cols)];
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let x = &preprojected.data()[start..end];
        let grad = &grad_projected.data()[start..end];
        let norm = x.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= f32::EPSILON {
            let factor = if scale > 0.0 { 1.0 / scale } else { 1.0 };
            for col in 0..cols {
                data[start + col] = grad[col] * factor;
            }
            continue;
        }
        let tanh = (norm / scale).tanh();
        let factor = tanh / norm;
        let sech2 = 1.0 - tanh * tanh;
        let radial = ((sech2 * norm / scale) - tanh) / (norm * norm * norm);
        let dot = x
            .iter()
            .zip(grad.iter())
            .map(|(&value, &grad)| value * grad)
            .sum::<f32>();
        for col in 0..cols {
            data[start + col] = factor * grad[col] + radial * x[col] * dot;
        }
    }
    Tensor::from_vec(rows, cols, data)
}

/// Hyperbolic feature gate that mixes LanguageWaveEncoder spectra with module tensors.
#[derive(Debug)]
pub struct WaveGate {
    gate: Parameter,
    bias: Parameter,
    topos: OpenCartesianTopos,
    encoder: LanguageWaveEncoder,
}

impl WaveGate {
    /// Creates a wave gate with deterministic small parameters and an inferred topos.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        curvature: f32,
        temperature: f32,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        let encoder = LanguageWaveEncoder::new(curvature, temperature)?;
        let depth = features.saturating_mul(64).max(64);
        let volume = features.saturating_mul(1024).max(1024);
        let topos = OpenCartesianTopos::new(curvature, 1e-6, 1e4, depth, volume)?;
        Self::with_topos(name, features, encoder, topos)
    }

    /// Builds a wave gate with explicit encoder/topos wiring.
    pub fn with_topos(
        name: impl Into<String>,
        features: usize,
        encoder: LanguageWaveEncoder,
        topos: OpenCartesianTopos,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if (encoder.curvature() - topos.curvature()).abs() > 1e-6 {
            return Err(TensorError::CurvatureMismatch {
                expected: topos.curvature(),
                got: encoder.curvature(),
            });
        }
        let name = name.into();
        let gate = Tensor::from_fn(1, features, |_r, c| ((c as f32 + 1.0) * 0.01).sin() * 0.1)?;
        let bias = Tensor::zeros(1, features)?;
        Ok(Self {
            gate: Parameter::new(format!("{name}::gate"), gate),
            bias: Parameter::new(format!("{name}::bias"), bias),
            topos,
            encoder,
        })
    }

    /// Returns the open-cartesian guard used by the gate.
    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    /// Returns the internal encoder used for text-driven updates.
    pub fn encoder(&self) -> &LanguageWaveEncoder {
        &self.encoder
    }

    /// Returns an immutable reference to the gate parameter.
    pub fn gate(&self) -> &Parameter {
        &self.gate
    }

    /// Returns an immutable reference to the bias parameter.
    pub fn bias(&self) -> &Parameter {
        &self.bias
    }

    fn gate_len(&self) -> usize {
        self.gate.value().shape().1
    }

    /// Runs backward with an explicit scale for broadcast parameter gradients.
    ///
    /// Standalone wave gates average over their input rows. Sequence wrappers can
    /// override that scale when rows contain unfolded timesteps rather than
    /// independent training samples.
    pub fn backward_with_parameter_gradient_scale(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
        parameter_gradient_scale: f32,
    ) -> PureResult<Tensor> {
        if parameter_gradient_scale <= 0.0 || !parameter_gradient_scale.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "wave_gate_parameter_gradient_scale",
            });
        }
        self.backward_impl(input, grad_output, Some(parameter_gradient_scale))
    }

    fn backward_impl(
        &mut self,
        input: &Tensor,
        grad_output: &Tensor,
        parameter_gradient_scale: Option<f32>,
    ) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if grad_output.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, cols),
            });
        }
        if cols != self.gate_len() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.gate.value().shape(),
            });
        }
        let requested_backend = current_tensor_util_backend();
        let reduction_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(requested_backend);
        if rows == 0 {
            let grad_input = Tensor::zeros(rows, cols)?;
            emit_wave_gate_backward_meta(
                rows,
                cols,
                "cpu",
                requested_backend,
                "wave_gate.scalar_backward",
                reduction_backend,
                self.topos.curvature(),
                self.topos.saturation(),
                self.topos.porosity(),
                true,
                true,
                None,
                None,
            );
            return Ok(grad_input);
        }
        let mut gate = self.gate.value().clone();
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("wave_gate_backward_gate_rewrite", &mut gate)?;
        let gate_data = gate.data();
        let bias_data = self.bias.value().data();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(reduction_backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available()
            {
                match wgpu_dense::wave_gate_backward(
                    input.data(),
                    grad_output.data(),
                    gate_data,
                    bias_data,
                    rows,
                    cols,
                    self.topos.curvature(),
                    self.topos.saturation(),
                    self.topos.porosity(),
                ) {
                    Ok((grad_input_data, grad_affine_data)) => {
                        let grad_affine = Tensor::from_vec(rows, cols, grad_affine_data)?;
                        let gradient_scale = parameter_gradient_scale.unwrap_or(1.0 / rows as f32);
                        let grad_gate_product =
                            grad_affine.hadamard_with_backend(input, reduction_backend)?;
                        let mut grad_gate = Tensor::from_vec(
                            1,
                            cols,
                            grad_gate_product.try_sum_axis0_scaled_with_backend(
                                gradient_scale,
                                reduction_backend,
                            )?,
                        )?;
                        let mut grad_bias = Tensor::from_vec(
                            1,
                            cols,
                            grad_affine.try_sum_axis0_scaled_with_backend(
                                gradient_scale,
                                reduction_backend,
                            )?,
                        )?;
                        let grad_input = Tensor::from_vec(rows, cols, grad_input_data)?;
                        monad.rewrite_tensor("wave_gate_grad_gate", &mut grad_gate)?;
                        monad.rewrite_tensor("wave_gate_grad_bias", &mut grad_bias)?;
                        self.gate.accumulate_euclidean(&grad_gate)?;
                        self.bias.accumulate_euclidean(&grad_bias)?;
                        self.topos
                            .guard_tensor("wave_gate_backward_out", &grad_input)?;
                        emit_wave_gate_backward_meta(
                            rows,
                            cols,
                            "wgpu_dense",
                            requested_backend,
                            "wave_gate.backward_wgpu",
                            reduction_backend,
                            self.topos.curvature(),
                            self.topos.saturation(),
                            self.topos.porosity(),
                            true,
                            true,
                            Some(gradient_scale),
                            None,
                        );
                        return Ok(grad_input);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(wave_gate_wgpu_error("wave_gate_backward", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut pre_saturation = Tensor::zeros(rows, cols)?;
        {
            let out_buf = pre_saturation.data_mut();
            let input_buf = input.data();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    out_buf[offset + c] = input_buf[offset + c] * gate_data[c] + bias_data[c];
                }
            }
        }
        let mut preprojected = pre_saturation.clone();
        monad.rewrite_tensor("wave_gate_backward_preprojected_rewrite", &mut preprojected)?;
        let grad_saturated =
            poincare_projection_backward(&preprojected, grad_output, self.topos.curvature())?;
        let grad_affine = porous_saturation_backward(
            &pre_saturation,
            &grad_saturated,
            self.topos.saturation(),
            self.topos.porosity(),
        )?;
        let gradient_scale = parameter_gradient_scale.unwrap_or(1.0 / (rows as f32));
        let grad_gate_product = grad_affine.hadamard_with_backend(input, reduction_backend)?;
        let mut grad_gate = Tensor::from_vec(
            1,
            cols,
            grad_gate_product
                .try_sum_axis0_scaled_with_backend(gradient_scale, reduction_backend)?,
        )?;
        let mut grad_bias = Tensor::from_vec(
            1,
            cols,
            grad_affine.try_sum_axis0_scaled_with_backend(gradient_scale, reduction_backend)?,
        )?;
        let mut grad_input = Tensor::zeros(rows, cols)?;
        {
            let grad_input_buf = grad_input.data_mut();
            let grad_affine_buf = grad_affine.data();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let go = grad_affine_buf[offset + c];
                    grad_input_buf[offset + c] = go * gate_data[c];
                }
            }
        }
        monad.rewrite_tensor("wave_gate_grad_gate", &mut grad_gate)?;
        monad.rewrite_tensor("wave_gate_grad_bias", &mut grad_bias)?;
        self.gate.accumulate_euclidean(&grad_gate)?;
        self.bias.accumulate_euclidean(&grad_bias)?;
        self.topos
            .guard_tensor("wave_gate_backward_out", &grad_input)?;
        emit_wave_gate_backward_meta(
            rows,
            cols,
            "cpu",
            requested_backend,
            "wave_gate.scalar_backward",
            reduction_backend,
            self.topos.curvature(),
            self.topos.saturation(),
            self.topos.porosity(),
            true,
            true,
            Some(gradient_scale),
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
        Ok(grad_input)
    }
}

impl Module for WaveGate {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.gate_len() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.gate.value().shape(),
            });
        }
        self.topos.guard_tensor("wave_gate_forward_in", input)?;
        let mut gate = self.gate.value().clone();
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("wave_gate_gate_rewrite", &mut gate)?;
        let gate_data = gate.data();
        let bias_data = self.bias.value().data();
        let requested_backend = current_tensor_util_backend();
        let selected_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend_label = tensor_util_backend_label(requested_backend);
        if matches!(selected_backend, TensorUtilBackend::GpuWgpu) {
            let projected = input.wave_gate_project_with_backend(
                gate_data,
                bias_data,
                self.topos.curvature(),
                self.topos.saturation(),
                self.topos.porosity(),
                selected_backend,
            )?;
            self.topos
                .guard_tensor("wave_gate_forward_projected", &projected)?;
            emit_wave_gate_forward_meta(
                rows,
                cols,
                "tensor_util",
                requested_backend_label,
                selected_backend,
                "wave_gate.forward_fused",
                self.topos.curvature(),
                self.topos.saturation(),
                self.topos.porosity(),
                true,
            );
            return Ok(projected);
        }
        let input_buf = input.data();
        let mut out = Tensor::zeros(rows, cols)?;
        {
            let out_buf = out.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let value = input_buf[offset + c] * gate_data[c] + bias_data[c];
                    out_buf[offset + c] = value;
                }
            }
        }
        monad.rewrite_tensor("wave_gate_forward_out", &mut out)?;
        let projected =
            out.project_to_poincare_with_backend(self.topos.curvature(), selected_backend)?;
        self.topos
            .guard_tensor("wave_gate_forward_projected", &projected)?;
        emit_wave_gate_forward_meta(
            rows,
            cols,
            "cpu",
            requested_backend_label,
            selected_backend,
            "wave_gate.forward_scalar",
            self.topos.curvature(),
            self.topos.saturation(),
            self.topos.porosity(),
            false,
        );
        Ok(projected)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.backward_impl(input, grad_output, None)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.gate)?;
        visitor(&self.bias)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.gate)?;
        visitor(&mut self.bias)?;
        Ok(())
    }

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        let encoder = self.encoder.clone();
        self.visit_parameters_mut(&mut |param| param.absorb_text(&encoder, text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::{BackendKind, DeviceCaps};
    #[cfg(feature = "wgpu")]
    use st_tensor::{AttentionBackend, LayerNormBackend, MatmulBackend, SoftmaxBackend};
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[cfg(feature = "wgpu")]
    #[track_caller]
    fn assert_tensor_close(lhs: &Tensor, rhs: &Tensor, tolerance: f32) {
        assert_eq!(lhs.shape(), rhs.shape());
        for (idx, (&left, &right)) in lhs.data().iter().zip(rhs.data()).enumerate() {
            let delta = (left - right).abs();
            assert!(
                delta <= tolerance,
                "tensor mismatch at {idx}: left={left} right={right} delta={delta} tolerance={tolerance}"
            );
        }
    }

    #[cfg(feature = "wgpu")]
    fn cpu_policy() -> BackendPolicy {
        BackendPolicy::explicit(
            DeviceCaps::cpu(),
            MatmulBackend::CpuNaive,
            MatmulBackend::CpuNaive,
            LayerNormBackend::Cpu,
            AttentionBackend::Cpu,
            SoftmaxBackend::Cpu,
        )
    }

    #[cfg(feature = "wgpu")]
    fn wgpu_policy() -> BackendPolicy {
        BackendPolicy::explicit(
            BackendKind::Wgpu.default_caps(),
            MatmulBackend::GpuWgpu,
            MatmulBackend::GpuWgpu,
            LayerNormBackend::Cpu,
            AttentionBackend::Cpu,
            SoftmaxBackend::Cpu,
        )
    }

    #[test]
    fn wave_gate_forwards_and_backwards() {
        let mut gate = WaveGate::new("wg", 4, -1.0, 0.5).unwrap();
        let input =
            Tensor::from_vec(2, 4, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]).unwrap();
        let output = gate.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let grad_out =
            Tensor::from_vec(2, 4, vec![0.2, -0.1, 0.05, -0.3, 0.4, -0.2, 0.1, -0.05]).unwrap();
        let grad_in = gate.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
        let before = gate.gate().value().clone();
        gate.apply_step(0.01).unwrap();
        let after = gate.gate().value();
        assert_ne!(before, *after);
    }

    #[test]
    fn wave_gate_attaches_hypergrad() {
        let mut gate = WaveGate::new("wg", 8, -0.75, 0.9).unwrap();
        let topos = gate.topos().clone();
        gate.attach_hypergrad_with_topos(-0.75, 0.04, topos)
            .unwrap();
        let encoder = gate.encoder().clone();
        gate.visit_parameters_mut(&mut |param| param.absorb_text(&encoder, "wave"))
            .unwrap();
        gate.apply_step(0.01).unwrap();
        assert!(gate.gate().value().squared_l2_norm() > 0.0);
    }

    #[test]
    fn wave_gate_backward_emits_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut gate = WaveGate::new("wg", 4, -1.0, 0.5).unwrap();
        let input =
            Tensor::from_vec(2, 4, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]).unwrap();
        let grad_out =
            Tensor::from_vec(2, 4, vec![0.2, -0.1, 0.05, -0.3, 0.4, -0.2, 0.1, -0.05]).unwrap();
        let _ = gate.backward(&input, &grad_out).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "wave_gate_backward" && data["rows"] == 2 && data["cols"] == 4
            })
            .expect("wave gate backward metadata event");
        assert_eq!(backward.1["backend"], "cpu");
        assert_eq!(backward.1["kind"], "broadcast_gate_backward");
        assert_eq!(backward.1["gradient_reduction_backend"], "auto");
        assert_eq!(backward.1["gradient_scale"], 0.5);
        assert_eq!(backward.1["parameter_gradient_scale"], 0.5);
        assert_eq!(backward.1["input_gradient_scale"], 1.0);
        assert_eq!(backward.1["trainable_parameters"], 8);
        assert_eq!(backward.1["effective_gate_rewrite"], true);
        assert_eq!(backward.1["projection_gradient"], true);
        assert_eq!(backward.1["saturation_gradient"], "porous_mix_exact");

        let hadamard = events
            .iter()
            .any(|(op_name, data)| *op_name == "hadamard" && data["rows"] == 2);
        let reduction = events
            .iter()
            .filter(|(op_name, data)| *op_name == "sum_axis0_scaled" && data["cols"] == 4)
            .count();
        assert!(hadamard);
        assert!(reduction >= 2);
    }

    #[test]
    fn wave_gate_forward_emits_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let gate = WaveGate::new("wg", 4, -1.0, 0.5).unwrap();
        let input =
            Tensor::from_vec(2, 4, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]).unwrap();
        let _ = gate.forward(&input).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "wave_gate_forward" && data["rows"] == 2 && data["cols"] == 4
            })
            .expect("wave gate forward metadata event");
        assert_eq!(forward.1["backend"], "cpu");
        assert_eq!(forward.1["requested_backend"], "auto");
        assert_eq!(forward.1["selected_backend"], "auto");
        assert_eq!(forward.1["kernel"], "wave_gate.forward_scalar");
        assert_eq!(forward.1["kind"], "broadcast_gate_forward");
        assert_eq!(forward.1["fused_projection"], false);
        assert_eq!(forward.1["trainable_parameters"], 8);
        assert_eq!(forward.1["effective_gate_rewrite"], true);
        assert_eq!(forward.1["projection"], "poincare");

        let projection = events
            .iter()
            .any(|(op_name, data)| *op_name == "project_to_poincare" && data["rows"] == 2);
        assert!(projection);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wave_gate_forward_wgpu_policy_reports_threshold_cpu_route() {
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1024");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let gate = WaveGate::new("wg", 4, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(1, 4, vec![0.1, -0.2, 0.3, -0.4]).unwrap();
        {
            let _guard = push_backend_policy(wgpu_policy());
            let _ = gate.forward(&input).unwrap();
        }

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        let events = events.lock().unwrap();
        let (_, route) = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "tensor_util_route"
                    && data["requested_backend"] == "wgpu"
                    && data["selected_backend"] == "cpu"
            })
            .expect("tensor util threshold route");
        assert_eq!(route["status"], "cpu_threshold");

        let (_, forward) = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "wave_gate_forward" && data["rows"] == 1 && data["cols"] == 4
            })
            .expect("wave gate forward metadata event");
        assert_eq!(forward["backend"], "cpu");
        assert_eq!(forward["requested_backend"], "wgpu");
        assert_eq!(forward["selected_backend"], "cpu");
        assert_eq!(forward["kernel"], "wave_gate.forward_scalar");
        assert_eq!(forward["fused_projection"], false);
    }

    #[test]
    fn wave_gate_parameter_gradients_are_batch_normalized() {
        let mut single = WaveGate::new("wg", 3, -1.0, 0.5).unwrap();
        let input_single = Tensor::from_vec(1, 3, vec![0.2, -0.3, 0.45]).unwrap();
        let grad_single = Tensor::from_vec(1, 3, vec![0.15, -0.25, 0.35]).unwrap();
        let grad_input_single = single.backward(&input_single, &grad_single).unwrap();
        assert_eq!(grad_input_single.shape(), input_single.shape());

        let mut repeated = WaveGate::new("wg", 3, -1.0, 0.5).unwrap();
        let input_repeated =
            Tensor::from_vec(2, 3, vec![0.2, -0.3, 0.45, 0.2, -0.3, 0.45]).unwrap();
        let grad_repeated =
            Tensor::from_vec(2, 3, vec![0.15, -0.25, 0.35, 0.15, -0.25, 0.35]).unwrap();
        let grad_input_repeated = repeated.backward(&input_repeated, &grad_repeated).unwrap();
        assert_eq!(grad_input_repeated.shape(), input_repeated.shape());

        let single_gate = single.gate().gradient().expect("single gate gradient");
        let repeated_gate = repeated.gate().gradient().expect("repeated gate gradient");
        let single_bias = single.bias().gradient().expect("single bias gradient");
        let repeated_bias = repeated.bias().gradient().expect("repeated bias gradient");

        for (idx, (&single_value, &repeated_value)) in single_gate
            .data()
            .iter()
            .zip(repeated_gate.data().iter())
            .enumerate()
        {
            let delta = (single_value - repeated_value).abs();
            assert!(
                delta <= 1.0e-6,
                "gate gradient mismatch at {idx}: single={single_value} repeated={repeated_value} delta={delta}"
            );
        }
        for (idx, (&single_value, &repeated_value)) in single_bias
            .data()
            .iter()
            .zip(repeated_bias.data().iter())
            .enumerate()
        {
            let delta = (single_value - repeated_value).abs();
            assert!(
                delta <= 1.0e-6,
                "bias gradient mismatch at {idx}: single={single_value} repeated={repeated_value} delta={delta}"
            );
        }
    }

    #[test]
    fn wave_gate_empty_batch_returns_empty_grad_without_parameter_updates() {
        let mut gate = WaveGate::new("wg", 3, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let grad_out = Tensor::from_vec(0, 3, Vec::new()).unwrap();

        let grad_input = gate.backward(&input, &grad_out).unwrap();

        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(gate.gate().gradient().is_none());
        assert!(gate.bias().gradient().is_none());
    }

    #[test]
    fn wave_gate_backward_uses_rewritten_effective_gate_for_grad_input() {
        let mut gate = WaveGate::new("wg", 2, -1.0, 0.5).unwrap();
        gate.gate
            .value_mut()
            .data_mut()
            .copy_from_slice(&[20_000.0, -30_000.0]);
        let input = Tensor::from_vec(1, 2, vec![0.2, -0.3]).unwrap();
        let grad_out = Tensor::from_vec(1, 2, vec![0.5, -0.25]).unwrap();

        let grad_in = gate.backward(&input, &grad_out).unwrap();
        let effective_gate = [
            gate.topos().saturate(20_000.0),
            gate.topos().saturate(-30_000.0),
        ];
        let preprojected = Tensor::from_vec(
            1,
            2,
            vec![
                input.data()[0] * effective_gate[0],
                input.data()[1] * effective_gate[1],
            ],
        )
        .unwrap();
        let grad_preprojected =
            poincare_projection_backward(&preprojected, &grad_out, gate.topos().curvature())
                .unwrap();
        let expected = [
            grad_preprojected.data()[0] * effective_gate[0],
            grad_preprojected.data()[1] * effective_gate[1],
        ];

        for (idx, (&actual, &expected)) in grad_in.data().iter().zip(expected.iter()).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                delta <= 1.0e-2,
                "grad_input mismatch at {idx}: actual={actual} expected={expected} delta={delta}"
            );
        }
        assert!(grad_in.data()[0].abs() < 10_000.0);
        assert!(grad_in.data()[1].abs() < 10_000.0);
    }

    #[test]
    fn wave_gate_backward_matches_projection_finite_difference_for_input() {
        let mut gate = WaveGate::new("wg", 3, -1.0, 0.7).unwrap();
        gate.gate
            .value_mut()
            .data_mut()
            .copy_from_slice(&[1.2, -0.8, 0.55]);
        gate.bias
            .value_mut()
            .data_mut()
            .copy_from_slice(&[0.15, -0.05, 0.2]);
        let input_values = vec![0.7, -0.4, 0.9];
        let input = Tensor::from_vec(1, 3, input_values.clone()).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![0.4, -0.25, 0.35]).unwrap();

        let _ = gate.forward(&input).unwrap();
        let grad_input = gate.backward(&input, &grad_out).unwrap();
        let analytic = grad_input.data()[0];

        let epsilon = 1.0e-3f32;
        let loss_at = |values: Vec<f32>| {
            let tensor = Tensor::from_vec(1, 3, values).unwrap();
            let out = gate.forward(&tensor).unwrap();
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
    fn wave_gate_backward_matches_saturation_finite_difference_for_input() {
        let encoder = LanguageWaveEncoder::new(-1.0, 0.7).unwrap();
        let topos = OpenCartesianTopos::new(-1.0, 1e-6, 0.5, 64, 1024)
            .unwrap()
            .with_porosity(0.8)
            .unwrap();
        let mut gate = WaveGate::with_topos("wg", 2, encoder, topos).unwrap();
        gate.gate
            .value_mut()
            .data_mut()
            .copy_from_slice(&[1.4, -1.1]);
        gate.bias
            .value_mut()
            .data_mut()
            .copy_from_slice(&[0.05, -0.02]);
        let input_values = vec![0.9, -0.75];
        let input = Tensor::from_vec(1, 2, input_values.clone()).unwrap();
        let grad_out = Tensor::from_vec(1, 2, vec![0.35, -0.2]).unwrap();

        let _ = gate.forward(&input).unwrap();
        let grad_input = gate.backward(&input, &grad_out).unwrap();
        let analytic = grad_input.data()[0];

        let epsilon = 1.0e-3f32;
        let loss_at = |values: Vec<f32>| {
            let tensor = Tensor::from_vec(1, 2, values).unwrap();
            let out = gate.forward(&tensor).unwrap();
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
            (analytic - finite_difference).abs() < 3.5e-3,
            "analytic={analytic} finite_difference={finite_difference}"
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wave_gate_forced_wgpu_forward_matches_cpu_reference() {
        let rows = 33;
        let cols = 32;
        let cpu_gate = WaveGate::new("wg", cols, -1.0, 0.5).unwrap();
        let wgpu_gate = WaveGate::new("wg", cols, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|idx| ((idx as f32 * 0.123).sin() * 0.8) - ((idx % 5) as f32 * 0.04))
                .collect(),
        )
        .unwrap();

        let cpu = {
            let _guard = push_backend_policy(cpu_policy());
            cpu_gate.forward(&input).unwrap()
        };
        let wgpu = {
            let _guard = push_backend_policy(wgpu_policy());
            wgpu_gate.forward(&input).unwrap()
        };

        assert_tensor_close(&cpu, &wgpu, 2e-5);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wave_gate_forced_wgpu_backward_matches_cpu_reference() {
        if !st_tensor::backend::wgpu_dense::is_available() {
            return;
        }
        let rows = 33;
        let cols = 32;
        let mut cpu_gate = WaveGate::new("wg", cols, -1.0, 0.5).unwrap();
        let mut wgpu_gate = WaveGate::new("wg", cols, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|idx| ((idx as f32 * 0.037).sin() * 0.6) - ((idx % 7) as f32 * 0.015))
                .collect(),
        )
        .unwrap();
        let grad_output = Tensor::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|idx| ((idx as f32 * 0.019).cos() * 0.4) + ((idx % 5) as f32 * 0.01))
                .collect(),
        )
        .unwrap();

        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy());
            cpu_gate.backward(&input, &grad_output).unwrap()
        };
        let wgpu_grad_input = {
            let _guard = push_backend_policy(wgpu_policy());
            wgpu_gate.backward(&input, &grad_output).unwrap()
        };

        assert_tensor_close(&cpu_grad_input, &wgpu_grad_input, 3e-4);
        assert_tensor_close(
            cpu_gate.gate().gradient().expect("cpu gate gradient"),
            wgpu_gate.gate().gradient().expect("wgpu gate gradient"),
            3e-4,
        );
        assert_tensor_close(
            cpu_gate.bias().gradient().expect("cpu bias gradient"),
            wgpu_gate.bias().gradient().expect("wgpu bias gradient"),
            3e-4,
        );
    }
}
