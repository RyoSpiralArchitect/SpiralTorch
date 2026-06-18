// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use rand::rngs::StdRng;
use rand::Rng;
use spiral_config::determinism;
#[cfg(feature = "wgpu")]
use st_tensor::wgpu_dense;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorUtilBackend};
use std::cell::RefCell;

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

#[cfg(feature = "wgpu")]
fn validate_finite_slice(label: &'static str, values: &[f32]) -> PureResult<()> {
    for &value in values {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue { label, value });
        }
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
fn strict_gpu_path() -> bool {
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(feature = "wgpu")]
fn dynamic_field_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_dynamic_field_route_meta(
    op_name: &'static str,
    field_model: &'static str,
    rows: usize,
    cols: usize,
    trainable_parameters: usize,
    estimated_ops_per_value: usize,
    backward: bool,
    gradient_scale: Option<f32>,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    input_gradient_backend: Option<&'static str>,
    gradient_reduction_backend: Option<&'static str>,
    gradient_scale_backend: Option<&'static str>,
    fallback: Option<String>,
) {
    emit_tensor_op(op_name, &[rows, cols], &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        let values = rows.saturating_mul(cols);
        let deterministic_backend = if field_model == "stochastic_schrodinger" && !backward {
            Some(if backend == "wgpu_dense" {
                "wgpu"
            } else {
                "cpu"
            })
        } else {
            None
        };
        let rng_backend = if field_model == "stochastic_schrodinger" && !backward {
            Some("cpu")
        } else {
            None
        };
        let mut data = serde_json::json!({
            "backend": backend,
            "requested_backend": requested_backend,
            "kernel": kernel,
            "deterministic_backend": deterministic_backend,
            "rng_backend": rng_backend,
            "kind": if backward { "dynamic_field_backward" } else { "dynamic_field_forward" },
            "field_model": field_model,
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "trainable_parameters": trainable_parameters,
            "gradient_scale": gradient_scale,
            "parameter_gradient_scale": gradient_scale,
            "input_gradient_backend": input_gradient_backend,
            "gradient_reduction_backend": gradient_reduction_backend,
            "gradient_scale_backend": gradient_scale_backend,
            "input_gradient_scale": if backward && gradient_scale.is_some() {
                Some(1.0f32)
            } else {
                None
            },
            "estimated_ops_per_value": estimated_ops_per_value,
            "estimated_total_ops": values.saturating_mul(estimated_ops_per_value),
            "empty": rows == 0 || cols == 0,
        });
        if let Some(message) = fallback {
            if let Some(map) = data.as_object_mut() {
                map.insert(
                    "fallback".to_string(),
                    serde_json::json!({"from": "wgpu", "message": message}),
                );
            }
        }
        data
    });
}

/// Propagates semantic amplitudes using a discrete Klein–Gordon style update
/// blended with a Dirac spinor coupling. This treats the incoming activations as
/// narrative field intensities and modulates them with a learnable mass/phase
/// pair.
#[derive(Debug)]
pub struct KleinGordonPropagation {
    mass: Parameter,
    spinor: Parameter,
    time_step: f32,
    damping: f32,
}

impl KleinGordonPropagation {
    /// Creates a propagation layer with deterministic initial parameters.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        time_step: f32,
        damping: f32,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if !time_step.is_finite() || time_step <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "klein_gordon_time_step",
            });
        }
        if !damping.is_finite() || damping < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "klein_gordon_damping",
            });
        }
        let name = name.into();
        let mass = Tensor::from_fn(1, features, |_r, c| 0.05 + (c as f32 * 0.01))?;
        let spinor = Tensor::from_fn(1, features, |_r, c| ((c as f32 + 1.0) * 0.37).sin() * 0.1)?;
        Ok(Self {
            mass: Parameter::new(format!("{name}::mass"), mass),
            spinor: Parameter::new(format!("{name}::spinor"), spinor),
            time_step,
            damping,
        })
    }

    /// Returns the configured time step used during propagation.
    pub fn time_step(&self) -> f32 {
        self.time_step
    }

    /// Returns the configured damping factor.
    pub fn damping(&self) -> f32 {
        self.damping
    }

    /// Returns a reference to the internal mass parameter.
    pub fn mass(&self) -> &Parameter {
        &self.mass
    }

    /// Returns a reference to the internal spinor parameter.
    pub fn spinor(&self) -> &Parameter {
        &self.spinor
    }
}

impl Module for KleinGordonPropagation {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.mass.value().shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.mass.value().shape(),
            });
        }
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let mass = self.mass.value().data();
        let spin = self.spinor.value().data();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::dynamic_klein_gordon_forward(
                    input.data(),
                    mass,
                    spin,
                    rows,
                    cols,
                    self.time_step,
                    self.damping,
                ) {
                    Ok(buffer) => {
                        validate_finite_slice("dynamic_klein_gordon_forward_output", &buffer)?;
                        let output = Tensor::from_vec(rows, cols, buffer)?;
                        emit_dynamic_field_route_meta(
                            "dynamic_field_klein_gordon_forward",
                            "klein_gordon",
                            rows,
                            cols,
                            2,
                            10,
                            false,
                            None,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_klein_gordon_forward",
                            None,
                            None,
                            None,
                            None,
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_klein_gordon_forward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut output = Tensor::zeros(rows, cols)?;
        let time_step = self.time_step;
        let damping = self.damping;
        {
            let input_buf = input.data();
            let out_buf = output.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let idx = offset + c;
                    let wave = input_buf[idx];
                    let amplitude = wave.tanh();
                    let kg_coeff = 1.0 - time_step * damping - time_step * time_step * mass[c];
                    let dirac_coeff = time_step * spin[c];
                    out_buf[idx] = wave * kg_coeff + dirac_coeff * amplitude;
                }
            }
        }
        emit_dynamic_field_route_meta(
            "dynamic_field_klein_gordon_forward",
            "klein_gordon",
            rows,
            cols,
            2,
            10,
            false,
            None,
            "cpu",
            requested_backend,
            "dynamic_field.scalar",
            None,
            None,
            None,
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if grad_output.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, cols),
            });
        }
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let scale_backend = current_tensor_util_backend_for_values(cols);
        let scale_backend_label = tensor_util_backend_label(scale_backend);
        let time_step = self.time_step;
        let damping = self.damping;
        let mass = self.mass.value().data();
        let spin = self.spinor.value().data();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::dynamic_klein_gordon_backward(
                    input.data(),
                    grad_output.data(),
                    mass,
                    spin,
                    rows,
                    cols,
                    time_step,
                    damping,
                ) {
                    Ok((grad_input_values, grad_mass_values, grad_spin_values)) => {
                        validate_finite_slice(
                            "dynamic_klein_gordon_backward_grad_input",
                            &grad_input_values,
                        )?;
                        validate_finite_slice(
                            "dynamic_klein_gordon_backward_grad_mass",
                            &grad_mass_values,
                        )?;
                        validate_finite_slice(
                            "dynamic_klein_gordon_backward_grad_spin",
                            &grad_spin_values,
                        )?;
                        let grad_input = Tensor::from_vec(rows, cols, grad_input_values)?;
                        let grad_mass = Tensor::from_vec(1, cols, grad_mass_values)?;
                        let grad_spin = Tensor::from_vec(1, cols, grad_spin_values)?;
                        let gradient_scale = if rows > 0 {
                            let scale = 1.0 / rows as f32;
                            self.mass.accumulate_euclidean(
                                &grad_mass.scale_with_backend(scale, scale_backend)?,
                            )?;
                            self.spinor.accumulate_euclidean(
                                &grad_spin.scale_with_backend(scale, scale_backend)?,
                            )?;
                            Some(scale)
                        } else {
                            None
                        };
                        emit_dynamic_field_route_meta(
                            "dynamic_field_klein_gordon_backward",
                            "klein_gordon",
                            rows,
                            cols,
                            2,
                            16,
                            true,
                            gradient_scale,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_klein_gordon_backward",
                            Some("wgpu"),
                            Some("wgpu"),
                            Some(scale_backend_label),
                            None,
                        );
                        return Ok(grad_input);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_klein_gordon_backward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut grad_input = Tensor::zeros(rows, cols)?;
        let mut grad_mass = Tensor::zeros(1, cols)?;
        let mut grad_spin = Tensor::zeros(1, cols)?;
        {
            let input_buf = input.data();
            let grad_output_buf = grad_output.data();
            let grad_input_buf = grad_input.data_mut();
            let grad_mass_buf = grad_mass.data_mut();
            let grad_spin_buf = grad_spin.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let idx = offset + c;
                    let wave = input_buf[idx];
                    let amplitude = wave.tanh();
                    let sech2 = 1.0 - amplitude * amplitude;
                    let kg_coeff = 1.0 - time_step * damping - time_step * time_step * mass[c];
                    let dirac_coeff = time_step * spin[c];
                    let go = grad_output_buf[idx];
                    grad_input_buf[idx] = go * (kg_coeff + dirac_coeff * sech2);
                    grad_mass_buf[c] += go * (-time_step * time_step * wave);
                    grad_spin_buf[c] += go * (time_step * amplitude);
                }
            }
        }
        let gradient_scale = if rows > 0 {
            let scale = 1.0 / rows as f32;
            self.mass
                .accumulate_euclidean(&grad_mass.scale_with_backend(scale, scale_backend)?)?;
            self.spinor
                .accumulate_euclidean(&grad_spin.scale_with_backend(scale, scale_backend)?)?;
            Some(scale)
        } else {
            None
        };
        emit_dynamic_field_route_meta(
            "dynamic_field_klein_gordon_backward",
            "klein_gordon",
            rows,
            cols,
            2,
            16,
            true,
            gradient_scale,
            "cpu",
            requested_backend,
            "dynamic_field.scalar",
            Some("cpu"),
            Some("cpu"),
            if gradient_scale.is_some() {
                Some(scale_backend_label)
            } else {
                None
            },
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.mass)?;
        visitor(&self.spinor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.mass)?;
        visitor(&mut self.spinor)?;
        Ok(())
    }
}

/// Hamilton–Jacobi style flow that nudges timelines towards minimal-action
/// trajectories. Each column receives an independent potential term that
/// encourages smoothness across the temporal dimension.
#[derive(Debug)]
pub struct HamiltonJacobiFlow {
    potential: Parameter,
    step_size: f32,
}

impl HamiltonJacobiFlow {
    /// Creates a flow model for the provided feature count.
    pub fn new(name: impl Into<String>, features: usize, step_size: f32) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if !step_size.is_finite() || step_size <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "hamilton_jacobi_step",
            });
        }
        let name = name.into();
        let potential = Tensor::from_fn(1, features, |_r, c| {
            ((c as f32 + 1.0) * 0.13).sin().abs() + 0.1
        })?;
        Ok(Self {
            potential: Parameter::new(format!("{name}::potential"), potential),
            step_size,
        })
    }

    /// Returns the internal potential parameter.
    pub fn potential(&self) -> &Parameter {
        &self.potential
    }

    /// Returns the configured gradient descent step size.
    pub fn step_size(&self) -> f32 {
        self.step_size
    }
}

impl Module for HamiltonJacobiFlow {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.potential.value().shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.potential.value().shape(),
            });
        }
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let step = self.step_size;
        let potential = self.potential.value().data();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::dynamic_hamilton_jacobi_forward(
                    input.data(),
                    potential,
                    rows,
                    cols,
                    step,
                ) {
                    Ok(buffer) => {
                        validate_finite_slice("dynamic_hamilton_jacobi_forward_output", &buffer)?;
                        let output = Tensor::from_vec(rows, cols, buffer)?;
                        emit_dynamic_field_route_meta(
                            "dynamic_field_hamilton_jacobi_forward",
                            "hamilton_jacobi",
                            rows,
                            cols,
                            1,
                            8,
                            false,
                            None,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_hamilton_jacobi_forward",
                            None,
                            None,
                            None,
                            None,
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_hamilton_jacobi_forward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut output = Tensor::zeros(rows, cols)?;
        {
            let input_buf = input.data();
            let out_buf = output.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                let input_row = &input_buf[offset..offset + cols];
                let prev_row = if r > 0 {
                    &input_buf[offset - cols..offset]
                } else {
                    input_row
                };
                let next_row = if r + 1 < rows {
                    &input_buf[offset + cols..offset + 2 * cols]
                } else {
                    input_row
                };
                let out_row = &mut out_buf[offset..offset + cols];

                for (((out, &current), &potential), (&prev, &next)) in out_row
                    .iter_mut()
                    .zip(input_row.iter())
                    .zip(potential.iter())
                    .zip(prev_row.iter().zip(next_row.iter()))
                {
                    let grad = (2.0 * current - prev - next) + potential * current;
                    *out = current - step * grad;
                }
            }
        }
        emit_dynamic_field_route_meta(
            "dynamic_field_hamilton_jacobi_forward",
            "hamilton_jacobi",
            rows,
            cols,
            1,
            8,
            false,
            None,
            "cpu",
            requested_backend,
            "dynamic_field.scalar",
            None,
            None,
            None,
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if grad_output.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, cols),
            });
        }
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let scale_backend = current_tensor_util_backend_for_values(cols);
        let scale_backend_label = tensor_util_backend_label(scale_backend);
        let step = self.step_size;
        let potential = self.potential.value().data();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::dynamic_hamilton_jacobi_backward(
                    input.data(),
                    grad_output.data(),
                    potential,
                    rows,
                    cols,
                    step,
                ) {
                    Ok((grad_input_values, grad_potential_values)) => {
                        validate_finite_slice(
                            "dynamic_hamilton_jacobi_backward_grad_input",
                            &grad_input_values,
                        )?;
                        validate_finite_slice(
                            "dynamic_hamilton_jacobi_backward_grad_potential",
                            &grad_potential_values,
                        )?;
                        let grad_input = Tensor::from_vec(rows, cols, grad_input_values)?;
                        let grad_potential = Tensor::from_vec(1, cols, grad_potential_values)?;
                        let gradient_scale = if rows > 0 {
                            let scale = 1.0 / rows as f32;
                            self.potential.accumulate_euclidean(
                                &grad_potential.scale_with_backend(scale, scale_backend)?,
                            )?;
                            Some(scale)
                        } else {
                            None
                        };
                        emit_dynamic_field_route_meta(
                            "dynamic_field_hamilton_jacobi_backward",
                            "hamilton_jacobi",
                            rows,
                            cols,
                            1,
                            12,
                            true,
                            gradient_scale,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_hamilton_jacobi_backward",
                            Some("wgpu"),
                            Some("wgpu"),
                            Some(scale_backend_label),
                            None,
                        );
                        return Ok(grad_input);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_hamilton_jacobi_backward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut grad_input = Tensor::zeros(rows, cols)?;
        let mut grad_potential = Tensor::zeros(1, cols)?;
        {
            let input_buf = input.data();
            let grad_output_buf = grad_output.data();
            let grad_input_buf = grad_input.data_mut();
            let grad_potential_buf = grad_potential.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let idx = offset + c;
                    let current = input_buf[idx];
                    let go = grad_output_buf[idx];
                    if rows == 1 {
                        grad_input_buf[idx] += go * (1.0 - step * potential[c]);
                    } else if r == 0 {
                        grad_input_buf[idx] += go * (1.0 - step * (1.0 + potential[c]));
                        grad_input_buf[idx + cols] += go * step;
                    } else if r + 1 == rows {
                        grad_input_buf[idx] += go * (1.0 - step * (1.0 + potential[c]));
                        grad_input_buf[idx - cols] += go * step;
                    } else {
                        grad_input_buf[idx] += go * (1.0 - step * (2.0 + potential[c]));
                        grad_input_buf[idx - cols] += go * step;
                        grad_input_buf[idx + cols] += go * step;
                    }
                    grad_potential_buf[c] += -step * current * go;
                }
            }
        }
        let gradient_scale = if rows > 0 {
            let scale = 1.0 / rows as f32;
            self.potential
                .accumulate_euclidean(&grad_potential.scale_with_backend(scale, scale_backend)?)?;
            Some(scale)
        } else {
            None
        };
        emit_dynamic_field_route_meta(
            "dynamic_field_hamilton_jacobi_backward",
            "hamilton_jacobi",
            rows,
            cols,
            1,
            12,
            true,
            gradient_scale,
            "cpu",
            requested_backend,
            "dynamic_field.scalar",
            Some("cpu"),
            Some("cpu"),
            if gradient_scale.is_some() {
                Some(scale_backend_label)
            } else {
                None
            },
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.potential)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.potential)
    }
}

/// Layer that models semantic interference through a stochastic Schrödinger
/// style update. It keeps track of decoherence factors that can later be
/// surfaced through [`NarrativeHint`] quantum metadata.
#[derive(Debug)]
pub struct StochasticSchrodingerLayer {
    coherence: Parameter,
    decoherence_rate: f32,
    noise_scale: f32,
    rng: RefCell<StdRng>,
    last_amplitude: RefCell<Option<Tensor>>,
    last_decoherence: RefCell<Option<Tensor>>,
}

impl StochasticSchrodingerLayer {
    /// Creates a stochastic layer using entropy-backed randomness.
    pub fn new(
        name: impl Into<String>,
        features: usize,
        decoherence_rate: f32,
        noise_scale: f32,
    ) -> PureResult<Self> {
        Self::with_seed(name, features, decoherence_rate, noise_scale, None)
    }

    /// Same as [`StochasticSchrodingerLayer::new`] but allows a deterministic seed
    /// for reproducible experiments and unit tests.
    pub fn with_seed(
        name: impl Into<String>,
        features: usize,
        decoherence_rate: f32,
        noise_scale: f32,
        seed: Option<u64>,
    ) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        if !decoherence_rate.is_finite() || decoherence_rate < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "schrodinger_decoherence",
            });
        }
        if !noise_scale.is_finite() || noise_scale < 0.0 {
            return Err(TensorError::InvalidValue {
                label: "schrodinger_noise",
            });
        }
        let name = name.into();
        let coherence = Tensor::from_fn(1, features, |_r, c| (0.85 - (c as f32 * 0.03)).max(0.1))?;
        let rng = determinism::rng_from_optional(seed, "st-nn/layers/stochastic_schrodinger");
        Ok(Self {
            coherence: Parameter::new(format!("{name}::coherence"), coherence),
            decoherence_rate,
            noise_scale,
            rng: RefCell::new(rng),
            last_amplitude: RefCell::new(None),
            last_decoherence: RefCell::new(None),
        })
    }

    /// Returns the configured decoherence rate.
    pub fn decoherence_rate(&self) -> f32 {
        self.decoherence_rate
    }

    /// Returns the coherence parameter.
    pub fn coherence(&self) -> &Parameter {
        &self.coherence
    }
}

impl Module for StochasticSchrodingerLayer {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.coherence.value().shape().1 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.coherence.value().shape(),
            });
        }
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let noise_scale = self.noise_scale;
        let rate = self.decoherence_rate;
        let coherence = self.coherence.value().data();
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::dynamic_schrodinger_forward(
                    input.data(),
                    coherence,
                    rows,
                    cols,
                    rate,
                ) {
                    Ok((mut output_values, amplitude_values, decoherence_values)) => {
                        validate_finite_slice(
                            "dynamic_schrodinger_forward_output",
                            &output_values,
                        )?;
                        validate_finite_slice(
                            "dynamic_schrodinger_forward_amplitude",
                            &amplitude_values,
                        )?;
                        validate_finite_slice(
                            "dynamic_schrodinger_forward_decoherence",
                            &decoherence_values,
                        )?;
                        {
                            let mut rng = self.rng.borrow_mut();
                            for value in output_values.iter_mut() {
                                let noise = (rng.gen::<f32>() - 0.5) * noise_scale;
                                *value += noise;
                                if !value.is_finite() {
                                    return Err(TensorError::NonFiniteValue {
                                        label: "dynamic_schrodinger_forward_output",
                                        value: *value,
                                    });
                                }
                            }
                        }
                        let output = Tensor::from_vec(rows, cols, output_values)?;
                        let amplitude = Tensor::from_vec(rows, cols, amplitude_values)?;
                        let decoherence = Tensor::from_vec(rows, cols, decoherence_values)?;
                        self.last_amplitude.borrow_mut().replace(amplitude);
                        self.last_decoherence.borrow_mut().replace(decoherence);
                        emit_dynamic_field_route_meta(
                            "dynamic_field_stochastic_schrodinger_forward",
                            "stochastic_schrodinger",
                            rows,
                            cols,
                            1,
                            14,
                            false,
                            None,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_schrodinger_forward",
                            None,
                            None,
                            None,
                            None,
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_stochastic_schrodinger_forward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut output = Tensor::zeros(rows, cols)?;
        let mut amplitude = Tensor::zeros(rows, cols)?;
        let mut decoherence = Tensor::zeros(rows, cols)?;
        {
            let input_buf = input.data();
            let out_buf = output.data_mut();
            let amp_buf = amplitude.data_mut();
            let deco_buf = decoherence.data_mut();
            let mut rng = self.rng.borrow_mut();
            for r in 0..rows {
                let offset = r * cols;
                let input_row = &input_buf[offset..offset + cols];
                let out_row = &mut out_buf[offset..offset + cols];
                let amp_row = &mut amp_buf[offset..offset + cols];
                let deco_row = &mut deco_buf[offset..offset + cols];
                for (((out, amp_slot), deco_slot), (&input, &coherence)) in out_row
                    .iter_mut()
                    .zip(amp_row.iter_mut())
                    .zip(deco_row.iter_mut())
                    .zip(input_row.iter().zip(coherence.iter()))
                {
                    let amp = input.tanh();
                    let deco = 1.0 / (1.0 + rate * amp.abs());
                    let interference = amp * coherence * deco;
                    let noise = (rng.gen::<f32>() - 0.5) * noise_scale;
                    *out = interference + noise;
                    *amp_slot = amp;
                    *deco_slot = deco;
                }
            }
        }
        self.last_amplitude.borrow_mut().replace(amplitude);
        self.last_decoherence.borrow_mut().replace(decoherence);
        emit_dynamic_field_route_meta(
            "dynamic_field_stochastic_schrodinger_forward",
            "stochastic_schrodinger",
            rows,
            cols,
            1,
            14,
            false,
            None,
            "cpu",
            requested_backend,
            "dynamic_field.scalar",
            None,
            None,
            None,
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(output)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let amplitude_guard = self.last_amplitude.borrow();
        let Some(amplitude) = amplitude_guard.as_ref() else {
            return Err(TensorError::InvalidValue {
                label: "schrodinger_amplitude_cache",
            });
        };
        let decoherence_guard = self.last_decoherence.borrow();
        let Some(decoherence) = decoherence_guard.as_ref() else {
            return Err(TensorError::InvalidValue {
                label: "schrodinger_decoherence_cache",
            });
        };
        let (rows, cols) = grad_output.shape();
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        let scale_backend = current_tensor_util_backend_for_values(cols);
        let scale_backend_label = tensor_util_backend_label(scale_backend);
        let coherence_values = self.coherence.value().data().to_vec();
        let rate = self.decoherence_rate;
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu)
                && rows > 0
                && cols > 0
                && wgpu_dense::is_available()
            {
                match wgpu_dense::dynamic_schrodinger_backward(
                    amplitude.data(),
                    decoherence.data(),
                    grad_output.data(),
                    &coherence_values,
                    rows,
                    cols,
                    rate,
                ) {
                    Ok((grad_input_values, grad_coherence_values)) => {
                        validate_finite_slice(
                            "dynamic_schrodinger_backward_grad_input",
                            &grad_input_values,
                        )?;
                        validate_finite_slice(
                            "dynamic_schrodinger_backward_grad_coherence",
                            &grad_coherence_values,
                        )?;
                        let grad_input = Tensor::from_vec(rows, cols, grad_input_values)?;
                        let grad_coherence = Tensor::from_vec(1, cols, grad_coherence_values)?;
                        let gradient_scale = if rows > 0 {
                            let scale = 1.0 / rows as f32;
                            self.coherence.accumulate_euclidean(
                                &grad_coherence.scale_with_backend(scale, scale_backend)?,
                            )?;
                            Some(scale)
                        } else {
                            None
                        };
                        emit_dynamic_field_route_meta(
                            "dynamic_field_stochastic_schrodinger_backward",
                            "stochastic_schrodinger",
                            rows,
                            cols,
                            1,
                            18,
                            true,
                            gradient_scale,
                            "wgpu_dense",
                            requested_backend,
                            "tensor_util.dynamic_schrodinger_backward",
                            Some("wgpu"),
                            Some("wgpu"),
                            Some(scale_backend_label),
                            None,
                        );
                        return Ok(grad_input);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(dynamic_field_wgpu_error(
                            "dynamic_field_stochastic_schrodinger_backward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut grad_input = Tensor::zeros(rows, cols)?;
        let mut grad_coherence = Tensor::zeros(1, cols)?;
        {
            let grad_output_buf = grad_output.data();
            let grad_input_buf = grad_input.data_mut();
            let amp_buf = amplitude.data();
            let deco_buf = decoherence.data();
            let grad_coherence_buf = grad_coherence.data_mut();
            for r in 0..rows {
                let offset = r * cols;
                for c in 0..cols {
                    let idx = offset + c;
                    let amp = amp_buf[idx];
                    let deco = deco_buf[idx];
                    let go = grad_output_buf[idx];
                    let denom = 1.0 + rate * amp.abs();
                    let sign = if amp >= 0.0 { 1.0 } else { -1.0 };
                    let d_deco_d_amp = -rate * sign / (denom * denom);
                    let base = deco + amp * d_deco_d_amp;
                    let d_amp_d_input = 1.0 - amp * amp;
                    grad_input_buf[idx] = go * coherence_values[c] * base * d_amp_d_input;
                    grad_coherence_buf[c] += go * (amp * deco);
                }
            }
        }
        let gradient_scale = if rows > 0 {
            let scale = 1.0 / rows as f32;
            self.coherence
                .accumulate_euclidean(&grad_coherence.scale_with_backend(scale, scale_backend)?)?;
            Some(scale)
        } else {
            None
        };
        emit_dynamic_field_route_meta(
            "dynamic_field_stochastic_schrodinger_backward",
            "stochastic_schrodinger",
            rows,
            cols,
            1,
            18,
            true,
            gradient_scale,
            "cpu",
            requested_backend,
            "dynamic_field.scalar",
            Some("cpu"),
            Some("cpu"),
            if gradient_scale.is_some() {
                Some(scale_backend_label)
            } else {
                None
            },
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.coherence)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.coherence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn assert_close(left: f32, right: f32) {
        let diff = (left - right).abs();
        assert!(diff < 1e-6, "expected {left} ≈ {right} (diff={diff})");
    }

    #[cfg(feature = "wgpu")]
    fn approx_eq(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
            let diff = (l - r).abs();
            assert!(diff < 1e-5, "idx={idx} left={l} right={r} diff={diff}");
        }
    }

    #[test]
    fn klein_gordon_shapes_match() {
        let layer = KleinGordonPropagation::new("kg", 4, 0.2, 0.1).unwrap();
        let input = Tensor::from_vec(3, 4, vec![0.1; 12]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn hamilton_jacobi_smooths_edges() {
        let mut layer = HamiltonJacobiFlow::new("hj", 3, 0.1).unwrap();
        let input = Tensor::from_vec(
            4,
            3,
            vec![
                1.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.0, 0.0, -1.0, 1.0, 0.5, -0.5,
            ],
        )
        .unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let grad = Tensor::from_vec(4, 3, vec![0.2; 12]).unwrap();
        let back = layer.backward(&input, &grad).unwrap();
        assert_eq!(back.shape(), input.shape());
    }

    #[test]
    fn dynamic_field_parameter_gradients_are_batch_normalized() {
        let input = Tensor::from_vec(2, 2, vec![0.5, -0.3, 0.7, 0.2]).unwrap();
        let grad = Tensor::from_vec(2, 2, vec![0.4, -0.1, 0.2, 0.5]).unwrap();

        let mut kg = KleinGordonPropagation::new("kg", 2, 0.2, 0.1).unwrap();
        let grad_input = kg.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        let mass_grad = kg.mass().gradient().expect("mass gradient");
        let spin_grad = kg.spinor().gradient().expect("spinor gradient");
        for col in 0..2 {
            let mut expected_mass = 0.0f32;
            let mut expected_spin = 0.0f32;
            for row in 0..2 {
                let idx = row * 2 + col;
                let wave = input.data()[idx];
                let go = grad.data()[idx];
                expected_mass += go * (-(0.2f32 * 0.2) * wave);
                expected_spin += go * (0.2 * wave.tanh());
            }
            assert_close(mass_grad.data()[col], expected_mass * 0.5);
            assert_close(spin_grad.data()[col], expected_spin * 0.5);
        }

        let mut hj = HamiltonJacobiFlow::new("hj", 2, 0.1).unwrap();
        let grad_input = hj.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        let potential_grad = hj.potential().gradient().expect("potential gradient");
        for col in 0..2 {
            let mut expected = 0.0f32;
            for row in 0..2 {
                let idx = row * 2 + col;
                expected += -0.1 * input.data()[idx] * grad.data()[idx];
            }
            assert_close(potential_grad.data()[col], expected * 0.5);
        }

        let mut qs = StochasticSchrodingerLayer::with_seed("qs", 2, 0.4, 0.0, Some(7)).unwrap();
        let _ = qs.forward(&input).unwrap();
        let grad_input = qs.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        let coherence_grad = qs.coherence().gradient().expect("coherence gradient");
        for col in 0..2 {
            let mut expected = 0.0f32;
            for row in 0..2 {
                let idx = row * 2 + col;
                let amp = input.data()[idx].tanh();
                let deco = 1.0 / (1.0 + 0.4 * amp.abs());
                expected += grad.data()[idx] * amp * deco;
            }
            assert_close(coherence_grad.data()[col], expected * 0.5);
        }
    }

    #[test]
    fn dynamic_field_empty_batches_skip_parameter_updates() {
        let input = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let grad = Tensor::from_vec(0, 2, Vec::new()).unwrap();

        let mut kg = KleinGordonPropagation::new("kg", 2, 0.2, 0.1).unwrap();
        let _ = kg.forward(&input).unwrap();
        let grad_input = kg.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(kg.mass().gradient().is_none());
        assert!(kg.spinor().gradient().is_none());

        let mut hj = HamiltonJacobiFlow::new("hj", 2, 0.1).unwrap();
        let _ = hj.forward(&input).unwrap();
        let grad_input = hj.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(hj.potential().gradient().is_none());

        let mut qs = StochasticSchrodingerLayer::with_seed("qs", 2, 0.4, 0.0, Some(7)).unwrap();
        let _ = qs.forward(&input).unwrap();
        let grad_input = qs.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(qs.coherence().gradient().is_none());
    }

    #[test]
    fn dynamic_field_layers_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_vec(2, 3, vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6]).unwrap();
        let grad = Tensor::from_vec(2, 3, vec![0.05; 6]).unwrap();

        let mut kg = KleinGordonPropagation::new("kg", 3, 0.2, 0.1).unwrap();
        let _ = kg.forward(&input).unwrap();
        let _ = kg.backward(&input, &grad).unwrap();

        let mut hj = HamiltonJacobiFlow::new("hj", 3, 0.1).unwrap();
        let _ = hj.forward(&input).unwrap();
        let _ = hj.backward(&input, &grad).unwrap();

        let mut qs = StochasticSchrodingerLayer::with_seed("qs", 3, 0.4, 0.05, Some(7)).unwrap();
        let _ = qs.forward(&input).unwrap();
        let _ = qs.backward(&input, &grad).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        for (op_name, field_model, kind) in [
            (
                "dynamic_field_klein_gordon_forward",
                "klein_gordon",
                "dynamic_field_forward",
            ),
            (
                "dynamic_field_klein_gordon_backward",
                "klein_gordon",
                "dynamic_field_backward",
            ),
            (
                "dynamic_field_hamilton_jacobi_forward",
                "hamilton_jacobi",
                "dynamic_field_forward",
            ),
            (
                "dynamic_field_hamilton_jacobi_backward",
                "hamilton_jacobi",
                "dynamic_field_backward",
            ),
            (
                "dynamic_field_stochastic_schrodinger_forward",
                "stochastic_schrodinger",
                "dynamic_field_forward",
            ),
            (
                "dynamic_field_stochastic_schrodinger_backward",
                "stochastic_schrodinger",
                "dynamic_field_backward",
            ),
        ] {
            let event = events
                .iter()
                .find(|(observed, data)| *observed == op_name && data["rows"] == 2)
                .unwrap_or_else(|| panic!("{op_name} metadata event"));
            assert_eq!(event.1["backend"], "cpu");
            assert_eq!(event.1["kind"], kind);
            assert_eq!(event.1["field_model"], field_model);
            if kind == "dynamic_field_backward" {
                assert_eq!(event.1["gradient_scale"], 0.5);
                assert_eq!(event.1["parameter_gradient_scale"], 0.5);
                assert_eq!(event.1["input_gradient_scale"], 1.0);
            }
            assert!(event.1["estimated_total_ops"].as_u64().unwrap_or(0) > 0);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn klein_gordon_forced_wgpu_matches_cpu_reference_and_emits_backend() {
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
            ((row * 11 + col * 5) % 17) as f32 * 0.041 - 0.27
        })
        .unwrap();
        let grad = Tensor::from_fn(4, 3, |row, col| {
            ((row * 7 + col * 3) % 13) as f32 * 0.029 - 0.18
        })
        .unwrap();
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_layer = KleinGordonPropagation::new("kg_cpu", 3, 0.2, 0.1).unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.backward(&input, &grad).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_layer = KleinGordonPropagation::new("kg_wgpu", 3, 0.2, 0.1).unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.forward(&input).unwrap(),
                wgpu_layer.backward(&input, &grad).unwrap(),
            )
        };

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        approx_eq(cpu_forward.data(), wgpu_forward.data());
        approx_eq(cpu_grad_input.data(), wgpu_grad_input.data());
        approx_eq(
            cpu_layer.mass().gradient().unwrap().data(),
            wgpu_layer.mass().gradient().unwrap().data(),
        );
        approx_eq(
            cpu_layer.spinor().gradient().unwrap().data(),
            wgpu_layer.spinor().gradient().unwrap().data(),
        );

        let events = events.lock().unwrap();
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_klein_gordon_forward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_klein_gordon_backward"
                && data["backend"] == "wgpu_dense"
                && data["input_gradient_backend"] == "wgpu"
                && data["gradient_reduction_backend"] == "wgpu"
        }));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn hamilton_jacobi_forced_wgpu_matches_cpu_reference_and_emits_backend() {
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

        let input = Tensor::from_fn(5, 3, |row, col| {
            ((row * 13 + col * 7) % 19) as f32 * 0.037 - 0.29
        })
        .unwrap();
        let grad = Tensor::from_fn(5, 3, |row, col| {
            ((row * 5 + col * 11) % 17) as f32 * 0.023 - 0.15
        })
        .unwrap();
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_layer = HamiltonJacobiFlow::new("hj_cpu", 3, 0.1).unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.backward(&input, &grad).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_layer = HamiltonJacobiFlow::new("hj_wgpu", 3, 0.1).unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.forward(&input).unwrap(),
                wgpu_layer.backward(&input, &grad).unwrap(),
            )
        };

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        approx_eq(cpu_forward.data(), wgpu_forward.data());
        approx_eq(cpu_grad_input.data(), wgpu_grad_input.data());
        approx_eq(
            cpu_layer.potential().gradient().unwrap().data(),
            wgpu_layer.potential().gradient().unwrap().data(),
        );

        let events = events.lock().unwrap();
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_hamilton_jacobi_forward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_hamilton_jacobi_backward"
                && data["backend"] == "wgpu_dense"
                && data["input_gradient_backend"] == "wgpu"
                && data["gradient_reduction_backend"] == "wgpu"
        }));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn stochastic_schrodinger_forced_wgpu_matches_cpu_reference_and_emits_backend() {
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
            ((row * 17 + col * 5) % 23) as f32 * 0.031 - 0.33
        })
        .unwrap();
        let grad = Tensor::from_fn(4, 3, |row, col| {
            ((row * 7 + col * 13) % 19) as f32 * 0.021 - 0.16
        })
        .unwrap();
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_layer =
            StochasticSchrodingerLayer::with_seed("qs_cpu", 3, 0.4, 0.05, Some(11)).unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.backward(&input, &grad).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_layer =
            StochasticSchrodingerLayer::with_seed("qs_wgpu", 3, 0.4, 0.05, Some(11)).unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.forward(&input).unwrap(),
                wgpu_layer.backward(&input, &grad).unwrap(),
            )
        };

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        approx_eq(cpu_forward.data(), wgpu_forward.data());
        approx_eq(cpu_grad_input.data(), wgpu_grad_input.data());
        approx_eq(
            cpu_layer.coherence().gradient().unwrap().data(),
            wgpu_layer.coherence().gradient().unwrap().data(),
        );

        let events = events.lock().unwrap();
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_stochastic_schrodinger_forward"
                && data["backend"] == "wgpu_dense"
                && data["requested_backend"] == "wgpu"
                && data["deterministic_backend"] == "wgpu"
                && data["rng_backend"] == "cpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dynamic_field_stochastic_schrodinger_backward"
                && data["backend"] == "wgpu_dense"
                && data["input_gradient_backend"] == "wgpu"
                && data["gradient_reduction_backend"] == "wgpu"
        }));
    }

    #[test]
    fn stochastic_schrodinger_is_reproducible() {
        let mut layer = StochasticSchrodingerLayer::with_seed("qs", 2, 0.4, 0.05, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.3, -0.7]).unwrap();
        let output_a = layer.forward(&input).unwrap();
        let output_b = layer.forward(&input).unwrap();
        assert_ne!(output_a.data(), output_b.data());
        let grad = Tensor::from_vec(1, 2, vec![0.1, -0.2]).unwrap();
        let back = layer.backward(&input, &grad).unwrap();
        assert_eq!(back.shape(), input.shape());
    }
}
