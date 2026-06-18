// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::{current_matmul_backend, current_tensor_util_backend_for_values};
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, MatmulBackend, TensorUtilBackend};
use std::cell::{Ref, RefCell};

fn emit_continuous_wavelet_meta(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    scales: usize,
    log_step: f32,
    omega0: f32,
    backward: bool,
    response_backend: MatmulBackend,
    mix_backend: TensorUtilBackend,
    gradient_scale: Option<f32>,
) {
    emit_tensor_op(op_name, &[rows, cols], &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        let response_work = rows
            .saturating_mul(cols)
            .saturating_mul(cols)
            .saturating_mul(scales);
        let mix_work = rows.saturating_mul(cols).saturating_mul(scales);
        let reduction_work = if backward {
            rows.saturating_mul(cols).saturating_mul(scales)
        } else {
            0
        };
        serde_json::json!({
            "backend": "composite",
            "requested_backend": response_backend.to_string(),
            "kernel": "continuous_wavelet.matmul_composite",
            "kind": if backward { "wavelet_backward" } else { "wavelet_forward" },
            "response_path": "matmul_per_scale",
            "response_backend": response_backend.to_string(),
            "mix_backend": mix_backend.to_string(),
            "gradient_scale": gradient_scale,
            "parameter_gradient_scale": gradient_scale,
            "input_gradient_scale": if backward && gradient_scale.is_some() {
                Some(1.0f32)
            } else {
                None
            },
            "rows": rows,
            "cols": cols,
            "values": rows.saturating_mul(cols),
            "output_rows": rows,
            "output_cols": cols,
            "output_values": rows.saturating_mul(cols),
            "scales": scales,
            "log_step": log_step,
            "omega0": omega0,
            "kernel_values": scales.saturating_mul(cols).saturating_mul(cols),
            "estimated_response_ops": response_work,
            "estimated_mix_ops": mix_work,
            "estimated_reduction_ops": reduction_work,
            "estimated_total_ops": response_work
                .saturating_add(mix_work)
                .saturating_add(reduction_work),
            "empty": rows == 0 || cols == 0 || scales == 0,
        })
    });
}

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
        let responses = self.compute_responses(input, current_matmul_backend())?;
        let focus = self.focus.value().data();
        let (rows, cols) = input.shape();
        let mut profile = vec![0.0f32; cols];
        for c in 0..cols {
            let mut energy = 0.0f32;
            for (scale_idx, response) in responses.iter().enumerate() {
                let mut column_energy = 0.0f32;
                for r in 0..rows {
                    let value = response.data()[r * cols + c];
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

    fn kernel_transpose_tensor(kernel: &[f32], cols: usize) -> PureResult<Tensor> {
        let mut data = vec![0.0f32; cols.saturating_mul(cols)];
        for center in 0..cols {
            for sample in 0..cols {
                data[sample * cols + center] = kernel[center * cols + sample];
            }
        }
        Tensor::from_vec(cols, cols, data)
    }

    fn weighted_kernel_tensor(
        kernels: &[Vec<f32>],
        focus: &[f32],
        cols: usize,
    ) -> PureResult<Tensor> {
        let mut data = vec![0.0f32; cols.saturating_mul(cols)];
        for (scale_idx, kernel) in kernels.iter().enumerate() {
            let weight = focus[scale_idx];
            for (dst, &value) in data.iter_mut().zip(kernel.iter()) {
                *dst += weight * value;
            }
        }
        Tensor::from_vec(cols, cols, data)
    }

    fn ensure_kernels(&self, cols: usize) -> Ref<'_, Vec<Vec<f32>>> {
        {
            let cached = self.kernels.borrow();
            if let Some(ref kernels) = *cached {
                if kernels.len() == self.scales.len()
                    && kernels
                        .first()
                        .map(|kernel| kernel.len() == cols * cols)
                        .unwrap_or(true)
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

    fn compute_responses(&self, input: &Tensor, backend: MatmulBackend) -> PureResult<Vec<Tensor>> {
        let (rows, cols) = input.shape();
        if cols != self.features {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (rows, self.features),
            });
        }
        let kernels = self.ensure_kernels(cols);
        let mut responses = Vec::with_capacity(kernels.len());
        for kernel in kernels.iter() {
            let kernel_t = Self::kernel_transpose_tensor(kernel, cols)?;
            responses.push(input.matmul_with_backend(&kernel_t, backend)?);
        }
        Ok(responses)
    }
}

impl Module for ContinuousWaveletTransform {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let matmul_backend = current_matmul_backend();
        let responses = self.compute_responses(input, matmul_backend)?;
        let focus = self.focus.value().data();
        let bias = self.bias.value().data();
        let (rows, cols) = input.shape();
        let mix_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let mut output = Tensor::zeros(rows, cols)?;
        output.add_row_inplace_with_backend(bias, mix_backend)?;
        for (scale_idx, response) in responses.iter().enumerate() {
            output.add_scaled_with_backend(response, focus[scale_idx], mix_backend)?;
        }
        emit_continuous_wavelet_meta(
            "continuous_wavelet_forward",
            rows,
            cols,
            self.scales.len(),
            self.log_step,
            self.omega0,
            false,
            matmul_backend,
            mix_backend,
            None,
        );
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if grad_output.shape() != input.shape() {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: input.shape(),
            });
        }
        let matmul_backend = current_matmul_backend();
        let (rows, cols) = input.shape();
        let reduction_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        if rows == 0 {
            let grad_input = Tensor::zeros(rows, cols)?;
            emit_continuous_wavelet_meta(
                "continuous_wavelet_backward",
                rows,
                cols,
                self.scales.len(),
                self.log_step,
                self.omega0,
                true,
                matmul_backend,
                reduction_backend,
                None,
            );
            return Ok(grad_input);
        }
        let responses = self.compute_responses(input, matmul_backend)?;
        let focus = self.focus.value().data().to_vec();
        let weighted_kernel = {
            let kernels = self.ensure_kernels(cols);
            Self::weighted_kernel_tensor(&kernels, &focus, cols)?
        };
        let grad_input = grad_output.matmul_with_backend(&weighted_kernel, matmul_backend)?;

        let gradient_scale = 1.0 / rows as f32;
        let mut grad_focus = vec![0.0f32; self.scales.len()];
        for (scale_idx, response) in responses.iter().enumerate() {
            let product = grad_output.hadamard_with_backend(response, reduction_backend)?;
            let column_sums =
                product.try_sum_axis0_scaled_with_backend(gradient_scale, reduction_backend)?;
            grad_focus[scale_idx] = column_sums.iter().sum();
        }

        let grad_bias =
            grad_output.try_sum_axis0_scaled_with_backend(gradient_scale, reduction_backend)?;

        let grad_focus_tensor = Tensor::from_vec(1, grad_focus.len(), grad_focus)?;
        let grad_bias_tensor = Tensor::from_vec(1, cols, grad_bias)?;
        self.focus.accumulate_euclidean(&grad_focus_tensor)?;
        self.bias.accumulate_euclidean(&grad_bias_tensor)?;

        emit_continuous_wavelet_meta(
            "continuous_wavelet_backward",
            rows,
            cols,
            self.scales.len(),
            self.log_step,
            self.omega0,
            true,
            matmul_backend,
            reduction_backend,
            Some(gradient_scale),
        );
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
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn assert_close(left: f32, right: f32, tolerance: f32) {
        let diff = (left - right).abs();
        assert!(
            diff <= tolerance,
            "expected {left} ≈ {right} (diff={diff}, tolerance={tolerance})"
        );
    }

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
    fn forward_backward_emit_wavelet_backend_meta() {
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
            ContinuousWaveletTransform::new("cwt", 4, vec![0.75, 1.5], 0.25, 5.0).unwrap();
        let input =
            Tensor::from_vec(2, 4, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2, -0.3]).unwrap();
        let grad_output = Tensor::from_vec(
            2,
            4,
            vec![0.01, -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, -0.03],
        )
        .unwrap();
        let _ = layer.forward(&input).unwrap();
        let _ = layer.backward(&input, &grad_output).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "continuous_wavelet_forward" && data["rows"] == 2 && data["cols"] == 4
            })
            .expect("continuous wavelet forward metadata event");
        assert_eq!(forward.1["backend"], "composite");
        assert_eq!(forward.1["kind"], "wavelet_forward");
        assert_eq!(forward.1["kernel"], "continuous_wavelet.matmul_composite");
        assert_eq!(forward.1["response_path"], "matmul_per_scale");
        assert_eq!(forward.1["response_backend"], "auto");
        assert_eq!(forward.1["mix_backend"], "auto");
        assert_eq!(forward.1["scales"], 2);
        assert!(forward.1["estimated_total_ops"].as_u64().unwrap_or(0) > 0);

        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "continuous_wavelet_backward" && data["rows"] == 2 && data["cols"] == 4
            })
            .expect("continuous wavelet backward metadata event");
        assert_eq!(backward.1["backend"], "composite");
        assert_eq!(backward.1["kind"], "wavelet_backward");
        assert_eq!(backward.1["response_path"], "matmul_per_scale");
        assert_eq!(backward.1["response_backend"], "auto");
        assert_eq!(backward.1["mix_backend"], "auto");
        assert_eq!(backward.1["gradient_scale"], 0.5);
        assert_eq!(backward.1["parameter_gradient_scale"], 0.5);
        assert_eq!(backward.1["input_gradient_scale"], 1.0);
        assert_eq!(backward.1["scales"], 2);
        assert!(
            backward.1["estimated_total_ops"].as_u64().unwrap_or(0)
                >= forward.1["estimated_total_ops"].as_u64().unwrap_or(0)
        );
        let matmul_events = events
            .iter()
            .filter(|(op_name, data)| *op_name == "matmul" && data["rows"] == 2)
            .count();
        assert!(matmul_events >= 5);
        let reduction_events = events
            .iter()
            .filter(|(op_name, data)| *op_name == "sum_axis0_scaled" && data["cols"] == 4)
            .count();
        assert!(reduction_events >= 3);
    }

    #[test]
    fn parameter_gradients_are_batch_normalized() {
        let mut single =
            ContinuousWaveletTransform::new("cwt", 3, vec![0.6, 1.4], 0.3, 4.0).unwrap();
        let input_single = Tensor::from_vec(1, 3, vec![0.35, -0.25, 0.6]).unwrap();
        let grad_single = Tensor::from_vec(1, 3, vec![0.2, -0.15, 0.4]).unwrap();
        let _ = single.backward(&input_single, &grad_single).unwrap();

        let mut repeated =
            ContinuousWaveletTransform::new("cwt", 3, vec![0.6, 1.4], 0.3, 4.0).unwrap();
        let input_repeated =
            Tensor::from_vec(2, 3, vec![0.35, -0.25, 0.6, 0.35, -0.25, 0.6]).unwrap();
        let grad_repeated = Tensor::from_vec(2, 3, vec![0.2, -0.15, 0.4, 0.2, -0.15, 0.4]).unwrap();
        let _ = repeated.backward(&input_repeated, &grad_repeated).unwrap();

        let single_focus = single.focus().gradient().expect("single focus gradient");
        let repeated_focus = repeated
            .focus()
            .gradient()
            .expect("repeated focus gradient");
        let single_bias = single.bias().gradient().expect("single bias gradient");
        let repeated_bias = repeated.bias().gradient().expect("repeated bias gradient");

        for (&single_value, &repeated_value) in
            single_focus.data().iter().zip(repeated_focus.data().iter())
        {
            assert_close(single_value, repeated_value, 1.0e-6);
        }
        for (&single_value, &repeated_value) in
            single_bias.data().iter().zip(repeated_bias.data().iter())
        {
            assert_close(single_value, repeated_value, 1.0e-6);
        }
    }

    #[test]
    fn empty_batch_returns_empty_grad_without_parameter_updates() {
        let mut layer =
            ContinuousWaveletTransform::new("cwt", 3, vec![0.6, 1.4], 0.3, 4.0).unwrap();
        let input = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let grad_output = Tensor::from_vec(0, 3, Vec::new()).unwrap();

        let output = layer.forward(&input).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();

        assert_eq!(output.shape(), input.shape());
        assert!(output.data().is_empty());
        assert_eq!(grad_input.shape(), input.shape());
        assert!(grad_input.data().is_empty());
        assert!(layer.focus().gradient().is_none());
        assert!(layer.bias().gradient().is_none());
    }

    #[test]
    fn backward_matches_finite_difference_for_input() {
        let mut layer =
            ContinuousWaveletTransform::new("cwt", 3, vec![0.6, 1.4], 0.3, 4.0).unwrap();
        let input_values = vec![0.35, -0.25, 0.6];
        let input = Tensor::from_vec(1, 3, input_values.clone()).unwrap();
        let grad_output = Tensor::from_vec(1, 3, vec![0.2, -0.15, 0.4]).unwrap();

        let grad_input = layer.backward(&input, &grad_output).unwrap();
        let analytic = grad_input.data()[0];

        let epsilon = 1.0e-3f32;
        let loss_at = |values: Vec<f32>| {
            let tensor = Tensor::from_vec(1, 3, values).unwrap();
            let out = layer.forward(&tensor).unwrap();
            out.data()
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
            (analytic - finite_difference).abs() < 2.0e-3,
            "analytic={analytic} finite_difference={finite_difference}"
        );
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
