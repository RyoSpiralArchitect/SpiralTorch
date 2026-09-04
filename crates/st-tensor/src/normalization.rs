use crate::{Layout, PureResult, Tensor, TensorError, TensorUtilBackend};

pub(crate) fn layer_norm_value(input: &[f32], residual: Option<&[f32]>, index: usize) -> f32 {
    input[index] + residual.map_or(0.0, |values| values[index])
}

// Dimensions and residual length must be validated before calling this helper.
pub(crate) fn validate_layer_norm_values(
    input: &[f32],
    residual: Option<&[f32]>,
    cols: usize,
    epsilon: f32,
) -> PureResult<()> {
    let mut first = 0.0;
    let mut constant = true;
    for index in 0..input.len() {
        let value = layer_norm_value(input, residual, index);
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "layer_norm_value",
                value,
            });
        }
        if epsilon == 0.0 {
            if index % cols == 0 {
                first = value;
                constant = true;
            } else {
                constant &= value == first;
            }
            if index % cols == cols - 1 && constant {
                return Err(TensorError::InvalidValue {
                    label: "layer_norm_zero_variance_without_epsilon",
                });
            }
        }
    }
    Ok(())
}

pub(crate) struct LayerNormRowStats {
    origin: f64,
    mean_offset: f64,
    denominator: f64,
}

impl LayerNormRowStats {
    pub(crate) fn new(input: &[f32], residual: Option<&[f32]>, epsilon: f32) -> Self {
        let origin = f64::from(layer_norm_value(input, residual, 0));
        let mean_offset = (0..input.len())
            .map(|index| f64::from(layer_norm_value(input, residual, index)) - origin)
            .sum::<f64>()
            / input.len() as f64;
        let variance = (0..input.len())
            .map(|index| {
                let centered =
                    (f64::from(layer_norm_value(input, residual, index)) - origin) - mean_offset;
                centered * centered
            })
            .sum::<f64>()
            / input.len() as f64;
        Self {
            origin,
            mean_offset,
            denominator: (variance + f64::from(epsilon)).sqrt(),
        }
    }

    pub(crate) fn normalize(&self, value: f32) -> f32 {
        self.normalize_f64(value) as f32
    }

    fn normalize_f64(&self, value: f32) -> f64 {
        ((f64::from(value) - self.origin) - self.mean_offset) / self.denominator
    }
}

impl Tensor {
    /// First-order affine LayerNorm VJP: `(input, gamma, beta)` gradients.
    /// Affine gradients sum rows; any loss reduction belongs to the upstream seed.
    /// CPU intermediates remain f64 until each requested final gradient is stored.
    pub fn layer_norm_affine_backward(
        &self,
        gamma: &Tensor,
        upstream: &Tensor,
        epsilon: f32,
    ) -> PureResult<(Tensor, Tensor, Tensor)> {
        self.layer_norm_affine_backward_with_backend(
            gamma,
            upstream,
            epsilon,
            1.0,
            TensorUtilBackend::Cpu,
        )
    }

    /// Shared VJP with an explicit affine-gradient scale and utility backend.
    /// The scale affects only gamma/beta, never the input gradient. `Auto` uses
    /// the stable CPU implementation. WGPU uses CPU moments and GPU tensor
    /// utilities, preserving the hybrid training path rather than claiming a
    /// fused GPU backward kernel. Only finite final gradients are returned.
    pub fn layer_norm_affine_backward_with_backend(
        &self,
        gamma: &Tensor,
        upstream: &Tensor,
        epsilon: f32,
        parameter_gradient_scale: f32,
        backend: TensorUtilBackend,
    ) -> PureResult<(Tensor, Tensor, Tensor)> {
        let [input, gamma, beta] = self.layer_norm_vjp(
            gamma,
            upstream,
            epsilon,
            parameter_gradient_scale,
            backend,
            [true; 3],
        )?;
        Ok((input.unwrap(), gamma.unwrap(), beta.unwrap()))
    }

    pub(crate) fn layer_norm_vjp(
        &self,
        gamma: &Tensor,
        upstream: &Tensor,
        epsilon: f32,
        parameter_gradient_scale: f32,
        backend: TensorUtilBackend,
        requested: [bool; 3],
    ) -> PureResult<[Option<Tensor>; 3]> {
        let (rows, cols) = self.shape();
        if cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        if gamma.shape() != (1, cols) || upstream.shape() != self.shape() {
            return Err(TensorError::ShapeMismatch {
                left: if gamma.shape() != (1, cols) {
                    gamma.shape()
                } else {
                    upstream.shape()
                },
                right: if gamma.shape() != (1, cols) {
                    (1, cols)
                } else {
                    self.shape()
                },
            });
        }
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "layernorm_epsilon",
                value: epsilon,
            });
        }
        finite(
            "layernorm_parameter_gradient_scale",
            parameter_gradient_scale,
        )?;
        let input = self.to_layout(Layout::RowMajor)?;
        let gamma = gamma.to_layout(Layout::RowMajor)?;
        let upstream = upstream.to_layout(Layout::RowMajor)?;
        validate_layer_norm_values(input.data(), None, cols, epsilon)?;
        for &value in gamma.data().iter().chain(upstream.data()) {
            finite("layernorm_backward_operand", value)?;
        }
        let output = if rows == 0 || !matches!(backend, TensorUtilBackend::GpuWgpu) {
            layer_norm_vjp_cpu(
                &input,
                &gamma,
                &upstream,
                epsilon,
                parameter_gradient_scale,
                requested,
            )?
        } else {
            layer_norm_vjp_gpu(
                &input,
                &gamma,
                &upstream,
                epsilon,
                parameter_gradient_scale,
                requested,
            )?
        };
        for tensor in output.iter().flatten() {
            for &value in tensor.data() {
                finite("layernorm_backward_output", value)?;
            }
        }
        crate::emit_tensor_op("layer_norm_affine_backward", &[rows, cols], &[rows, cols]);
        crate::emit_tensor_op_meta("layer_norm_affine_backward", || {
            serde_json::json!({
                "semantic_owner": "st-tensor", "rows": rows, "cols": cols,
                "backend": if rows > 0 && matches!(backend, TensorUtilBackend::GpuWgpu) { "hybrid" } else { "cpu" },
                "normalization_backend": "cpu", "parameter_gradient_scale": parameter_gradient_scale,
                "input_gradient_scale": 1.0, "requested_gradients": requested,
            })
        });
        Ok(output)
    }

    /// Un-affined row-normalized values and per-row inverse standard deviation.
    ///
    /// Returns tensors shaped `(rows, cols)` and `(rows, 1)`. This CPU helper
    /// uses the same centered f64 moments as affine LayerNorm, so backward
    /// callers need not reconstruct a rounded f32 mean. Final outputs must fit
    /// finite f32, and constant rows with zero epsilon are rejected.
    pub fn layer_norm_stats(&self, epsilon: f32) -> PureResult<(Tensor, Tensor)> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(TensorError::NonFiniteValue {
                label: "layernorm_epsilon",
                value: epsilon,
            });
        }
        let (rows, cols) = self.shape();
        if cols == 0 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        let input = self.to_layout(Layout::RowMajor)?;
        validate_layer_norm_values(input.data(), None, cols, epsilon)?;
        let mut normalized = Vec::with_capacity(input.len());
        let mut inverse_std = Vec::with_capacity(rows);
        for row in input.data().chunks_exact(cols) {
            let stats = LayerNormRowStats::new(row, None, epsilon);
            normalized.extend(row.iter().map(|&value| stats.normalize(value)));
            inverse_std.push((1.0 / stats.denominator) as f32);
        }
        for &value in normalized.iter().chain(&inverse_std) {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "layer_norm_stats_output",
                    value,
                });
            }
        }
        let normalized = Tensor::from_vec(rows, cols, normalized)?;
        let inverse_std = Tensor::from_vec(rows, 1, inverse_std)?;
        crate::emit_tensor_op("layer_norm_stats", &[rows, cols], &[rows, cols]);
        crate::emit_tensor_op_meta("layer_norm_stats", || {
            serde_json::json!({
                "backend": "cpu", "kernel": "centered_f64", "accumulator_dtype": "f64",
                "semantic_owner": "st-tensor", "rows": rows, "cols": cols,
            })
        });
        Ok((normalized, inverse_std))
    }
}

fn finite(label: &'static str, value: f32) -> PureResult<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(TensorError::NonFiniteValue { label, value })
    }
}

fn layer_norm_vjp_cpu(
    input: &Tensor,
    gamma: &Tensor,
    upstream: &Tensor,
    epsilon: f32,
    scale: f32,
    requested: [bool; 3],
) -> PureResult<[Option<Tensor>; 3]> {
    let (rows, cols) = input.shape();
    let mut dx = requested[0].then(|| Vec::with_capacity(input.len()));
    let mut dg = requested[1].then(|| vec![0.0f64; cols]);
    let mut db = requested[2].then(|| vec![0.0f64; cols]);
    for (row, seed) in input
        .data()
        .chunks_exact(cols)
        .zip(upstream.data().chunks_exact(cols))
    {
        let stats = LayerNormRowStats::new(row, None, epsilon);
        if let Some(dx) = &mut dx {
            // Center the weighted seed before reduction: constant cotangents
            // then cancel exactly, even when their f32 product would overflow.
            let origin = f64::from(seed[0]) * f64::from(gamma.data()[0]);
            let mut mean = 0.0;
            let mut projection = 0.0;
            for col in 0..cols {
                let g = f64::from(seed[col]) * f64::from(gamma.data()[col]) - origin;
                mean += g;
                projection += g * stats.normalize_f64(row[col]);
            }
            mean /= cols as f64;
            projection /= cols as f64;
            for col in 0..cols {
                let g = f64::from(seed[col]) * f64::from(gamma.data()[col]) - origin;
                dx.push(finite(
                    "layernorm_backward_input_grad",
                    ((g - mean - stats.normalize_f64(row[col]) * projection) / stats.denominator)
                        as f32,
                )?);
            }
        }
        for col in 0..cols {
            if let Some(dg) = &mut dg {
                dg[col] += f64::from(seed[col]) * stats.normalize_f64(row[col]);
            }
            if let Some(db) = &mut db {
                db[col] += f64::from(seed[col]);
            }
        }
    }
    let affine = |values: Option<Vec<f64>>| -> PureResult<Option<Tensor>> {
        values
            .map(|values| {
                let values = values
                    .into_iter()
                    .map(|value| {
                        finite(
                            "layernorm_backward_affine_grad",
                            (value * f64::from(scale)) as f32,
                        )
                    })
                    .collect::<PureResult<Vec<_>>>()?;
                Tensor::from_vec(1, cols, values)
            })
            .transpose()
    };
    Ok([
        dx.map(|values| Tensor::from_vec(rows, cols, values))
            .transpose()?,
        affine(dg)?,
        affine(db)?,
    ])
}

fn layer_norm_vjp_gpu(
    input: &Tensor,
    gamma: &Tensor,
    upstream: &Tensor,
    epsilon: f32,
    scale: f32,
    requested: [bool; 3],
) -> PureResult<[Option<Tensor>; 3]> {
    let backend = TensorUtilBackend::GpuWgpu;
    let (_, cols) = input.shape();
    let (normed, inverse_std) = input.layer_norm_stats(epsilon)?;
    let dx = if requested[0] {
        let weighted = upstream
            .mul_row_with_backend(gamma.data(), backend)?
            .transpose_with_backend(backend)?;
        let normalized = normed.transpose_with_backend(backend)?;
        let sum = weighted.try_sum_axis0_with_backend(backend)?;
        let projection = weighted
            .hadamard_with_backend(&normalized, backend)?
            .try_sum_axis0_with_backend(backend)?;
        let mut correction = normalized.mul_row_with_backend(&projection, backend)?;
        correction.add_row_inplace_with_backend(&sum, backend)?;
        let correction = correction.scale_with_backend(1.0 / cols as f32, backend)?;
        let mut output = weighted;
        output.add_scaled_with_backend(&correction, -1.0, backend)?;
        Some(
            output
                .mul_row_with_backend(inverse_std.data(), backend)?
                .transpose_with_backend(backend)?,
        )
    } else {
        None
    };
    let dg = if requested[1] {
        Some(Tensor::from_vec(
            1,
            cols,
            upstream
                .hadamard_with_backend(&normed, backend)?
                .try_sum_axis0_scaled_with_backend(scale, backend)?,
        )?)
    } else {
        None
    };
    let db = if requested[2] {
        Some(Tensor::from_vec(
            1,
            cols,
            upstream.try_sum_axis0_scaled_with_backend(scale, backend)?,
        )?)
    } else {
        None
    };
    Ok([dx, dg, db])
}
