use crate::{Layout, PureResult, Tensor, TensorError};

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
        (((f64::from(value) - self.origin) - self.mean_offset) / self.denominator) as f32
    }
}

impl Tensor {
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
