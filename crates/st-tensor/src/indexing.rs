//! Exact integer row indexing shared by embeddings and reverse-mode graphs.

use crate::{Layout, PureResult, Tensor, TensorError, TensorUtilBackend};

fn volume(rows: usize, cols: usize) -> PureResult<usize> {
    rows.checked_mul(cols)
        .filter(|&len| len <= isize::MAX as usize / size_of::<f32>())
        .ok_or(TensorError::InvalidDimensions { rows, cols })
}

fn validate_indices(indices: &[usize], rows: usize) -> PureResult<()> {
    if indices.iter().any(|&index| index >= rows) {
        return Err(TensorError::InvalidValue {
            label: "row_index_out_of_bounds",
        });
    }
    Ok(())
}

fn finite(values: &[f32], label: &'static str) -> PureResult<()> {
    for &value in values {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue { label, value });
        }
    }
    Ok(())
}

fn zeros(len: usize) -> PureResult<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| TensorError::InvalidValue {
            label: "row_index_allocation",
        })?;
    values.resize(len, 0.0);
    Ok(values)
}

impl Tensor {
    /// Select rows by exact integer IDs, preserving order and duplicates.
    /// IDs are never rounded or clamped. Selected values must be finite.
    pub fn gather_rows(&self, indices: &[usize]) -> PureResult<Self> {
        self.gather_rows_with_backend(indices, TensorUtilBackend::Cpu)
    }

    /// `Auto` uses CPU. Explicit WGPU is strict; it never silently falls back.
    /// Empty results perform no device work, but still validate every ID.
    pub fn gather_rows_with_backend(
        &self,
        indices: &[usize],
        backend: TensorUtilBackend,
    ) -> PureResult<Self> {
        let (rows, cols) = self.shape();
        validate_indices(indices, rows)?;
        let len = volume(indices.len(), cols)?;
        let input = self.to_layout(Layout::RowMajor)?;
        let gpu = len > 0 && matches!(backend, TensorUtilBackend::GpuWgpu);
        let values = if gpu {
            gather_gpu(input.data(), indices, rows, cols)?
        } else {
            let mut values = zeros(len)?;
            if cols > 0 {
                for (out, &index) in values.chunks_exact_mut(cols).zip(indices) {
                    out.copy_from_slice(&input.data()[index * cols..(index + 1) * cols]);
                }
            }
            values
        };
        finite(&values, "gather_rows_output")?;
        let output = Self::from_vec(indices.len(), cols, values)?;
        emit("gather_rows", self.shape(), output.shape(), gpu, 1.0);
        Ok(output)
    }

    /// Sum input rows into a fresh zero table; duplicates accumulate in order.
    /// No batch average is implicit. CPU accumulation uses f64 intermediates.
    pub fn scatter_add_rows(&self, indices: &[usize], output_rows: usize) -> PureResult<Self> {
        self.scatter_add_rows_with_backend(indices, output_rows, TensorUtilBackend::Cpu)
    }

    pub fn scatter_add_rows_with_backend(
        &self,
        indices: &[usize],
        output_rows: usize,
        backend: TensorUtilBackend,
    ) -> PureResult<Self> {
        self.scatter_add_rows_scaled_with_backend(indices, output_rows, 1.0, backend)
    }

    /// Explicit scale applied to the row sum, for callers owning a reduction.
    /// CPU scales before final f32 conversion; WGPU uses deterministic f32 sums.
    /// Nonfinite final values fail without mutating any operand.
    pub fn scatter_add_rows_scaled_with_backend(
        &self,
        indices: &[usize],
        output_rows: usize,
        scale: f32,
        backend: TensorUtilBackend,
    ) -> PureResult<Self> {
        let (rows, cols) = self.shape();
        if indices.len() != rows {
            return Err(TensorError::DataLength {
                expected: rows,
                got: indices.len(),
            });
        }
        validate_indices(indices, output_rows)?;
        finite(&[scale], "scatter_add_rows_scale")?;
        let len = volume(output_rows, cols)?;
        let input = self.to_layout(Layout::RowMajor)?;
        finite(input.data(), "scatter_add_rows_input")?;
        let gpu = rows > 0 && len > 0 && matches!(backend, TensorUtilBackend::GpuWgpu);
        let values = if gpu {
            scatter_gpu(input.data(), indices, output_rows, cols, scale)?
        } else {
            let mut values = zeros(len)?;
            if rows > 0 && cols > 0 {
                let (offsets, positions) = grouped_rows(indices, output_rows)?;
                for row in 0..output_rows {
                    for col in 0..cols {
                        let sum: f64 = positions[offsets[row]..offsets[row + 1]]
                            .iter()
                            .map(|&position| f64::from(input.data()[position * cols + col]))
                            .sum();
                        values[row * cols + col] = (sum * f64::from(scale)) as f32;
                    }
                }
            }
            values
        };
        finite(&values, "scatter_add_rows_output")?;
        let output = Self::from_vec(output_rows, cols, values)?;
        emit("scatter_add_rows", self.shape(), output.shape(), gpu, scale);
        Ok(output)
    }
}

// Stable CSR grouping avoids scanning all tokens for each vocabulary row.
// IDs must already have been bounds-checked by the caller.
pub(crate) fn grouped_rows(indices: &[usize], rows: usize) -> PureResult<(Vec<usize>, Vec<usize>)> {
    let count = rows.checked_add(1).ok_or(TensorError::InvalidValue {
        label: "row_index_offset_overflow",
    })?;
    let allocate = |len| -> PureResult<Vec<usize>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| TensorError::InvalidValue {
                label: "row_index_allocation",
            })?;
        values.resize(len, 0);
        Ok(values)
    };
    let mut offsets = allocate(count)?;
    for &index in indices {
        offsets[index + 1] += 1;
    }
    for row in 1..count {
        offsets[row] += offsets[row - 1];
    }
    let mut cursor = allocate(rows)?;
    cursor.copy_from_slice(&offsets[..rows]);
    let mut positions = allocate(indices.len())?;
    for (position, &index) in indices.iter().enumerate() {
        positions[cursor[index]] = position;
        cursor[index] += 1;
    }
    Ok((offsets, positions))
}

fn emit(op: &'static str, input: (usize, usize), output: (usize, usize), gpu: bool, scale: f32) {
    crate::emit_tensor_op(op, &[input.0, input.1], &[output.0, output.1]);
    crate::emit_tensor_op_meta(op, || {
        serde_json::json!({
            "semantic_owner": "st-tensor", "index_dtype": "usize",
            "backend": if gpu { "wgpu" } else { "cpu" },
            "accumulator_dtype": if op == "gather_rows" { None } else { Some(if gpu { "f32" } else { "f64" }) },
            "index_transport": if gpu { "u32" } else { "usize" },
            "scale": scale,
        })
    });
}

fn gather_gpu(input: &[f32], indices: &[usize], rows: usize, cols: usize) -> PureResult<Vec<f32>> {
    #[cfg(feature = "wgpu_dense")]
    {
        crate::wgpu_dense::gather_rows(input, indices, rows, cols).map_err(gpu_error)
    }
    #[cfg(not(feature = "wgpu_dense"))]
    {
        let _ = (input, indices, rows, cols);
        Err(gpu_error("wgpu_dense feature is disabled".into()))
    }
}

fn scatter_gpu(
    input: &[f32],
    indices: &[usize],
    rows: usize,
    cols: usize,
    scale: f32,
) -> PureResult<Vec<f32>> {
    #[cfg(feature = "wgpu_dense")]
    {
        crate::wgpu_dense::scatter_add_rows(input, indices, rows, cols, scale).map_err(gpu_error)
    }
    #[cfg(not(feature = "wgpu_dense"))]
    {
        let _ = (input, indices, rows, cols, scale);
        Err(gpu_error("wgpu_dense feature is disabled".into()))
    }
}

fn gpu_error(message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message,
    }
}
