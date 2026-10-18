//! Explicit wide-accumulator CPU projection over existing frozen packed weights.
use super::*;

impl Tensor {
    /// f32 inputs and output with four fixed f64 partial sums per dot product.
    /// CPU-only, including WASM CPU. No implicit WGPU or foreign fallback.
    pub fn matmul_prepacked_wide(&self, packed: &PackedB) -> PureResult<Self> {
        if self.cols != packed.inner() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape(),
                right: (packed.inner(), packed.cols()),
            });
        }
        self.layout.expect_row_major("wide matmul lhs")?;
        if !matches!(packed.layout(), PackedLayout::ColMajor) {
            return Err(TensorError::UnsupportedLayout {
                label: "wide matmul expects col-major packed rhs",
            });
        }
        let (rows, inner, cols) = (self.rows, self.cols, packed.cols());
        rows.checked_mul(cols)
            .and_then(AlignedVec::checked_layout)
            .ok_or(TensorError::InvalidValue {
                label: "wide matmul output exceeds addressable storage",
            })?;
        let execution = prepare_component_execution(
            TensorExecutionWorkload::PrepackedMatmul {
                rows: execution_dimension(rows)?,
                inner: execution_dimension(inner)?,
                cols: execution_dimension(cols)?,
                bias: false,
            },
            "matmul_prepacked",
            TensorExecutionBackend::CpuSimd,
        )?;
        let mut output = Tensor::zeros(rows, cols)?;
        let completion = if rows == 0 || cols == 0 {
            completed_no_op_execution(execution)?
        } else {
            cpu_dense::matmul_packed_wide_into(
                output.data_mut(),
                self.data(),
                packed.as_slice(),
                rows,
                inner,
                cols,
            )
            .map_err(|message| TensorError::BackendFailure {
                backend: "cpu_simd",
                message,
            })?;
            Self::validate_finite_tensor_util_slice("matmul_prepacked_output", output.data())?;
            completed_component_execution(execution, "cpu_simd", false)?
        };
        crate::emit_tensor_op(
            "matmul_prepacked",
            &[rows, inner, inner, cols],
            &[rows, cols],
        );
        emit_tensor_execution_receipt(completion, |data| {
            data.insert("rows".into(), serde_json::json!(rows));
            data.insert("inner".into(), serde_json::json!(inner));
            data.insert("cols".into(), serde_json::json!(cols));
            data.insert("accumulator_dtype".into(), serde_json::json!("f64"));
            data.insert(
                "kernel".into(),
                serde_json::json!("packed_column_dot_wide_v1"),
            );
            data.insert("rhs_layout".into(), serde_json::json!("packed"));
            data.insert(
                "packed_layout".into(),
                serde_json::json!(packed.layout().as_str()),
            );
        });
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(rows: usize, cols: usize, values: Vec<f32>) -> PackedB {
        PackedB::from_tensor(
            &Tensor::from_vec(rows, cols, values).unwrap(),
            Tile::col_major(),
        )
        .unwrap()
    }

    #[test]
    fn wide_projection_recovers_cancellation_without_changing_default_contract() {
        let lhs = Tensor::from_vec(1, 4, vec![1.0; 4]).unwrap();
        let rhs = packed(4, 1, vec![1e8, 1., -1e8, 1.]);
        assert_eq!(lhs.matmul_prepacked_wide(&rhs).unwrap().data(), &[2.]);
        assert_eq!(
            lhs.matmul_prepacked_with_backend(&rhs, MatmulBackend::CpuSimd)
                .unwrap()
                .data(),
            &[1.]
        );
        let lhs = Tensor::from_vec(1, 2, vec![2., 2.]).unwrap();
        let rhs = packed(2, 1, vec![f32::MAX, -f32::MAX]);
        assert_eq!(lhs.matmul_prepacked_wide(&rhs).unwrap().data(), &[0.]);
    }

    #[test]
    fn wide_projection_rejects_bad_shapes_layouts_and_nonfinite_results() {
        let lhs = Tensor::from_vec(1, 2, vec![2., 2.]).unwrap();
        assert!(lhs.matmul_prepacked_wide(&packed(1, 1, vec![1.])).is_err());
        for values in [
            vec![f32::MAX; 2],
            vec![f32::NAN, 1.],
            vec![f32::INFINITY, 1.],
        ] {
            assert!(lhs.matmul_prepacked_wide(&packed(2, 1, values)).is_err());
        }
        let mut rhs = packed(2, 1, vec![1., 1.]);
        rhs.layout = PackedLayout::Tiled {
            tm: 2,
            tn: 1,
            tk: 2,
        };
        assert!(lhs.matmul_prepacked_wide(&rhs).is_err());
        let mut column_lhs = lhs;
        column_lhs.layout = Layout::ColMajor;
        assert!(column_lhs
            .matmul_prepacked_wide(&packed(2, 1, vec![1., 1.]))
            .is_err());
    }

    #[test]
    fn wide_projection_handles_zero_extents() {
        for (rows, inner, cols) in [(0, 4, 2), (2, 4, 0), (2, 0, 3)] {
            let lhs = Tensor::zeros(rows, inner).unwrap();
            let rhs = packed(inner, cols, vec![0.; inner * cols]);
            let out = lhs.matmul_prepacked_wide(&rhs).unwrap();
            assert_eq!(out.shape(), (rows, cols));
            assert!(out.data().iter().all(|&v| v == 0.));
        }
    }

    #[test]
    fn wide_projection_rejects_overflow_before_allocating_empty_inner_product() {
        let byte_boundary = isize::MAX as usize / std::mem::size_of::<f32>();
        for (rows, cols) in [(usize::MAX, 2), (byte_boundary, 2), (byte_boundary, 1)] {
            let rhs = packed(0, cols, vec![]);
            let lhs = Tensor::zeros(rows, 0).unwrap();
            assert!(matches!(
                lhs.matmul_prepacked_wide(&rhs),
                Err(TensorError::InvalidValue {
                    label: "wide matmul output exceeds addressable storage"
                })
            ));
        }
    }
}
