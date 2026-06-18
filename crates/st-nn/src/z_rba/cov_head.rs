// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Covariance reconstruction head used by the Z-RBA module.

use crate::execution::{
    current_backend_policy, current_matmul_backend, current_tensor_util_backend_for_values,
};
use crate::{PureResult, Tensor, TensorError};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, MatmulBackend};
use std::cmp::Ordering;

fn validate_finite_value(label: &'static str, value: f32) -> PureResult<()> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
}

fn checked_value(label: &'static str, value: f32) -> PureResult<f32> {
    validate_finite_value(label, value)?;
    Ok(value)
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

fn validate_finite_matrix(label: &'static str, matrix: &DMatrix<f32>) -> PureResult<()> {
    for &value in matrix.iter() {
        validate_finite_value(label, value)?;
    }
    Ok(())
}

fn relabel_non_finite<T>(result: PureResult<T>, label: &'static str) -> PureResult<T> {
    match result {
        Err(TensorError::NonFiniteValue { value, .. }) => {
            Err(TensorError::NonFiniteValue { label, value })
        }
        other => other,
    }
}

fn matmul_backend_label(backend: MatmulBackend) -> &'static str {
    match backend {
        MatmulBackend::Auto => "auto",
        MatmulBackend::CpuFaer => "faer",
        MatmulBackend::CpuSimd => "cpu_simd",
        MatmulBackend::CpuNaive => "naive",
        #[cfg(feature = "wgpu")]
        MatmulBackend::GpuWgpu => "wgpu",
        #[cfg(feature = "hip")]
        MatmulBackend::GpuHip => "hip",
        #[allow(unreachable_patterns)]
        _ => "gpu",
    }
}

/// Telemetry exposing key PSD statistics.
#[derive(Clone, Debug)]
pub struct CovHeadTelemetry {
    pub min_eigenvalue: f32,
    pub max_eigenvalue: f32,
    pub condition_number: f32,
    pub stabiliser: f32,
}

/// Covariance output bundle.
#[derive(Clone, Debug)]
pub struct CovHeadOutput {
    pub covariance: Tensor,
    pub telemetry: CovHeadTelemetry,
}

/// Covariance reconstruction using low-rank factors + diagonal correction.
#[derive(Debug)]
pub struct CovHead {
    rank: usize,
    stabiliser: f32,
}

impl CovHead {
    pub fn new(rank: usize) -> Self {
        Self {
            rank: rank.max(1),
            stabiliser: 1e-4,
        }
    }

    pub fn forward(&self, mu: &Tensor, sigma_diag: &Tensor) -> PureResult<CovHeadOutput> {
        let (rows, cols) = mu.shape();
        if sigma_diag.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: sigma_diag.shape(),
                right: (rows, cols),
            });
        }
        if rows == 0 {
            return Err(TensorError::EmptyInput("cov_head::forward"));
        }

        validate_finite_tensor("zrba_cov_mu", mu)?;
        validate_finite_tensor("zrba_cov_sigma_diag", sigma_diag)?;
        let mu_data = mu.data();
        let reduction_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let inv_rows = 1.0 / rows as f32;
        let mean = relabel_non_finite(
            mu.try_sum_axis0_scaled_with_backend(inv_rows, reduction_backend),
            "zrba_cov_mean",
        )?;
        validate_finite_slice("zrba_cov_mean", &mean)?;

        let mut centered = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let offset = row * cols;
            for col in 0..cols {
                centered.push(checked_value(
                    "zrba_cov_centered_mu",
                    mu_data[offset + col] - mean[col],
                )?);
            }
        }
        let centered = Tensor::from_vec(rows, cols, centered)?;
        validate_finite_tensor("zrba_cov_centered_mu", &centered)?;
        let matmul_backend = current_matmul_backend();
        let sample_cov_tensor = relabel_non_finite(
            centered.matmul_lhs_transpose_scaled_with_backend(&centered, inv_rows, matmul_backend),
            "zrba_covariance_product",
        )?;
        validate_finite_tensor("zrba_covariance", &sample_cov_tensor)?;

        let mut diag_variance = relabel_non_finite(
            sigma_diag.try_sum_axis0_scaled_with_backend(inv_rows, reduction_backend),
            "zrba_cov_diag_variance",
        )?;
        validate_finite_slice("zrba_cov_diag_variance", &diag_variance)?;
        for value in &mut diag_variance {
            *value = (*value).max(1e-6);
            validate_finite_value("zrba_cov_diag_variance", *value)?;
        }

        let sample_cov = DMatrix::from_vec(cols, cols, sample_cov_tensor.data().to_vec());
        validate_finite_matrix("zrba_covariance_matrix", &sample_cov)?;
        let truncated = self.low_rank_projection(sample_cov.clone())?;
        let diag = DMatrix::from_diagonal(&DVector::from_vec(diag_variance));
        validate_finite_matrix("zrba_cov_diag_matrix", &diag)?;
        let combined = truncated + diag;
        validate_finite_matrix("zrba_cov_combined_matrix", &combined)?;
        let (psd, telemetry) = self.make_psd(combined)?;
        validate_finite_matrix("zrba_cov_psd_matrix", &psd)?;
        telemetry.validate()?;
        let tensor = Tensor::from_vec(cols, cols, psd.iter().copied().collect())?;
        validate_finite_tensor("zrba_covariance_output", &tensor)?;
        emit_cov_head_meta(
            rows,
            cols,
            self.rank,
            self.stabiliser,
            current_backend_policy()
                .map(|policy| policy.device_backend_label())
                .unwrap_or("auto"),
            reduction_backend.to_string(),
            matmul_backend_label(matmul_backend),
            &telemetry,
        );
        Ok(CovHeadOutput {
            covariance: tensor,
            telemetry,
        })
    }

    fn low_rank_projection(&self, matrix: DMatrix<f32>) -> PureResult<DMatrix<f32>> {
        validate_finite_matrix("zrba_cov_low_rank_input", &matrix)?;
        let eigen = SymmetricEigen::new(matrix.clone());
        for &value in eigen.eigenvalues.iter() {
            validate_finite_value("zrba_cov_low_rank_eigenvalue", value)?;
        }
        validate_finite_matrix("zrba_cov_low_rank_eigenvectors", &eigen.eigenvectors)?;
        let mut pairs: Vec<(usize, f32)> = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(idx, &value)| (idx, value))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let mut result = DMatrix::zeros(matrix.nrows(), matrix.ncols());
        for (rank, (idx, value)) in pairs.into_iter().enumerate() {
            if rank >= self.rank {
                break;
            }
            let eigenvector = eigen.eigenvectors.column(idx);
            let value = value.max(0.0);
            result += value * (eigenvector * eigenvector.transpose());
            validate_finite_matrix("zrba_cov_low_rank_projection", &result)?;
        }
        Ok(result)
    }

    fn make_psd(&self, matrix: DMatrix<f32>) -> PureResult<(DMatrix<f32>, CovHeadTelemetry)> {
        validate_finite_matrix("zrba_cov_psd_input", &matrix)?;
        let eigen = SymmetricEigen::new(matrix.clone());
        for &value in eigen.eigenvalues.iter() {
            validate_finite_value("zrba_cov_psd_eigenvalue", value)?;
        }
        validate_finite_matrix("zrba_cov_psd_eigenvectors", &eigen.eigenvectors)?;
        let mut adjusted = eigen.eigenvalues.clone();
        let mut min_eigen = f32::INFINITY;
        let mut max_eigen = f32::NEG_INFINITY;
        for value in adjusted.iter_mut() {
            min_eigen = min_eigen.min(*value);
            max_eigen = max_eigen.max(*value);
            if *value < self.stabiliser {
                *value = self.stabiliser;
            }
            validate_finite_value("zrba_cov_adjusted_eigenvalue", *value)?;
        }
        let diag = DMatrix::from_diagonal(&adjusted);
        let psd = &eigen.eigenvectors * diag * eigen.eigenvectors.transpose();
        validate_finite_matrix("zrba_cov_psd_matrix", &psd)?;
        let mut min_adj = f32::INFINITY;
        let mut max_adj = f32::NEG_INFINITY;
        for value in adjusted.iter() {
            min_adj = min_adj.min(*value);
            max_adj = max_adj.max(*value);
        }
        let condition = if min_adj > 0.0 {
            max_adj / min_adj
        } else {
            f32::INFINITY
        };
        validate_finite_value("zrba_cov_condition_number", condition)?;
        let telemetry = CovHeadTelemetry {
            min_eigenvalue: min_adj,
            max_eigenvalue: max_adj,
            condition_number: condition,
            stabiliser: self.stabiliser,
        };
        telemetry.validate()?;
        Ok((psd, telemetry))
    }
}

impl CovHeadTelemetry {
    fn validate(&self) -> PureResult<()> {
        validate_finite_value("zrba_cov_min_eigenvalue", self.min_eigenvalue)?;
        validate_finite_value("zrba_cov_max_eigenvalue", self.max_eigenvalue)?;
        validate_finite_value("zrba_cov_condition_number", self.condition_number)?;
        validate_finite_value("zrba_cov_stabiliser", self.stabiliser)
    }
}

fn emit_cov_head_meta(
    rows: usize,
    cols: usize,
    rank: usize,
    stabiliser: f32,
    requested_backend: &'static str,
    reduction_backend: String,
    covariance_backend: &'static str,
    telemetry: &CovHeadTelemetry,
) {
    emit_tensor_op(
        "zrba_cov_head_forward",
        &[rows, cols, rows, cols],
        &[cols, cols],
    );
    emit_tensor_op_meta("zrba_cov_head_forward", || {
        let values = rows.saturating_mul(cols);
        let covariance_values = cols.saturating_mul(cols);
        let effective_rank = rank.min(cols);
        let eigen_ops = 2usize
            .saturating_mul(cols)
            .saturating_mul(cols)
            .saturating_mul(cols);
        let mut data = serde_json::json!({
            "backend": "hybrid",
            "requested_backend": requested_backend,
            "kernel": "zrba.cov_head.cpu_eigen",
            "kind": "covariance_psd_projection",
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": cols,
            "output_cols": cols,
            "output_values": covariance_values,
            "rank": rank,
            "effective_rank": effective_rank,
            "stabiliser": stabiliser,
            "reduction_backend": reduction_backend,
            "covariance_centering_backend": "cpu",
            "covariance_accumulation_backend": covariance_backend,
            "low_rank_projection_backend": "cpu_eigen",
            "psd_projection_backend": "cpu_eigen",
            "min_eigenvalue": telemetry.min_eigenvalue,
            "max_eigenvalue": telemetry.max_eigenvalue,
            "condition_number": telemetry.condition_number,
            "telemetry_stabiliser": telemetry.stabiliser,
            "estimated_mean_reduction_values": values,
            "estimated_diag_reduction_values": values,
            "estimated_covariance_centering_values": values,
            "estimated_covariance_ops": rows.saturating_mul(cols).saturating_mul(cols),
            "estimated_eigen_matrix_values": covariance_values,
            "empty": rows == 0 || cols == 0,
        });
        if let Some(object) = data.as_object_mut() {
            object.insert(
                "eigen_solver_backend".to_string(),
                serde_json::json!("nalgebra_cpu"),
            );
            object.insert(
                "eigen_precision_backend".to_string(),
                serde_json::json!("f32_cpu"),
            );
            object.insert(
                "low_rank_projection_mode".to_string(),
                serde_json::json!("topk_symmetric_eigen_outer_products"),
            );
            object.insert(
                "psd_projection_mode".to_string(),
                serde_json::json!("symmetric_eigenvalue_clamp"),
            );
            object.insert(
                "psd_projection_blocker".to_string(),
                serde_json::json!("symmetric_eigen_decomposition_and_dense_reconstruction"),
            );
            object.insert(
                "estimated_eigen_decompositions".to_string(),
                serde_json::json!(2),
            );
            object.insert(
                "estimated_eigen_ops".to_string(),
                serde_json::json!(eigen_ops),
            );
            object.insert(
                "estimated_low_rank_reconstruction_ops".to_string(),
                serde_json::json!(effective_rank.saturating_mul(cols).saturating_mul(cols)),
            );
            object.insert(
                "estimated_psd_reconstruction_ops".to_string(),
                serde_json::json!(cols.saturating_mul(cols).saturating_mul(cols)),
            );
        }
        data
    });
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

    #[test]
    fn covariance_head_outputs_psd() {
        let mu = Tensor::from_vec(
            3,
            4,
            vec![
                0.1, 0.2, 0.3, 0.4, 0.0, -0.1, 0.2, 0.3, -0.2, 0.1, 0.0, -0.3,
            ],
        )
        .unwrap();
        let sigma = Tensor::from_vec(
            3,
            4,
            vec![
                0.05, 0.02, 0.03, 0.01, 0.04, 0.03, 0.02, 0.01, 0.06, 0.05, 0.04, 0.03,
            ],
        )
        .unwrap();
        let head = CovHead::new(2);
        let output = head.forward(&mu, &sigma).unwrap();
        assert_eq!(output.covariance.shape(), (4, 4));
        assert!(output.telemetry.min_eigenvalue > 0.0);
        assert!(output.telemetry.condition_number.is_finite());
    }

    #[test]
    fn covariance_head_rejects_non_finite_mu_before_reduction() {
        let mu = Tensor::from_vec(2, 2, vec![0.1, f32::NAN, 0.2, 0.3]).unwrap();
        let sigma = Tensor::from_vec(2, 2, vec![0.05, 0.02, 0.03, 0.01]).unwrap();
        let head = CovHead::new(1);

        let err = head.forward(&mu, &sigma).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zrba_cov_mu",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn covariance_head_rejects_overflowing_covariance_product() {
        let mu = Tensor::from_vec(2, 1, vec![1.0e20, -1.0e20]).unwrap();
        let sigma = Tensor::from_vec(2, 1, vec![0.05, 0.02]).unwrap();
        let head = CovHead::new(1);

        let err = head.forward(&mu, &sigma).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "zrba_covariance_product",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn covariance_head_emits_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mu = Tensor::from_vec(
            3,
            4,
            vec![
                0.1, 0.2, 0.3, 0.4, 0.0, -0.1, 0.2, 0.3, -0.2, 0.1, 0.0, -0.3,
            ],
        )
        .unwrap();
        let sigma = Tensor::from_vec(
            3,
            4,
            vec![
                0.05, 0.02, 0.03, 0.01, 0.04, 0.03, 0.02, 0.01, 0.06, 0.05, 0.04, 0.03,
            ],
        )
        .unwrap();
        let head = CovHead::new(2);
        let output = head.forward(&mu, &sigma).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(output.covariance.shape(), (4, 4));
        let events = events.lock().unwrap();
        let cov = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zrba_cov_head_forward" && data["rows"] == 3 && data["cols"] == 4
            })
            .expect("zrba covariance head metadata event");
        assert_eq!(cov.1["backend"], "hybrid");
        assert_eq!(cov.1["requested_backend"], "auto");
        assert_eq!(cov.1["kind"], "covariance_psd_projection");
        assert_eq!(cov.1["rank"], 2);
        assert_eq!(cov.1["effective_rank"], 2);
        assert_eq!(cov.1["reduction_backend"], "auto");
        assert_eq!(cov.1["covariance_centering_backend"], "cpu");
        assert_eq!(cov.1["covariance_accumulation_backend"], "auto");
        assert_eq!(cov.1["low_rank_projection_backend"], "cpu_eigen");
        assert_eq!(cov.1["psd_projection_backend"], "cpu_eigen");
        assert_eq!(cov.1["eigen_solver_backend"], "nalgebra_cpu");
        assert_eq!(cov.1["eigen_precision_backend"], "f32_cpu");
        assert_eq!(
            cov.1["low_rank_projection_mode"],
            "topk_symmetric_eigen_outer_products"
        );
        assert_eq!(cov.1["psd_projection_mode"], "symmetric_eigenvalue_clamp");
        assert_eq!(
            cov.1["psd_projection_blocker"],
            "symmetric_eigen_decomposition_and_dense_reconstruction"
        );
        assert_eq!(cov.1["estimated_eigen_decompositions"], 2);
        assert_eq!(cov.1["estimated_eigen_ops"], 2 * 4 * 4 * 4);
        assert_eq!(cov.1["estimated_low_rank_reconstruction_ops"], 2 * 4 * 4);
        assert_eq!(cov.1["estimated_psd_reconstruction_ops"], 4 * 4 * 4);
        assert_eq!(cov.1["output_rows"], 4);
        assert!(cov.1["condition_number"]
            .as_f64()
            .unwrap_or(0.0)
            .is_finite());

        let reductions = events
            .iter()
            .filter(|(op_name, data)| *op_name == "sum_axis0_scaled" && data["cols"] == 4)
            .count();
        assert!(reductions >= 2);
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "matmul_lhs_transpose_scaled" && data["rows"] == 4 && data["cols"] == 4
        }));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn covariance_head_forced_wgpu_routes_sample_covariance() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mu =
            Tensor::from_vec(3, 3, vec![0.1, 0.2, 0.3, 0.0, -0.1, 0.2, -0.2, 0.1, 0.0]).unwrap();
        let sigma = Tensor::from_vec(
            3,
            3,
            vec![0.05, 0.02, 0.03, 0.04, 0.03, 0.02, 0.06, 0.05, 0.04],
        )
        .unwrap();
        {
            let _guard = push_backend_policy(policy);
            let _ = CovHead::new(2).forward(&mu, &sigma).unwrap();
        }
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "matmul_lhs_transpose_scaled" && data["backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "zrba_cov_head_forward"
                && data["requested_backend"] == "wgpu"
                && data["covariance_centering_backend"] == "cpu"
                && data["covariance_accumulation_backend"] == "wgpu"
                && data["psd_projection_backend"] == "cpu_eigen"
        }));
    }
}
