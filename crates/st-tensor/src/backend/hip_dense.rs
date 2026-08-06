// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "hip")]

use std::sync::atomic::{AtomicBool, Ordering};

use st_backend_hip as hip;

static HIP_READY: AtomicBool = AtomicBool::new(false);

pub(crate) fn ensure_runtime() -> Result<(), String> {
    if !hip::real_backend_compiled() {
        return Err(
            "HIP execution requires the 'hip-real' feature; the default backend is a CPU contract reference"
                .to_string(),
        );
    }
    if HIP_READY.load(Ordering::Relaxed) && hip::runtime().is_some() {
        return Ok(());
    }

    match hip::init() {
        Ok(_) => {
            HIP_READY.store(true, Ordering::Relaxed);
            Ok(())
        }
        Err(err) => Err(err.to_string()),
    }
}

pub fn is_available() -> bool {
    if !hip::real_backend_compiled() {
        return false;
    }
    if HIP_READY.load(Ordering::Relaxed) && hip::runtime().is_some() {
        return true;
    }

    hip::init()
        .map(|runtime| {
            HIP_READY.store(true, Ordering::Relaxed);
            runtime.device_count() > 0
        })
        .unwrap_or(false)
}

pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
    let volume = rows.saturating_mul(cols);
    volume >= 256 && inner >= 16
}

fn matmul_shape_supported(rows: usize, inner: usize, cols: usize) -> bool {
    if rows == 0 || inner == 0 || cols == 0 {
        return false;
    }
    if rows > i32::MAX as usize || inner > i32::MAX as usize || cols > i32::MAX as usize {
        return false;
    }
    [
        rows.checked_mul(inner),
        inner.checked_mul(cols),
        rows.checked_mul(cols),
    ]
    .into_iter()
    .all(|values| {
        values
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .is_some()
    })
}

/// Preflight the static rocBLAS shape contract and local HIP runtime.
pub fn supports_matmul(rows: usize, inner: usize, cols: usize) -> bool {
    matmul_shape_supported(rows, inner, cols) && is_available()
}

pub fn matmul_into(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), String> {
    ensure_runtime()?;
    validate_output_len(out, rows, cols)?;
    hip::gemm_f32(rows, cols, inner, lhs, rhs, out).map_err(|err| err.to_string())
}

pub fn matmul_scaled_into(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
    scale: f32,
) -> Result<(), String> {
    ensure_runtime()?;
    validate_output_len(out, rows, cols)?;
    hip::gemm_scaled_f32(rows, cols, inner, scale, lhs, rhs, out).map_err(|err| err.to_string())
}

pub fn matmul_lhs_transpose_scaled_into(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
    scale: f32,
) -> Result<(), String> {
    ensure_runtime()?;
    validate_output_len(out, rows, cols)?;
    hip::gemm_lhs_transpose_scaled_f32(rows, cols, inner, scale, lhs, rhs, out)
        .map_err(|err| err.to_string())
}

#[allow(clippy::too_many_arguments)]
fn matmul_bias_activation_into(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
    activation: hip::GemmActivation,
) -> Result<(), String> {
    ensure_runtime()?;
    validate_output_len(out, rows, cols)?;
    hip::gemm_bias_activation_f32(rows, cols, inner, activation, lhs, rhs, bias, residual, out)
        .map_err(|err| err.to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_bias_relu_into(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), String> {
    matmul_bias_activation_into(
        lhs,
        rhs,
        bias,
        None,
        out,
        rows,
        inner,
        cols,
        hip::GemmActivation::Relu,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_bias_gelu_into(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), String> {
    matmul_bias_activation_into(
        lhs,
        rhs,
        bias,
        None,
        out,
        rows,
        inner,
        cols,
        hip::GemmActivation::Gelu,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_bias_add_relu_into(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    residual: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), String> {
    matmul_bias_activation_into(
        lhs,
        rhs,
        bias,
        Some(residual),
        out,
        rows,
        inner,
        cols,
        hip::GemmActivation::Relu,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_bias_add_gelu_into(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    residual: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), String> {
    matmul_bias_activation_into(
        lhs,
        rhs,
        bias,
        Some(residual),
        out,
        rows,
        inner,
        cols,
        hip::GemmActivation::Gelu,
    )
}

fn validate_output_len(out: &[f32], rows: usize, cols: usize) -> Result<(), String> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| "matmul rows*cols overflow".to_string())?;
    if out.len() != expected {
        return Err(format!(
            "output buffer length {} does not match rows*cols={}",
            out.len(),
            expected
        ));
    }
    Ok(())
}

pub fn matmul(
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let output_len = rows
        .checked_mul(cols)
        .ok_or_else(|| "matmul rows*cols overflow".to_owned())?;
    let mut out = vec![0.0f32; output_len];
    matmul_into(lhs, rhs, &mut out, rows, inner, cols)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::matmul_shape_supported;

    #[test]
    fn matmul_preflight_rejects_invalid_rocblas_shapes_without_initializing_hip() {
        assert!(matmul_shape_supported(2, 3, 4));
        assert!(!matmul_shape_supported(0, 3, 4));
        assert!(!matmul_shape_supported(i32::MAX as usize + 1, 1, 1));
        assert!(!matmul_shape_supported(usize::MAX, 2, 2));
    }
}
