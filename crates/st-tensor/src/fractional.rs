// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-tensor/src/fractional.rs

use crate::pure::{PureResult, TensorError};
use st_frac::{FracErr, Pad};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PadMode {
    /// Extend the signal with a fixed exterior value.
    Constant(f32),
    /// Extend the signal with its nearest edge sample.
    Edge,
}

impl From<PadMode> for Pad {
    fn from(value: PadMode) -> Self {
        match value {
            PadMode::Constant(constant) => Pad::Constant(constant),
            PadMode::Edge => Pad::Edge,
        }
    }
}

fn map_frac_error(error: FracErr) -> TensorError {
    match error {
        FracErr::Kernel => TensorError::InvalidDimensions { rows: 0, cols: 0 },
        FracErr::Alpha { .. } => TensorError::InvalidValue {
            label: "fractional alpha must be finite and positive",
        },
        FracErr::Step { .. } => TensorError::InvalidValue {
            label: "fractional step h must be finite and positive",
        },
        FracErr::NonFiniteParameter { label, value }
        | FracErr::NonFiniteSample { label, value, .. } => {
            TensorError::NonFiniteValue { label, value }
        }
        FracErr::NumericOverflow { label, value } => TensorError::NonFiniteValue {
            label,
            value: if value.is_sign_negative() {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            },
        },
        FracErr::Axis => TensorError::InvalidValue {
            label: "fractional axis out of range",
        },
        FracErr::Tol => TensorError::InvalidValue {
            label: "fractional tolerance must be finite and positive",
        },
        FracErr::Capacity { .. } => TensorError::InvalidValue {
            label: "fractional buffer capacity exceeds the addressable range",
        },
    }
}

pub(crate) fn derivative_contract(
    values: &[f32],
    alpha: f32,
    h: f32,
) -> PureResult<st_frac::GlStepScale> {
    if let Some(&value) = values.iter().find(|value| !value.is_finite()) {
        return Err(TensorError::NonFiniteValue {
            label: "fractional derivative input",
            value,
        });
    }
    st_frac::gl_step_scale(alpha, h).map_err(map_frac_error)
}

#[cfg(feature = "wgpu_frac")]
pub(crate) fn validate_fractional_output(values: &[f32]) -> PureResult<()> {
    if let Some(&value) = values.iter().find(|value| !value.is_finite()) {
        return Err(TensorError::NonFiniteValue {
            label: "fractional derivative output",
            value,
        });
    }
    Ok(())
}

/// Grünwald–Letnikov の 1D fractional difference（軽量CPU実装）
/// `y[n] = Σ_{k=0..K-1} c_k · x[n-k], c_k = (-1)^k · C(α, k)`
/// C(α,k) は一般化二項係数。K は kernel_len で打ち切り。
pub fn fracdiff_gl_1d(
    xs: &[f32],
    alpha: f32,
    kernel_len: usize,
    pad: PadMode,
) -> PureResult<Vec<f32>> {
    st_frac::fracdiff_gl_1d(xs, alpha, kernel_len, pad.into(), None).map_err(map_frac_error)
}

/// Grünwald–Letnikov coefficients `w[k]`.
pub fn gl_coeffs(alpha: f32, m: usize) -> PureResult<Vec<f32>> {
    let len = m.checked_add(1).ok_or(TensorError::InvalidValue {
        label: "gl_coeffs length overflow",
    })?;
    st_frac::gl_coeffs(alpha, len).map_err(map_frac_error)
}

/// Left-sided GL 1D derivative (forward form) with zero-extension boundaries.
/// `y[i] ≈ (1/h^α) * Σ_{k=0..min(i,m)} w[k]*x[i-k]`
pub fn fracdiff1d_gl(x: &[f32], alpha: f32, h: f32, m: usize) -> PureResult<Vec<f32>> {
    let n = x.len();
    let scale = derivative_contract(x, alpha, h)?;
    let m = m.min(n.saturating_sub(1));
    st_frac::fracdiff_gl_1d(x, alpha, m + 1, Pad::Zero, Some(scale.inverse_h_alpha()))
        .map_err(map_frac_error)
}

/// Backward pass (VJP): `∂L/∂x[j] += Σ_k w[k] * ∂L/∂y[j+k] / h^α`.
pub fn fracdiff1d_gl_vjp(gy: &[f32], alpha: f32, h: f32, m: usize) -> PureResult<Vec<f32>> {
    let n = gy.len();
    let scale = derivative_contract(gy, alpha, h)?;
    let m = m.min(n.saturating_sub(1));
    st_frac::fracdiff_gl_1d_vjp(gy, alpha, m + 1, Pad::Zero, Some(scale.inverse_h_alpha()))
        .map_err(map_frac_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_fractional_surface_delegates_to_st_frac() {
        let input = [1.0, 2.0, 4.0, 8.0];
        let tensor = fracdiff_gl_1d(&input, 0.7, 4, PadMode::Edge).unwrap();
        let core = st_frac::fracdiff_gl_1d(&input, 0.7, 4, Pad::Edge, None).unwrap();
        assert_eq!(tensor, core);
        assert_eq!(
            gl_coeffs(0.7, 3).unwrap(),
            st_frac::gl_coeffs(0.7, 4).unwrap()
        );
    }

    #[test]
    fn derivative_contract_rejects_invalid_step_and_signal() {
        assert!(matches!(
            fracdiff1d_gl(&[1.0], 0.5, 0.0, 2),
            Err(TensorError::InvalidValue { .. })
        ));
        assert!(matches!(
            fracdiff1d_gl(&[f32::NAN], 0.5, 1.0, 2),
            Err(TensorError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            fracdiff1d_gl(&[], f32::NAN, 1.0, 2),
            Err(TensorError::InvalidValue { .. })
        ));
    }

    #[test]
    fn derivative_vjp_satisfies_the_adjoint_identity() {
        let input = [0.5, -1.0, 2.0, 0.25];
        let gradient = [1.5, 0.25, -0.75, 2.0];
        let output = fracdiff1d_gl(&input, 0.65, 0.25, 8).unwrap();
        let input_gradient = fracdiff1d_gl_vjp(&gradient, 0.65, 0.25, 8).unwrap();
        let lhs: f64 = output
            .iter()
            .zip(&gradient)
            .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
            .sum();
        let rhs: f64 = input
            .iter()
            .zip(&input_gradient)
            .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
            .sum();
        assert!((lhs - rhs).abs() < 2e-6, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    fn empty_signals_preserve_the_validated_zero_dimensional_operator() {
        assert_eq!(
            fracdiff_gl_1d(&[], 0.5, 3, PadMode::Constant(0.0)).unwrap(),
            Vec::<f32>::new()
        );
        assert_eq!(fracdiff1d_gl(&[], 0.5, 1.0, 3).unwrap(), Vec::<f32>::new());
        assert_eq!(
            fracdiff1d_gl_vjp(&[], 0.5, 1.0, 3).unwrap(),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn coefficient_capacity_overflow_is_an_error() {
        assert!(matches!(
            gl_coeffs(0.5, usize::MAX),
            Err(TensorError::InvalidValue { .. })
        ));
    }
}
