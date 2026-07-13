// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Stable CPU entry points for fractional GL operators.
//!
//! `st-frac` is the sole owner of coefficient, padding, numeric-validation,
//! and adjoint semantics. This module deliberately remains a thin `st-core`
//! facade so runtime users do not need to reconstruct those contracts.

use ndarray::ArrayD;

pub use st_frac::{FracErr, FracdiffGlConfig, GlScale, GlStepScale, Pad};

pub type FracResult<T> = Result<T, FracErr>;

/// Apply an unscaled GL fractional difference along `axis`.
pub fn fracdiff_gl_cpu(
    x: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
) -> FracResult<ArrayD<f32>> {
    fracdiff_gl_cpu_config(x, FracdiffGlConfig::new(alpha, axis, kernel_len, pad))
}

/// Apply a GL fractional difference with an explicit, alpha-independent scale.
pub fn fracdiff_gl_cpu_with_scale(
    x: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> FracResult<ArrayD<f32>> {
    let mut config = FracdiffGlConfig::new(alpha, axis, kernel_len, pad);
    if let Some(scale) = scale {
        config = config.with_fixed_scale(scale);
    }
    fracdiff_gl_cpu_config(x, config)
}

/// Apply a backend-independent GL operator contract on CPU.
pub fn fracdiff_gl_cpu_config(
    x: &ArrayD<f32>,
    config: FracdiffGlConfig,
) -> FracResult<ArrayD<f32>> {
    st_frac::fracdiff_gl_nd_config(x, config)
}

/// Apply the exact VJP of [`fracdiff_gl_cpu`].
pub fn fracdiff_gl_cpu_vjp(
    gy: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
) -> FracResult<ArrayD<f32>> {
    fracdiff_gl_cpu_vjp_config(gy, FracdiffGlConfig::new(alpha, axis, kernel_len, pad))
}

/// Apply the exact VJP with the same fixed scale used by the forward operator.
pub fn fracdiff_gl_cpu_vjp_with_scale(
    gy: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> FracResult<ArrayD<f32>> {
    let mut config = FracdiffGlConfig::new(alpha, axis, kernel_len, pad);
    if let Some(scale) = scale {
        config = config.with_fixed_scale(scale);
    }
    fracdiff_gl_cpu_vjp_config(gy, config)
}

/// Apply the exact input VJP for a backend-independent GL operator contract.
pub fn fracdiff_gl_cpu_vjp_config(
    gy: &ArrayD<f32>,
    config: FracdiffGlConfig,
) -> FracResult<ArrayD<f32>> {
    st_frac::fracdiff_gl_nd_vjp_config(gy, config)
}

/// Differentiate [`fracdiff_gl_cpu`] exactly with respect to `alpha`.
pub fn fracdiff_gl_cpu_alpha_derivative(
    x: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
) -> FracResult<ArrayD<f32>> {
    fracdiff_gl_cpu_alpha_derivative_config(x, FracdiffGlConfig::new(alpha, axis, kernel_len, pad))
}

/// Differentiate the GL operator with a fixed, alpha-independent scale.
pub fn fracdiff_gl_cpu_alpha_derivative_with_scale(
    x: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
    scale: Option<f32>,
) -> FracResult<ArrayD<f32>> {
    let mut config = FracdiffGlConfig::new(alpha, axis, kernel_len, pad);
    if let Some(scale) = scale {
        config = config.with_fixed_scale(scale);
    }
    fracdiff_gl_cpu_alpha_derivative_config(x, config)
}

/// Differentiate a backend-independent GL contract exactly with respect to `alpha`.
pub fn fracdiff_gl_cpu_alpha_derivative_config(
    x: &ArrayD<f32>,
    config: FracdiffGlConfig,
) -> FracResult<ArrayD<f32>> {
    st_frac::fracdiff_gl_nd_alpha_derivative_config(x, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn core_facade_preserves_the_shared_fractional_contract() {
        let input =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.25, -1.0, 2.0, 1.5, 0.5, -0.75]).unwrap();
        let gradient =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.25, -0.5, 0.75, -1.25, 2.0])
                .unwrap();

        assert_eq!(
            fracdiff_gl_cpu(&input, 0.65, 1, 5, Pad::Wrap).unwrap(),
            st_frac::fracdiff_gl_nd(&input, 0.65, 1, 5, Pad::Wrap, None).unwrap()
        );
        assert_eq!(
            fracdiff_gl_cpu_vjp(&gradient, 0.65, 1, 5, Pad::Wrap).unwrap(),
            st_frac::fracdiff_gl_nd_backward(&gradient, 0.65, 1, 5, Pad::Wrap, None).unwrap()
        );
        assert_eq!(
            fracdiff_gl_cpu_alpha_derivative(&input, 0.65, 1, 5, Pad::Wrap).unwrap(),
            st_frac::fracdiff_gl_nd_alpha_derivative(&input, 0.65, 1, 5, Pad::Wrap, None,).unwrap()
        );

        let config = FracdiffGlConfig::new(0.65, 1, 5, Pad::Wrap).with_step(0.25);
        assert_eq!(
            fracdiff_gl_cpu_config(&input, config).unwrap(),
            st_frac::fracdiff_gl_nd_config(&input, config).unwrap()
        );
        assert_eq!(
            fracdiff_gl_cpu_vjp_config(&gradient, config).unwrap(),
            st_frac::fracdiff_gl_nd_vjp_config(&gradient, config).unwrap()
        );
        assert_eq!(
            fracdiff_gl_cpu_alpha_derivative_config(&input, config).unwrap(),
            st_frac::fracdiff_gl_nd_alpha_derivative_config(&input, config).unwrap()
        );
    }

    #[test]
    fn core_facade_propagates_structured_contract_errors() {
        let input = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap();
        assert!(matches!(
            fracdiff_gl_cpu(&input, 0.5, 0, 3, Pad::Zero),
            Err(FracErr::NonFiniteSample { .. })
        ));
        assert_eq!(
            fracdiff_gl_cpu(&input, 0.5, 1, 3, Pad::Zero),
            Err(FracErr::Axis)
        );
    }
}
