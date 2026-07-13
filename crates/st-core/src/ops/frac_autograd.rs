// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Autograd contract for fractional GL operators.
//!
//! Input gradients use the exact padding-aware VJP owned by `st-frac`.
//! Alpha gradients are analytic by default; central differences remain
//! available as an explicit diagnostic method rather than a silent fallback.

use super::frac::{self, FracErr, FracdiffGlConfig, Pad};
use ndarray::ArrayD;

const DEFAULT_ABS_EPS: f32 = 1e-3;
const DEFAULT_REL_EPS: f32 = 1e-3;
const MIN_EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AlphaGradMethod {
    /// Exact derivative of the GL coefficient recurrence.
    #[default]
    Analytic,
    /// Central difference retained for diagnostics and cross-checks.
    CentralDiff {
        eps: Option<f32>,
        rel_eps: Option<f32>,
    },
}

impl AlphaGradMethod {
    pub fn auto() -> Self {
        Self::Analytic
    }

    pub fn with_abs_eps(eps: f32) -> Self {
        Self::CentralDiff {
            eps: Some(eps),
            rel_eps: None,
        }
    }

    pub fn with_eps_and_rel(eps: f32, rel_eps: f32) -> Self {
        Self::CentralDiff {
            eps: Some(eps),
            rel_eps: Some(rel_eps),
        }
    }

    fn finite_difference_step(self, alpha: f32) -> Result<f32, FracAutogradError> {
        let Self::CentralDiff { eps, rel_eps } = self else {
            return Err(FracAutogradError::MethodMismatch);
        };
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(FracErr::Alpha { alpha }.into());
        }

        let absolute = validate_epsilon("absolute epsilon", eps.unwrap_or(DEFAULT_ABS_EPS))?;
        let relative = validate_epsilon("relative epsilon", rel_eps.unwrap_or(DEFAULT_REL_EPS))?;
        let scaled = f64::from(alpha).abs().max(1.0) * f64::from(relative);
        let requested = f64::from(absolute).max(scaled).max(MIN_EPS);
        let step = requested.min(0.5 * f64::from(alpha)) as f32;
        let lower = alpha - step;
        let upper = alpha + step;

        if !step.is_finite()
            || step <= 0.0
            || !lower.is_finite()
            || lower <= 0.0
            || !upper.is_finite()
            || lower == alpha
            || upper == alpha
        {
            return Err(FracAutogradError::UnrepresentableFiniteDifference { alpha, step });
        }
        Ok(step)
    }
}

fn validate_epsilon(label: &'static str, value: f32) -> Result<f32, FracAutogradError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(FracAutogradError::InvalidEpsilon { label, value });
    }
    Ok(value)
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum FracAutogradError {
    #[error(transparent)]
    Fractional(#[from] FracErr),
    #[error("input shape {input:?} does not match output-gradient shape {output_gradient:?}")]
    ShapeMismatch {
        input: Vec<usize>,
        output_gradient: Vec<usize>,
    },
    #[error("{label} must be finite and positive, got {value}")]
    InvalidEpsilon { label: &'static str, value: f32 },
    #[error("cannot represent a central-difference step for alpha={alpha} (step={step})")]
    UnrepresentableFiniteDifference { alpha: f32, step: f32 },
    #[error("analytic mode does not have a finite-difference step")]
    MethodMismatch,
    #[error("alpha gradient cannot be represented as finite f32 (value={value})")]
    AlphaGradientOverflow { value: f64 },
}

#[derive(Debug, PartialEq)]
pub struct FracGrad {
    pub gx: ArrayD<f32>,
    pub galpha: f32,
}

/// Compute gradients with respect to the input and fractional order.
pub fn backward_with_alpha(
    x: &ArrayD<f32>,
    alpha: f32,
    axis: usize,
    kernel_len: usize,
    pad: Pad,
    gy: &ArrayD<f32>,
    method: AlphaGradMethod,
) -> Result<FracGrad, FracAutogradError> {
    backward_with_alpha_config(
        x,
        FracdiffGlConfig::new(alpha, axis, kernel_len, pad),
        gy,
        method,
    )
}

/// Compute gradients for a backend-independent fractional operator contract.
pub fn backward_with_alpha_config(
    x: &ArrayD<f32>,
    config: FracdiffGlConfig,
    gy: &ArrayD<f32>,
    method: AlphaGradMethod,
) -> Result<FracGrad, FracAutogradError> {
    if x.shape() != gy.shape() {
        return Err(FracAutogradError::ShapeMismatch {
            input: x.shape().to_vec(),
            output_gradient: gy.shape().to_vec(),
        });
    }

    let gx = frac::fracdiff_gl_cpu_vjp_config(gy, config)?;
    let galpha = match method {
        AlphaGradMethod::Analytic => {
            let derivative = frac::fracdiff_gl_cpu_alpha_derivative_config(x, config)?;
            checked_inner_product(gy, &derivative)?
        }
        AlphaGradMethod::CentralDiff { .. } => {
            let epsilon = method.finite_difference_step(config.alpha)?;
            let plus = frac::fracdiff_gl_cpu_config(
                x,
                FracdiffGlConfig {
                    alpha: config.alpha + epsilon,
                    ..config
                },
            )?;
            let minus = frac::fracdiff_gl_cpu_config(
                x,
                FracdiffGlConfig {
                    alpha: config.alpha - epsilon,
                    ..config
                },
            )?;
            checked_central_difference_inner_product(gy, &plus, &minus, epsilon)?
        }
    };

    Ok(FracGrad { gx, galpha })
}

fn checked_inner_product(lhs: &ArrayD<f32>, rhs: &ArrayD<f32>) -> Result<f32, FracAutogradError> {
    let value = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(&lhs, &rhs)| f64::from(lhs) * f64::from(rhs))
        .sum();
    checked_alpha_gradient(value)
}

fn checked_central_difference_inner_product(
    gradient: &ArrayD<f32>,
    plus: &ArrayD<f32>,
    minus: &ArrayD<f32>,
    epsilon: f32,
) -> Result<f32, FracAutogradError> {
    let inverse_width = 0.5 / f64::from(epsilon);
    let value = gradient
        .iter()
        .zip(plus.iter())
        .zip(minus.iter())
        .map(|((&gradient, &plus), &minus)| {
            f64::from(gradient) * (f64::from(plus) - f64::from(minus)) * inverse_width
        })
        .sum();
    checked_alpha_gradient(value)
}

fn checked_alpha_gradient(value: f64) -> Result<f32, FracAutogradError> {
    if !value.is_finite() || value > f64::from(f32::MAX) || value < f64::from(f32::MIN) {
        return Err(FracAutogradError::AlphaGradientOverflow { value });
    }
    Ok(value as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn array(shape: &[usize], values: Vec<f32>) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap()
    }

    #[test]
    fn auto_method_selects_the_analytic_contract() {
        assert_eq!(AlphaGradMethod::auto(), AlphaGradMethod::Analytic);
        assert_eq!(AlphaGradMethod::default(), AlphaGradMethod::Analytic);
    }

    #[test]
    fn analytic_alpha_gradient_matches_explicit_central_difference() {
        let input = array(&[2, 3], vec![0.5, -1.0, 2.0, 1.25, 0.25, -0.75]);
        let gradient = array(&[2, 3], vec![1.0, 0.5, -0.25, 2.0, -1.0, 0.75]);
        let pads = [
            Pad::Zero,
            Pad::Reflect,
            Pad::Edge,
            Pad::Wrap,
            Pad::Constant(1.5),
        ];

        for pad in pads {
            let analytic = backward_with_alpha(
                &input,
                0.63,
                1,
                5,
                pad,
                &gradient,
                AlphaGradMethod::Analytic,
            )
            .unwrap();
            let numeric = backward_with_alpha(
                &input,
                0.63,
                1,
                5,
                pad,
                &gradient,
                AlphaGradMethod::with_abs_eps(1e-3),
            )
            .unwrap();
            let tolerance = 4e-3 * analytic.galpha.abs().max(numeric.galpha.abs()).max(1.0);
            assert!(
                (analytic.galpha - numeric.galpha).abs() <= tolerance,
                "pad={pad:?}, analytic={}, numeric={}",
                analytic.galpha,
                numeric.galpha
            );
            assert_eq!(analytic.gx, numeric.gx);
        }
    }

    #[test]
    fn analytic_alpha_gradient_includes_step_scale_derivative() {
        let input = array(&[2, 3], vec![0.5, -1.0, 2.0, 1.25, 0.25, -0.75]);
        let gradient = array(&[2, 3], vec![1.0, 0.5, -0.25, 2.0, -1.0, 0.75]);
        let config = FracdiffGlConfig::new(0.63, 1, 5, Pad::Edge).with_step(0.25);

        let analytic =
            backward_with_alpha_config(&input, config, &gradient, AlphaGradMethod::Analytic)
                .unwrap();
        let numeric = backward_with_alpha_config(
            &input,
            config,
            &gradient,
            AlphaGradMethod::with_abs_eps(1e-3),
        )
        .unwrap();
        let tolerance = 5e-3 * analytic.galpha.abs().max(numeric.galpha.abs()).max(1.0);
        assert!(
            (analytic.galpha - numeric.galpha).abs() <= tolerance,
            "analytic={}, numeric={}",
            analytic.galpha,
            numeric.galpha
        );
        assert_eq!(analytic.gx, numeric.gx);
    }

    #[test]
    fn input_gradient_satisfies_the_affine_adjoint_identity() {
        let input = array(&[4], vec![0.25, -1.5, 2.0, 0.75]);
        let zero = array(&[4], vec![0.0; 4]);
        let gradient = array(&[4], vec![1.25, -0.5, 0.75, 2.0]);
        let forward = frac::fracdiff_gl_cpu(&input, 0.6, 0, 6, Pad::Constant(2.5)).unwrap();
        let baseline = frac::fracdiff_gl_cpu(&zero, 0.6, 0, 6, Pad::Constant(2.5)).unwrap();
        let backward = backward_with_alpha(
            &input,
            0.6,
            0,
            6,
            Pad::Constant(2.5),
            &gradient,
            AlphaGradMethod::Analytic,
        )
        .unwrap();

        let lhs: f64 = forward
            .iter()
            .zip(baseline.iter())
            .zip(gradient.iter())
            .map(|((&value, &offset), &gradient)| f64::from(value - offset) * f64::from(gradient))
            .sum();
        let rhs: f64 = input
            .iter()
            .zip(backward.gx.iter())
            .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
            .sum();
        assert!((lhs - rhs).abs() < 2e-6, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    fn backward_rejects_shape_mismatch_and_invalid_diagnostic_epsilon() {
        let input = array(&[2], vec![1.0, 2.0]);
        let wrong_gradient = array(&[1], vec![1.0]);
        assert!(matches!(
            backward_with_alpha(
                &input,
                0.5,
                0,
                3,
                Pad::Zero,
                &wrong_gradient,
                AlphaGradMethod::Analytic,
            ),
            Err(FracAutogradError::ShapeMismatch { .. })
        ));

        let gradient = array(&[2], vec![0.25, -0.5]);
        assert!(matches!(
            backward_with_alpha(
                &input,
                0.5,
                0,
                3,
                Pad::Zero,
                &gradient,
                AlphaGradMethod::with_abs_eps(0.0),
            ),
            Err(FracAutogradError::InvalidEpsilon { .. })
        ));
    }
}
