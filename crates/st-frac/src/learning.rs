//! Immutable learning snapshots of the existing causal GL operator.

use crate::{
    checked_f32, fracdiff_gl_nd_alpha_derivative_config, fracdiff_gl_nd_config,
    fracdiff_gl_nd_vjp_config, fracdiff_gl_nd_vjp_with_coeffs, fracdiff_gl_nd_with_coeffs,
    gl_coeffs_and_scaled_alpha_derivative, validate_alpha, validate_slice, FracErr,
    FracdiffGlConfig, Pad,
};
use ndarray::{ArrayD, IxDyn};

#[derive(Debug, thiserror::Error)]
pub enum FractionalLearningError {
    #[error("fractional learning needs nonempty rank 1..=16 shapes and a valid axis")]
    Shape,
    #[error("fractional learning value/product budget exceeded or invalid")]
    Budget,
    #[error(transparent)]
    Operator(#[from] FracErr),
}

type Result<T> = std::result::Result<T, FractionalLearningError>;

/// Zero-padded, backward-looking GL convolution with bounded host allocations.
/// `max_products` bounds `input.len() * kernel_len` for each convolution.
#[derive(Clone, Debug)]
pub struct FractionalGlKernel {
    kernel_len: usize,
    step: f32,
    max_values: usize,
    max_products: usize,
}

/// Captures the exact recipe and alpha differential, not live input/parameters.
#[derive(Clone, Debug)]
pub struct FractionalGlLearningBatch {
    config: FracdiffGlConfig,
    output: ArrayD<f32>,
    alpha_derivative: ArrayD<f32>,
    // Some(empty) is the zero map: no coefficient or step-scale evaluation is needed.
    history_coefficients: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct FractionalGlGradients {
    pub input: Vec<f32>,
    pub alpha: f32,
}

impl FractionalGlKernel {
    pub fn new(
        kernel_len: usize,
        step: f32,
        max_values: usize,
        max_products: usize,
    ) -> Result<Self> {
        if kernel_len == 0 {
            return Err(FracErr::Kernel.into());
        }
        if !step.is_finite() || step <= 0.0 {
            return Err(FracErr::Step { h: step }.into());
        }
        if max_values == 0 || max_products == 0 || kernel_len > max_products {
            return Err(FractionalLearningError::Budget);
        }
        Ok(Self {
            kernel_len,
            step,
            max_values,
            max_products,
        })
    }

    pub fn kernel_len(&self) -> usize {
        self.kernel_len
    }
    pub fn step(&self) -> f32 {
        self.step
    }
    pub fn max_values(&self) -> usize {
        self.max_values
    }
    pub fn max_products(&self) -> usize {
        self.max_products
    }

    pub fn forward(
        &self,
        input: &[f32],
        shape: &[usize],
        axis: usize,
        alpha: f32,
    ) -> Result<FractionalGlLearningBatch> {
        self.forward_part(input, shape, axis, alpha, false)
    }

    /// Strictly past contribution: omit GL's zero-lag tap before convolution.
    /// This is `GL(x) - h^-alpha*x` mathematically, without subtractive cancellation.
    /// A length-one kernel or selected axis returns zero with zero differentials.
    pub fn forward_history(
        &self,
        input: &[f32],
        shape: &[usize],
        axis: usize,
        alpha: f32,
    ) -> Result<FractionalGlLearningBatch> {
        self.forward_part(input, shape, axis, alpha, true)
    }

    fn forward_part(
        &self,
        input: &[f32],
        shape: &[usize],
        axis: usize,
        alpha: f32,
        history_only: bool,
    ) -> Result<FractionalGlLearningBatch> {
        if shape.is_empty()
            || shape.len() > 16
            || shape.contains(&0)
            || axis >= shape.len()
            || shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(input.len())
        {
            return Err(FractionalLearningError::Shape);
        }
        if input.len() > self.max_values
            || input
                .len()
                .checked_mul(self.kernel_len)
                .is_none_or(|n| n > self.max_products)
        {
            return Err(FractionalLearningError::Budget);
        }
        let config =
            FracdiffGlConfig::new(alpha, axis, self.kernel_len, Pad::Zero).with_step(self.step);
        if history_only && (self.kernel_len == 1 || shape[axis] == 1) {
            validate_alpha(alpha)?;
            validate_slice("fractional input", input)?;
            return Ok(FractionalGlLearningBatch {
                config,
                output: ArrayD::zeros(IxDyn(shape)),
                alpha_derivative: ArrayD::zeros(IxDyn(shape)),
                history_coefficients: Some(Vec::new()),
            });
        }
        let input = ArrayD::from_shape_vec(IxDyn(shape), input.to_vec())
            .map_err(|_| FractionalLearningError::Shape)?;
        let (output, alpha_derivative, history_coefficients) = if history_only {
            let (mut coefficients, mut derivatives, scale) =
                gl_coeffs_and_scaled_alpha_derivative(config)?;
            // Remove both zero-lag terms, including the sample-spacing derivative.
            coefficients[0] = 0.0;
            derivatives[0] = 0.0;
            (
                fracdiff_gl_nd_with_coeffs(&input, axis, &coefficients, Pad::Zero, Some(scale))?,
                fracdiff_gl_nd_with_coeffs(&input, axis, &derivatives, Pad::Zero, None)?,
                Some(coefficients),
            )
        } else {
            (
                fracdiff_gl_nd_config(&input, config)?,
                fracdiff_gl_nd_alpha_derivative_config(&input, config)?,
                None,
            )
        };
        Ok(FractionalGlLearningBatch {
            config,
            output,
            alpha_derivative,
            history_coefficients,
        })
    }
}

impl FractionalGlLearningBatch {
    pub fn output(&self) -> &ArrayD<f32> {
        &self.output
    }

    fn validate_direction(&self, values: &[f32]) -> Result<()> {
        if values.len() != self.output.len() {
            return Err(FractionalLearningError::Shape);
        }
        validate_slice("fractional learning direction", values)?;
        Ok(())
    }

    fn shaped(&self, values: &[f32]) -> Result<ArrayD<f32>> {
        self.validate_direction(values)?;
        ArrayD::from_shape_vec(self.output.raw_dim(), values.to_vec())
            .map_err(|_| FractionalLearningError::Shape)
    }

    fn input_pullback(&self, upstream: &ArrayD<f32>) -> Result<Vec<f32>> {
        let input = match &self.history_coefficients {
            Some(coefficients) if coefficients.is_empty() => ArrayD::zeros(self.output.raw_dim()),
            Some(coefficients) => fracdiff_gl_nd_vjp_with_coeffs(
                upstream,
                self.config.axis,
                coefficients,
                self.config.pad,
                Some(self.config.scale_multiplier()?),
            )?,
            None => fracdiff_gl_nd_vjp_config(upstream, self.config)?,
        };
        Ok(input.iter().copied().collect())
    }

    fn alpha_pullback(&self, upstream: &[f32]) -> Result<f32> {
        let alpha = upstream
            .iter()
            .zip(self.alpha_derivative.iter())
            .map(|(&g, &d)| f64::from(g) * f64::from(d))
            .sum();
        Ok(checked_f32("fractional learning alpha gradient", alpha)?)
    }

    /// True input/shared-alpha VJP; no optimizer policy or gradient normalization.
    pub fn vjp(&self, upstream: &[f32]) -> Result<FractionalGlGradients> {
        let shaped = self.shaped(upstream)?;
        Ok(FractionalGlGradients {
            input: self.input_pullback(&shaped)?,
            alpha: self.alpha_pullback(upstream)?,
        })
    }

    /// Input VJP only. Does not reduce or check the unrequested alpha gradient.
    pub fn vjp_input(&self, upstream: &[f32]) -> Result<Vec<f32>> {
        self.input_pullback(&self.shaped(upstream)?)
    }

    /// Shared-alpha VJP only: no input adjoint convolution or input-gradient allocation.
    /// The direction is still shape/finite checked; only this requested component
    /// must be representable as f32. Forward snapshots and joint VJP are unchanged.
    pub fn vjp_alpha(&self, upstream: &[f32]) -> Result<f32> {
        self.validate_direction(upstream)?;
        self.alpha_pullback(upstream)
    }

    pub fn jvp(&self, input_tangent: &[f32], alpha_tangent: f32) -> Result<Vec<f32>> {
        validate_slice("fractional alpha tangent", &[alpha_tangent])?;
        let tangent = self.shaped(input_tangent)?;
        let input = match &self.history_coefficients {
            Some(coefficients) if coefficients.is_empty() => ArrayD::zeros(self.output.raw_dim()),
            Some(coefficients) => fracdiff_gl_nd_with_coeffs(
                &tangent,
                self.config.axis,
                coefficients,
                self.config.pad,
                Some(self.config.scale_multiplier()?),
            )?,
            None => fracdiff_gl_nd_config(&tangent, self.config)?,
        };
        input
            .iter()
            .zip(self.alpha_derivative.iter())
            .map(|(&dx, &da)| {
                checked_f32(
                    "fractional learning tangent",
                    f64::from(dx) + f64::from(alpha_tangent) * f64::from(da),
                )
                .map_err(Into::into)
            })
            .collect()
    }
}
