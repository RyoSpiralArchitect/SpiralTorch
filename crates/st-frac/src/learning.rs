//! Immutable learning snapshots of the existing causal GL operator.

use crate::{
    checked_f32, fracdiff_gl_nd_config, fracdiff_gl_nd_vjp_config, fracdiff_gl_nd_vjp_with_coeffs,
    fracdiff_gl_nd_with_coeffs, gl_coeffs_and_scaled_alpha_derivative, validate_alpha,
    validate_slice, zeroed_vec, FracErr, FracdiffGlConfig, Pad,
};
use ndarray::{ArrayD, IxDyn};

#[derive(Debug, thiserror::Error)]
pub enum FractionalLearningError {
    #[error("fractional learning needs nonempty rank 1..=16 shapes and a valid axis")]
    Shape,
    #[error("fractional learning value/product budget exceeded or invalid")]
    Budget,
    #[error("fractional history L2 gain must be finite and positive")]
    NormalizationGain,
    #[error("fractional history log-gain must be finite with positive finite f32 exp(log_gain)")]
    LogGain,
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
    // Normalized history cancels the positive h^-alpha multiplier analytically.
    input_scale: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct FractionalGlGradients {
    pub input: Vec<f32>,
    pub alpha: f32,
}

/// Normalized history shape with an independently learned positive amplitude.
/// The logarithmic amplitude differential is the captured output itself.
#[derive(Clone, Debug)]
pub struct FractionalGlGainLearningBatch {
    history: FractionalGlLearningBatch,
    gain: f32,
}

#[derive(Clone, Debug)]
pub struct FractionalGlGainGradients {
    pub input: Vec<f32>,
    pub alpha: f32,
    pub log_gain: f32,
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

    /// Strictly-past GL with coefficient L2 norm fixed to `gain`.
    ///
    /// For c = (0, c_1(alpha), ..., c_{K-1}(alpha)), use gain*c/||c||_2.
    /// The denominator covers the declared kernel, not just taps available at a
    /// prefix boundary. This fixes filter energy, not variance on correlated data.
    /// Positive sample-spacing scale cancels, including its alpha differential.
    /// After factoring out -alpha, the coefficient polynomial and its derivative
    /// must fit the f32 range. Recurrences and normalization accumulate in f64;
    /// this bounded operator does not clamp orders or normalize gradients.
    pub fn forward_history_l2(
        &self,
        input: &[f32],
        shape: &[usize],
        axis: usize,
        alpha: f32,
        gain: f32,
    ) -> Result<FractionalGlLearningBatch> {
        self.validate_shape_budget(input, shape, axis)?;
        validate_alpha(alpha)?;
        validate_slice("fractional input", input)?;
        if !gain.is_finite() || gain <= 0.0 {
            return Err(FractionalLearningError::NormalizationGain);
        }
        let config =
            FracdiffGlConfig::new(alpha, axis, self.kernel_len, Pad::Zero).with_step(self.step);
        if self.kernel_len == 1 || shape[axis] == 1 {
            return Ok(FractionalGlLearningBatch {
                config,
                output: ArrayD::zeros(IxDyn(shape)),
                alpha_derivative: ArrayD::zeros(IxDyn(shape)),
                history_coefficients: Some(Vec::new()),
                input_scale: Some(1.0),
            });
        }
        let (coefficients, derivatives) = history_l2_coefficients(alpha, self.kernel_len, gain)?;
        let (output, alpha_derivative) =
            paired_zero_forward(input, shape, axis, &coefficients, &derivatives, 1.0)?;
        Ok(FractionalGlLearningBatch {
            config,
            output,
            alpha_derivative,
            history_coefficients: Some(coefficients),
            input_scale: Some(1.0),
        })
    }

    /// `exp(log_gain) * c(alpha) / ||c(alpha)||_2`, with both scalar VJPs.
    /// No order/amplitude clipping or optimizer policy is applied. The positive
    /// amplitude must remain representable in f32, including on a zero map.
    pub fn forward_history_log_gain(
        &self,
        input: &[f32],
        shape: &[usize],
        axis: usize,
        alpha: f32,
        log_gain: f32,
    ) -> Result<FractionalGlGainLearningBatch> {
        let gain = log_gain.exp();
        if !log_gain.is_finite() || !gain.is_finite() || gain <= 0.0 {
            return Err(FractionalLearningError::LogGain);
        }
        Ok(FractionalGlGainLearningBatch {
            history: self.forward_history_l2(input, shape, axis, alpha, gain)?,
            gain,
        })
    }

    fn validate_shape_budget(&self, input: &[f32], shape: &[usize], axis: usize) -> Result<()> {
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
        Ok(())
    }

    fn forward_part(
        &self,
        input: &[f32],
        shape: &[usize],
        axis: usize,
        alpha: f32,
        history_only: bool,
    ) -> Result<FractionalGlLearningBatch> {
        self.validate_shape_budget(input, shape, axis)?;
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
                input_scale: None,
            });
        }
        let (mut coefficients, mut derivatives, scale) =
            gl_coeffs_and_scaled_alpha_derivative(config)?;
        if history_only {
            // Remove both zero-lag terms, including the sample-spacing derivative.
            coefficients[0] = 0.0;
            derivatives[0] = 0.0;
        }
        let (output, alpha_derivative) =
            paired_zero_forward(input, shape, axis, &coefficients, &derivatives, scale)?;
        Ok(FractionalGlLearningBatch {
            config,
            output,
            alpha_derivative,
            history_coefficients: history_only.then_some(coefficients),
            input_scale: None,
        })
    }
}

fn history_l2_coefficients(alpha: f32, len: usize, gain: f32) -> Result<(Vec<f32>, Vec<f32>)> {
    // c_k = -alpha*p_k for k >= 1. Cancel positive alpha analytically rather
    // than subtracting two O(1/alpha) derivative terms near zero.
    let mut polynomial = zeroed_vec::<f64>("history L2 polynomial", len)?;
    let mut differential = zeroed_vec::<f64>("history L2 differential", len)?;
    polynomial[1] = 1.0;
    for index in 2..len {
        let order = index as f64;
        let factor = (order - 1.0 - f64::from(alpha)) / order;
        polynomial[index] = factor * polynomial[index - 1];
        differential[index] = factor * differential[index - 1] - polynomial[index - 1] / order;
        checked_f32("history L2 polynomial", polynomial[index])?;
        checked_f32("history L2 differential", differential[index])?;
    }
    let norm_squared: f64 = polynomial.iter().map(|p| p * p).sum();
    let radial = polynomial
        .iter()
        .zip(&differential)
        .map(|(p, d)| p * d)
        .sum::<f64>()
        / norm_squared;
    let factor = -f64::from(gain) / norm_squared.sqrt();
    let mut coefficients = zeroed_vec("normalized history coefficient", len)?;
    let mut derivatives = zeroed_vec("normalized history alpha derivative", len)?;
    for index in 1..len {
        coefficients[index] =
            checked_f32("normalized history coefficient", factor * polynomial[index])?;
        derivatives[index] = checked_f32(
            "normalized history alpha derivative",
            factor * (differential[index] - polynomial[index] * radial),
        )?;
    }
    Ok((coefficients, derivatives))
}

// Learning inputs are checked, C-order slices. Share reads across both maps,
// keeping each accumulator's original increasing-lag order and f64 arithmetic.
fn paired_zero_forward(
    input: &[f32],
    shape: &[usize],
    axis: usize,
    coefficients: &[f32],
    derivatives: &[f32],
    scale: f32,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    validate_slice("fractional input", input)?;
    let inner: usize = shape[axis + 1..].iter().product();
    let axis_len = shape[axis];
    let lane_block = inner * axis_len;
    let mut output = zeroed_vec("fractional learning output", input.len())?;
    let mut differential = zeroed_vec("fractional learning differential", input.len())?;
    let scale = f64::from(scale);
    for base in (0..input.len()).step_by(lane_block) {
        for time in 0..axis_len {
            let taps = coefficients.len().min(time + 1);
            let destination = base + time * inner;
            if inner == 1 {
                let mut value = 0.0f64;
                let mut derivative = 0.0f64;
                for lag in 0..taps {
                    let sample = f64::from(input[destination - lag]);
                    value += f64::from(coefficients[lag]) * sample;
                    derivative += f64::from(derivatives[lag]) * sample;
                }
                output[destination] = checked_f32("fractional output", scale * value)?;
                differential[destination] = checked_f32("fractional output", derivative)?;
                continue;
            }
            const TILE: usize = 64;
            for first in (0..inner).step_by(TILE) {
                let width = TILE.min(inner - first);
                let mut values = [0.0f64; TILE];
                let mut differentials = [0.0f64; TILE];
                for lag in 0..taps {
                    let source = destination - lag * inner + first;
                    let coefficient = f64::from(coefficients[lag]);
                    let derivative = f64::from(derivatives[lag]);
                    for ((value, differential), &sample) in values[..width]
                        .iter_mut()
                        .zip(&mut differentials[..width])
                        .zip(&input[source..source + width])
                    {
                        let sample = f64::from(sample);
                        *value += coefficient * sample;
                        *differential += derivative * sample;
                    }
                }
                for index in 0..width {
                    let destination = destination + first + index;
                    output[destination] = checked_f32("fractional output", scale * values[index])?;
                    differential[destination] =
                        checked_f32("fractional output", differentials[index])?;
                }
            }
        }
    }
    let shaped = |values| {
        ArrayD::from_shape_vec(IxDyn(shape), values).map_err(|_| FractionalLearningError::Shape)
    };
    Ok((shaped(output)?, shaped(differential)?))
}

impl FractionalGlLearningBatch {
    pub fn output(&self) -> &ArrayD<f32> {
        &self.output
    }

    fn input_multiplier(&self) -> Result<f32> {
        match self.input_scale {
            Some(scale) => Ok(scale),
            None => Ok(self.config.scale_multiplier()?),
        }
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
                Some(self.input_multiplier()?),
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

    fn input_pushforward(&self, input_tangent: &[f32]) -> Result<ArrayD<f32>> {
        let tangent = self.shaped(input_tangent)?;
        Ok(match &self.history_coefficients {
            Some(coefficients) if coefficients.is_empty() => ArrayD::zeros(self.output.raw_dim()),
            Some(coefficients) => fracdiff_gl_nd_with_coeffs(
                &tangent,
                self.config.axis,
                coefficients,
                self.config.pad,
                Some(self.input_multiplier()?),
            )?,
            None => fracdiff_gl_nd_config(&tangent, self.config)?,
        })
    }

    pub fn jvp(&self, input_tangent: &[f32], alpha_tangent: f32) -> Result<Vec<f32>> {
        validate_slice("fractional alpha tangent", &[alpha_tangent])?;
        self.input_pushforward(input_tangent)?
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

impl FractionalGlGainLearningBatch {
    pub fn output(&self) -> &ArrayD<f32> {
        self.history.output()
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn vjp(&self, upstream: &[f32]) -> Result<FractionalGlGainGradients> {
        let gradient = self.history.vjp(upstream)?;
        Ok(FractionalGlGainGradients {
            input: gradient.input,
            alpha: gradient.alpha,
            log_gain: self.vjp_log_gain(upstream)?,
        })
    }

    pub fn vjp_input(&self, upstream: &[f32]) -> Result<Vec<f32>> {
        self.history.vjp_input(upstream)
    }

    pub fn vjp_alpha(&self, upstream: &[f32]) -> Result<f32> {
        self.history.vjp_alpha(upstream)
    }

    /// Only reduce the requested amplitude component; no input adjoint allocation.
    pub fn vjp_log_gain(&self, upstream: &[f32]) -> Result<f32> {
        self.history.validate_direction(upstream)?;
        let gradient = upstream
            .iter()
            .zip(self.output())
            .map(|(&g, &y)| f64::from(g) * f64::from(y))
            .sum();
        Ok(checked_f32("fractional log-gain gradient", gradient)?)
    }

    /// Shared scalar gradients without computing the unrequested input VJP.
    pub fn vjp_parameters(&self, upstream: &[f32]) -> Result<(f32, f32)> {
        self.history.validate_direction(upstream)?;
        let (mut alpha, mut log_gain) = (0.0f64, 0.0f64);
        for ((&g, &da), &y) in upstream
            .iter()
            .zip(&self.history.alpha_derivative)
            .zip(self.output())
        {
            alpha += f64::from(g) * f64::from(da);
            log_gain += f64::from(g) * f64::from(y);
        }
        Ok((
            checked_f32("fractional learning alpha gradient", alpha)?,
            checked_f32("fractional log-gain gradient", log_gain)?,
        ))
    }

    pub fn jvp(
        &self,
        input_tangent: &[f32],
        alpha_tangent: f32,
        log_gain_tangent: f32,
    ) -> Result<Vec<f32>> {
        validate_slice(
            "fractional scalar tangents",
            &[alpha_tangent, log_gain_tangent],
        )?;
        self.history
            .input_pushforward(input_tangent)?
            .iter()
            .zip(&self.history.alpha_derivative)
            .zip(self.output())
            .map(|((&dx, &da), &y)| {
                checked_f32(
                    "fractional gain learning tangent",
                    f64::from(dx)
                        + f64::from(alpha_tangent) * f64::from(da)
                        + f64::from(log_gain_tangent) * f64::from(y),
                )
                .map_err(Into::into)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        input: &[f32],
        shape: &[usize],
        axis: usize,
        alpha: f32,
        kernel_len: usize,
        step: f32,
        history: bool,
    ) -> FractionalGlLearningBatch {
        let config = FracdiffGlConfig::new(alpha, axis, kernel_len, Pad::Zero).with_step(step);
        let input = ArrayD::from_shape_vec(IxDyn(shape), input.to_vec()).unwrap();
        let (output, alpha_derivative, history_coefficients) = if history {
            let (mut coefficients, mut derivatives, scale) =
                gl_coeffs_and_scaled_alpha_derivative(config).unwrap();
            coefficients[0] = 0.;
            derivatives[0] = 0.;
            (
                fracdiff_gl_nd_with_coeffs(&input, axis, &coefficients, Pad::Zero, Some(scale))
                    .unwrap(),
                fracdiff_gl_nd_with_coeffs(&input, axis, &derivatives, Pad::Zero, None).unwrap(),
                Some(coefficients),
            )
        } else {
            (
                fracdiff_gl_nd_config(&input, config).unwrap(),
                crate::fracdiff_gl_nd_alpha_derivative_config(&input, config).unwrap(),
                None,
            )
        };
        FractionalGlLearningBatch {
            config,
            output,
            alpha_derivative,
            history_coefficients,
            input_scale: None,
        }
    }

    fn bits(values: impl IntoIterator<Item = f32>) -> Vec<u32> {
        values.into_iter().map(f32::to_bits).collect()
    }

    fn compare(
        shape: &[usize],
        axis: usize,
        alpha: f32,
        kernel_len: usize,
        step: f32,
        history: bool,
    ) {
        let len: usize = shape.iter().product();
        let input: Vec<f32> = (0..len)
            .map(|i| (i * 37 % 127) as f32 / 97. - 0.65)
            .collect();
        let direction: Vec<f32> = (0..len).map(|i| (i * 13 % 31) as f32 / 31. - 0.5).collect();
        let kernel = FractionalGlKernel::new(kernel_len, step, len, len * kernel_len).unwrap();
        let actual = if history {
            kernel.forward_history(&input, shape, axis, alpha)
        } else {
            kernel.forward(&input, shape, axis, alpha)
        }
        .unwrap();
        let reference = reference(&input, shape, axis, alpha, kernel_len, step, history);
        assert_eq!(
            bits(actual.output.iter().copied()),
            bits(reference.output.iter().copied())
        );
        assert_eq!(
            bits(actual.alpha_derivative.iter().copied()),
            bits(reference.alpha_derivative.iter().copied())
        );
        let a = actual.vjp(&direction).unwrap();
        let r = reference.vjp(&direction).unwrap();
        assert_eq!(bits(a.input), bits(r.input));
        assert_eq!(a.alpha.to_bits(), r.alpha.to_bits());
        assert_eq!(
            bits(actual.jvp(&direction, 0.3).unwrap()),
            bits(reference.jvp(&direction, 0.3).unwrap())
        );
    }

    #[test]
    fn paired_forward_is_bit_exact_across_axes_and_tile_boundaries() {
        for shape in [
            vec![9],
            vec![2, 5, 3],
            vec![2, 7, 63],
            vec![2, 7, 64],
            vec![2, 7, 65],
            vec![2, 3, 129],
            vec![1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1],
        ] {
            for axis in 0..shape.len() {
                for alpha in [0.01, 0.5, 1., 2., 2.1, 4.] {
                    for step in [0.7, 1., 1.4] {
                        for kernel_len in [1, 2, 5, 32] {
                            for history in [false, true] {
                                compare(&shape, axis, alpha, kernel_len, step, history);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn paired_forward_is_bit_exact_at_lm_training_shape() {
        for history in [false, true] {
            compare(&[2, 128, 768], 1, 0.9, 32, 1., history);
        }
    }

    #[test]
    fn paired_forward_preserves_signed_zero_and_subnormal_values() {
        let input: Vec<f32> = [
            0.,
            -0.,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
        ]
        .into_iter()
        .cycle()
        .take(130)
        .collect();
        for axis in 0..2 {
            for history in [false, true] {
                let kernel = FractionalGlKernel::new(8, 0.7, 130, 1040).unwrap();
                let actual = if history {
                    kernel.forward_history(&input, &[2, 65], axis, 1.)
                } else {
                    kernel.forward(&input, &[2, 65], axis, 1.)
                }
                .unwrap();
                let reference = reference(&input, &[2, 65], axis, 1., 8, 0.7, history);
                assert_eq!(
                    bits(actual.output.iter().copied()),
                    bits(reference.output.iter().copied())
                );
                assert_eq!(
                    bits(actual.alpha_derivative.iter().copied()),
                    bits(reference.alpha_derivative.iter().copied())
                );
            }
        }
    }

    fn near(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} (tolerance {tolerance})"
        );
    }

    #[test]
    fn history_l2_fixes_coefficient_norm_and_removes_radial_derivative() {
        for alpha in [f32::from_bits(1), 0.01, 0.5, 1., 2., 4.] {
            for len in [2, 3, 5, 32] {
                for gain in [0.3, 1., 5.] {
                    let (c, d) = history_l2_coefficients(alpha, len, gain).unwrap();
                    assert_eq!(c[0], 0.);
                    assert_eq!(d[0], 0.);
                    let norm = c.iter().map(|&c| f64::from(c).powi(2)).sum::<f64>().sqrt();
                    near(norm, f64::from(gain), f64::from(gain) * 1e-7);
                    let radial = c
                        .iter()
                        .zip(&d)
                        .map(|(&c, &d)| f64::from(c) * f64::from(d))
                        .sum();
                    near(radial, 0., f64::from(gain).powi(2) * 1e-7);
                }
            }
        }
        let (c, d) = history_l2_coefficients(2., 5, 1.).unwrap();
        assert_eq!(c[3], 0.);
        near(f64::from(d[3]), -1. / (3. * 5f64.sqrt()), 1e-8);
    }

    #[test]
    fn history_l2_one_past_tap_has_exactly_zero_order_gradient() {
        for alpha in [f32::from_bits(1), 1e-30, 0.01, 1., 4., f32::MAX] {
            let kernel = FractionalGlKernel::new(2, f32::MIN_POSITIVE, 4, 8).unwrap();
            let batch = kernel
                .forward_history_l2(&[1., 2., 3., 4.], &[4], 0, alpha, 0.5)
                .unwrap();
            assert_eq!(batch.output.as_slice().unwrap(), &[0., -0.5, -1., -1.5]);
            assert_eq!(batch.vjp_alpha(&[1.; 4]).unwrap(), 0.);
            assert_eq!(batch.jvp(&[0.; 4], 1.).unwrap(), [0.; 4]);
        }
    }

    #[test]
    fn history_l2_matches_finite_differences_and_adjoint_across_axes() {
        for shape in [vec![9], vec![2, 5, 3], vec![2, 5, 65]] {
            let n: usize = shape.iter().product();
            let x: Vec<f32> = (0..n).map(|i| (i % 13) as f32 / 11. - 0.4).collect();
            let dx: Vec<f32> = (0..n).map(|i| (i % 7) as f32 / 5. - 0.6).collect();
            let upstream: Vec<f32> = (0..n).map(|i| (i % 11) as f32 / 9. - 0.5).collect();
            let kernel = FractionalGlKernel::new(5, 0.7, n, n * 5).unwrap();
            for axis in 0..shape.len() {
                for alpha in [0.1, 1., 2., 4.] {
                    let batch = kernel
                        .forward_history_l2(&x, &shape, axis, alpha, 1.7)
                        .unwrap();
                    let grad = batch.vjp(&upstream).unwrap();
                    assert_eq!(grad.input, batch.vjp_input(&upstream).unwrap());
                    assert_eq!(grad.alpha, batch.vjp_alpha(&upstream).unwrap());
                    let tangent = batch.jvp(&dx, 0.3).unwrap();
                    let lhs: f64 = tangent
                        .iter()
                        .zip(&upstream)
                        .map(|(&d, &u)| f64::from(d) * f64::from(u))
                        .sum();
                    let rhs: f64 = grad
                        .input
                        .iter()
                        .zip(&dx)
                        .map(|(&g, &d)| f64::from(g) * f64::from(d))
                        .sum::<f64>()
                        + f64::from(grad.alpha) * 0.3;
                    near(lhs, rhs, 1e-5);
                    let eps = 0.001;
                    let plus: Vec<f32> = x.iter().zip(&dx).map(|(&x, &d)| x + eps * d).collect();
                    let minus: Vec<f32> = x.iter().zip(&dx).map(|(&x, &d)| x - eps * d).collect();
                    let p = kernel
                        .forward_history_l2(&plus, &shape, axis, alpha + eps * 0.3, 1.7)
                        .unwrap();
                    let m = kernel
                        .forward_history_l2(&minus, &shape, axis, alpha - eps * 0.3, 1.7)
                        .unwrap();
                    for ((&p, &m), &d) in p.output.iter().zip(m.output.iter()).zip(&tangent) {
                        near(f64::from((p - m) / (2. * eps)), f64::from(d), 3e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn history_l2_cancels_step_and_uses_whole_kernel_at_prefix_boundaries() {
        let mut baseline: Option<(Vec<u32>, Vec<u32>, FractionalGlGradients)> = None;
        for step in [f32::from_bits(1), 0.7, 1., 1e38] {
            let kernel = FractionalGlKernel::new(5, step, 8, 40).unwrap();
            let batch = kernel
                .forward_history_l2(&[1., 0., 0., 0., 0.], &[5], 0, 4., 1.)
                .unwrap();
            let short = kernel
                .forward_history_l2(&[1., 0.], &[2], 0, 4., 1.)
                .unwrap();
            assert_eq!(short.output[IxDyn(&[1])], batch.output[IxDyn(&[1])]);
            let result = (
                bits(batch.output.iter().copied()),
                bits(batch.jvp(&[0.2; 5], 0.3).unwrap()),
                batch.vjp(&[0.5; 5]).unwrap(),
            );
            if let Some((values, tangent, gradient)) = &baseline {
                assert_eq!(&result.0, values);
                assert_eq!(&result.1, tangent);
                assert_eq!(result.2.input, gradient.input);
                assert_eq!(result.2.alpha, gradient.alpha);
            } else {
                baseline = Some(result);
            }
        }
    }

    #[test]
    fn history_l2_has_no_current_future_or_cross_sample_dependency() {
        let kernel = FractionalGlKernel::new(5, 1., 6, 30).unwrap();
        let batch = kernel
            .forward_history_l2(&[1., 2., 3., 4., 5., 6.], &[2, 3], 1, 0.7, 1.)
            .unwrap();
        let changed = kernel
            .forward_history_l2(&[1., 80., 90., -1., -2., -3.], &[2, 3], 1, 0.7, 1.)
            .unwrap();
        assert_eq!(batch.output[IxDyn(&[0, 1])], changed.output[IxDyn(&[0, 1])]);
        let dx = batch.vjp_input(&[0., 1., 0., 0., 0., 0.]).unwrap();
        assert_ne!(dx[0], 0.);
        assert_eq!(&dx[1..], &[0.; 5]);
    }

    #[test]
    fn history_l2_empty_maps_still_validate_inputs_and_directions() {
        for (len, shape) in [(1, vec![2, 3]), (8, vec![2, 1])] {
            let n: usize = shape.iter().product();
            let kernel = FractionalGlKernel::new(len, 0.1, n, n * len).unwrap();
            let batch = kernel
                .forward_history_l2(&vec![1.; n], &shape, 1, f32::MAX, 1.)
                .unwrap();
            assert!(batch.output.iter().all(|&v| v == 0.));
            assert_eq!(batch.vjp_alpha(&vec![1.; n]).unwrap(), 0.);
            assert_eq!(batch.jvp(&vec![1.; n], 1.).unwrap(), vec![0.; n]);
            assert!(batch.vjp_input(&[1.]).is_err());
            assert!(batch.vjp_alpha(&vec![f32::NAN; n]).is_err());
            assert!(batch.jvp(&vec![0.; n], f32::INFINITY).is_err());
            assert!(kernel
                .forward_history_l2(&vec![f32::NAN; n], &shape, 1, 1., 1.)
                .is_err());
            for invalid in [0., -1., f32::NAN, f32::INFINITY] {
                assert!(kernel
                    .forward_history_l2(&vec![1.; n], &shape, 1, invalid, 1.)
                    .is_err());
                assert!(kernel
                    .forward_history_l2(&vec![1.; n], &shape, 1, 1., invalid)
                    .is_err());
            }
        }
    }

    #[test]
    fn history_l2_rejects_shapes_budgets_and_unrepresentable_arithmetic() {
        let kernel = FractionalGlKernel::new(5, 1., 3, 15).unwrap();
        assert!(kernel
            .forward_history_l2(&[1.; 3], &[3], 1, 1., 1.)
            .is_err());
        assert!(kernel
            .forward_history_l2(&[1.; 3], &[4], 0, 1., 1.)
            .is_err());
        assert!(kernel
            .forward_history_l2(&[1.; 4], &[4], 0, 1., 1.)
            .is_err());
        let limited = FractionalGlKernel::new(5, 1., 3, 5).unwrap();
        assert!(limited
            .forward_history_l2(&[1.; 2], &[2], 0, 1., 1.)
            .is_err());
        assert!(kernel
            .forward_history_l2(&[f32::MAX; 3], &[3], 0, 1., 2.)
            .is_err());
        assert!(kernel
            .forward_history_l2(&[1.; 3], &[3], 0, f32::MAX, 1.)
            .is_err());
    }

    #[test]
    fn log_gain_matches_fixed_energy_forward_and_existing_differentials() {
        let kernel = FractionalGlKernel::new(6, 0.2, 24, 144).unwrap();
        let x: Vec<_> = (0..24).map(|i| (i as f32 * 0.3).sin()).collect();
        for log_gain in [-1., 0., 0.8] {
            let learned = kernel
                .forward_history_log_gain(&x, &[2, 4, 3], 1, 2., log_gain)
                .unwrap();
            let fixed = kernel
                .forward_history_l2(&x, &[2, 4, 3], 1, 2., learned.gain())
                .unwrap();
            assert_eq!(
                bits(learned.output().iter().copied()),
                bits(fixed.output().iter().copied())
            );
            assert_eq!(
                bits(learned.vjp_input(&x).unwrap()),
                bits(fixed.vjp_input(&x).unwrap())
            );
            assert_eq!(
                learned.vjp_alpha(&x).unwrap().to_bits(),
                fixed.vjp_alpha(&x).unwrap().to_bits()
            );
            assert_eq!(
                bits(learned.jvp(&x, 0.3, 0.).unwrap()),
                bits(fixed.jvp(&x, 0.3).unwrap())
            );
            let joint = learned.vjp(&x).unwrap();
            assert_eq!(
                learned.vjp_parameters(&x).unwrap(),
                (joint.alpha, joint.log_gain)
            );
        }
    }

    #[test]
    fn log_gain_joint_jvp_matches_finite_differences_and_adjoint() {
        let shape = [2, 3, 4];
        let x: Vec<_> = (0..24).map(|i| (i as f32 * 0.31).sin()).collect();
        let dx: Vec<_> = x.iter().map(|x| x.cos()).collect();
        let u: Vec<_> = x.iter().map(|x| 0.2 - x * 0.3).collect();
        let kernel = FractionalGlKernel::new(5, 0.7, 24, 120).unwrap();
        let dot = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
        };
        for axis in 0..3 {
            for alpha in [0.1, 0.7, 1., 2., 3.2] {
                let h = kernel
                    .forward_history_log_gain(&x, &shape, axis, alpha, 0.3)
                    .unwrap();
                let vjp = h.vjp(&u).unwrap();
                let jvp = h.jvp(&dx, 0.2, -0.4).unwrap();
                let adjoint = dot(&u, &jvp) - dot(&vjp.input, &dx) - 0.2 * f64::from(vjp.alpha)
                    + 0.4 * f64::from(vjp.log_gain);
                assert!(adjoint.abs() < 2e-6, "{adjoint}");
                let eps = 0.001;
                let plus: Vec<_> = x.iter().zip(&dx).map(|(x, dx)| x + eps * dx).collect();
                let minus: Vec<_> = x.iter().zip(&dx).map(|(x, dx)| x - eps * dx).collect();
                let yp = kernel
                    .forward_history_log_gain(
                        &plus,
                        &shape,
                        axis,
                        alpha + eps * 0.2,
                        0.3 - eps * 0.4,
                    )
                    .unwrap();
                let ym = kernel
                    .forward_history_log_gain(
                        &minus,
                        &shape,
                        axis,
                        alpha - eps * 0.2,
                        0.3 + eps * 0.4,
                    )
                    .unwrap();
                for ((dy, yp), ym) in jvp.iter().zip(yp.output()).zip(ym.output()) {
                    assert!((dy - (yp - ym) / (2. * eps)).abs() < 3e-4);
                }
            }
        }
    }

    #[test]
    fn log_gain_selective_vjps_do_not_evaluate_unrequested_overflow() {
        let kernel = FractionalGlKernel::new(2, 1., 2, 4).unwrap();
        let h = kernel
            .forward_history_log_gain(&[f32::MAX, 0.], &[2], 0, 1., 0.)
            .unwrap();
        assert_eq!(h.vjp_input(&[0., 2.]).unwrap(), vec![-2., 0.]);
        assert_eq!(h.vjp_alpha(&[0., 2.]).unwrap(), 0.);
        assert!(h.vjp_log_gain(&[0., 2.]).is_err());
        assert!(h.vjp(&[0., 2.]).is_err());
        let h = kernel
            .forward_history_log_gain(&[0., 0.], &[2], 0, 1., 2f32.ln())
            .unwrap();
        assert!(h.vjp_input(&[0., f32::MAX]).is_err());
        assert_eq!(h.vjp_parameters(&[0., f32::MAX]).unwrap(), (0., 0.));
    }

    #[test]
    fn log_gain_empty_maps_still_validate_gain_and_directions() {
        for (len, shape) in [(1, [2, 3]), (8, [6, 1])] {
            let kernel = FractionalGlKernel::new(len, 1e-30, 6, 48).unwrap();
            let h = kernel
                .forward_history_log_gain(&[1.; 6], &shape, 1, f32::MAX, 0.)
                .unwrap();
            assert_eq!(h.vjp_parameters(&[1.; 6]).unwrap(), (0., 0.));
            assert_eq!(h.jvp(&[1.; 6], 1., 1.).unwrap(), vec![0.; 6]);
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 100., -200.] {
                assert!(kernel
                    .forward_history_log_gain(&[1.; 6], &shape, 1, 1., bad)
                    .is_err());
            }
            assert!(h.vjp_log_gain(&[1.]).is_err());
            assert!(h.vjp_parameters(&[f32::NAN; 6]).is_err());
            assert!(h.jvp(&[1.; 6], 0., f32::INFINITY).is_err());
        }
    }

    #[test]
    fn log_gain_is_owned_causal_and_sample_spacing_invariant() {
        let mut x = vec![1., 2., 3., 4., 5., 6.];
        let mut reference = None;
        for step in [1e-40, 1., 1e38] {
            let kernel = FractionalGlKernel::new(5, step, 6, 30).unwrap();
            let h = kernel
                .forward_history_log_gain(&x, &[2, 3], 1, 0.7, 0.3)
                .unwrap();
            let current = (h.output().clone(), h.vjp_parameters(&x).unwrap());
            if let Some(ref old) = reference {
                assert_eq!(&current, old);
            }
            reference = Some(current);
            let saved = h.output().clone();
            let old = x.clone();
            x[1..].fill(90.);
            let changed = kernel
                .forward_history_log_gain(&x, &[2, 3], 1, 0.7, 0.3)
                .unwrap();
            assert_eq!(h.output(), &saved);
            assert_eq!(changed.output()[IxDyn(&[0, 0])], saved[IxDyn(&[0, 0])]);
            assert_eq!(changed.output()[IxDyn(&[0, 1])], saved[IxDyn(&[0, 1])]);
            x = old;
        }
    }
}
