use std::ops::Range;

use super::{
    history_l2_coefficients, paired_zero_forward, FractionalGlGainLearningBatch,
    FractionalGlKernel, FractionalGlLearningBatch, FractionalLearningError, Result,
};
use crate::{validate_alpha, validate_slice, FracdiffGlConfig, Pad};
use ndarray::{ArrayD, IxDyn};

impl FractionalGlKernel {
    /// Half-open strictly-past lags. Empty windows are valid zero maps.
    pub fn validate_history_window(&self, lags: Range<usize>) -> Result<()> {
        if lags.start == 0 || lags.start > lags.end || lags.end > self.kernel_len {
            return Err(FractionalLearningError::HistoryWindow);
        }
        Ok(())
    }

    /// Select lags AFTER full declared-kernel normalization, without renormalizing.
    ///
    /// Both coefficients and their full-normalization alpha differentials are
    /// masked. Thus complementary windows decompose the original linear map
    /// and its first-order derivatives up to final f32 rounding. This is not
    /// equivalent to constructing a shorter kernel. Budgets still cover K.
    /// Empty/unobservable windows validate inputs/controls then return zero;
    /// they do not evaluate unused coefficient recurrences. No lag-zero tap,
    /// clipping, learned mask or optimizer policy is introduced.
    #[allow(clippy::too_many_arguments)] // Match the existing ND map plus one lag range.
    pub fn forward_history_log_gain_window(
        &self,
        input: &[f32],
        shape: &[usize],
        axis: usize,
        alpha: f32,
        log_gain: f32,
        lags: Range<usize>,
    ) -> Result<FractionalGlGainLearningBatch> {
        self.validate_history_window(lags.clone())?;
        if lags == (1..self.kernel_len) {
            return self.forward_history_log_gain(input, shape, axis, alpha, log_gain);
        }
        let gain = Self::gain_from_log_gain(log_gain)?;
        self.validate_shape_budget(input, shape, axis)?;
        validate_alpha(alpha)?;
        validate_slice("fractional input", input)?;
        let config =
            FracdiffGlConfig::new(alpha, axis, self.kernel_len, Pad::Zero).with_step(self.step);
        if lags.is_empty() || lags.start >= shape[axis] {
            return Ok(FractionalGlGainLearningBatch {
                history: FractionalGlLearningBatch {
                    config,
                    output: ArrayD::zeros(IxDyn(shape)),
                    alpha_derivative: ArrayD::zeros(IxDyn(shape)),
                    history_coefficients: Some(Vec::new()),
                    input_scale: Some(1.0),
                },
                gain,
            });
        }
        let (mut coefficients, mut derivatives) =
            history_l2_coefficients(alpha, self.kernel_len, gain)?;
        for values in [&mut coefficients, &mut derivatives] {
            values[..lags.start].fill(0.0);
            values[lags.end..].fill(0.0);
        }
        let (output, alpha_derivative) =
            paired_zero_forward(input, shape, axis, &coefficients, &derivatives, 1.0)?;
        Ok(FractionalGlGainLearningBatch {
            history: FractionalGlLearningBatch {
                config,
                output,
                alpha_derivative,
                history_coefficients: Some(coefficients),
                input_scale: Some(1.0),
            },
            gain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support_tests::{bits, dense_line, dense_vjp};
    use ndarray::Axis;

    fn dense_nd(
        input: &[f32],
        shape: &[usize],
        axis: usize,
        coeff: &[f32],
        adjoint: bool,
    ) -> Vec<f32> {
        let input = ArrayD::from_shape_vec(IxDyn(shape), input.to_vec()).unwrap();
        let mut output = ArrayD::<f32>::zeros(IxDyn(shape));
        for (source, mut destination) in input
            .lanes(Axis(axis))
            .into_iter()
            .zip(output.lanes_mut(Axis(axis)))
        {
            let source: Vec<_> = source.iter().copied().collect();
            let values = if adjoint {
                dense_vjp(&source, coeff, Pad::Zero, 1.)
            } else {
                dense_line(&source, coeff, Pad::Zero, 1.)
            };
            for (slot, value) in destination.iter_mut().zip(values) {
                *slot = value;
            }
        }
        output.into_raw_vec()
    }

    fn compare_dense(shape: &[usize], axis: usize, alpha: f32, lags: Range<usize>) {
        let len: usize = shape.iter().product();
        let input: Vec<_> = (0..len)
            .map(|i| match i % 13 {
                0 => -0.,
                1 => f32::from_bits(1),
                2 => -f32::from_bits(1),
                _ => (i * 37 % 127) as f32 / 97. - 0.65,
            })
            .collect();
        let direction: Vec<_> = (0..len).map(|i| (i * 13 % 31) as f32 / 31. - 0.5).collect();
        let kernel = FractionalGlKernel::new(32, 0.7, len, len * 32).unwrap();
        let actual = kernel
            .forward_history_log_gain_window(&input, shape, axis, alpha, 0.3, lags.clone())
            .unwrap();
        let (mut coeff, mut derivative) =
            history_l2_coefficients(alpha, 32, actual.gain()).unwrap();
        for values in [&mut coeff, &mut derivative] {
            values[..lags.start].fill(0.);
            values[lags.end..].fill(0.);
        }
        let expected = dense_nd(&input, shape, axis, &coeff, false);
        let differential = dense_nd(&input, shape, axis, &derivative, false);
        assert_eq!(
            bits(actual.output().iter().copied()),
            bits(expected.clone())
        );
        assert_eq!(
            bits(actual.history.alpha_derivative.iter().copied()),
            bits(differential.clone())
        );
        let gradient = actual.vjp(&direction).unwrap();
        let input_gradient = dense_nd(&direction, shape, axis, &coeff, true);
        assert_eq!(bits(gradient.input), bits(input_gradient.clone()));
        assert_eq!(
            bits(actual.vjp_input(&direction).unwrap()),
            bits(input_gradient)
        );
        let dot = |values: &[f32]| -> f32 {
            direction
                .iter()
                .zip(values)
                .map(|(&g, &v)| f64::from(g) * f64::from(v))
                .sum::<f64>() as f32
        };
        assert_eq!(gradient.alpha.to_bits(), dot(&differential).to_bits());
        assert_eq!(gradient.log_gain.to_bits(), dot(&expected).to_bits());
        let parameters = actual.vjp_parameters(&direction).unwrap();
        assert_eq!(parameters.0.to_bits(), gradient.alpha.to_bits());
        assert_eq!(parameters.1.to_bits(), gradient.log_gain.to_bits());
        let tangent: Vec<_> = dense_nd(&direction, shape, axis, &coeff, false)
            .into_iter()
            .zip(&differential)
            .zip(&expected)
            .map(|((dx, &da), &y)| {
                (f64::from(dx)
                    + f64::from(0.2f32) * f64::from(da)
                    + f64::from(-0.3f32) * f64::from(y)) as f32
            })
            .collect();
        assert_eq!(
            bits(actual.jvp(&direction, 0.2, -0.3).unwrap()),
            bits(tangent)
        );
    }

    #[test]
    fn sparse_windows_preserve_dense_bits_across_axes_tiles_and_integer_orders() {
        for shape in [
            vec![9],
            vec![2, 5, 3],
            vec![2, 7, 63],
            vec![2, 7, 64],
            vec![2, 7, 65],
        ] {
            for axis in 0..shape.len() {
                for alpha in [f32::from_bits(1), 0.08, 0.7, 1., 2., 2.6] {
                    for window in [1..3, 3..32, 5..7, 31..32, 32..32, 1..32] {
                        compare_dense(&shape, axis, alpha, window);
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_windows_preserve_dense_bits_at_lm_training_shape() {
        for (alpha, lags) in [(0.09, 1..3), (0.09, 1..32), (0.55, 3..32), (2., 3..32)] {
            compare_dense(&[2, 128, 768], 1, alpha, lags);
        }
    }
}
