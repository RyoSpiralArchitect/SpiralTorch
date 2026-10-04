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
