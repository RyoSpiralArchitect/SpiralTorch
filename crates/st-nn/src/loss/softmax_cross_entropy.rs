use super::Loss;
use crate::{PureResult, Tensor};
use st_tensor::pure::TensorError;

/// Row-wise softmax cross entropy for multi-class logits.
///
/// Targets may be one-hot rows, probability distributions, or sparse class ids
/// shaped as `(batch, 1)`. Sparse rows can optionally use an ignored sentinel
/// (for example `-1`) so padded language-model rows contribute no loss or
/// gradient. Optional label smoothing blends each active target row with a
/// uniform distribution over classes. Gradients are normalized by the active
/// row count, matching the mean row loss.
#[derive(Debug, Default, Clone, Copy)]
pub struct SoftmaxCrossEntropy {
    ignore_index: Option<i32>,
    label_smoothing: f32,
}

/// Top-1 diagnostics for sparse class-index targets.
///
/// `active_rows` counts rows that are not masked by `ignore_index`; ignored
/// rows do not contribute to `correct`, `accuracy`, `mean_loss`, or
/// `perplexity`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseClassificationMetrics {
    pub active_rows: usize,
    pub correct: usize,
    pub accuracy: f32,
    pub mean_loss: f32,
    pub perplexity: f32,
}

/// Before/after diagnostic deltas for sparse class-index metrics.
///
/// Positive `loss_delta` and `perplexity_delta` mean the value decreased after
/// training or fine-tuning. Positive `accuracy_delta` means top-1 accuracy
/// improved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseClassificationDelta {
    pub before: SparseClassificationMetrics,
    pub after: SparseClassificationMetrics,
    pub loss_delta: f32,
    pub accuracy_delta: f32,
    pub perplexity_delta: f32,
}

impl SparseClassificationMetrics {
    /// Builds aggregate sparse metrics from active-row counts and summed row loss.
    pub fn from_totals(active_rows: usize, correct: usize, total_loss: f32) -> PureResult<Self> {
        if correct > active_rows {
            return Err(TensorError::IoError {
                message: format!(
                    "sparse classification correct count {correct} exceeds active rows {active_rows}"
                ),
            });
        }
        if !total_loss.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "sparse_classification_total_loss",
                value: total_loss,
            });
        }
        let mean_loss = if active_rows == 0 {
            0.0
        } else {
            total_loss / active_rows as f32
        };
        let perplexity = mean_loss.exp();
        if !perplexity.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "sparse_classification_perplexity",
                value: perplexity,
            });
        }
        let accuracy = if active_rows == 0 {
            0.0
        } else {
            correct as f32 / active_rows as f32
        };
        Ok(Self {
            active_rows,
            correct,
            accuracy,
            mean_loss,
            perplexity,
        })
    }

    /// Compares these metrics against a later evaluation.
    pub fn delta_to(self, after: Self) -> SparseClassificationDelta {
        SparseClassificationDelta {
            before: self,
            after,
            loss_delta: self.mean_loss - after.mean_loss,
            accuracy_delta: after.accuracy - self.accuracy,
            perplexity_delta: self.perplexity - after.perplexity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetMode {
    Dense,
    SparseClassIndex,
}

impl SoftmaxCrossEntropy {
    /// Creates a new softmax cross entropy loss instance.
    pub fn new() -> Self {
        Self {
            ignore_index: None,
            label_smoothing: 0.0,
        }
    }

    /// Creates a sparse-target loss that ignores the provided sentinel class id.
    pub fn with_ignore_index(ignore_index: i32) -> Self {
        Self {
            ignore_index: Some(ignore_index),
            label_smoothing: 0.0,
        }
    }

    /// Creates a loss with label smoothing enabled.
    pub fn with_label_smoothing(label_smoothing: f32) -> PureResult<Self> {
        let mut loss = Self::new();
        loss.set_label_smoothing(label_smoothing)?;
        Ok(loss)
    }

    /// Creates a sparse-target loss with both an ignored sentinel and smoothing.
    pub fn with_ignore_index_and_label_smoothing(
        ignore_index: i32,
        label_smoothing: f32,
    ) -> PureResult<Self> {
        let mut loss = Self::with_ignore_index(ignore_index);
        loss.set_label_smoothing(label_smoothing)?;
        Ok(loss)
    }

    /// Returns the sparse target sentinel ignored by this loss, if configured.
    pub fn ignore_index(&self) -> Option<i32> {
        self.ignore_index
    }

    /// Sets or clears the sparse target sentinel ignored by this loss.
    pub fn set_ignore_index(&mut self, ignore_index: Option<i32>) {
        self.ignore_index = ignore_index;
    }

    /// Returns the amount of target mass blended into a uniform distribution.
    pub fn label_smoothing(&self) -> f32 {
        self.label_smoothing
    }

    /// Sets label smoothing. The value must be finite and within `[0, 1)`.
    pub fn set_label_smoothing(&mut self, label_smoothing: f32) -> PureResult<()> {
        if label_smoothing < 0.0 || label_smoothing >= 1.0 || !label_smoothing.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "softmax_cross_entropy_label_smoothing",
                value: label_smoothing,
            });
        }
        self.label_smoothing = label_smoothing;
        Ok(())
    }

    fn validate_shapes(
        prediction: &Tensor,
        target: &Tensor,
    ) -> PureResult<(usize, usize, TargetMode)> {
        let (rows, cols) = prediction.shape();
        let target_shape = target.shape();
        if target_shape == (rows, cols) {
            return Ok((rows, cols, TargetMode::Dense));
        }
        if target_shape == (rows, 1) {
            return Ok((rows, cols, TargetMode::SparseClassIndex));
        }
        Err(TensorError::ShapeMismatch {
            left: prediction.shape(),
            right: target_shape,
        })
    }

    fn validate_sparse_shapes(prediction: &Tensor, target: &Tensor) -> PureResult<(usize, usize)> {
        let (rows, cols) = prediction.shape();
        let target_shape = target.shape();
        if target_shape == (rows, 1) {
            return Ok((rows, cols));
        }
        Err(TensorError::ShapeMismatch {
            left: (rows, 1),
            right: target_shape,
        })
    }

    fn row_logsumexp(row: &[f32]) -> PureResult<(f32, f32)> {
        let mut max_logit = f32::NEG_INFINITY;
        for &value in row {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue {
                    label: "softmax_cross_entropy_logit",
                    value,
                });
            }
            max_logit = max_logit.max(value);
        }
        let mut exp_sum = 0.0f32;
        for &value in row {
            exp_sum += (value - max_logit).exp();
        }
        if exp_sum <= 0.0 || !exp_sum.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "softmax_cross_entropy_exp_sum",
                value: exp_sum,
            });
        }
        Ok((max_logit + exp_sum.ln(), exp_sum))
    }

    fn row_argmax(row: &[f32]) -> usize {
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &value) in row.iter().enumerate() {
            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }
        best_index
    }

    fn class_index(value: f32, classes: usize) -> PureResult<usize> {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "softmax_cross_entropy_class_index",
                value,
            });
        }
        let rounded = value.round();
        if (value - rounded).abs() > f32::EPSILON || rounded < 0.0 || rounded >= classes as f32 {
            return Err(TensorError::InvalidClassIndex {
                index: value,
                classes,
            });
        }
        Ok(rounded as usize)
    }

    fn sparse_class_index(&self, value: f32, classes: usize) -> PureResult<Option<usize>> {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "softmax_cross_entropy_class_index",
                value,
            });
        }
        let rounded = value.round();
        if let Some(ignore_index) = self.ignore_index {
            if (value - rounded).abs() <= f32::EPSILON && rounded == ignore_index as f32 {
                return Ok(None);
            }
        }
        Self::class_index(value, classes).map(Some)
    }

    fn sparse_row_loss(&self, logits: &[f32], class: usize, logsumexp: f32) -> f32 {
        let hard_loss = logsumexp - logits[class];
        if self.label_smoothing == 0.0 {
            hard_loss
        } else {
            let mean_logit = logits.iter().copied().sum::<f32>() / logits.len() as f32;
            let uniform_loss = logsumexp - mean_logit;
            hard_loss * (1.0 - self.label_smoothing) + uniform_loss * self.label_smoothing
        }
    }

    fn active_rows(&self, prediction: &Tensor, target: &Tensor) -> PureResult<usize> {
        let (rows, cols, mode) = Self::validate_shapes(prediction, target)?;
        match mode {
            TargetMode::Dense => Ok(rows),
            TargetMode::SparseClassIndex => {
                let mut active_rows = 0usize;
                for row in 0..rows {
                    if self.sparse_class_index(target.data()[row], cols)?.is_some() {
                        active_rows += 1;
                    }
                }
                Ok(active_rows)
            }
        }
    }

    fn smoothed_dense_target(&self, target_value: f32, target_sum: f32, classes: usize) -> f32 {
        let uniform = target_sum / classes as f32;
        target_value * (1.0 - self.label_smoothing) + uniform * self.label_smoothing
    }

    /// Computes active-row top-1 accuracy, mean loss, and perplexity for sparse targets.
    ///
    /// This is intentionally sparse-only: dense probability targets do not have
    /// a single ground-truth class id, while tokenizerless/LM contracts do.
    /// Ignored rows are skipped before logit validation, matching `forward()`
    /// and `backward()` behavior for padded rows.
    pub fn sparse_metrics(
        &self,
        prediction: &Tensor,
        target: &Tensor,
    ) -> PureResult<SparseClassificationMetrics> {
        let (rows, cols) = Self::validate_sparse_shapes(prediction, target)?;
        let mut active_rows = 0usize;
        let mut correct = 0usize;
        let mut total_loss = 0.0f32;
        for row in 0..rows {
            let Some(class) = self.sparse_class_index(target.data()[row], cols)? else {
                continue;
            };
            let start = row * cols;
            let end = start + cols;
            let logits = &prediction.data()[start..end];
            let (logsumexp, _) = Self::row_logsumexp(logits)?;
            total_loss += self.sparse_row_loss(logits, class, logsumexp);
            if Self::row_argmax(logits) == class {
                correct += 1;
            }
            active_rows += 1;
        }
        SparseClassificationMetrics::from_totals(active_rows, correct, total_loss)
    }
}

impl Loss for SoftmaxCrossEntropy {
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        let (rows, cols, mode) = Self::validate_shapes(prediction, target)?;
        let mut total = 0.0f32;
        let mut active_rows = 0usize;
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            if mode == TargetMode::SparseClassIndex
                && self.sparse_class_index(target.data()[row], cols)?.is_none()
            {
                continue;
            }
            let logits = &prediction.data()[start..end];
            let (logsumexp, _) = Self::row_logsumexp(logits)?;
            match mode {
                TargetMode::Dense => {
                    let targets = &target.data()[start..end];
                    let mut target_sum = 0.0f32;
                    for &target_value in targets {
                        if !target_value.is_finite() {
                            return Err(TensorError::NonFiniteValue {
                                label: "softmax_cross_entropy_target",
                                value: target_value,
                            });
                        }
                        target_sum += target_value;
                    }
                    let mut row_loss = 0.0f32;
                    for (&logit, &target_value) in logits.iter().zip(targets.iter()) {
                        let smoothed_target =
                            self.smoothed_dense_target(target_value, target_sum, cols);
                        if smoothed_target != 0.0 {
                            row_loss += -smoothed_target * (logit - logsumexp);
                        }
                    }
                    total += row_loss;
                    active_rows += 1;
                }
                TargetMode::SparseClassIndex => {
                    if let Some(class) = self.sparse_class_index(target.data()[row], cols)? {
                        total += self.sparse_row_loss(logits, class, logsumexp);
                        active_rows += 1;
                    }
                }
            }
        }
        let mean = if active_rows == 0 {
            0.0
        } else {
            total / active_rows as f32
        };
        Tensor::from_vec(1, 1, vec![mean])
    }

    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        let (rows, cols, mode) = Self::validate_shapes(prediction, target)?;
        let mut data = vec![0.0f32; rows * cols];
        let inv_active_rows = match mode {
            TargetMode::Dense => 1.0f32 / rows as f32,
            TargetMode::SparseClassIndex => {
                let active_rows = self.active_rows(prediction, target)?;
                if active_rows == 0 {
                    return Tensor::from_vec(rows, cols, data);
                }
                1.0f32 / active_rows as f32
            }
        };
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            if mode == TargetMode::SparseClassIndex
                && self.sparse_class_index(target.data()[row], cols)?.is_none()
            {
                continue;
            }
            let logits = &prediction.data()[start..end];
            let (logsumexp, _) = Self::row_logsumexp(logits)?;
            match mode {
                TargetMode::Dense => {
                    let targets = &target.data()[start..end];
                    let mut target_sum = 0.0f32;
                    for &target_value in targets {
                        if !target_value.is_finite() {
                            return Err(TensorError::NonFiniteValue {
                                label: "softmax_cross_entropy_target",
                                value: target_value,
                            });
                        }
                        target_sum += target_value;
                    }
                    for col in 0..cols {
                        let idx = start + col;
                        let probability = (prediction.data()[idx] - logsumexp).exp();
                        let smoothed_target =
                            self.smoothed_dense_target(target.data()[idx], target_sum, cols);
                        data[idx] = (probability * target_sum - smoothed_target) * inv_active_rows;
                    }
                }
                TargetMode::SparseClassIndex => {
                    if let Some(class) = self.sparse_class_index(target.data()[row], cols)? {
                        let uniform_target = self.label_smoothing / cols as f32;
                        for col in 0..cols {
                            let idx = start + col;
                            let mut target_value = uniform_target;
                            if col == class {
                                target_value += 1.0 - self.label_smoothing;
                            }
                            data[idx] = ((prediction.data()[idx] - logsumexp).exp() - target_value)
                                * inv_active_rows;
                        }
                    }
                }
            }
        }
        Tensor::from_vec(rows, cols, data)
    }

    fn reduction_rows(&self, prediction: &Tensor, target: &Tensor) -> PureResult<usize> {
        self.active_rows(prediction, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_cross_entropy_forward_backward_one_hot() {
        let mut loss = SoftmaxCrossEntropy::new();
        let prediction = Tensor::from_vec(1, 3, vec![2.0, 0.0, -2.0]).unwrap();
        let target = Tensor::from_vec(1, 3, vec![1.0, 0.0, 0.0]).unwrap();
        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0] > 0.0);
        assert!(value.data()[0] < 0.2);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (1, 3));
        assert!(grad.data()[0] < 0.0);
        assert!(grad.data()[1] > 0.0);
        assert!(grad.data()[2] > 0.0);
        let sum = grad.data().iter().copied().sum::<f32>();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn softmax_cross_entropy_forward_backward_sparse_indices() {
        let mut loss = SoftmaxCrossEntropy::new();
        let prediction = Tensor::from_vec(2, 3, vec![2.0, 0.0, -2.0, -1.0, 3.0, 0.0]).unwrap();
        let target = Tensor::from_vec(2, 1, vec![0.0, 1.0]).unwrap();
        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0] > 0.0);
        assert!(value.data()[0] < 0.2);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (2, 3));
        assert!(grad.data()[0] < 0.0);
        assert!(grad.data()[4] < 0.0);
        let first_row_sum = grad.data()[0..3].iter().copied().sum::<f32>();
        let second_row_sum = grad.data()[3..6].iter().copied().sum::<f32>();
        assert!(first_row_sum.abs() < 1e-6);
        assert!(second_row_sum.abs() < 1e-6);
    }

    #[test]
    fn softmax_cross_entropy_label_smoothing_softens_sparse_targets() {
        let prediction = Tensor::from_vec(1, 2, vec![5.0, 0.0]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![0.0]).unwrap();

        let mut hard = SoftmaxCrossEntropy::new();
        let hard_value = hard.forward(&prediction, &target).unwrap();
        let hard_grad = hard.backward(&prediction, &target).unwrap();

        let mut smooth = SoftmaxCrossEntropy::with_label_smoothing(0.2).unwrap();
        assert_eq!(smooth.label_smoothing(), 0.2);
        let smooth_value = smooth.forward(&prediction, &target).unwrap();
        let smooth_grad = smooth.backward(&prediction, &target).unwrap();

        assert!(smooth_value.data()[0] > hard_value.data()[0]);
        assert!(smooth_grad.data()[0] > hard_grad.data()[0]);
        assert!(smooth_grad.data()[1] < hard_grad.data()[1]);
        let sum = smooth_grad.data().iter().copied().sum::<f32>();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn softmax_cross_entropy_label_smoothing_supports_dense_targets() {
        let prediction = Tensor::from_vec(1, 2, vec![4.0, -1.0]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();

        let mut hard = SoftmaxCrossEntropy::new();
        let hard_value = hard.forward(&prediction, &target).unwrap();

        let mut smooth = SoftmaxCrossEntropy::with_label_smoothing(0.1).unwrap();
        let smooth_value = smooth.forward(&prediction, &target).unwrap();
        let smooth_grad = smooth.backward(&prediction, &target).unwrap();

        assert!(smooth_value.data()[0] > hard_value.data()[0]);
        assert!(smooth_grad.data()[0] > 0.0);
        assert!(smooth_grad.data()[1] < 0.0);
        let sum = smooth_grad.data().iter().copied().sum::<f32>();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn softmax_cross_entropy_rejects_invalid_label_smoothing() {
        let err = SoftmaxCrossEntropy::with_label_smoothing(1.0).unwrap_err();
        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "softmax_cross_entropy_label_smoothing",
                ..
            }
        ));
    }

    #[test]
    fn softmax_cross_entropy_ignores_sparse_padding_rows() {
        let mut masked = SoftmaxCrossEntropy::with_ignore_index(-1);
        let prediction = Tensor::from_vec(2, 3, vec![2.0, 0.0, -2.0, 0.0, 4.0, -4.0]).unwrap();
        let target = Tensor::from_vec(2, 1, vec![0.0, -1.0]).unwrap();
        let value = masked.forward(&prediction, &target).unwrap();

        let mut single = SoftmaxCrossEntropy::new();
        let single_prediction = Tensor::from_vec(1, 3, vec![2.0, 0.0, -2.0]).unwrap();
        let single_target = Tensor::from_vec(1, 1, vec![0.0]).unwrap();
        let single_value = single.forward(&single_prediction, &single_target).unwrap();
        assert!((value.data()[0] - single_value.data()[0]).abs() < 1e-6);

        let grad = masked.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (2, 3));
        assert!(grad.data()[0] < 0.0);
        assert!(grad.data()[1] > 0.0);
        assert!(grad.data()[2] > 0.0);
        assert_eq!(&grad.data()[3..6], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn softmax_cross_entropy_all_ignored_sparse_rows_are_zero_loss() {
        let mut loss = SoftmaxCrossEntropy::with_ignore_index(-1);
        let prediction = Tensor::from_vec(2, 2, vec![0.0, 1.0, 2.0, -2.0]).unwrap();
        let target = Tensor::from_vec(2, 1, vec![-1.0, -1.0]).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert_eq!(value.data(), &[0.0]);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (2, 2));
        assert_eq!(grad.data(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn softmax_cross_entropy_skips_ignored_sparse_logits() {
        let mut loss = SoftmaxCrossEntropy::with_ignore_index(-1);
        let prediction = Tensor::from_vec(1, 2, vec![f32::NAN, f32::NAN]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![-1.0]).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert_eq!(value.data(), &[0.0]);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.data(), &[0.0, 0.0]);
    }

    #[test]
    fn softmax_cross_entropy_sparse_metrics_respect_ignore_index() {
        let loss = SoftmaxCrossEntropy::with_ignore_index(-1);
        let prediction =
            Tensor::from_vec(3, 3, vec![4.0, 1.0, 0.0, -2.0, 3.0, 0.0, 0.0, 0.0, 5.0]).unwrap();
        let target = Tensor::from_vec(3, 1, vec![0.0, -1.0, 1.0]).unwrap();
        let metrics = loss.sparse_metrics(&prediction, &target).unwrap();

        assert_eq!(metrics.active_rows, 2);
        assert_eq!(metrics.correct, 1);
        assert!((metrics.accuracy - 0.5).abs() < 1e-6);

        let mut forward_loss = loss;
        let value = forward_loss.forward(&prediction, &target).unwrap();
        assert!((metrics.mean_loss - value.data()[0]).abs() < 1e-6);
        assert!((metrics.perplexity - metrics.mean_loss.exp()).abs() < 1e-6);
    }

    #[test]
    fn softmax_cross_entropy_sparse_metrics_all_ignored_are_stable() {
        let loss = SoftmaxCrossEntropy::with_ignore_index(-1);
        let prediction = Tensor::from_vec(2, 2, vec![f32::NAN, f32::NAN, 4.0, -1.0]).unwrap();
        let target = Tensor::from_vec(2, 1, vec![-1.0, -1.0]).unwrap();
        let metrics = loss.sparse_metrics(&prediction, &target).unwrap();

        assert_eq!(metrics.active_rows, 0);
        assert_eq!(metrics.correct, 0);
        assert_eq!(metrics.accuracy, 0.0);
        assert_eq!(metrics.mean_loss, 0.0);
        assert_eq!(metrics.perplexity, 1.0);
    }

    #[test]
    fn softmax_cross_entropy_sparse_metrics_track_label_smoothing_loss() {
        let prediction = Tensor::from_vec(1, 2, vec![5.0, 0.0]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![0.0]).unwrap();

        let hard = SoftmaxCrossEntropy::new();
        let hard_metrics = hard.sparse_metrics(&prediction, &target).unwrap();

        let smooth = SoftmaxCrossEntropy::with_label_smoothing(0.2).unwrap();
        let smooth_metrics = smooth.sparse_metrics(&prediction, &target).unwrap();

        assert_eq!(hard_metrics.correct, 1);
        assert_eq!(smooth_metrics.correct, 1);
        assert_eq!(hard_metrics.accuracy, 1.0);
        assert_eq!(smooth_metrics.accuracy, 1.0);
        assert!(smooth_metrics.mean_loss > hard_metrics.mean_loss);
        assert!(smooth_metrics.perplexity > hard_metrics.perplexity);
    }

    #[test]
    fn sparse_classification_metrics_report_improvement_deltas() {
        let before = SparseClassificationMetrics {
            active_rows: 4,
            correct: 1,
            accuracy: 0.25,
            mean_loss: 2.0,
            perplexity: 7.0,
        };
        let after = SparseClassificationMetrics {
            active_rows: 4,
            correct: 3,
            accuracy: 0.75,
            mean_loss: 1.5,
            perplexity: 4.0,
        };

        let delta = before.delta_to(after);

        assert_eq!(delta.before, before);
        assert_eq!(delta.after, after);
        assert!((delta.loss_delta - 0.5).abs() < 1e-6);
        assert!((delta.accuracy_delta - 0.5).abs() < 1e-6);
        assert!((delta.perplexity_delta - 3.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_cross_entropy_sparse_metrics_reject_dense_targets() {
        let loss = SoftmaxCrossEntropy::new();
        let prediction = Tensor::from_vec(1, 3, vec![2.0, 0.0, -2.0]).unwrap();
        let target = Tensor::from_vec(1, 3, vec![1.0, 0.0, 0.0]).unwrap();
        let err = loss.sparse_metrics(&prediction, &target).unwrap_err();
        assert!(matches!(err, TensorError::ShapeMismatch { .. }));
    }

    #[test]
    fn softmax_cross_entropy_can_clear_ignore_index() {
        let mut loss = SoftmaxCrossEntropy::with_ignore_index(-1);
        assert_eq!(loss.ignore_index(), Some(-1));
        loss.set_ignore_index(None);
        assert_eq!(loss.ignore_index(), None);

        let prediction = Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![-1.0]).unwrap();
        let err = loss.forward(&prediction, &target).unwrap_err();
        assert!(matches!(
            err,
            TensorError::InvalidClassIndex {
                index: -1.0,
                classes: 2
            }
        ));
    }

    #[test]
    fn softmax_cross_entropy_rejects_non_finite_logits() {
        let mut loss = SoftmaxCrossEntropy::new();
        let prediction = Tensor::from_vec(1, 2, vec![f32::NAN, 0.0]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        let err = loss.forward(&prediction, &target).unwrap_err();
        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "softmax_cross_entropy_logit",
                ..
            }
        ));
    }

    #[test]
    fn softmax_cross_entropy_rejects_out_of_range_sparse_index() {
        let mut loss = SoftmaxCrossEntropy::new();
        let prediction = Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![2.0]).unwrap();
        let err = loss.forward(&prediction, &target).unwrap_err();
        assert!(matches!(
            err,
            TensorError::InvalidClassIndex {
                index: 2.0,
                classes: 2
            }
        ));
    }
}
