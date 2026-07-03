mod hyperbolic_cross_entropy;
mod mean_squared_error;
mod softmax_cross_entropy;

use crate::{PureResult, Tensor};

pub use hyperbolic_cross_entropy::HyperbolicCrossEntropy;
pub use mean_squared_error::MeanSquaredError;
pub use softmax_cross_entropy::{
    SoftmaxCrossEntropy, SparseClassificationDelta, SparseClassificationMetrics,
};

/// Trait implemented by differentiable losses that operate directly on
/// SpiralTorch tensors.
pub trait Loss {
    /// Computes the loss value for the given predictions and targets.
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor>;

    /// Returns the gradient of the loss with respect to the predictions.
    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor>;

    /// Returns how many rows contributed to the current loss reduction.
    ///
    /// Most losses reduce over every prediction row. Losses with masks or
    /// ignored sparse targets can override this so trainer histories and
    /// validation-best selection weight only active rows.
    fn reduction_rows(&self, prediction: &Tensor, _target: &Tensor) -> PureResult<usize> {
        Ok(prediction.shape().0)
    }
}
