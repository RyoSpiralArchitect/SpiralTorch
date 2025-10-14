mod hyperbolic_cross_entropy;
mod mean_squared_error;

use crate::{PureResult, Tensor};

pub use hyperbolic_cross_entropy::HyperbolicCrossEntropy;
pub use mean_squared_error::MeanSquaredError;

/// Trait implemented by differentiable losses that operate directly on
/// SpiralTorch tensors.
pub trait Loss {
    /// Computes the loss value for the given predictions and targets.
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor>;

    /// Returns the gradient of the loss with respect to the predictions.
    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor>;
}
