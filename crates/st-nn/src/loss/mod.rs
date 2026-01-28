// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

mod categorical_cross_entropy;
mod contrastive_loss;
mod focal_loss;
mod hyperbolic_cross_entropy;
mod mean_squared_error;
mod triplet_loss;

use crate::{PureResult, Tensor};

pub use categorical_cross_entropy::CategoricalCrossEntropy;
pub use contrastive_loss::ContrastiveLoss;
pub use focal_loss::FocalLoss;
pub use hyperbolic_cross_entropy::HyperbolicCrossEntropy;
pub use mean_squared_error::MeanSquaredError;
pub use triplet_loss::TripletLoss;

/// Trait implemented by differentiable losses that operate directly on
/// SpiralTorch tensors.
pub trait Loss {
    /// Computes the loss value for the given predictions and targets.
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor>;

    /// Returns the gradient of the loss with respect to the predictions.
    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor>;
}
