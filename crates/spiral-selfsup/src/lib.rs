//! Self-supervised learning objectives used throughout SpiralTorch.

pub mod contrastive;
pub mod dataset;
pub mod masked;
pub mod metrics;
pub mod trainer;

use st_tensor::TensorError;
use thiserror::Error;

/// Errors surfaced by the self-supervised objectives crate.
#[derive(Debug, Error, PartialEq)]
pub enum ObjectiveError {
    /// Raised when inputs have mismatched batch or feature dimensions.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// Raised when an invalid parameter (temperature, etc.) is provided.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, ObjectiveError>;

impl From<TensorError> for ObjectiveError {
    fn from(err: TensorError) -> Self {
        match err {
            TensorError::InvalidDimensions { rows, cols } => {
                ObjectiveError::Shape(format!("invalid tensor dimensions: {rows}x{cols}"))
            }
            TensorError::DataLength { expected, got } => ObjectiveError::Shape(format!(
                "data length mismatch: expected {expected}, got {got}"
            )),
            TensorError::ShapeMismatch { left, right } => {
                ObjectiveError::Shape(format!("shape mismatch: left={left:?}, right={right:?}"))
            }
            other => ObjectiveError::InvalidArgument(other.to_string()),
        }
    }
}
