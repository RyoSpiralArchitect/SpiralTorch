//! Self-supervised learning objectives used throughout SpiralTorch.

pub mod contrastive;
pub mod masked;
pub mod metrics;

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
