//! SpiralTorch-rs v1.0.1 (einsum DP-general + segment_* + logprod)

mod error;
mod dtype;
mod device;
mod tensor;
mod autograd;
pub mod ops;

pub use crate::dtype::DType;
pub use crate::device::Device;
pub use crate::error::{SpiralError, Result};
pub use crate::tensor::Tensor;
