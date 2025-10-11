
pub mod error;
pub mod dtype;
pub mod device;
pub mod tensor;
pub mod autograd;
pub mod backend;
pub mod ops;

#[cfg(feature="cuda")]
pub mod cuda_support {
    pub use crate::backend::*;
}

#[cfg(feature="wgpu")]
pub mod wgpu_support {
    pub use crate::backend::*;
}

#[cfg(feature="mps")]
pub mod mps_support {
    pub use crate::backend::*;
}
