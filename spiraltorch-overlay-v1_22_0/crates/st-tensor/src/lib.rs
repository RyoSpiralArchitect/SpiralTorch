// crates/st-tensor/src/lib.rs
pub mod fractional;

#[cfg(feature = "wgpu_frac")]
pub mod backend;

#[cfg(feature = "wgpu_frac")]
mod util;

#[cfg(feature = "wgpu_frac")]
pub use backend::*;
