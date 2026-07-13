#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

mod cobol;
mod cosmology;
mod fractal_field;
mod reports;
mod scale_stack;
mod topos_control;
mod topos_policy;
mod topos_route;
mod training_projection;
mod zspace_fusion;

#[cfg(target_arch = "wasm32")]
mod canvas;
#[cfg(target_arch = "wasm32")]
mod cobol_bridge;
#[cfg(target_arch = "wasm32")]
mod fft;
#[cfg(target_arch = "wasm32")]
mod mellin;
#[cfg(target_arch = "wasm32")]
mod tuner;
#[cfg(target_arch = "wasm32")]
mod utils;

#[cfg(target_arch = "wasm32")]
pub use canvas::*;
pub use cobol::*;
#[cfg(target_arch = "wasm32")]
pub use cobol_bridge::*;
pub use cosmology::*;
#[cfg(target_arch = "wasm32")]
pub use fft::*;
pub use fractal_field::*;
#[cfg(target_arch = "wasm32")]
pub use mellin::*;
pub use reports::*;
pub use scale_stack::*;
pub use topos_control::*;
pub use topos_policy::*;
pub use topos_route::*;
pub use training_projection::*;
#[cfg(target_arch = "wasm32")]
pub use tuner::*;
pub use zspace_fusion::*;
