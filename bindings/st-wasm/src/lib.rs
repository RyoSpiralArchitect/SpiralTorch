#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

mod cobol;
mod concept_diffusion;
mod cosmology;
mod fractal_field;
mod generation_control;
mod reports;
mod scale_stack;
mod temperature_control;
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
pub use concept_diffusion::*;
pub use cosmology::*;
#[cfg(target_arch = "wasm32")]
pub use fft::*;
pub use fractal_field::*;
pub use generation_control::*;
#[cfg(target_arch = "wasm32")]
pub use mellin::*;
pub use reports::*;
pub use scale_stack::*;
pub use temperature_control::*;
pub use topos_control::*;
pub use topos_policy::*;
pub use topos_route::*;
pub use training_projection::*;
#[cfg(target_arch = "wasm32")]
pub use tuner::*;
pub use zspace_fusion::*;
