#![cfg_attr(feature = "simd", feature(portable_simd))]
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-tensor/src/lib.rs
pub mod fractional;

pub mod backend;
pub mod dlpack;
mod hardmax;
mod memory;
pub mod observability;

#[cfg(feature = "wgpu_frac")]
mod util;

pub use backend::faer_dense;

#[cfg(feature = "wgpu_frac")]
pub use backend::wgpu_frac;

#[cfg(feature = "wgpu_dense")]
pub use backend::wgpu_dense;

mod pure;

#[doc = "Re-exported for convenience."]
pub use pure::*;

pub use observability::{
    emit_tensor_op, emit_tensor_op_meta, set_tensor_op_meta_observer, set_tensor_op_observer,
    TensorOpEvent, TensorOpMetaEvent, TensorOpMetaObserver, TensorOpObserver,
};

#[doc = "Expose the hardmax backend selector."]
pub use hardmax::HardmaxBackend;

#[doc = "Re-exported for convenience."]
pub use pure::wasm_canvas;
