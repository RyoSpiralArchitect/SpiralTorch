// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-tensor/src/lib.rs
pub mod fractional;

#[cfg(feature = "wgpu_frac")]
pub mod backend;

#[cfg(feature = "wgpu_frac")]
mod util;

#[cfg(feature = "wgpu_frac")]
pub use backend::*;

pub mod pure;
