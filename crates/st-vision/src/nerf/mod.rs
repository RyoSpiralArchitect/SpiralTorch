// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight NeRF field definitions built on top of the core SpiralTorch
//! neural network layers.

mod encoding;
mod field;

pub use encoding::PositionalEncoding;
pub use field::{FieldSampleLayout, NerfField, NerfFieldConfig};
