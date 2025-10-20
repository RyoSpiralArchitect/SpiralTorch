// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

pub mod faer_dense;

#[cfg(feature = "wgpu_frac")]
pub mod wgpu_frac;

#[cfg(feature = "wgpu")]
pub mod wgpu_dense;

#[cfg(feature = "wgpu")]
pub mod wgpu_conv;
