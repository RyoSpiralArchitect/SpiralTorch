// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Compatibility exports for the backend-owned WGPU runtime.

pub use st_backend_wgpu::runtime::{
    checked_byte_len, empty_buffer, read_buffer, upload_slice, WgpuContext, WgpuRuntimeError,
};
