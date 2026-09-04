// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-tensor/src/util.rs
#![cfg(feature = "wgpu_frac")]
use wgpu::{Buffer, Device, Queue};

pub fn readback_f32(
    device: &Device,
    queue: &Queue,
    src: &Buffer,
    len: usize,
) -> Result<Vec<f32>, String> {
    st_backend_wgpu::runtime::read_buffer(device, queue, src, len, "st.tensor.readback_f32")
        .map_err(|error| error.to_string())
}
