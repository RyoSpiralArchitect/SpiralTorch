// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "wgpu")]

use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages, Device, MapMode, Queue};

/// Wrapper that keeps a device/queue pair alive for buffer uploads.
#[derive(Clone, Debug)]
pub struct WgpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl WgpuContext {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self { device, queue }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}

/// Create a storage buffer initialised with the provided `f32` slice.
pub fn upload_slice(device: &Device, label: &str, data: &[f32], usage: BufferUsages) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

/// Allocate an uninitialised storage buffer sized to `elements` `f32` values.
pub fn empty_buffer(device: &Device, label: &str, elements: usize, usage: BufferUsages) -> Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (elements * std::mem::size_of::<f32>()) as u64,
        usage,
        mapped_at_creation: false,
    })
}

/// Read back a storage buffer into host memory.
pub fn read_buffer(
    device: &Device,
    queue: &Queue,
    buffer: &Buffer,
    elements: usize,
) -> Result<Vec<f32>, String> {
    let size = (elements * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.tensor.wgpu.readback"),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu.readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    futures_lite::future::block_on(async {
        receiver
            .receive()
            .await
            .ok_or_else(|| "map_async was cancelled".to_string())?
            .map_err(|_| "failed to map WGPU buffer".to_string())
    })?;
    let data = slice.get_mapped_range();
    let mut output = vec![0.0f32; elements];
    output.copy_from_slice(bytemuck::cast_slice(&data));
    drop(data);
    staging.unmap();
    Ok(output)
}
