// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-tensor/src/util.rs
#![cfg(any(feature = "wgpu", feature = "wgpu_frac"))]
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};
use wgpu::*;

fn wait_for_map(slice: &BufferSlice, device: &Device) {
    // 0 => pending, 1 => success, 2 => error
    let status = Arc::new(AtomicU8::new(0));
    let flag = Arc::clone(&status);
    slice.map_async(MapMode::Read, move |result| {
        let code = if result.is_ok() { 1 } else { 2 };
        flag.store(code, Ordering::SeqCst);
    });

    loop {
        match status.load(Ordering::SeqCst) {
            0 => {
                let _ = device.poll(Maintain::Wait);
            }
            1 => break,
            2 => panic!("buffer map failed"),
            _ => unreachable!("unexpected map_async completion flag"),
        }
    }
}

pub fn readback_f32(device: &Device, queue: &Queue, src: &Buffer, len: usize) -> Vec<f32> {
    let size_bytes = (len * std::mem::size_of::<f32>()) as u64;
    let rb = device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: size_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback-enc"),
    });
    enc.copy_buffer_to_buffer(src, 0, &rb, 0, size_bytes);
    queue.submit(Some(enc.finish()));

    let slice = rb.slice(..);
    wait_for_map(&slice, device);

    let data = slice.get_mapped_range();
    let mut out = vec![0.0f32; len];
    out.copy_from_slice(bytemuck::cast_slice(&data));
    drop(data);
    rb.unmap();
    out
}
