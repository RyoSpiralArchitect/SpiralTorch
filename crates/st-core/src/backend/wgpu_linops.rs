// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! WGPU DeviceOps minimal impl (v1.8.0)
#![allow(unused)]
use crate::ops::hypergrad_gpu::{DeviceBuf, DeviceOps};

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
use super::wgpu_rt;

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
pub struct WgpuOps {
    pub ctx: std::sync::Arc<wgpu_rt::WgpuCtx>,
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
impl WgpuOps {
    pub fn new(ctx: std::sync::Arc<wgpu_rt::WgpuCtx>) -> Self {
        wgpu_rt::install_ctx(ctx.clone());
        Self { ctx }
    }

    pub fn wrap_buffer(&self, buffer: std::sync::Arc<wgpu::Buffer>, len: usize) -> DeviceBuf {
        DeviceBuf::from_wgpu(len, buffer)
    }

    fn ensure_len(&self, label: &str, buf: &DeviceBuf, n: usize) -> Result<(), String> {
        if buf.len < n {
            Err(format!(
                "{label} buffer length {} smaller than required {n}",
                buf.len
            ))
        } else {
            Ok(())
        }
    }

    fn require_wgpu<'a>(&self, label: &str, buf: &'a DeviceBuf) -> Result<&'a wgpu::Buffer, String> {
        buf.as_wgpu()
            .ok_or_else(|| format!("{label} buffer is not backed by wgpu"))
    }
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
impl DeviceOps for WgpuOps {
    fn dot(&self, n: usize, x: &DeviceBuf, y: &DeviceBuf) -> Result<f32, String> {
        self.ensure_len("x", x, n)?;
        self.ensure_len("y", y, n)?;
        let n_u32 = u32::try_from(n).map_err(|_| "vector length exceeds u32 range".to_string())?;

        let buf_x = self.require_wgpu("x", x)?;
        let buf_y = self.require_wgpu("y", y)?;

        let partials = (n_u32 + 255) / 256;
        let scratch_elems = partials.max(1);
        let scratch = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("st.lin.dot.scratch"),
            size: (scratch_elems as u64) * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        wgpu_rt::dispatch_lin_dot(n_u32, buf_x, buf_y, &scratch)?;

        let readback = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("st.lin.dot.readback"),
            size: std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("st.lin.dot.copy"),
            });
        encoder.copy_buffer_to_buffer(
            &scratch,
            0,
            &readback,
            0,
            std::mem::size_of::<f32>() as u64,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
        self.ctx.device.poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..std::mem::size_of::<f32>() as u64);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);

        match receiver
            .recv()
            .map_err(|_| "failed to receive map completion".to_string())?
        {
            Ok(()) => {}
            Err(e) => return Err(format!("map_async failed: {e:?}")),
        }

        let value = {
            let data = slice.get_mapped_range();
            let mut bytes = [0u8; std::mem::size_of::<f32>()];
            bytes.copy_from_slice(&data[..std::mem::size_of::<f32>()]);
            f32::from_ne_bytes(bytes)
        };

        readback.unmap();
        Ok(value)
    }

    fn axpy(&self, n: usize, alpha: f32, x: &DeviceBuf, y: &DeviceBuf, out: &DeviceBuf) -> Result<(), String> {
        self.ensure_len("x", x, n)?;
        self.ensure_len("y", y, n)?;
        self.ensure_len("out", out, n)?;
        let n_u32 = u32::try_from(n).map_err(|_| "vector length exceeds u32 range".to_string())?;

        let buf_x = self.require_wgpu("x", x)?;
        let buf_y = self.require_wgpu("y", y)?;
        let buf_out = self.require_wgpu("out", out)?;

        wgpu_rt::dispatch_lin_axpy(n_u32, alpha, buf_x, buf_y, buf_out)
    }

    fn copy(&self, n: usize, src: &DeviceBuf, dst: &DeviceBuf) -> Result<(), String> {
        self.ensure_len("src", src, n)?;
        self.ensure_len("dst", dst, n)?;
        let n_u32 = u32::try_from(n).map_err(|_| "vector length exceeds u32 range".to_string())?;

        let buf_src = self.require_wgpu("src", src)?;
        let buf_dst = self.require_wgpu("dst", dst)?;

        wgpu_rt::dispatch_lin_copy(n_u32, buf_src, buf_dst)
    }
}
