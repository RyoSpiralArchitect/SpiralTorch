// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "wgpu")]

use crate::util::readback_f32;
use std::sync::{Arc, OnceLock};
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, BufferUsages, ComputePipeline, Device, Queue};

const IM2COL_WGSL: &str = include_str!("../wgpu_shaders/conv_im2col.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Im2colParams {
    dims0: [u32; 4],
    dims1: [u32; 4],
    dims2: [i32; 4],
    dims3: [u32; 4],
    dims4: [u32; 4],
}

struct ConvContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: ComputePipeline,
    bind_layout: BindGroupLayout,
}

impl ConvContext {
    fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
        })
        .ok_or_else(|| "no suitable WGPU adapter".to_string())?;

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("st.tensor.wgpu_conv.device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                    },
                    None,
                )
                .await
        })
        .map_err(|err| err.to_string())?;

        let device: Arc<Device> = Arc::new(device);
        let queue: Arc<Queue> = Arc::new(queue);

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("st.tensor.wgpu_conv.bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("st.tensor.wgpu_conv.pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("st.tensor.wgpu_conv.shader"),
            source: wgpu::ShaderSource::Wgsl(IM2COL_WGSL.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("st.tensor.wgpu_conv.pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_layout,
        })
    }

    fn device(&self) -> &Device {
        self.device.as_ref()
    }

    fn queue(&self) -> &Queue {
        self.queue.as_ref()
    }

    fn bind_group(&self, input: &Buffer, patches: &Buffer, params: &Buffer) -> BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_conv.bind_group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: patches.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }
}

static CONTEXT: OnceLock<Arc<ConvContext>> = OnceLock::new();

fn conv_context() -> Result<Arc<ConvContext>, String> {
    if let Some(ctx) = CONTEXT.get() {
        return Ok(ctx.clone());
    }
    let ctx = Arc::new(ConvContext::new()?);
    let _ = CONTEXT.set(ctx.clone());
    Ok(ctx)
}

pub fn is_available() -> bool {
    conv_context().is_ok()
}

pub fn should_use(batch: usize, oh: usize, ow: usize, kernel_elems: usize) -> bool {
    if batch == 0 || oh == 0 || ow == 0 || kernel_elems == 0 {
        return false;
    }

    let rows = batch.saturating_mul(oh).saturating_mul(ow);
    let volume = rows.saturating_mul(kernel_elems);

    volume >= 64 * 64 * 32
}

#[allow(clippy::too_many_arguments)]
pub fn im2col(
    input: &[f32],
    batch: usize,
    channels: usize,
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    oh: usize,
    ow: usize,
) -> Result<Vec<f32>, String> {
    if batch == 0
        || channels == 0
        || input_h == 0
        || input_w == 0
        || kernel_h == 0
        || kernel_w == 0
        || oh == 0
        || ow == 0
    {
        return Err("im2col dimensions must be positive".into());
    }

    let expected_input = batch
        .saturating_mul(channels)
        .saturating_mul(input_h)
        .saturating_mul(input_w);
    if input.len() != expected_input {
        return Err(format!(
            "input length mismatch: expected {} elements, got {}",
            expected_input,
            input.len()
        ));
    }

    let rows = batch * oh * ow;
    let kernel_elems = channels * kernel_h * kernel_w;
    let output_len = rows * kernel_elems;

    let ctx = conv_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_conv.input"),
        contents: bytemuck::cast_slice(input),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let patches_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.tensor.wgpu_conv.patches"),
        size: (output_len * std::mem::size_of::<f32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Im2colParams {
        dims0: [
            batch as u32,
            channels as u32,
            input_h as u32,
            input_w as u32,
        ],
        dims1: [
            kernel_h as u32,
            kernel_w as u32,
            stride_h as u32,
            stride_w as u32,
        ],
        dims2: [
            pad_h as i32,
            pad_w as i32,
            dilation_h as i32,
            dilation_w as i32,
        ],
        dims3: [oh as u32, ow as u32, kernel_elems as u32, (oh * ow) as u32],
        dims4: [
            (channels * input_h * input_w) as u32,
            (input_h * input_w) as u32,
            kernel_elems as u32,
            rows as u32,
        ],
    };

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_conv.params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = ctx.bind_group(&input_buf, &patches_buf, &params_buf);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_conv.encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_conv.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = 16u32;
        let wg_y = 4u32;
        let groups_x = ((kernel_elems as u32) + wg_x - 1) / wg_x;
        let groups_y = (((oh * ow) as u32) + wg_y - 1) / wg_y;
        let groups_z = batch as u32;
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), groups_z.max(1));
    }

    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &patches_buf, output_len)
}
