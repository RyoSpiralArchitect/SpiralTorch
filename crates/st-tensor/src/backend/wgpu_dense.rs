// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::sync::{mpsc, Arc, OnceLock};

use bytemuck::{cast_slice, Pod, Zeroable};
use pollster::block_on;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatmulParams {
    rows: u32,
    inner: u32,
    cols: u32,
    _pad: u32,
}

struct DenseContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl DenseContext {
    fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|| {
            block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })
        .ok_or_else(|| "wgpu adapter unavailable".to_string())?;

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("st-tensor.linear.device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|err| err.to_string())?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("st-tensor.linear.matmul"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../wgpu_shaders/matmul.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("st-tensor.linear.layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("st-tensor.linear.pipeline"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("st-tensor.linear.matmul"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            layout,
            pipeline,
        })
    }
}

static CONTEXT: OnceLock<Result<DenseContext, String>> = OnceLock::new();

fn context() -> Result<&'static DenseContext, String> {
    match CONTEXT.get_or_init(DenseContext::new) {
        Ok(ctx) => Ok(ctx),
        Err(err) => Err(err.clone()),
    }
}

fn dispatch_dimensions(rows: u32, cols: u32) -> (u32, u32) {
    let x = (cols + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let y = (rows + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    (x.max(1), y.max(1))
}

pub fn is_available() -> bool {
    context().is_ok()
}

pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
    rows * cols >= 1024 && inner >= 16
}

pub fn matmul(
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if rows == 0 || cols == 0 || inner == 0 {
        return Ok(vec![0.0; rows * cols]);
    }

    let ctx = context()?;
    let device = &*ctx.device;
    let queue = &*ctx.queue;

    let lhs_bytes = (lhs.len() * std::mem::size_of::<f32>()) as u64;
    let rhs_bytes = (rhs.len() * std::mem::size_of::<f32>()) as u64;
    let out_bytes = (rows * cols * std::mem::size_of::<f32>()) as u64;

    let lhs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st-tensor.linear.lhs"),
        size: lhs_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let rhs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st-tensor.linear.rhs"),
        size: rhs_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st-tensor.linear.out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st-tensor.linear.stage"),
        size: out_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    queue.write_buffer(&lhs_buffer, 0, cast_slice(lhs));
    queue.write_buffer(&rhs_buffer, 0, cast_slice(rhs));

    let params = MatmulParams {
        rows: rows as u32,
        inner: inner as u32,
        cols: cols as u32,
        _pad: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st-tensor.linear.params"),
        contents: cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st-tensor.linear.bind"),
        layout: &ctx.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st-tensor.linear.encode"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st-tensor.linear.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y) = dispatch_dimensions(params.rows, params.cols);
        pass.dispatch_workgroups(x, y, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging_buffer, 0, out_bytes);

    queue.submit(Some(encoder.finish()));
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    match receiver
        .recv()
        .map_err(|_| "map_async callback dropped".to_string())?
    {
        Ok(()) => {}
        Err(err) => return Err(err.to_string()),
    }
    let data = slice.get_mapped_range();
    let mut result = vec![0.0f32; rows * cols];
    result.copy_from_slice(cast_slice(&data));
    drop(data);
    staging_buffer.unmap();

    Ok(result)
}
