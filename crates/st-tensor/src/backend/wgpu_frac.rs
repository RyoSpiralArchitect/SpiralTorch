// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-tensor/src/backend/wgpu_frac.rs
#![cfg(feature = "wgpu_frac")]
use crate::fractional::gl_coeffs;
use crate::util::readback_f32;
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use wgpu::*;

pub struct Frac1dKernel {
    pipeline: ComputePipeline,
    bind_layout: BindGroupLayout,
}

impl Frac1dKernel {
    pub fn new(device: &Device, shader_src: &str) -> Result<Self, String> {
        let module = catch_unwind(AssertUnwindSafe(|| {
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("frac_gl_1d"),
                source: ShaderSource::Wgsl(shader_src.into()),
            })
        }))
        .map_err(|payload| format!("WGSL parse error: {}", panic_payload_to_string(payload)))?;
        let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("frac_gl_1d_bind"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("frac_gl_1d_pl"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("frac_gl_1d"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Ok(Self {
            pipeline,
            bind_layout,
        })
    }

    pub fn dispatch(
        &self,
        device: &Device,
        queue: &Queue,
        x: &[f32],
        alpha: f32,
        h: f32,
        m: usize,
    ) -> Result<Vec<f32>, String> {
        let n = x.len();
        let m = m.min(n.saturating_sub(1));
        let w = gl_coeffs(alpha, m);
        let h_alpha = h.powf(alpha);

        use wgpu::util::DeviceExt;
        let xb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("X"),
            contents: bytemuck::cast_slice(x),
            usage: BufferUsages::STORAGE,
        });
        let wb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("W"),
            contents: bytemuck::cast_slice(&w),
            usage: BufferUsages::STORAGE,
        });
        let yb = device.create_buffer(&BufferDescriptor {
            label: Some("Y"),
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n: u32,
            m: u32,
            h_alpha: f32,
            _pad: f32,
        }
        let params = Params {
            n: n as u32,
            m: m as u32,
            h_alpha,
            _pad: 0.0,
        };
        let pb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("P"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("frac_gl_1d_bg"),
            layout: &self.bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: xb.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wb.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: yb.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pb.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("frac_gl_1d_enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("frac_gl_1d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let wg = 256u32;
            let n_groups = ((n as u32) + wg - 1) / wg;
            pass.dispatch_workgroups(n_groups, 1, 1);
        }
        queue.submit(Some(enc.finish()));

        readback_f32(device, queue, &yb, n)
    }
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}
