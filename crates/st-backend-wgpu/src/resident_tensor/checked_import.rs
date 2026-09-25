//! Validate externally produced GPU values without copying or mapping them.
use super::{grid, TensorError};
use crate::runtime::WgpuContext;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub(crate) struct CheckedImportKernels {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl CheckedImportKernels {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor.checked_import.layout"),
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
            label: Some("tensor.checked_import.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor.checked_import.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/checked_import.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tensor.checked_import"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Self { layout, pipeline }
    }

    pub(crate) fn check(
        &self,
        context: &WgpuContext,
        values: &wgpu::Buffer,
        flags: &wgpu::Buffer,
        len: usize,
    ) -> Result<(), TensorError> {
        let [x, y, _] = grid(len, &context.device().limits())?;
        let meta = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor.checked_import.meta"),
                contents: bytemuck::cast_slice(&[len as u32, x, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let binding = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tensor.checked_import.binding"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: flags.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: meta.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tensor.checked_import.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        context.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}
