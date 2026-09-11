//! Freeze all checked NN stage flags without copying the result values.
use super::*;

pub(crate) struct GuardCapture {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl GuardCapture {
    pub(crate) fn new(gpu: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..2)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding == 0,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph.guard_capture.bindings"),
            entries: &entries,
        });
        let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("graph.guard_capture.layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("graph.guard_capture.shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/graph_guard_capture.wgsl")
                    .replace("INVALID_TENSOR_FLAG", &format!("{INVALID_TENSOR_FLAG}u"))
                    .into(),
            ),
        });
        let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("graph.guard_capture.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Self { layout, pipeline }
    }

    pub(crate) fn bind(
        &self,
        gpu: &wgpu::Device,
        upstream: &wgpu::Buffer,
        destination: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("graph.guard_capture"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: upstream.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: destination.as_entire_binding(),
                },
            ],
        })
    }

    /// Upstream stages have finished; this output version is not externally
    /// visible yet. atomicStore overwrites a recycled destination's entire guard.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, binding: &wgpu::BindGroup) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("graph.guard_capture.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, binding, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
