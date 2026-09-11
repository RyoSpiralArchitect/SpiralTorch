//! Private immutable snapshots of fixed, contiguous graph scratch buffers.
use super::*;

struct Slot {
    layout: NdLayout,
    source: wgpu::BindGroup,
    grid: [u32; 2],
}

/// Only source bindings and shape metadata persist. Every encode owns new values
/// and a new shared guard, so neither graph reuse nor another capture can mutate it.
pub(crate) struct PreparedCapture {
    device: TensorDevice,
    pipeline: wgpu::ComputePipeline,
    output_layout: wgpu::BindGroupLayout,
    slots: Vec<Slot>,
}

impl PreparedCapture {
    pub(crate) fn new(
        device: &TensorDevice,
        sources: &[(&NdLayout, &wgpu::Buffer)],
        upstream: &wgpu::Buffer,
    ) -> Result<Self, TensorError> {
        let gpu = device.runtime().context().device();
        let limits = gpu.limits();
        if sources.is_empty() {
            return Err(TensorError::Operands);
        }
        if limits.max_bind_groups < 2 || limits.max_uniform_buffer_binding_size < 16 {
            return Err(TensorError::Limit("graph capture pipeline"));
        }
        storage_limit((upstream.size() / 4) as usize, &limits)?;
        for (layout, values) in sources {
            validate_view(layout, (values.size() / 4) as usize, &limits)?;
            if !layout.is_contiguous() || layout.offset() != 0 {
                return Err(TensorError::Limit("graph capture requires packed storage"));
            }
            grid(layout.len(), &limits)?;
        }
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let source_layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph.capture.sources"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, wgpu::BufferBindingType::Uniform),
            ],
        });
        let output_layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph.capture.outputs"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("graph.capture.pipeline_layout"),
            bind_group_layouts: &[&source_layout, &output_layout],
            push_constant_ranges: &[],
        });
        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("graph.capture.shader"),
            source: wgpu::ShaderSource::Wgsl(
                substitute_ops(include_str!("../shaders/graph_capture.wgsl").replace(
                    "CHECKED_ELEMENTWISE",
                    include_str!("../shaders/checked_elementwise.wgsl"),
                ))
                .into(),
            ),
        });
        let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("graph.capture.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let mut slots = Vec::with_capacity(sources.len());
        for (layout, values) in sources {
            let [x, y, groups] = grid(layout.len(), &limits)?;
            let params = runtime::upload_slice(
                gpu,
                "graph.capture.shape",
                &[layout.len() as u32, x, groups, 0],
                wgpu::BufferUsages::UNIFORM,
            )?;
            let source = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("graph.capture.source"),
                layout: &source_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: upstream.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
            slots.push(Slot {
                layout: (*layout).clone(),
                source,
                grid: [x, y],
            });
        }
        Ok(Self {
            device: device.clone(),
            pipeline,
            output_layout,
            slots,
        })
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<Vec<ResidentTensor>, TensorError> {
        let gpu = self.device.runtime().context().device();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let flags = Shared::new(runtime::empty_buffer::<u32>(
            gpu,
            "graph.capture.flags",
            1,
            usage,
        )?);
        let mut outputs = Vec::with_capacity(self.slots.len());
        let mut bindings = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let values = runtime::empty_buffer::<f32>(
                gpu,
                "graph.capture.values",
                slot.layout.len().max(1),
                usage,
            )?;
            bindings.push(gpu.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("graph.capture.output"),
                layout: &self.output_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: flags.as_entire_binding(),
                    },
                ],
            }));
            outputs.push(ResidentTensor {
                storage: Shared::new(Storage {
                    values,
                    flags: flags.clone(),
                }),
                layout: slot.layout.clone(),
                device: self.device.clone(),
            });
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("graph.capture"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (slot, binding) in self.slots.iter().zip(&bindings) {
            pass.set_bind_group(0, &slot.source, &[]);
            pass.set_bind_group(1, binding, &[]);
            pass.dispatch_workgroups(slot.grid[0], slot.grid[1], 1);
        }
        drop(pass);
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests;
