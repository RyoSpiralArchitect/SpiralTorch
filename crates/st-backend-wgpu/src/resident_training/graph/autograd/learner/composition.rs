//! Reusable four-source composition. No per-update tensor-sized allocations.
use super::*;
use std::num::NonZeroU64;

const LANES: usize = 4;
const CHUNK_BYTES: u64 = 32;

pub(super) struct Composition {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    guard: wgpu::ComputePipeline,
    guard_group: wgpu::BindGroup,
    sources: wgpu::Buffer,
    flags: wgpu::Buffer,
    weights: wgpu::Buffer,
    stride: usize,
    shapes: Vec<(wgpu::Buffer, [u32; 3])>,
}

fn entry(binding: u32, uniform: bool, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if uniform {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    source: String,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
    })
}

impl Composition {
    pub(super) fn new(g: &ResidentGraphTraining) -> Result<Self, TrainingError> {
        let device = g.device.runtime().context().device();
        let limits = device.limits();
        let stride = (CHUNK_BYTES as usize)
            .div_ceil(limits.min_uniform_buffer_offset_alignment as usize)
            * limits.min_uniform_buffer_offset_alignment as usize;
        let bytes = stride
            .checked_mul(MAX_TERMS / LANES)
            .ok_or(TrainingError::Overflow)?;
        if bytes as u64 > limits.max_buffer_size
            || limits.max_uniform_buffer_binding_size < CHUNK_BYTES as u32
        {
            return Err(MatmulError::DeviceLimit("gradient composition uniforms").into());
        }
        storage_limit(MAX_TERMS, &limits)?;
        let weights = runtime::empty_buffer::<u8>(
            device,
            "learner.weights",
            bytes,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        let sources = runtime::empty_buffer::<u32>(
            device,
            "learner.source_flags",
            MAX_TERMS,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        let flags = runtime::empty_buffer::<u32>(
            device,
            "learner.composition_flags",
            1,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("learner.composition"),
            entries: &(0..8).map(|i| entry(i, i >= 6, i < 4)).collect::<Vec<_>>(),
        });
        let source = crate::resident_tensor::substitute_ops(
            include_str!("../../../../shaders/gradient_composition.wgsl").replace(
                "CHECKED_ELEMENTWISE",
                include_str!("../../../../shaders/checked_elementwise.wgsl"),
            ),
        );
        let pipeline = pipeline(device, "learner.composition", &layout, source);
        let guard_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("learner.source_guard"),
            entries: &[
                entry(0, false, true),
                entry(1, false, false),
                entry(2, true, true),
            ],
        });
        let guard = self::pipeline(
            device,
            "learner.source_guard",
            &guard_layout,
            include_str!("../../../../shaders/gradient_source_guard.wgsl").to_owned(),
        );
        let guard_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("learner.source_guard"),
            layout: &guard_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sources.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &weights,
                        offset: 0,
                        size: NonZeroU64::new(CHUNK_BYTES),
                    }),
                },
            ],
        });
        let shapes = g
            .raw_gradients
            .iter()
            .map(|p| {
                let n = (p.size() / 4) as usize;
                let grid = groups(n, &limits)?;
                Ok((
                    runtime::upload_slice(
                        device,
                        "learner.parameter_shape",
                        &[n as u32, grid[0], 0, 0],
                        wgpu::BufferUsages::UNIFORM,
                    )?,
                    grid,
                ))
            })
            .collect::<Result<_, TrainingError>>()?;
        Ok(Self {
            layout,
            pipeline,
            guard,
            guard_group,
            sources,
            flags,
            weights,
            stride,
            shapes,
        })
    }

    pub(super) fn encode(
        &self,
        g: &ResidentGraphTraining,
        terms: &[(&GraphGradients, f32)],
        encoder: &mut wgpu::CommandEncoder,
    ) -> Vec<u32> {
        let device = g.device.runtime().context().device();
        let stride = self.stride / 4;
        let mut weights = vec![0u32; terms.len().div_ceil(LANES) * stride];
        for (chunk, sources) in terms.chunks(LANES).enumerate() {
            for (i, (_, weight)) in sources.iter().enumerate() {
                weights[chunk * stride + i] = weight.to_bits();
            }
            weights[chunk * stride + 4] = (chunk * LANES) as u32;
            weights[chunk * stride + 5] = sources.len() as u32;
            weights[chunk * stride + 6] = terms.len() as u32;
        }
        encoder.clear_buffer(&self.flags, 0, None);
        // Entire VJP flags are independent of parameters, including zero weights.
        for (i, (source, _)) in terms.iter().enumerate() {
            encoder.copy_buffer_to_buffer(source.input.flags(), 0, &self.sources, i as u64 * 4, 4);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("learner.source_guard"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.guard);
            pass.set_bind_group(0, &self.guard_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        for (id, output) in g.raw_gradients.iter().enumerate() {
            if terms.len() == 1 && terms[0].1 == 1. {
                encoder.copy_buffer_to_buffer(
                    terms[0].0.parameters[id].values(),
                    0,
                    output,
                    0,
                    output.size(),
                );
                continue;
            }
            let (shape, grid) = &self.shapes[id];
            for (chunk, sources) in terms.chunks(LANES).enumerate() {
                let mut entries: Vec<_> = (0..LANES)
                    .map(|slot| wgpu::BindGroupEntry {
                        binding: slot as u32,
                        resource: sources[slot.min(sources.len() - 1)].0.parameters[id]
                            .values()
                            .as_entire_binding(),
                    })
                    .collect();
                entries.extend([
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.flags.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: shape.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.weights,
                            offset: (chunk * self.stride) as u64,
                            size: NonZeroU64::new(CHUNK_BYTES),
                        }),
                    },
                ]);
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("learner.composition"),
                    layout: &self.layout,
                    entries: &entries,
                });
                // Separate passes preserve read/write dependencies across chunks.
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("learner.composition"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(grid[0], grid[1], 1);
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.flags,
            0,
            &g.validation,
            (g.nodes.len() + 2) as u64 * 4,
            4,
        );
        weights
    }

    pub(super) fn write_weights(&self, queue: &wgpu::Queue, values: &[u32]) {
        queue.write_buffer(&self.weights, 0, bytemuck::cast_slice(values));
    }
}
